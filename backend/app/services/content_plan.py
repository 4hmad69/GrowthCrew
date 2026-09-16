"""Content Planning agent: business rules and transaction boundaries.

Unlike Market Research, Competitor Analysis, and Marketing Strategy, this
agent does not run the CRAG graph - it doesn't need external grounding,
only synthesis over a marketing strategy that already exists. Same shape
as BusinessUnderstandingService: one direct structured LLM call via
LLMGateway.structured_with_usage(), no embeddings or web-search gateway
dependency at all.

generate() hard-requires a Marketing Strategy to already exist for the
workspace - the same "no partial-data fallback" reasoning Marketing
Strategy itself used for its own three prerequisites. Deliberately does
NOT additionally require the business profile: the workspace's own name
(always present once the workspace itself exists) is enough context, and
requiring the profile too would reintroduce exactly the kind of extra,
avoidable failure path Marketing Strategy already sidesteps by carrying
no foreign key to any of its own inputs.
"""

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from sqlalchemy.orm.exc import StaleDataError

from backend.app.config import Settings
from backend.app.db.errors import DatabaseOperationError
from backend.app.db.models.content_plan import ContentPlan
from backend.app.db.models.marketing_strategy import MarketingStrategy
from backend.app.db.models.workspace import Workspace
from backend.app.db.repositories.content_plan import ContentPlanRepository
from backend.app.db.repositories.marketing_strategy import MarketingStrategyRepository
from backend.app.db.repositories.workspaces import WorkspaceRepository
from backend.app.exceptions import ResourceNotFoundError, StaleResourceError
from backend.app.llm.gateway import LLMGateway, LLMUsage


class _ContentCalendarEntryDraft(BaseModel):
    """LLM-facing schema for one scheduled calendar entry."""

    model_config = ConfigDict(extra="forbid")

    day: int = Field(ge=1, le=30)
    week: int = Field(ge=1, le=4)
    channel: str = Field(min_length=1, max_length=80)
    content_type: str = Field(min_length=1, max_length=80)
    pillar: str = Field(min_length=1, max_length=160)
    topic: str = Field(min_length=1, max_length=300)
    cta: str = Field(min_length=1, max_length=200)


class _ContentPlanDraft(BaseModel):
    """LLM-facing output schema.

    entries deliberately has no min_length. LocalStructuredRunnable (the
    deterministic "local" provider Postgres integration tests use) fills
    every list-typed field with [] regardless of the nested schema or any
    Field constraints on it - a min_length here would make every
    "local"-backed call fail Pydantic validation before it ever reaches
    the service, the same way every other agent's list fields
    (key_differentiators, likely_customer_pain_points, ...) already avoid
    min_length for exactly this reason. The real 30-entry target is
    stated in the prompt and verified against genuine Ollama Cloud output
    in test_content_plan_llm_integration.py - the same place every prior
    agent's real generation quality is verified, not hard-coded into the
    LLM-facing schema.
    """

    model_config = ConfigDict(extra="forbid")

    overview: str = Field(min_length=1, max_length=1000)
    entries: list[_ContentCalendarEntryDraft] = Field(default_factory=list, max_length=30)


class ContentPlanService:
    """Coordinate Content Planning generation and persistence."""

    def __init__(
        self,
        session: Session,
        gateway: LLMGateway,
        settings: Settings,
    ) -> None:
        self._session = session
        self._gateway = gateway
        self._settings = settings
        self._plans = ContentPlanRepository(session)
        self._strategies = MarketingStrategyRepository(session)
        self._workspaces = WorkspaceRepository(session)

    def generate(
        self,
        workspace_id: UUID,
        *,
        force_regenerate: bool = False,
    ) -> ContentPlan:
        """Return the current content plan, generating one if needed.

        If a plan already exists and force_regenerate is False, the
        existing record is returned without calling the LLM at all -
        checked *before* requiring a marketing strategy to exist, so an
        already-generated plan stays readable even if the strategy it
        was built from is later regenerated or removed.
        """

        workspace = self._require_workspace(workspace_id)

        existing = self._plans.get_by_workspace(workspace_id)
        if existing is not None and not force_regenerate:
            return existing

        strategy = self._strategies.get_by_workspace(workspace_id)
        if strategy is None:
            raise ResourceNotFoundError("Marketing strategy has not been generated yet.")

        prompt = self._build_prompt(workspace, strategy)
        draft, usage = self._gateway.structured_with_usage(prompt, _ContentPlanDraft)

        plan = existing if existing is not None else ContentPlan(workspace_id=workspace_id)
        self._apply_draft(plan, draft, usage)

        if existing is None:
            self._plans.add(plan)

        try:
            self._session.commit()
        except StaleDataError as exc:
            self._session.rollback()
            raise StaleResourceError(
                "Content plan changed while it was being regenerated."
            ) from exc
        except SQLAlchemyError as exc:
            self._session.rollback()
            raise DatabaseOperationError("Content plan generation failed.") from exc

        self._session.refresh(plan)
        return plan

    def get(self, workspace_id: UUID) -> ContentPlan:
        """Return the current content plan for a workspace.

        Deliberately does not require the marketing strategy to still
        exist - a generated plan is a self-contained, already-written
        document, not a live view over the strategy it originally
        synthesized.
        """

        self._require_workspace(workspace_id)

        plan = self._plans.get_by_workspace(workspace_id)
        if plan is None:
            raise ResourceNotFoundError("Content plan has not been generated yet.")

        return plan

    def _require_workspace(self, workspace_id: UUID) -> Workspace:
        """Ensure the parent workspace exists, and return it."""

        workspace = self._workspaces.get(workspace_id)
        if workspace is None:
            raise ResourceNotFoundError("Workspace not found.")
        return workspace

    def _build_prompt(self, workspace: Workspace, strategy: MarketingStrategy) -> str:
        """Build the calendar-synthesis prompt from an existing marketing strategy.

        Unlike MarketingStrategyService's _build_queries(), this pastes full
        section text directly into the prompt rather than a short excerpt.
        That discipline existed there because the CRAG graph's rewrite_query
        node compresses (and can discard) anything not distilled before
        generation ever sees it; there is no such node here - this is one
        direct structured call, so the model should see the whole strategy
        it is turning into a calendar. Each field sits on its own labeled
        line (matching BusinessUnderstandingService._build_prompt's style),
        not stitched into a hand-written sentence, so this sidesteps the
        stray-period/article-grammar bug class entirely.
        """

        return (
            "You are turning an existing marketing strategy into a concrete, "
            "actionable 30-day content calendar. Do not invent a new "
            "strategy - synthesize it into a day-by-day plan. Produce "
            "exactly 30 entries, one per day, that put the strategy below "
            "into action. Every entry needs: day (1-30), week (1-4), "
            "channel, content_type (e.g. Reel, Carousel, Blog post, Email, "
            "Story), pillar (which content pillar it serves), topic "
            "(specific, not generic), and cta (the call to action).\n\n"
            f"Business: {workspace.name}\n\n"
            "Recommended channels and tactics:\n"
            f"{strategy.recommended_channels_and_tactics}\n\n"
            "Content and messaging pillars:\n"
            f"{strategy.content_and_messaging_pillars}\n\n"
            "90-day roadmap (use only what applies to the first 30 days):\n"
            f"{strategy.ninety_day_roadmap}\n\n"
            "Budget allocation and KPIs:\n"
            f"{strategy.budget_allocation_and_kpis}"
        )

    def _apply_draft(
        self,
        plan: ContentPlan,
        draft: _ContentPlanDraft,
        usage: LLMUsage,
    ) -> None:
        """Copy a generated draft plus usage metadata onto an ORM record."""

        plan.overview = draft.overview
        plan.entries = [entry.model_dump() for entry in draft.entries]
        plan.model_used = self._settings.llm_model
        plan.input_tokens = usage.input_tokens
        plan.output_tokens = usage.output_tokens