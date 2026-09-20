"""Customer Personas agent: business rules and transaction boundaries.

Like Content Planning, this agent does not run the CRAG graph - it needs
no external grounding, only synthesis over reports that already exist.
Same shape as BusinessUnderstandingService and ContentPlanService: one
direct structured LLM call via LLMGateway.structured_with_usage(), no
embeddings or web-search gateway dependency at all.

generate() hard-requires the business profile, its Business
Understanding, and Market Research to already exist for the workspace -
the same "no partial-data fallback" reasoning Marketing Strategy used
for its own prerequisites. Personas are grounded in Market Research's
target customer segments and Business Understanding's inferred pain
points; without them the model would be inventing customers instead of
turning researched segments into usable personas. Requiring Business
Understanding already implies the profile exists (it is keyed by
profile_id), so unlike Content Planning this adds no extra, avoidable
failure path.
"""

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from sqlalchemy.orm.exc import StaleDataError

from backend.app.config import Settings
from backend.app.db.errors import DatabaseOperationError
from backend.app.db.models.business_profile import BusinessProfile
from backend.app.db.models.business_understanding import BusinessUnderstanding
from backend.app.db.models.customer_persona_set import CustomerPersonaSet
from backend.app.db.models.market_research import MarketResearch
from backend.app.db.repositories.business_profiles import BusinessProfileRepository
from backend.app.db.repositories.business_understanding import (
    BusinessUnderstandingRepository,
)
from backend.app.db.repositories.customer_persona_set import CustomerPersonaSetRepository
from backend.app.db.repositories.market_research import MarketResearchRepository
from backend.app.db.repositories.workspaces import WorkspaceRepository
from backend.app.exceptions import ResourceNotFoundError, StaleResourceError
from backend.app.llm.gateway import LLMGateway, LLMUsage


class _PersonaDraft(BaseModel):
    """LLM-facing schema for one customer persona.

    Field names must stay identical to PersonaSchema (the read-side
    shape); tests/test_customer_personas_draft.py enforces that so the
    two cannot drift apart silently.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=120)
    segment: str = Field(min_length=1, max_length=200)
    summary: str = Field(min_length=1, max_length=600)
    demographics: str = Field(min_length=1, max_length=300)
    goals: list[str] = Field(default_factory=list, max_length=6)
    pain_points: list[str] = Field(default_factory=list, max_length=6)
    buying_triggers: list[str] = Field(default_factory=list, max_length=6)
    objections: list[str] = Field(default_factory=list, max_length=6)
    preferred_channels: list[str] = Field(default_factory=list, max_length=6)


class _CustomerPersonasDraft(BaseModel):
    """LLM-facing output schema.

    personas deliberately has no min_length, and neither do the list
    fields on _PersonaDraft. LocalStructuredRunnable (the deterministic
    "local" provider Postgres integration tests use) fills every
    list-typed field with [] regardless of any Field constraints - a
    min_length would make every "local"-backed call fail Pydantic
    validation before it ever reaches the service, the same reason
    ContentPlan's entries and every other agent's list fields avoid it.
    The real "exactly 3 personas" target is stated in the prompt and
    verified against genuine Ollama Cloud output in the LLM integration
    test, not hard-coded into the schema. The max_length caps only bound
    the worst case.
    """

    model_config = ConfigDict(extra="forbid")

    overview: str = Field(min_length=1, max_length=1000)
    personas: list[_PersonaDraft] = Field(default_factory=list, max_length=5)


class CustomerPersonasService:
    """Coordinate Customer Personas generation and persistence."""

    def __init__(
        self,
        session: Session,
        gateway: LLMGateway,
        settings: Settings,
    ) -> None:
        self._session = session
        self._gateway = gateway
        self._settings = settings
        self._persona_sets = CustomerPersonaSetRepository(session)
        self._profiles = BusinessProfileRepository(session)
        self._understandings = BusinessUnderstandingRepository(session)
        self._research = MarketResearchRepository(session)
        self._workspaces = WorkspaceRepository(session)

    def generate(
        self,
        workspace_id: UUID,
        *,
        force_regenerate: bool = False,
    ) -> CustomerPersonaSet:
        """Return the current persona set, generating one if needed.

        If a persona set already exists and force_regenerate is False,
        the existing record is returned without calling the LLM at all -
        checked *before* requiring any prerequisite to exist, so an
        already-generated set stays readable even if the reports it was
        built from are later regenerated or removed.
        """

        self._require_workspace(workspace_id)

        existing = self._persona_sets.get_by_workspace(workspace_id)
        if existing is not None and not force_regenerate:
            return existing

        profile = self._profiles.get_by_workspace(workspace_id)
        if profile is None:
            raise ResourceNotFoundError("Business profile not found.")

        understanding = self._understandings.get_by_business_profile(profile.id)
        if understanding is None:
            raise ResourceNotFoundError("Business understanding has not been generated yet.")

        research = self._research.get_by_workspace(workspace_id)
        if research is None:
            raise ResourceNotFoundError("Market research has not been generated yet.")

        prompt = self._build_prompt(profile, understanding, research)
        draft, usage = self._gateway.structured_with_usage(prompt, _CustomerPersonasDraft)

        persona_set = (
            existing if existing is not None else CustomerPersonaSet(workspace_id=workspace_id)
        )
        self._apply_draft(persona_set, draft, usage)

        if existing is None:
            self._persona_sets.add(persona_set)

        try:
            self._session.commit()
        except StaleDataError as exc:
            self._session.rollback()
            raise StaleResourceError(
                "Customer personas changed while they were being regenerated."
            ) from exc
        except SQLAlchemyError as exc:
            self._session.rollback()
            raise DatabaseOperationError("Customer personas generation failed.") from exc

        self._session.refresh(persona_set)
        return persona_set

    def get(self, workspace_id: UUID) -> CustomerPersonaSet:
        """Return the current persona set for a workspace.

        Deliberately does not require any prerequisite report to still
        exist - a generated persona set is a self-contained,
        already-written document, not a live view over the reports it
        originally synthesized.
        """

        self._require_workspace(workspace_id)

        persona_set = self._persona_sets.get_by_workspace(workspace_id)
        if persona_set is None:
            raise ResourceNotFoundError("Customer personas have not been generated yet.")

        return persona_set

    def _require_workspace(self, workspace_id: UUID) -> None:
        """Ensure the parent workspace exists."""

        workspace = self._workspaces.get(workspace_id)
        if workspace is None:
            raise ResourceNotFoundError("Workspace not found.")

    def _build_prompt(
        self,
        profile: BusinessProfile,
        understanding: BusinessUnderstanding,
        research: MarketResearch,
    ) -> str:
        """Build the persona-synthesis prompt from existing reports.

        Like ContentPlanService._build_prompt(), this pastes full text
        rather than excerpts: there is no CRAG rewrite step here to lose
        anything to, and the model should see the whole set of researched
        customer segments it is turning into personas. Each field sits on
        its own labeled line (matching BusinessUnderstandingService's
        style) rather than being stitched into a hand-written sentence,
        sidestepping the stray-period/article-grammar bug class.
        """

        lines = [
            "You are turning researched customer segments into concrete, "
            "usable customer personas for a business. Do not invent "
            "customers unrelated to the research below - ground every "
            "persona in the target customer segments and pain points "
            "provided.",
            "Produce exactly 3 personas. Put the primary (highest-value) "
            "persona first, and make the three clearly distinct from one "
            "another.",
            "Every persona needs: name (a short fictional first name plus a "
            "descriptor), segment (which researched customer segment it "
            "represents), summary (2-3 sentences), demographics (one line), "
            "goals, pain_points, buying_triggers, objections, and "
            "preferred_channels (specific platforms, not generic advice). "
            "Give each list 3-5 short items, one sentence at most.",
            "Also write an overview: which persona is primary and why, in a few sentences.",
            "",
            "Business profile:",
        ]

        profile_fields = {
            "Business name": profile.business_name,
            "Product or service": profile.product_or_service,
            "Industry": profile.industry,
            "Country": profile.country,
            "Target market": profile.target_market,
            "Target customer": profile.target_customer,
            "Price range": profile.price_range,
        }
        for label, value in profile_fields.items():
            if value:
                lines.append(f"- {label}: {value}")

        lines.extend(
            [
                "",
                "Business understanding:",
                f"- Summary: {understanding.summary}",
                "- Key differentiators:",
                *[f"  - {item}" for item in understanding.key_differentiators],
                "- Likely customer pain points:",
                *[f"  - {item}" for item in understanding.likely_customer_pain_points],
                "",
                "Market research - target customer segments:",
                research.target_customer_segments,
            ]
        )

        return "\n".join(lines)

    def _apply_draft(
        self,
        persona_set: CustomerPersonaSet,
        draft: _CustomerPersonasDraft,
        usage: LLMUsage,
    ) -> None:
        """Copy a generated draft plus usage metadata onto an ORM record."""

        persona_set.overview = draft.overview
        persona_set.personas = [persona.model_dump() for persona in draft.personas]
        persona_set.model_used = self._settings.llm_model
        persona_set.input_tokens = usage.input_tokens
        persona_set.output_tokens = usage.output_tokens
