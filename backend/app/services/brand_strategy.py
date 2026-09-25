"""Brand Strategy agent: business rules and transaction boundaries.

Like Customer Personas and Content Planning, this agent does not run the
CRAG graph - it needs no external grounding, only synthesis over reports
that already did their own grounding (Business Understanding, Competitor
Analysis, and Customer Personas). Same shape as CustomerPersonasService:
one direct structured LLM call via LLMGateway.structured_with_usage(),
no embeddings or web-search gateway dependency at all.

generate() hard-requires the business profile, its Business
Understanding, Competitor Analysis, and Customer Personas to already
exist for the workspace - the roadmap's "needs to know the customer and
the competition before it can position against them." Requiring
Business Understanding already implies the profile exists (it is keyed
by profile_id), same reasoning CustomerPersonasService used.

The real dependency graph is deeper than these four direct checks:
Customer Personas is itself downstream of Market Research (see
CustomerPersonasService's docstring), so generating a brand strategy
transitively requires Market Research to have existed at some point too.
This service does not re-check Market Research directly, though - it
only needs the Personas record to exist right now, for the same reason
Personas does not re-check Business Understanding's freshness: a
generated artifact is a self-contained document, not a live view over
its inputs.
"""

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from sqlalchemy.orm.exc import StaleDataError

from backend.app.config import Settings
from backend.app.db.errors import DatabaseOperationError
from backend.app.db.models.brand_strategy import BrandStrategy
from backend.app.db.models.business_profile import BusinessProfile
from backend.app.db.models.business_understanding import BusinessUnderstanding
from backend.app.db.models.competitor_analysis import CompetitorAnalysis
from backend.app.db.models.customer_persona_set import CustomerPersonaSet
from backend.app.db.repositories.brand_strategy import BrandStrategyRepository
from backend.app.db.repositories.business_profiles import BusinessProfileRepository
from backend.app.db.repositories.business_understanding import (
    BusinessUnderstandingRepository,
)
from backend.app.db.repositories.competitor_analysis import CompetitorAnalysisRepository
from backend.app.db.repositories.customer_persona_set import CustomerPersonaSetRepository
from backend.app.db.repositories.workspaces import WorkspaceRepository
from backend.app.exceptions import ResourceNotFoundError, StaleResourceError
from backend.app.llm.gateway import LLMGateway, LLMUsage


class _BrandStrategyDraft(BaseModel):
    """LLM-facing output schema.

    Field names must stay identical to the content fields on
    BrandStrategyResponse (excluding id/workspace_id/model_used/
    token counts/version/timestamps, which are never LLM output);
    tests/test_brand_strategy_draft.py enforces that so the two
    cannot drift apart silently.

    brand_pillars and tagline_options deliberately have no min_length,
    matching every other list field on every other agent's draft
    schema: the deterministic "local" provider fills every list-typed
    field with [] regardless of any Field constraints, and a
    min_length would make every local-backed call fail Pydantic
    validation before it ever reaches the service. The max_length caps
    only bound the worst case.
    """

    model_config = ConfigDict(extra="forbid")

    overview: str = Field(min_length=1, max_length=1000)
    positioning_statement: str = Field(min_length=1, max_length=500)
    value_proposition: str = Field(min_length=1, max_length=600)
    brand_voice_and_tone: str = Field(min_length=1, max_length=800)
    brand_pillars: list[str] = Field(default_factory=list, max_length=6)
    tagline_options: list[str] = Field(default_factory=list, max_length=5)


class BrandStrategyService:
    """Coordinate Brand Strategy generation and persistence."""

    def __init__(
        self,
        session: Session,
        gateway: LLMGateway,
        settings: Settings,
    ) -> None:
        self._session = session
        self._gateway = gateway
        self._settings = settings
        self._brand_strategies = BrandStrategyRepository(session)
        self._profiles = BusinessProfileRepository(session)
        self._understandings = BusinessUnderstandingRepository(session)
        self._competitor_analyses = CompetitorAnalysisRepository(session)
        self._persona_sets = CustomerPersonaSetRepository(session)
        self._workspaces = WorkspaceRepository(session)

    def generate(
        self,
        workspace_id: UUID,
        *,
        force_regenerate: bool = False,
    ) -> BrandStrategy:
        """Return the current brand strategy, generating one if needed.

        If a brand strategy already exists and force_regenerate is
        False, the existing record is returned without calling the LLM
        at all - checked *before* requiring any prerequisite to exist,
        so an already-generated strategy stays readable even if the
        reports it was built from are later regenerated or removed.
        """

        self._require_workspace(workspace_id)

        existing = self._brand_strategies.get_by_workspace(workspace_id)
        if existing is not None and not force_regenerate:
            return existing

        profile = self._profiles.get_by_workspace(workspace_id)
        if profile is None:
            raise ResourceNotFoundError("Business profile not found.")

        understanding = self._understandings.get_by_business_profile(profile.id)
        if understanding is None:
            raise ResourceNotFoundError("Business understanding has not been generated yet.")

        analysis = self._competitor_analyses.get_by_workspace(workspace_id)
        if analysis is None:
            raise ResourceNotFoundError("Competitor analysis has not been generated yet.")

        persona_set = self._persona_sets.get_by_workspace(workspace_id)
        if persona_set is None:
            raise ResourceNotFoundError("Customer personas have not been generated yet.")

        prompt = self._build_prompt(profile, understanding, analysis, persona_set)
        draft, usage = self._gateway.structured_with_usage(prompt, _BrandStrategyDraft)

        brand_strategy = (
            existing if existing is not None else BrandStrategy(workspace_id=workspace_id)
        )
        self._apply_draft(brand_strategy, draft, usage)

        if existing is None:
            self._brand_strategies.add(brand_strategy)

        try:
            self._session.commit()
        except StaleDataError as exc:
            self._session.rollback()
            raise StaleResourceError(
                "Brand strategy changed while it was being regenerated."
            ) from exc
        except SQLAlchemyError as exc:
            self._session.rollback()
            raise DatabaseOperationError("Brand strategy generation failed.") from exc

        self._session.refresh(brand_strategy)
        return brand_strategy

    def get(self, workspace_id: UUID) -> BrandStrategy:
        """Return the current brand strategy for a workspace.

        Deliberately does not require any prerequisite report to still
        exist - a generated brand strategy is a self-contained,
        already-written document, not a live view over the reports it
        originally synthesized.
        """

        self._require_workspace(workspace_id)

        brand_strategy = self._brand_strategies.get_by_workspace(workspace_id)
        if brand_strategy is None:
            raise ResourceNotFoundError("Brand strategy has not been generated yet.")

        return brand_strategy

    def _require_workspace(self, workspace_id: UUID) -> None:
        """Ensure the parent workspace exists."""

        workspace = self._workspaces.get(workspace_id)
        if workspace is None:
            raise ResourceNotFoundError("Workspace not found.")

    def _build_prompt(
        self,
        profile: BusinessProfile,
        understanding: BusinessUnderstanding,
        analysis: CompetitorAnalysis,
        persona_set: CustomerPersonaSet,
    ) -> str:
        """Build the positioning-synthesis prompt from existing reports.

        Like CustomerPersonasService._build_prompt(), this pastes full
        text rather than excerpts: there is no CRAG rewrite step here to
        lose anything to, and the model should see the whole competitive
        and customer picture it is positioning against. Each field sits
        on its own labeled line rather than being stitched into a
        hand-written sentence, sidestepping the stray-period/article-
        grammar bug class documented for every prior query-builder.
        """

        lines = [
            "You are defining a brand strategy for a business, grounded "
            "in its researched competitors and customer personas. Do not "
            "invent positioning unrelated to the material below - every "
            "claim must be defensible against the competitive "
            "differentiation opportunities and customer pain points "
            "provided.",
            "Produce: an overview (a few sentences on which positioning "
            "direction you chose and why), a positioning_statement in the "
            "form 'For [target customer], [brand] is the [category] that "
            "[key benefit], because [reason to believe]', a "
            "value_proposition (2-4 sentences), brand_voice_and_tone "
            "(personality plus concrete writing-style guidance), "
            "brand_pillars (3-5 short, distinct core themes), and "
            "tagline_options (3-5 short candidate taglines).",
            "",
            "Business profile:",
        ]

        profile_fields = {
            "Business name": profile.business_name,
            "Product or service": profile.product_or_service,
            "Industry": profile.industry,
            "Target market": profile.target_market,
            "Target customer": profile.target_customer,
            "Price range": profile.price_range,
            "Existing brand tone": profile.brand_tone,
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
                "Competitor analysis - overview:",
                analysis.competitor_overview,
                "",
                "Competitor analysis - strengths and weaknesses:",
                analysis.strengths_and_weaknesses,
                "",
                "Competitor analysis - pricing and positioning:",
                analysis.pricing_and_positioning,
                "",
                "Competitor analysis - differentiation opportunities:",
                analysis.differentiation_opportunities,
                "",
                "Customer personas - overview:",
                persona_set.overview,
                "",
                "Customer personas:",
            ]
        )

        for persona in persona_set.personas:
            lines.extend(
                [
                    f"- {persona['name']} ({persona['segment']}):",
                    f"  Summary: {persona['summary']}",
                    f"  Goals: {', '.join(persona['goals'])}",
                    f"  Pain points: {', '.join(persona['pain_points'])}",
                    f"  Objections: {', '.join(persona['objections'])}",
                ]
            )

        return "\n".join(lines)

    def _apply_draft(
        self,
        brand_strategy: BrandStrategy,
        draft: _BrandStrategyDraft,
        usage: LLMUsage,
    ) -> None:
        """Copy a generated draft plus usage metadata onto an ORM record."""

        brand_strategy.overview = draft.overview
        brand_strategy.positioning_statement = draft.positioning_statement
        brand_strategy.value_proposition = draft.value_proposition
        brand_strategy.brand_voice_and_tone = draft.brand_voice_and_tone
        brand_strategy.brand_pillars = list(draft.brand_pillars)
        brand_strategy.tagline_options = list(draft.tagline_options)
        brand_strategy.model_used = self._settings.llm_model
        brand_strategy.input_tokens = usage.input_tokens
        brand_strategy.output_tokens = usage.output_tokens
