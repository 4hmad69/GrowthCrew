"""Business Understanding agent: business rules and transaction boundaries.

Builds a prompt from an existing business profile, calls the LLM gateway
for a synthesis (not a restatement of fields already on the form), and
persists the result with optimistic concurrency, mirroring
BusinessProfileService's transaction-boundary pattern exactly.
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
from backend.app.db.repositories.business_profiles import BusinessProfileRepository
from backend.app.db.repositories.business_understanding import (
    BusinessUnderstandingRepository,
)
from backend.app.db.repositories.workspaces import WorkspaceRepository
from backend.app.exceptions import ResourceNotFoundError, StaleResourceError
from backend.app.llm.gateway import LLMGateway, LLMUsage


class _BusinessUnderstandingDraft(BaseModel):
    """LLM-facing output schema.

    The service adds the ID, profile link, model/token metadata, and
    timestamps afterward - the LLM only produces what it's actually
    synthesizing, not database bookkeeping.
    """

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1, max_length=2000)
    inferred_business_stage: str = Field(min_length=1, max_length=160)
    competitive_category: str = Field(min_length=1, max_length=160)
    key_differentiators: list[str] = Field(default_factory=list, max_length=10)
    likely_customer_pain_points: list[str] = Field(default_factory=list, max_length=10)
    confidence_notes: str | None = Field(default=None, max_length=1000)


class BusinessUnderstandingService:
    """Coordinate Business Understanding generation and persistence."""

    def __init__(
        self,
        session: Session,
        gateway: LLMGateway,
        settings: Settings,
    ) -> None:
        self._session = session
        self._gateway = gateway
        self._settings = settings
        self._understandings = BusinessUnderstandingRepository(session)
        self._profiles = BusinessProfileRepository(session)
        self._workspaces = WorkspaceRepository(session)

    def generate(
        self,
        workspace_id: UUID,
        *,
        force_regenerate: bool = False,
    ) -> BusinessUnderstanding:
        """Return the current understanding, generating one if needed.

        If an understanding already exists and force_regenerate is False,
        the existing record is returned without calling the LLM at all -
        generation costs real tokens, so nothing gets re-generated just
        because it was asked for again.
        """

        self._require_workspace(workspace_id)

        profile = self._profiles.get_by_workspace(workspace_id)
        if profile is None:
            raise ResourceNotFoundError("Business profile not found.")

        existing = self._understandings.get_by_business_profile(profile.id)
        if existing is not None and not force_regenerate:
            return existing

        prompt = self._build_prompt(profile)
        draft, usage = self._gateway.structured_with_usage(prompt, _BusinessUnderstandingDraft)

        understanding = (
            existing if existing is not None else BusinessUnderstanding(profile_id=profile.id)
        )
        self._apply_draft(understanding, draft, usage)

        if existing is None:
            self._understandings.add(understanding)

        try:
            self._session.commit()
        except StaleDataError as exc:
            self._session.rollback()
            raise StaleResourceError(
                "Business understanding changed while it was being regenerated."
            ) from exc
        except SQLAlchemyError as exc:
            self._session.rollback()
            raise DatabaseOperationError("Business understanding generation failed.") from exc

        self._session.refresh(understanding)
        return understanding

    def get(self, workspace_id: UUID) -> BusinessUnderstanding:
        """Return the current understanding for a workspace's business profile."""

        self._require_workspace(workspace_id)

        profile = self._profiles.get_by_workspace(workspace_id)
        if profile is None:
            raise ResourceNotFoundError("Business profile not found.")

        understanding = self._understandings.get_by_business_profile(profile.id)
        if understanding is None:
            raise ResourceNotFoundError("Business understanding has not been generated yet.")

        return understanding

    def _require_workspace(self, workspace_id: UUID) -> None:
        """Ensure the parent workspace exists."""

        workspace = self._workspaces.get(workspace_id)
        if workspace is None:
            raise ResourceNotFoundError("Workspace not found.")

    def _build_prompt(self, profile: BusinessProfile) -> str:
        """Build the synthesis prompt from a business profile's known fields."""

        lines = [
            "You are analyzing a business based on its onboarding profile.",
            "Synthesize and infer - do not just restate the fields below.",
            "Produce a narrative summary, and infer the business stage, "
            "competitive category, likely differentiators, and likely "
            "customer pain points, even where they were not stated directly.",
            "",
            "Known profile fields:",
        ]

        budget = (
            f"{profile.monthly_marketing_budget} {profile.marketing_budget_currency}"
            if profile.monthly_marketing_budget is not None
            else None
        )
        channels = ", ".join(profile.existing_channels) if profile.existing_channels else None
        competitors = ", ".join(profile.known_competitors) if profile.known_competitors else None

        fields = {
            "Business name": profile.business_name,
            "Website": profile.website,
            "Product or service": profile.product_or_service,
            "Industry": profile.industry,
            "Country": profile.country,
            "Target market": profile.target_market,
            "Target customer": profile.target_customer,
            "Price range": profile.price_range,
            "Monthly marketing budget": budget,
            "Main marketing goal": profile.main_marketing_goal,
            "Existing channels": channels,
            "Brand tone": profile.brand_tone,
            "Known competitors": competitors,
            "Current challenges": profile.current_challenges,
            "Additional instructions": profile.additional_instructions,
        }

        for label, value in fields.items():
            if value:
                lines.append(f"- {label}: {value}")

        return "\n".join(lines)

    def _apply_draft(
        self,
        understanding: BusinessUnderstanding,
        draft: _BusinessUnderstandingDraft,
        usage: LLMUsage,
    ) -> None:
        """Copy a generated draft plus usage metadata onto an ORM record."""

        understanding.summary = draft.summary
        understanding.inferred_business_stage = draft.inferred_business_stage
        understanding.competitive_category = draft.competitive_category
        understanding.key_differentiators = list(draft.key_differentiators)
        understanding.likely_customer_pain_points = list(draft.likely_customer_pain_points)
        understanding.confidence_notes = draft.confidence_notes
        understanding.model_used = self._settings.llm_model
        understanding.input_tokens = usage.input_tokens
        understanding.output_tokens = usage.output_tokens
