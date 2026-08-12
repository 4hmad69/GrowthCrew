"""Business-profile business rules and transaction boundaries."""

from typing import Any
from uuid import UUID

from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session
from sqlalchemy.orm.exc import StaleDataError

from backend.app.db.errors import DatabaseOperationError
from backend.app.db.models.business_profile import BusinessProfile
from backend.app.db.repositories.business_profiles import (
    BusinessProfileRepository,
)
from backend.app.db.repositories.workspaces import WorkspaceRepository
from backend.app.exceptions import (
    ResourceConflictError,
    ResourceNotFoundError,
    StaleResourceError,
)
from backend.app.schemas.business_profile import (
    BusinessProfileCreate,
    BusinessProfileUpdate,
)


class BusinessProfileService:
    """Coordinate onboarding-profile rules and persistence."""

    def __init__(self, session: Session) -> None:
        self._session = session
        self._profiles = BusinessProfileRepository(session)
        self._workspaces = WorkspaceRepository(session)

    def create(
        self,
        workspace_id: UUID,
        payload: BusinessProfileCreate,
    ) -> BusinessProfile:
        """Create the single onboarding profile for an active workspace."""

        self._require_active_workspace(workspace_id)

        existing = self._profiles.get_by_workspace(workspace_id)
        if existing is not None:
            raise ResourceConflictError("A business profile already exists for this workspace.")

        profile = BusinessProfile(
            workspace_id=workspace_id,
            business_name=payload.business_name,
            website=self._serialize_url(payload.website),
            product_or_service=payload.product_or_service,
            industry=payload.industry,
            country=payload.country,
            target_market=payload.target_market,
            target_customer=payload.target_customer,
            price_range=payload.price_range,
            monthly_marketing_budget=(payload.monthly_marketing_budget),
            marketing_budget_currency=(payload.marketing_budget_currency),
            main_marketing_goal=payload.main_marketing_goal,
            existing_channels=list(payload.existing_channels),
            brand_tone=payload.brand_tone,
            known_competitors=list(payload.known_competitors),
            current_challenges=payload.current_challenges,
            additional_instructions=payload.additional_instructions,
        )

        self._profiles.add(profile)

        try:
            self._session.commit()
        except IntegrityError as exc:
            self._session.rollback()
            raise ResourceConflictError(
                "A business profile already exists for this workspace."
            ) from exc
        except SQLAlchemyError as exc:
            self._session.rollback()
            raise DatabaseOperationError("Business profile creation failed.") from exc

        self._session.refresh(profile)
        return profile

    def get(self, workspace_id: UUID) -> BusinessProfile:
        """Return a workspace's current business profile."""

        self._require_workspace(workspace_id)

        profile = self._profiles.get_by_workspace(workspace_id)
        if profile is None:
            raise ResourceNotFoundError("Business profile not found.")

        return profile

    def update(
        self,
        workspace_id: UUID,
        payload: BusinessProfileUpdate,
    ) -> BusinessProfile:
        """Patch an onboarding profile using optimistic concurrency."""

        self._require_active_workspace(workspace_id)

        profile = self._profiles.get_by_workspace(workspace_id)
        if profile is None:
            raise ResourceNotFoundError("Business profile not found.")

        if profile.version != payload.version:
            raise StaleResourceError("Business profile has changed. Reload it and try again.")

        changes = self._prepare_changes(payload)

        for field_name, value in changes.items():
            setattr(profile, field_name, value)

        try:
            self._session.commit()
        except StaleDataError as exc:
            self._session.rollback()
            raise StaleResourceError(
                "Business profile changed while the update was being processed."
            ) from exc
        except SQLAlchemyError as exc:
            self._session.rollback()
            raise DatabaseOperationError("Business profile update failed.") from exc

        self._session.refresh(profile)
        return profile

    def delete(
        self,
        workspace_id: UUID,
        *,
        expected_version: int,
    ) -> None:
        """Delete the current onboarding profile."""

        self._require_active_workspace(workspace_id)

        profile = self._profiles.get_by_workspace(workspace_id)
        if profile is None:
            raise ResourceNotFoundError("Business profile not found.")

        if profile.version != expected_version:
            raise StaleResourceError("Business profile has changed. Reload it and try again.")

        self._profiles.delete(profile)

        try:
            self._session.commit()
        except StaleDataError as exc:
            self._session.rollback()
            raise StaleResourceError(
                "Business profile changed while deletion was being processed."
            ) from exc
        except SQLAlchemyError as exc:
            self._session.rollback()
            raise DatabaseOperationError("Business profile deletion failed.") from exc

    def _require_workspace(self, workspace_id: UUID) -> None:
        """Ensure the parent workspace exists."""

        workspace = self._workspaces.get(workspace_id)
        if workspace is None:
            raise ResourceNotFoundError("Workspace not found.")

    def _require_active_workspace(
        self,
        workspace_id: UUID,
    ) -> None:
        """Ensure the workspace exists and accepts mutations."""

        workspace = self._workspaces.get(workspace_id)

        if workspace is None:
            raise ResourceNotFoundError("Workspace not found.")

        if workspace.status != "active":
            raise ResourceConflictError("Archived workspaces cannot be modified.")

    @staticmethod
    def _serialize_url(value: Any) -> str | None:
        """Convert Pydantic URLs to database strings."""

        if value is None:
            return None
        return str(value)

    @classmethod
    def _prepare_changes(
        cls,
        payload: BusinessProfileUpdate,
    ) -> dict[str, Any]:
        """Convert a PATCH schema into ORM-compatible values."""

        changes = payload.model_dump(
            exclude_unset=True,
        )
        changes.pop("version", None)

        if "website" in changes and changes["website"] is not None:
            changes["website"] = cls._serialize_url(changes["website"])

        return changes
