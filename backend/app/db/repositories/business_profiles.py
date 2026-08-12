"""Persistence operations for business profiles."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.db.models.business_profile import BusinessProfile


class BusinessProfileRepository:
    """Perform business-profile persistence operations."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, profile: BusinessProfile) -> None:
        """Stage a new business profile."""

        self._session.add(profile)

    def get_by_workspace(
        self,
        workspace_id: UUID,
    ) -> BusinessProfile | None:
        """Return the current business profile for a workspace."""

        statement = select(BusinessProfile).where(BusinessProfile.workspace_id == workspace_id)
        return self._session.scalar(statement)

    def delete(self, profile: BusinessProfile) -> None:
        """Stage a business profile for deletion."""

        self._session.delete(profile)
