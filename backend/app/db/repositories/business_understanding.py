"""Persistence operations for Business Understanding agent output."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.db.models.business_understanding import BusinessUnderstanding


class BusinessUnderstandingRepository:
    """Perform business-understanding persistence operations."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, understanding: BusinessUnderstanding) -> None:
        """Stage a new business understanding record."""

        self._session.add(understanding)

    def get_by_business_profile(
        self,
        business_profile_id: UUID,
    ) -> BusinessUnderstanding | None:
        """Return the current understanding for a business profile, if any."""

        statement = select(BusinessUnderstanding).where(
            BusinessUnderstanding.profile_id == business_profile_id
        )
        return self._session.scalar(statement)

    def delete(self, understanding: BusinessUnderstanding) -> None:
        """Stage a business understanding record for deletion."""

        self._session.delete(understanding)
