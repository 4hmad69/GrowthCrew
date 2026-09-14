"""Persistence operations for Marketing Strategy agent output."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.db.models.marketing_strategy import MarketingStrategy


class MarketingStrategyRepository:
    """Perform marketing-strategy persistence operations."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, strategy: MarketingStrategy) -> None:
        """Stage a new marketing strategy record."""

        self._session.add(strategy)

    def get_by_workspace(self, workspace_id: UUID) -> MarketingStrategy | None:
        """Return the current marketing strategy for a workspace, if any."""

        statement = select(MarketingStrategy).where(MarketingStrategy.workspace_id == workspace_id)
        return self._session.scalar(statement)

    def delete(self, strategy: MarketingStrategy) -> None:
        """Stage a marketing strategy record for deletion."""

        self._session.delete(strategy)
