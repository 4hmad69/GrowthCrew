"""Persistence operations for Market Research agent output."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.db.models.market_research import MarketResearch


class MarketResearchRepository:
    """Perform market-research persistence operations."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, research: MarketResearch) -> None:
        """Stage a new market research record."""

        self._session.add(research)

    def get_by_workspace(self, workspace_id: UUID) -> MarketResearch | None:
        """Return the current market research report for a workspace, if any."""

        statement = select(MarketResearch).where(MarketResearch.workspace_id == workspace_id)
        return self._session.scalar(statement)

    def delete(self, research: MarketResearch) -> None:
        """Stage a market research record for deletion."""

        self._session.delete(research)
