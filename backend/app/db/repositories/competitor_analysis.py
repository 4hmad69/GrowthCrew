"""Persistence operations for Competitor Analysis agent output."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.db.models.competitor_analysis import CompetitorAnalysis


class CompetitorAnalysisRepository:
    """Perform competitor-analysis persistence operations."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, analysis: CompetitorAnalysis) -> None:
        """Stage a new competitor analysis record."""

        self._session.add(analysis)

    def get_by_workspace(self, workspace_id: UUID) -> CompetitorAnalysis | None:
        """Return the current competitor analysis report for a workspace, if any."""

        statement = select(CompetitorAnalysis).where(
            CompetitorAnalysis.workspace_id == workspace_id
        )
        return self._session.scalar(statement)

    def delete(self, analysis: CompetitorAnalysis) -> None:
        """Stage a competitor analysis record for deletion."""

        self._session.delete(analysis)
