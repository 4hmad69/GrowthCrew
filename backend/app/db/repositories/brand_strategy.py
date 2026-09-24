"""Persistence operations for Brand Strategy agent output."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.db.models.brand_strategy import BrandStrategy


class BrandStrategyRepository:
    """Perform brand-strategy persistence operations."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, brand_strategy: BrandStrategy) -> None:
        """Stage a new brand strategy record."""

        self._session.add(brand_strategy)

    def get_by_workspace(self, workspace_id: UUID) -> BrandStrategy | None:
        """Return the current brand strategy for a workspace, if any."""

        statement = select(BrandStrategy).where(BrandStrategy.workspace_id == workspace_id)
        return self._session.scalar(statement)

    def delete(self, brand_strategy: BrandStrategy) -> None:
        """Stage a brand strategy record for deletion."""

        self._session.delete(brand_strategy)
