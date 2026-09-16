"""Persistence operations for Content Planning agent output."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.db.models.content_plan import ContentPlan


class ContentPlanRepository:
    """Perform content-plan persistence operations."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, plan: ContentPlan) -> None:
        """Stage a new content plan record."""

        self._session.add(plan)

    def get_by_workspace(self, workspace_id: UUID) -> ContentPlan | None:
        """Return the current content plan for a workspace, if any."""

        statement = select(ContentPlan).where(ContentPlan.workspace_id == workspace_id)
        return self._session.scalar(statement)

    def delete(self, plan: ContentPlan) -> None:
        """Stage a content plan record for deletion."""

        self._session.delete(plan)
