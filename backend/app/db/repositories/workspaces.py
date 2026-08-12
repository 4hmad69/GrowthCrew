"""Persistence operations for workspaces."""

from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.db.models.workspace import Workspace


class WorkspaceRepository:
    """Perform workspace persistence without business-policy decisions."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, workspace: Workspace) -> None:
        """Stage a workspace for insertion."""

        self._session.add(workspace)

    def get(self, workspace_id: UUID) -> Workspace | None:
        """Return a workspace by identifier."""

        return self._session.get(Workspace, workspace_id)

    def list(
        self,
        *,
        limit: int,
        offset: int,
        include_archived: bool,
    ) -> tuple[list[Workspace], int]:
        """Return a page of workspaces and the matching total count."""

        item_statement = select(Workspace)
        count_statement = select(func.count()).select_from(Workspace)

        if not include_archived:
            item_statement = item_statement.where(Workspace.status == "active")
            count_statement = count_statement.where(Workspace.status == "active")

        item_statement = (
            item_statement.order_by(
                Workspace.created_at.desc(),
                Workspace.id,
            )
            .offset(offset)
            .limit(limit)
        )

        items = list(self._session.scalars(item_statement))
        total = int(self._session.scalar(count_statement) or 0)

        return items, total
