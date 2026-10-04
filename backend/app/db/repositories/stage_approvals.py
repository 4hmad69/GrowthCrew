"""Persistence operations for stage approvals."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.db.models.stage_approval import StageApproval


class StageApprovalRepository:
    """Perform stage-approval persistence operations."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, approval: StageApproval) -> None:
        """Stage a new approval record."""

        self._session.add(approval)

    def get_by_workspace_and_stage(
        self,
        workspace_id: UUID,
        stage: str,
    ) -> StageApproval | None:
        """Return the approval for one stage of a workspace, if any."""

        statement = select(StageApproval).where(
            StageApproval.workspace_id == workspace_id,
            StageApproval.stage == stage,
        )
        return self._session.scalar(statement)

    def list_by_workspace(self, workspace_id: UUID) -> list[StageApproval]:
        """Return every approval recorded for a workspace."""

        statement = select(StageApproval).where(StageApproval.workspace_id == workspace_id)
        return list(self._session.scalars(statement))

    def delete(self, approval: StageApproval) -> None:
        """Stage an approval record for deletion."""

        self._session.delete(approval)
