"""Persistence operations for stage snapshots.

Deliberately append-only: there is no update and no delete here. History
is only ever added to, and the database rejects UPDATE on the table as a
second line of defence. Rows disappear only when their workspace does.
"""

from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.db.models.stage_snapshot import StageSnapshot


class StageSnapshotRepository:
    """Perform stage-snapshot persistence operations."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, snapshot: StageSnapshot) -> None:
        """Stage a new snapshot row."""

        self._session.add(snapshot)

    def get(self, workspace_id: UUID, stage: str, generation: int) -> StageSnapshot | None:
        """Return one specific generation of a stage, if it exists."""

        statement = select(StageSnapshot).where(
            StageSnapshot.workspace_id == workspace_id,
            StageSnapshot.stage == stage,
            StageSnapshot.generation == generation,
        )
        return self._session.scalar(statement)

    def get_latest(self, workspace_id: UUID, stage: str) -> StageSnapshot | None:
        """Return the newest generation of a stage, if any."""

        statement = (
            select(StageSnapshot)
            .where(
                StageSnapshot.workspace_id == workspace_id,
                StageSnapshot.stage == stage,
            )
            .order_by(StageSnapshot.generation.desc())
            .limit(1)
        )
        return self._session.scalar(statement)

    def next_generation(self, workspace_id: UUID, stage: str) -> int:
        """Return the generation number the next snapshot of a stage must use.

        Two writers can compute the same number; the unique constraint
        on (workspace, stage, generation) makes the second one fail
        instead of silently forking the history.
        """

        statement = select(func.max(StageSnapshot.generation)).where(
            StageSnapshot.workspace_id == workspace_id,
            StageSnapshot.stage == stage,
        )
        return (self._session.scalar(statement) or 0) + 1

    def list_by_stage(self, workspace_id: UUID, stage: str) -> list[StageSnapshot]:
        """Return every generation of a stage, newest first."""

        statement = (
            select(StageSnapshot)
            .where(
                StageSnapshot.workspace_id == workspace_id,
                StageSnapshot.stage == stage,
            )
            .order_by(StageSnapshot.generation.desc())
        )
        return list(self._session.scalars(statement))

    def list_latest_by_workspace(self, workspace_id: UUID) -> list[StageSnapshot]:
        """Return the newest snapshot of each stage that has one."""

        statement = (
            select(StageSnapshot)
            .where(StageSnapshot.workspace_id == workspace_id)
            .distinct(StageSnapshot.stage)
            .order_by(StageSnapshot.stage, StageSnapshot.generation.desc())
        )
        return list(self._session.scalars(statement))
