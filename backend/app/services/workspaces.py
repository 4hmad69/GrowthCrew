"""Workspace business logic and transaction boundaries."""

from uuid import UUID

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from sqlalchemy.orm.exc import StaleDataError

from backend.app.db.errors import DatabaseOperationError
from backend.app.db.models.workspace import Workspace
from backend.app.db.repositories.workspaces import WorkspaceRepository
from backend.app.exceptions import (
    ResourceConflictError,
    ResourceNotFoundError,
    StaleResourceError,
)
from backend.app.schemas.workspace import WorkspaceCreate, WorkspaceUpdate


class WorkspaceService:
    """Coordinate workspace rules and persistence."""

    def __init__(self, session: Session) -> None:
        self._session = session
        self._repository = WorkspaceRepository(session)

    def create(self, payload: WorkspaceCreate) -> Workspace:
        """Create and persist an active workspace."""

        workspace = Workspace(name=payload.name)
        self._repository.add(workspace)

        self._commit(
            stale_message="Workspace creation became stale.",
        )
        self._session.refresh(workspace)

        return workspace

    def get(self, workspace_id: UUID) -> Workspace:
        """Return a workspace or raise a safe not-found error."""

        workspace = self._repository.get(workspace_id)
        if workspace is None:
            raise ResourceNotFoundError("Workspace not found.")

        return workspace

    def list(
        self,
        *,
        limit: int,
        offset: int,
        include_archived: bool,
    ) -> tuple[list[Workspace], int]:
        """Return a paginated workspace collection."""

        try:
            return self._repository.list(
                limit=limit,
                offset=offset,
                include_archived=include_archived,
            )
        except SQLAlchemyError as exc:
            self._session.rollback()
            raise DatabaseOperationError("Workspace listing failed.") from exc

    def update(
        self,
        workspace_id: UUID,
        payload: WorkspaceUpdate,
    ) -> Workspace:
        """Update an active workspace using optimistic concurrency."""

        workspace = self.get(workspace_id)

        if workspace.status != "active":
            raise ResourceConflictError("Archived workspaces cannot be modified.")

        self._require_version(
            workspace=workspace,
            expected_version=payload.version,
        )

        if payload.name is not None:
            workspace.name = payload.name

        self._commit(
            stale_message=("Workspace changed while the update was being processed."),
        )
        self._session.refresh(workspace)

        return workspace

    def archive(
        self,
        workspace_id: UUID,
        *,
        expected_version: int,
    ) -> None:
        """Archive a workspace without physically deleting its data."""

        workspace = self.get(workspace_id)

        if workspace.status == "archived":
            return

        self._require_version(
            workspace=workspace,
            expected_version=expected_version,
        )

        workspace.status = "archived"

        self._commit(
            stale_message=("Workspace changed while the archive was being processed."),
        )

    @staticmethod
    def _require_version(
        *,
        workspace: Workspace,
        expected_version: int,
    ) -> None:
        """Reject stale client state before mutation."""

        if workspace.version != expected_version:
            raise StaleResourceError("Workspace has changed. Reload it and try again.")

    def _commit(self, *, stale_message: str) -> None:
        """Commit the transaction and translate persistence failures."""

        try:
            self._session.commit()
        except StaleDataError as exc:
            self._session.rollback()
            raise StaleResourceError(stale_message) from exc
        except SQLAlchemyError as exc:
            self._session.rollback()
            raise DatabaseOperationError("Workspace database operation failed.") from exc
