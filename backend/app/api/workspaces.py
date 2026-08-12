"""Workspace REST API routes."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Response, status
from sqlalchemy.orm import Session

from backend.app.db.dependencies import get_db_session
from backend.app.schemas.workspace import (
    WorkspaceCreate,
    WorkspaceListResponse,
    WorkspaceResponse,
    WorkspaceUpdate,
)
from backend.app.services.workspaces import WorkspaceService

router = APIRouter(
    prefix="/workspaces",
    tags=["workspaces"],
)

DbSession = Annotated[
    Session,
    Depends(get_db_session),
]


@router.post(
    "",
    response_model=WorkspaceResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a workspace",
)
def create_workspace(
    payload: WorkspaceCreate,
    session: DbSession,
) -> WorkspaceResponse:
    """Create a new GrowthCrew workspace."""

    workspace = WorkspaceService(session).create(payload)
    return WorkspaceResponse.model_validate(workspace)


@router.get(
    "",
    response_model=WorkspaceListResponse,
    summary="List workspaces",
)
def list_workspaces(
    session: DbSession,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
    include_archived: bool = False,
) -> WorkspaceListResponse:
    """Return a bounded page of workspaces."""

    items, total = WorkspaceService(session).list(
        limit=limit,
        offset=offset,
        include_archived=include_archived,
    )

    return WorkspaceListResponse(
        items=[WorkspaceResponse.model_validate(item) for item in items],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{workspace_id}",
    response_model=WorkspaceResponse,
    summary="Read a workspace",
)
def get_workspace(
    workspace_id: UUID,
    session: DbSession,
) -> WorkspaceResponse:
    """Return one workspace."""

    workspace = WorkspaceService(session).get(workspace_id)
    return WorkspaceResponse.model_validate(workspace)


@router.patch(
    "/{workspace_id}",
    response_model=WorkspaceResponse,
    summary="Update a workspace",
)
def update_workspace(
    workspace_id: UUID,
    payload: WorkspaceUpdate,
    session: DbSession,
) -> WorkspaceResponse:
    """Update workspace properties using optimistic concurrency."""

    workspace = WorkspaceService(session).update(
        workspace_id,
        payload,
    )
    return WorkspaceResponse.model_validate(workspace)


@router.delete(
    "/{workspace_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Archive a workspace",
)
def archive_workspace(
    workspace_id: UUID,
    session: DbSession,
    version: Annotated[int, Query(ge=1)],
) -> Response:
    """Soft-delete a workspace by marking it archived."""

    WorkspaceService(session).archive(
        workspace_id,
        expected_version=version,
    )

    return Response(status_code=status.HTTP_204_NO_CONTENT)
