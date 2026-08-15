"""Business-profile REST API routes."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Response, status
from sqlalchemy.orm import Session

from backend.app.db.dependencies import get_db_session
from backend.app.schemas.business_profile import (
    BusinessProfileCreate,
    BusinessProfileResponse,
    BusinessProfileUpdate,
)
from backend.app.services.business_profiles import BusinessProfileService

router = APIRouter(
    prefix="/workspaces/{workspace_id}/business-profile",
    tags=["business profiles"],
)

DbSession = Annotated[
    Session,
    Depends(get_db_session),
]


@router.post(
    "",
    response_model=BusinessProfileResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a business profile",
)
def create_business_profile(
    workspace_id: UUID,
    payload: BusinessProfileCreate,
    session: DbSession,
) -> BusinessProfileResponse:
    """Create user-provided onboarding information."""

    profile = BusinessProfileService(session).create(
        workspace_id,
        payload,
    )

    return BusinessProfileResponse.model_validate(profile)


@router.get(
    "",
    response_model=BusinessProfileResponse,
    summary="Read a business profile",
)
def get_business_profile(
    workspace_id: UUID,
    session: DbSession,
) -> BusinessProfileResponse:
    """Return a workspace's onboarding profile."""

    profile = BusinessProfileService(session).get(workspace_id)

    return BusinessProfileResponse.model_validate(profile)


@router.patch(
    "",
    response_model=BusinessProfileResponse,
    summary="Update a business profile",
)
def update_business_profile(
    workspace_id: UUID,
    payload: BusinessProfileUpdate,
    session: DbSession,
) -> BusinessProfileResponse:
    """Patch user-provided onboarding information."""

    profile = BusinessProfileService(session).update(
        workspace_id,
        payload,
    )

    return BusinessProfileResponse.model_validate(profile)


@router.delete(
    "",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a business profile",
)
def delete_business_profile(
    workspace_id: UUID,
    session: DbSession,
    version: Annotated[int, Query(ge=1)],
) -> Response:
    """Delete the current onboarding profile."""

    BusinessProfileService(session).delete(
        workspace_id,
        expected_version=version,
    )

    return Response(status_code=status.HTTP_204_NO_CONTENT)
