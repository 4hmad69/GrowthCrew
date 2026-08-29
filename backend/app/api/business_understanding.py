"""Business Understanding agent REST API routes."""

from typing import Annotated, cast
from uuid import UUID

from fastapi import APIRouter, Depends, Request, status
from sqlalchemy.orm import Session

from backend.app.config import Settings
from backend.app.db.dependencies import get_db_session
from backend.app.llm.dependencies import get_llm_gateway
from backend.app.llm.gateway import LLMGateway
from backend.app.schemas.business_understanding import (
    BusinessUnderstandingGenerateRequest,
    BusinessUnderstandingResponse,
)
from backend.app.services.business_understanding import BusinessUnderstandingService

router = APIRouter(
    prefix="/workspaces/{workspace_id}/business-profile/understanding",
    tags=["business understanding"],
)

DbSession = Annotated[Session, Depends(get_db_session)]
Gateway = Annotated[LLMGateway, Depends(get_llm_gateway)]


@router.post(
    "",
    response_model=BusinessUnderstandingResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate or fetch the Business Understanding agent's synthesis",
)
def generate_business_understanding(
    workspace_id: UUID,
    payload: BusinessUnderstandingGenerateRequest,
    session: DbSession,
    gateway: Gateway,
    request: Request,
) -> BusinessUnderstandingResponse:
    """Generate a new understanding, or return the existing one.

    Returns the existing record without calling the LLM unless
    force_regenerate is set - generation costs real tokens.
    """

    settings = cast(Settings, request.app.state.settings)
    understanding = BusinessUnderstandingService(session, gateway, settings).generate(
        workspace_id,
        force_regenerate=payload.force_regenerate,
    )

    return BusinessUnderstandingResponse.model_validate(understanding)


@router.get(
    "",
    response_model=BusinessUnderstandingResponse,
    summary="Read the current Business Understanding",
)
def get_business_understanding(
    workspace_id: UUID,
    session: DbSession,
    gateway: Gateway,
    request: Request,
) -> BusinessUnderstandingResponse:
    """Return the current understanding for a workspace's business profile."""

    settings = cast(Settings, request.app.state.settings)
    understanding = BusinessUnderstandingService(session, gateway, settings).get(workspace_id)

    return BusinessUnderstandingResponse.model_validate(understanding)
