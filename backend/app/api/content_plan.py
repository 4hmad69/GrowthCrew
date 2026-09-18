"""Content Planning agent REST API routes."""

from typing import Annotated, cast
from uuid import UUID

from fastapi import APIRouter, Depends, Request, status
from sqlalchemy.orm import Session

from backend.app.config import Settings
from backend.app.db.dependencies import get_db_session
from backend.app.llm.dependencies import get_llm_gateway
from backend.app.llm.gateway import LLMGateway
from backend.app.schemas.content_plan import (
    ContentPlanGenerateRequest,
    ContentPlanResponse,
)
from backend.app.services.content_plan import ContentPlanService

router = APIRouter(
    prefix="/workspaces/{workspace_id}/content-plan",
    tags=["content plan"],
)

DbSession = Annotated[Session, Depends(get_db_session)]
Gateway = Annotated[LLMGateway, Depends(get_llm_gateway)]


@router.post(
    "",
    response_model=ContentPlanResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate or fetch the Content Planning agent's 30-day calendar",
)
def generate_content_plan(
    workspace_id: UUID,
    payload: ContentPlanGenerateRequest,
    session: DbSession,
    gateway: Gateway,
    request: Request,
) -> ContentPlanResponse:
    """Generate a new content plan, or return the existing one.

    Returns the existing record without calling the LLM unless
    force_regenerate is set - generation costs real tokens. Requires a
    Marketing Strategy to already exist for this workspace; 404s if it
    doesn't. Unlike Marketing Strategy, this needs only the LLM gateway -
    no embeddings or web search dependency, since generation is one
    direct structured call, not a CRAG graph run.
    """

    settings = cast(Settings, request.app.state.settings)
    plan = ContentPlanService(session, gateway, settings).generate(
        workspace_id,
        force_regenerate=payload.force_regenerate,
    )

    return ContentPlanResponse.model_validate(plan)


@router.get(
    "",
    response_model=ContentPlanResponse,
    summary="Read the current Content Plan",
)
def get_content_plan(
    workspace_id: UUID,
    session: DbSession,
    gateway: Gateway,
    request: Request,
) -> ContentPlanResponse:
    """Return the current content plan for a workspace."""

    settings = cast(Settings, request.app.state.settings)
    plan = ContentPlanService(session, gateway, settings).get(workspace_id)

    return ContentPlanResponse.model_validate(plan)
