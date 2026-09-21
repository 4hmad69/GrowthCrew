"""Customer Personas agent REST API routes."""

from typing import Annotated, cast
from uuid import UUID

from fastapi import APIRouter, Depends, Request, status
from sqlalchemy.orm import Session

from backend.app.config import Settings
from backend.app.db.dependencies import get_db_session
from backend.app.llm.dependencies import get_llm_gateway
from backend.app.llm.gateway import LLMGateway
from backend.app.schemas.customer_persona_set import (
    CustomerPersonaSetGenerateRequest,
    CustomerPersonaSetResponse,
)
from backend.app.services.customer_personas import CustomerPersonasService

router = APIRouter(
    prefix="/workspaces/{workspace_id}/personas",
    tags=["customer personas"],
)

DbSession = Annotated[Session, Depends(get_db_session)]
Gateway = Annotated[LLMGateway, Depends(get_llm_gateway)]


@router.post(
    "",
    response_model=CustomerPersonaSetResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate or fetch the Customer Personas agent's persona set",
)
def generate_customer_personas(
    workspace_id: UUID,
    payload: CustomerPersonaSetGenerateRequest,
    session: DbSession,
    gateway: Gateway,
    request: Request,
) -> CustomerPersonaSetResponse:
    """Generate a new persona set, or return the existing one.

    Returns the existing record without calling the LLM unless
    force_regenerate is set - generation costs real tokens. Requires the
    workspace's business profile, Business Understanding, and Market
    Research to already exist; 404s if any of them doesn't. Like Content
    Planning, this needs only the LLM gateway - no embeddings or web
    search dependency, since generation is one direct structured call,
    not a CRAG graph run.
    """

    settings = cast(Settings, request.app.state.settings)
    persona_set = CustomerPersonasService(session, gateway, settings).generate(
        workspace_id,
        force_regenerate=payload.force_regenerate,
    )

    return CustomerPersonaSetResponse.model_validate(persona_set)


@router.get(
    "",
    response_model=CustomerPersonaSetResponse,
    summary="Read the current Customer Personas",
)
def get_customer_personas(
    workspace_id: UUID,
    session: DbSession,
    gateway: Gateway,
    request: Request,
) -> CustomerPersonaSetResponse:
    """Return the current persona set for a workspace."""

    settings = cast(Settings, request.app.state.settings)
    persona_set = CustomerPersonasService(session, gateway, settings).get(workspace_id)

    return CustomerPersonaSetResponse.model_validate(persona_set)
