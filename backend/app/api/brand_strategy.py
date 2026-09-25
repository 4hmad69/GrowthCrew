"""Brand Strategy agent REST API routes."""

from typing import Annotated, cast
from uuid import UUID

from fastapi import APIRouter, Depends, Request, status
from sqlalchemy.orm import Session

from backend.app.config import Settings
from backend.app.db.dependencies import get_db_session
from backend.app.llm.dependencies import get_llm_gateway
from backend.app.llm.gateway import LLMGateway
from backend.app.schemas.brand_strategy import (
    BrandStrategyGenerateRequest,
    BrandStrategyResponse,
)
from backend.app.services.brand_strategy import BrandStrategyService

router = APIRouter(
    prefix="/workspaces/{workspace_id}/brand-strategy",
    tags=["brand strategy"],
)

DbSession = Annotated[Session, Depends(get_db_session)]
Gateway = Annotated[LLMGateway, Depends(get_llm_gateway)]


@router.post(
    "",
    response_model=BrandStrategyResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate or fetch the Brand Strategy agent's output",
)
def generate_brand_strategy(
    workspace_id: UUID,
    payload: BrandStrategyGenerateRequest,
    session: DbSession,
    gateway: Gateway,
    request: Request,
) -> BrandStrategyResponse:
    """Generate a new brand strategy, or return the existing one.

    Returns the existing record without calling the LLM unless
    force_regenerate is set - generation costs real tokens. Requires the
    workspace's business profile, Business Understanding, Competitor
    Analysis, and Customer Personas to already exist; 404s if any of
    them doesn't. Like Customer Personas, this needs only the LLM
    gateway - no embeddings or web search dependency, since generation
    is one direct structured call, not a CRAG graph run.
    """

    settings = cast(Settings, request.app.state.settings)
    brand_strategy = BrandStrategyService(session, gateway, settings).generate(
        workspace_id,
        force_regenerate=payload.force_regenerate,
    )

    return BrandStrategyResponse.model_validate(brand_strategy)


@router.get(
    "",
    response_model=BrandStrategyResponse,
    summary="Read the current Brand Strategy",
)
def get_brand_strategy(
    workspace_id: UUID,
    session: DbSession,
    gateway: Gateway,
    request: Request,
) -> BrandStrategyResponse:
    """Return the current brand strategy for a workspace."""

    settings = cast(Settings, request.app.state.settings)
    brand_strategy = BrandStrategyService(session, gateway, settings).get(workspace_id)

    return BrandStrategyResponse.model_validate(brand_strategy)
