"""Marketing Strategy agent REST API routes."""

from typing import Annotated, cast
from uuid import UUID

from fastapi import APIRouter, Depends, Request, status
from sqlalchemy.orm import Session

from backend.app.config import Settings
from backend.app.db.dependencies import get_db_session
from backend.app.embeddings.dependencies import get_embeddings_gateway
from backend.app.embeddings.gateway import EmbeddingsGateway
from backend.app.llm.dependencies import get_llm_gateway
from backend.app.llm.gateway import LLMGateway
from backend.app.schemas.marketing_strategy import (
    MarketingStrategyGenerateRequest,
    MarketingStrategyResponse,
)
from backend.app.services.marketing_strategy import MarketingStrategyService
from backend.app.websearch.dependencies import get_web_search_gateway
from backend.app.websearch.gateway import WebSearchGateway

router = APIRouter(
    prefix="/workspaces/{workspace_id}/marketing-strategy",
    tags=["marketing strategy"],
)

DbSession = Annotated[Session, Depends(get_db_session)]
Gateway = Annotated[LLMGateway, Depends(get_llm_gateway)]
Embeddings = Annotated[EmbeddingsGateway, Depends(get_embeddings_gateway)]
WebSearch = Annotated[WebSearchGateway, Depends(get_web_search_gateway)]


@router.post(
    "",
    response_model=MarketingStrategyResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate or fetch the Marketing Strategy agent's report",
)
def generate_marketing_strategy(
    workspace_id: UUID,
    payload: MarketingStrategyGenerateRequest,
    session: DbSession,
    gateway: Gateway,
    embeddings: Embeddings,
    web_search: WebSearch,
    request: Request,
) -> MarketingStrategyResponse:
    """Generate a new strategy, or return the existing one.

    Returns the existing record without touching the LLM, retrieval, or
    web search unless force_regenerate is set - each section is its own
    full CRAG graph run, so regeneration is genuinely not free. Requires
    Business Understanding, Market Research, and Competitor Analysis to
    already exist for this workspace; 404s naming whichever is missing.
    """

    settings = cast(Settings, request.app.state.settings)
    strategy = MarketingStrategyService(
        session, gateway, embeddings, web_search, settings
    ).generate(
        workspace_id,
        force_regenerate=payload.force_regenerate,
    )

    return MarketingStrategyResponse.model_validate(strategy)


@router.get(
    "",
    response_model=MarketingStrategyResponse,
    summary="Read the current Marketing Strategy report",
)
def get_marketing_strategy(
    workspace_id: UUID,
    session: DbSession,
    gateway: Gateway,
    embeddings: Embeddings,
    web_search: WebSearch,
    request: Request,
) -> MarketingStrategyResponse:
    """Return the current marketing strategy for a workspace."""

    settings = cast(Settings, request.app.state.settings)
    strategy = MarketingStrategyService(session, gateway, embeddings, web_search, settings).get(
        workspace_id
    )

    return MarketingStrategyResponse.model_validate(strategy)
