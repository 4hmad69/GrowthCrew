"""Market Research agent REST API routes."""

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
from backend.app.schemas.market_research import (
    MarketResearchGenerateRequest,
    MarketResearchResponse,
)
from backend.app.services.market_research import MarketResearchService
from backend.app.websearch.dependencies import get_web_search_gateway
from backend.app.websearch.gateway import WebSearchGateway

router = APIRouter(
    prefix="/workspaces/{workspace_id}/market-research",
    tags=["market research"],
)

DbSession = Annotated[Session, Depends(get_db_session)]
Gateway = Annotated[LLMGateway, Depends(get_llm_gateway)]
Embeddings = Annotated[EmbeddingsGateway, Depends(get_embeddings_gateway)]
WebSearch = Annotated[WebSearchGateway, Depends(get_web_search_gateway)]


@router.post(
    "",
    response_model=MarketResearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate or fetch the Market Research agent's report",
)
def generate_market_research(
    workspace_id: UUID,
    payload: MarketResearchGenerateRequest,
    session: DbSession,
    gateway: Gateway,
    embeddings: Embeddings,
    web_search: WebSearch,
    request: Request,
) -> MarketResearchResponse:
    """Generate a new report, or return the existing one.

    Returns the existing record without touching the LLM, retrieval, or
    web search unless force_regenerate is set - each section is its own
    full CRAG graph run, so regeneration is genuinely not free.
    """

    settings = cast(Settings, request.app.state.settings)
    research = MarketResearchService(session, gateway, embeddings, web_search, settings).generate(
        workspace_id,
        force_regenerate=payload.force_regenerate,
    )

    return MarketResearchResponse.model_validate(research)


@router.get(
    "",
    response_model=MarketResearchResponse,
    summary="Read the current Market Research report",
)
def get_market_research(
    workspace_id: UUID,
    session: DbSession,
    gateway: Gateway,
    embeddings: Embeddings,
    web_search: WebSearch,
    request: Request,
) -> MarketResearchResponse:
    """Return the current market research report for a workspace."""

    settings = cast(Settings, request.app.state.settings)
    research = MarketResearchService(session, gateway, embeddings, web_search, settings).get(
        workspace_id
    )

    return MarketResearchResponse.model_validate(research)
