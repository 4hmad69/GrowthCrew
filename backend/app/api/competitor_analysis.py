"""Competitor Analysis agent REST API routes."""

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
from backend.app.schemas.competitor_analysis import (
    CompetitorAnalysisGenerateRequest,
    CompetitorAnalysisResponse,
)
from backend.app.services.competitor_analysis import CompetitorAnalysisService
from backend.app.websearch.dependencies import get_web_search_gateway
from backend.app.websearch.gateway import WebSearchGateway

router = APIRouter(
    prefix="/workspaces/{workspace_id}/competitor-analysis",
    tags=["competitor analysis"],
)

DbSession = Annotated[Session, Depends(get_db_session)]
Gateway = Annotated[LLMGateway, Depends(get_llm_gateway)]
Embeddings = Annotated[EmbeddingsGateway, Depends(get_embeddings_gateway)]
WebSearch = Annotated[WebSearchGateway, Depends(get_web_search_gateway)]


@router.post(
    "",
    response_model=CompetitorAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate or fetch the Competitor Analysis agent's report",
)
def generate_competitor_analysis(
    workspace_id: UUID,
    payload: CompetitorAnalysisGenerateRequest,
    session: DbSession,
    gateway: Gateway,
    embeddings: Embeddings,
    web_search: WebSearch,
    request: Request,
) -> CompetitorAnalysisResponse:
    """Generate a new report, or return the existing one.

    Returns the existing record without touching the LLM, retrieval, or
    web search unless force_regenerate is set - each section is its own
    full CRAG graph run, so regeneration is genuinely not free.
    """

    settings = cast(Settings, request.app.state.settings)
    analysis = CompetitorAnalysisService(
        session, gateway, embeddings, web_search, settings
    ).generate(
        workspace_id,
        force_regenerate=payload.force_regenerate,
    )

    return CompetitorAnalysisResponse.model_validate(analysis)


@router.get(
    "",
    response_model=CompetitorAnalysisResponse,
    summary="Read the current Competitor Analysis report",
)
def get_competitor_analysis(
    workspace_id: UUID,
    session: DbSession,
    gateway: Gateway,
    embeddings: Embeddings,
    web_search: WebSearch,
    request: Request,
) -> CompetitorAnalysisResponse:
    """Return the current competitor analysis report for a workspace."""

    settings = cast(Settings, request.app.state.settings)
    analysis = CompetitorAnalysisService(session, gateway, embeddings, web_search, settings).get(
        workspace_id
    )

    return CompetitorAnalysisResponse.model_validate(analysis)
