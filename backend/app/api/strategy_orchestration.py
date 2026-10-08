"""Strategy orchestration REST API routes."""

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
from backend.app.schemas.strategy_orchestration import (
    StrategyGenerateRequest,
    StrategyGenerateResponse,
    StrategyStatusResponse,
)
from backend.app.services.stage_approval import StageApprovalService
from backend.app.services.strategy_orchestration import (
    StrategyOrchestrationService,
    build_stage_generators,
)
from backend.app.services.strategy_status import StrategyStatusService
from backend.app.websearch.dependencies import get_web_search_gateway
from backend.app.websearch.gateway import WebSearchGateway

router = APIRouter(
    prefix="/workspaces/{workspace_id}",
    tags=["strategy orchestration"],
)

DbSession = Annotated[Session, Depends(get_db_session)]
Gateway = Annotated[LLMGateway, Depends(get_llm_gateway)]
Embeddings = Annotated[EmbeddingsGateway, Depends(get_embeddings_gateway)]
WebSearch = Annotated[WebSearchGateway, Depends(get_web_search_gateway)]


@router.get(
    "/strategy-status",
    response_model=StrategyStatusResponse,
    summary="Read the state of the whole strategy chain",
    responses={404: {"description": "Workspace not found."}},
)
def get_strategy_status(
    workspace_id: UUID,
    session: DbSession,
) -> StrategyStatusResponse:
    """Report which of the seven stages exist and what blocks the rest.

    Read-only and free: no LLM, retrieval, or web-search call is made.
    For each stage, in generation order, returns whether it has been
    generated (and its version) and whether its prerequisites currently
    exist, naming any that are missing. A workspace with no business
    profile is a normal state here, not an error - the profile shows up
    as a missing prerequisite.
    """

    return StrategyStatusService(session).get_status(workspace_id)


@router.post(
    "/generate-full-strategy",
    response_model=StrategyGenerateResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate every missing stage of the strategy chain, in order",
    responses={404: {"description": "Workspace not found."}},
)
def generate_full_strategy(
    workspace_id: UUID,
    payload: StrategyGenerateRequest,
    session: DbSession,
    gateway: Gateway,
    embeddings: Embeddings,
    web_search: WebSearch,
    request: Request,
) -> StrategyGenerateResponse:
    """Run all seven agents in dependency order for one workspace.

    Synchronous: the request returns when the run finishes or stops, and
    a full run includes several LLM-backed stages, so it can take a long
    time - clients should use a generous timeout. Stages that already
    exist are skipped at no cost unless force_regenerate is set, which
    regenerates every stage.

    The run stops at the first stage that fails, or whose prerequisites
    still await approval. Either is reported in the body with HTTP 200,
    not as an error status: the response lists every stage as generated,
    skipped, failed, awaiting_approval, or not_attempted, and each stage
    saves independently, so calling again without force_regenerate
    resumes from the stage that stopped the run. A stage awaiting
    approval names the stages to approve first (unapproved_prerequisites);
    approve them through the approval routes and call again. Only a
    missing workspace is a 404.

    auto_approve keeps the run going past those checkpoints without a
    person: each stage the run generates is approved as soon as it is
    saved, recorded as an automatic approval. It never approves a stage
    that was only skipped, since a person may not have reviewed it.
    """

    settings = cast(Settings, request.app.state.settings)
    orchestrator = StrategyOrchestrationService(
        StrategyStatusService(session),
        build_stage_generators(session, settings, gateway, embeddings, web_search),
        approver=StageApprovalService(session),
    )

    return orchestrator.generate_full_strategy(
        workspace_id,
        force_regenerate=payload.force_regenerate,
        auto_approve=payload.auto_approve,
    )
