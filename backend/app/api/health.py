"""Process, database, LLM gateway, and embeddings gateway health-check routes."""

from typing import cast

from fastapi import APIRouter, Request, status

from backend.app.config import Settings
from backend.app.db.health import DatabaseHealthChecker
from backend.app.embeddings.health import EmbeddingsHealthChecker
from backend.app.llm.health import LLMHealthChecker
from backend.app.schemas.health import (
    DatabaseHealthResponse,
    EmbeddingsHealthResponse,
    HealthResponse,
    LLMHealthResponse,
)

router = APIRouter(prefix="/health", tags=["health"])


@router.get(
    "",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Check API process health",
)
def read_health(request: Request) -> HealthResponse:
    """Confirm that the FastAPI process is running and configured."""

    settings = cast(Settings, request.app.state.settings)
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
    )


@router.get(
    "/database",
    response_model=DatabaseHealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Check database readiness",
    responses={503: {"description": "PostgreSQL or pgvector is not ready."}},
)
def read_database_health(request: Request) -> DatabaseHealthResponse:
    """Confirm PostgreSQL connectivity and pgvector availability."""

    checker = cast(
        DatabaseHealthChecker,
        request.app.state.database_health_checker,
    )
    snapshot = checker.check()

    return DatabaseHealthResponse(
        status="ok",
        database=snapshot.database,
        pgvector=snapshot.pgvector,
    )


@router.get(
    "/llm",
    response_model=LLMHealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Check LLM gateway readiness",
    responses={503: {"description": "The configured LLM provider is not reachable."}},
)
def read_llm_health(request: Request) -> LLMHealthResponse:
    """Confirm the configured LLM provider responds to a minimal prompt."""

    checker = cast(LLMHealthChecker, request.app.state.llm_health_checker)
    snapshot = checker.check()

    return LLMHealthResponse(
        status="ok",
        provider=snapshot.provider,
        model=snapshot.model,
        reachable=snapshot.reachable,
    )


@router.get(
    "/embeddings",
    response_model=EmbeddingsHealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Check embeddings gateway readiness",
    responses={503: {"description": "The configured embeddings provider is not reachable."}},
)
def read_embeddings_health(request: Request) -> EmbeddingsHealthResponse:
    """Confirm the configured embeddings provider responds with the expected dimension."""

    checker = cast(
        EmbeddingsHealthChecker,
        request.app.state.embeddings_health_checker,
    )
    snapshot = checker.check()

    return EmbeddingsHealthResponse(
        status="ok",
        provider=snapshot.provider,
        model=snapshot.model,
        dimension=snapshot.dimension,
        reachable=snapshot.reachable,
    )
