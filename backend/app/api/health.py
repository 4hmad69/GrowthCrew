"""Process and database health-check routes."""

from typing import cast

from fastapi import APIRouter, Request, status

from backend.app.config import Settings
from backend.app.db.health import DatabaseHealthChecker
from backend.app.schemas import DatabaseHealthResponse, HealthResponse

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
