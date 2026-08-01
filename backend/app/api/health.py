"""Health-check route."""

from typing import cast

from fastapi import APIRouter, Request, status

from backend.app.config import Settings
from backend.app.schemas import HealthResponse

router = APIRouter(prefix="/health", tags=["health"])


@router.get(
    "",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Check API health",
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
