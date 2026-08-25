"""Health and readiness API schemas."""

from typing import Literal

from backend.app.schemas.common import StrictSchema


class HealthResponse(StrictSchema):
    """Response returned by the process-health endpoint."""

    status: Literal["ok"]
    service: str
    version: str
    environment: str


class DatabaseHealthResponse(StrictSchema):
    """Response returned when PostgreSQL and pgvector are ready."""

    status: Literal["ok"]
    database: Literal["reachable"]
    pgvector: Literal["available"]


class LLMHealthResponse(StrictSchema):
    """Response returned when the configured LLM provider is reachable."""

    status: Literal["ok"]
    provider: str
    model: str
    reachable: Literal[True]
