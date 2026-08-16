"""Typed HTTP client used by Streamlit to call the GrowthCrew backend."""

from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from frontend.services.backend_http import (
    BackendHttpClient,
    BackendResponseError,
    BackendServiceUnavailableError,
    BackendUnavailableError,
)

__all__ = [
    "BackendDatabaseHealth",
    "BackendHealth",
    "BackendResponseError",
    "BackendServiceUnavailableError",
    "BackendUnavailableError",
    "GrowthCrewApiClient",
]


class BackendHealth(BaseModel):
    """Frontend copy of the process health-response contract."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"]
    service: str
    version: str
    environment: str


class BackendDatabaseHealth(BaseModel):
    """Frontend copy of the database readiness-response contract."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"]
    database: Literal["reachable"]
    pgvector: Literal["available"]


class GrowthCrewApiClient:
    """Small synchronous API client for Streamlit's request model."""

    def __init__(
        self,
        base_url: str,
        timeout_seconds: float,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._http = BackendHttpClient(base_url, timeout_seconds, transport)

    def get_health(self) -> BackendHealth:
        """Fetch and validate the backend process-health response."""

        payload = self._http.get("/api/v1/health")
        try:
            return BackendHealth.model_validate(payload)
        except ValidationError as exc:
            raise BackendResponseError(
                "The backend returned an invalid process-health response."
            ) from exc

    def get_database_health(self) -> BackendDatabaseHealth:
        """Fetch and validate PostgreSQL and pgvector readiness."""

        payload = self._http.get("/api/v1/health/database")
        try:
            return BackendDatabaseHealth.model_validate(payload)
        except ValidationError as exc:
            raise BackendResponseError(
                "The backend returned an invalid database-health response."
            ) from exc
