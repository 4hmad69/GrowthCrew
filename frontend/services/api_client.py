"""Typed HTTP client used by Streamlit to call the GrowthCrew backend."""

from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError


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


class BackendUnavailableError(RuntimeError):
    """Raised when the backend cannot be reached."""


class BackendResponseError(RuntimeError):
    """Raised when the backend returns an invalid or unsuccessful response."""


class BackendServiceUnavailableError(BackendResponseError):
    """Raised when a required backend dependency is not ready."""


class GrowthCrewApiClient:
    """Small synchronous API client for Streamlit's request model."""

    def __init__(
        self,
        base_url: str,
        timeout_seconds: float,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._transport = transport

    def get_health(self) -> BackendHealth:
        """Fetch and validate the backend process-health response."""

        payload = self._get_json("/api/v1/health")
        try:
            return BackendHealth.model_validate(payload)
        except ValidationError as exc:
            raise BackendResponseError(
                "The backend returned an invalid process-health response."
            ) from exc

    def get_database_health(self) -> BackendDatabaseHealth:
        """Fetch and validate PostgreSQL and pgvector readiness."""

        payload = self._get_json("/api/v1/health/database")
        try:
            return BackendDatabaseHealth.model_validate(payload)
        except ValidationError as exc:
            raise BackendResponseError(
                "The backend returned an invalid database-health response."
            ) from exc

    def _get_json(self, path: str) -> dict[str, Any]:
        """Perform one GET request and return a validated JSON object shape."""

        try:
            with httpx.Client(
                base_url=self._base_url,
                timeout=self._timeout_seconds,
                transport=self._transport,
            ) as client:
                response = client.get(path)
        except httpx.HTTPError as exc:
            raise BackendUnavailableError(
                "The GrowthCrew backend could not be reached."
            ) from exc

        if response.status_code == 503:
            raise BackendServiceUnavailableError(
                "A required GrowthCrew service is not ready."
            )

        if response.status_code != 200:
            raise BackendResponseError(
                f"The backend endpoint returned HTTP {response.status_code}."
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise BackendResponseError(
                "The backend returned invalid JSON."
            ) from exc

        if not isinstance(payload, dict):
            raise BackendResponseError(
                "The backend returned an invalid JSON object."
            )

        return payload