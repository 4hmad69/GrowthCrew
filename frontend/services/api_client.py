"""Typed HTTP client used by Streamlit to call the GrowthCrew backend."""

from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError


class BackendHealth(BaseModel):
    """Frontend copy of the backend health-response contract."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"]
    service: str
    version: str
    environment: str


class BackendUnavailableError(RuntimeError):
    """Raised when the backend cannot be reached."""


class BackendResponseError(RuntimeError):
    """Raised when the backend returns an invalid or unsuccessful response."""


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
        """Fetch and validate the backend health response."""

        try:
            with httpx.Client(
                base_url=self._base_url,
                timeout=self._timeout_seconds,
                transport=self._transport,
            ) as client:
                response = client.get("/api/v1/health")
        except httpx.HTTPError as exc:
            raise BackendUnavailableError("The GrowthCrew backend could not be reached.") from exc

        if response.status_code != 200:
            raise BackendResponseError(
                f"The backend health endpoint returned HTTP {response.status_code}."
            )

        try:
            return BackendHealth.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            raise BackendResponseError("The backend returned an invalid health response.") from exc
