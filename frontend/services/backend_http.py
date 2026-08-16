"""Shared synchronous JSON/HTTP plumbing for the GrowthCrew frontend clients.

Every Streamlit-facing API client (health checks, onboarding, and future
domain clients) sends requests through :class:`BackendHttpClient` so that
connection handling, timeout configuration, and status-code translation stay
in exactly one place.
"""

from typing import Any

import httpx


class BackendUnavailableError(RuntimeError):
    """Raised when the backend cannot be reached at all."""


class BackendResponseError(RuntimeError):
    """Raised when the backend returns an invalid or unsuccessful response."""


class BackendServiceUnavailableError(BackendResponseError):
    """Raised when a required backend dependency is not ready (HTTP 503)."""


class BackendConflictError(BackendResponseError):
    """Raised when a request conflicts with current backend state (HTTP 409)."""


class BackendValidationError(BackendResponseError):
    """Raised when the backend rejects a request payload (HTTP 422)."""

    def __init__(self, message: str, *, detail: Any = None) -> None:
        super().__init__(message)
        self.detail = detail


_SUCCESS_STATUS_CODES = frozenset({200, 201})


class BackendHttpClient:
    """Thin synchronous JSON client shared by every GrowthCrew API client."""

    def __init__(
        self,
        base_url: str,
        timeout_seconds: float,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._transport = transport

    def get(self, path: str) -> dict[str, Any]:
        """Perform a GET request and return a validated JSON object shape."""

        return self._request("GET", path)

    def post(self, path: str, json_body: dict[str, Any]) -> dict[str, Any]:
        """Perform a POST request and return a validated JSON object shape."""

        return self._request("POST", path, json_body=json_body)

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send one request and translate the response into JSON or an error."""

        try:
            with httpx.Client(
                base_url=self._base_url,
                timeout=self._timeout_seconds,
                transport=self._transport,
            ) as client:
                response = client.request(method, path, json=json_body)
        except httpx.HTTPError as exc:
            raise BackendUnavailableError("The GrowthCrew backend could not be reached.") from exc

        self._raise_for_status(response)

        try:
            payload = response.json()
        except ValueError as exc:
            raise BackendResponseError("The backend returned invalid JSON.") from exc

        if not isinstance(payload, dict):
            raise BackendResponseError("The backend returned an invalid JSON object.")

        return payload

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        """Translate a non-success status code into a typed frontend error."""

        if response.status_code in _SUCCESS_STATUS_CODES:
            return

        detail = _error_detail(response)

        if response.status_code == 409:
            raise BackendConflictError(detail or "The request conflicts with the current state.")

        if response.status_code == 422:
            raise BackendValidationError(
                detail or "The backend rejected the request payload.",
                detail=_validation_errors(response),
            )

        if response.status_code == 503:
            raise BackendServiceUnavailableError("A required GrowthCrew service is not ready.")

        raise BackendResponseError(f"The backend endpoint returned HTTP {response.status_code}.")


def _safe_json(response: httpx.Response) -> Any:
    """Parse a response body as JSON, tolerating a missing or invalid body."""

    try:
        return response.json()
    except ValueError:
        return None


def _validation_errors(response: httpx.Response) -> list[dict[str, Any]] | None:
    """Return FastAPI's per-field validation error list, if the body has one.

    FastAPI's default 422 body is ``{"detail": [{"loc": ..., "msg": ...}, ...]}``.
    Exposing just the list lets callers map errors onto form fields without
    having to know about the wrapping ``detail`` key.
    """

    payload = _safe_json(response)
    if not isinstance(payload, dict):
        return None

    detail = payload.get("detail")
    if isinstance(detail, list):
        return detail

    return None


def _error_detail(response: httpx.Response) -> str | None:
    """Extract a human-readable message from a FastAPI-style error body."""

    payload = _safe_json(response)
    if not isinstance(payload, dict):
        return None

    detail = payload.get("detail")

    if isinstance(detail, str):
        return detail

    if isinstance(detail, list) and detail:
        first_error = detail[0]
        if isinstance(first_error, dict) and isinstance(first_error.get("msg"), str):
            return first_error["msg"]

    return None
