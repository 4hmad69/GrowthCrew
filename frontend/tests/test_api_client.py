"""Tests for the frontend-to-backend API client."""

import httpx
import pytest

from frontend.services.api_client import (
    BackendResponseError,
    BackendUnavailableError,
    GrowthCrewApiClient,
)


def test_get_health_returns_validated_model() -> None:
    """A valid backend response should become a typed frontend model."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/health"
        return httpx.Response(
            200,
            json={
                "status": "ok",
                "service": "GrowthCrew API",
                "version": "0.1.0",
                "environment": "test",
            },
        )

    client = GrowthCrewApiClient(
        base_url="http://testserver",
        timeout_seconds=1.0,
        transport=httpx.MockTransport(handler),
    )

    health = client.get_health()

    assert health.status == "ok"
    assert health.service == "GrowthCrew API"


def test_get_health_rejects_invalid_payload() -> None:
    """The frontend should reject a response that breaks the API contract."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/health"
        return httpx.Response(200, json={"status": "ok"})

    client = GrowthCrewApiClient(
        base_url="http://testserver",
        timeout_seconds=1.0,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(BackendResponseError):
        client.get_health()


def test_get_health_reports_connection_failure() -> None:
    """Connection errors should become a user-safe frontend exception."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused", request=request)

    client = GrowthCrewApiClient(
        base_url="http://testserver",
        timeout_seconds=1.0,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(BackendUnavailableError):
        client.get_health()
