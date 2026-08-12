"""Contract tests for the foundation API."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_application


def build_test_application() -> FastAPI:
    """Create a test-configured application."""

    return create_application(
        Settings(
            app_name="GrowthCrew Test API",
            app_version="0.2.0-test",
            environment="test",
        )
    )


def test_health_endpoint_returns_expected_contract() -> None:
    """The process health endpoint should expose a stable typed response."""

    application = build_test_application()

    with TestClient(application) as client:
        response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "GrowthCrew Test API",
        "version": "0.2.0-test",
        "environment": "test",
    }


def test_unexpected_errors_are_not_exposed_to_clients() -> None:
    """Unexpected exceptions should return a generic safe message."""

    application = build_test_application()

    @application.get("/test-only-error", include_in_schema=False)
    def raise_test_error() -> None:
        raise RuntimeError("sensitive internal detail")

    with TestClient(application, raise_server_exceptions=False) as client:
        response = client.get("/test-only-error")

    assert response.status_code == 500
    assert response.json() == {"detail": "An unexpected server error occurred."}
    assert "sensitive internal detail" not in response.text
