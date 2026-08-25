"""Unit tests for the LLM gateway readiness API contract."""

from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.llm.errors import LLMProviderUnavailableError
from backend.app.llm.health import LLMHealthSnapshot
from backend.app.main import create_application


class HealthyLLMChecker:
    """Return a deterministic healthy LLM gateway result."""

    def check(self) -> LLMHealthSnapshot:
        return LLMHealthSnapshot(provider="ollama", model="gpt-oss:120b-cloud")


class UnavailableLLMChecker:
    """Simulate an LLM provider connectivity failure."""

    def check(self) -> LLMHealthSnapshot:
        raise LLMProviderUnavailableError("internal connection detail")


def test_llm_health_endpoint_returns_expected_contract() -> None:
    """A healthy LLM gateway should return a minimal readiness contract."""

    application = create_application(
        Settings(environment="test"),
        llm_health_checker=HealthyLLMChecker(),
    )

    with TestClient(application) as client:
        response = client.get("/api/v1/health/llm")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "provider": "ollama",
        "model": "gpt-oss:120b-cloud",
        "reachable": True,
    }


def test_llm_health_endpoint_sanitizes_failures() -> None:
    """LLM gateway failures should return 503 without internal details."""

    application = create_application(
        Settings(environment="test"),
        llm_health_checker=UnavailableLLMChecker(),
    )

    with TestClient(
        application,
        raise_server_exceptions=False,
    ) as client:
        response = client.get("/api/v1/health/llm")

    assert response.status_code == 503
    assert response.json() == {"detail": "A required LLM operation could not be completed."}
    assert "internal connection detail" not in response.text


def test_llm_health_service_skips_the_probe_call_for_the_local_provider() -> None:
    """The local provider has nothing real to reach, so no call should be made."""

    from backend.app.llm.gateway import LLMGateway
    from backend.app.llm.health import LLMHealthService

    settings = Settings(environment="test", llm_provider="local")
    service = LLMHealthService(LLMGateway(settings), settings)

    snapshot = service.check()

    assert snapshot.provider == "local"
    assert snapshot.reachable is True
