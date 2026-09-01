"""Unit tests for the embeddings gateway readiness API contract."""

from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.embeddings.errors import EmbeddingsProviderUnavailableError
from backend.app.embeddings.health import EmbeddingsHealthSnapshot
from backend.app.main import create_application


class HealthyEmbeddingsChecker:
    """Return a deterministic healthy embeddings gateway result."""

    def check(self) -> EmbeddingsHealthSnapshot:
        return EmbeddingsHealthSnapshot(provider="ollama", model="nomic-embed-text", dimension=768)


class UnavailableEmbeddingsChecker:
    """Simulate an embeddings provider connectivity failure."""

    def check(self) -> EmbeddingsHealthSnapshot:
        raise EmbeddingsProviderUnavailableError("internal connection detail")


def test_embeddings_health_endpoint_returns_expected_contract() -> None:
    """A healthy embeddings gateway should return a minimal readiness contract."""

    application = create_application(
        Settings(environment="test"),
        embeddings_health_checker=HealthyEmbeddingsChecker(),
    )

    with TestClient(application) as client:
        response = client.get("/api/v1/health/embeddings")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "provider": "ollama",
        "model": "nomic-embed-text",
        "dimension": 768,
        "reachable": True,
    }


def test_embeddings_health_endpoint_sanitizes_failures() -> None:
    """Embeddings gateway failures should return 503 without internal details."""

    application = create_application(
        Settings(environment="test"),
        embeddings_health_checker=UnavailableEmbeddingsChecker(),
    )

    with TestClient(
        application,
        raise_server_exceptions=False,
    ) as client:
        response = client.get("/api/v1/health/embeddings")

    assert response.status_code == 503
    assert response.json() == {"detail": "A required embeddings operation could not be completed."}
    assert "internal connection detail" not in response.text


def test_embeddings_health_service_skips_the_probe_call_for_the_local_provider() -> None:
    """The local provider has nothing real to reach, so no call should be made."""

    from backend.app.embeddings.gateway import EmbeddingsGateway
    from backend.app.embeddings.health import EmbeddingsHealthService

    settings = Settings(environment="test", embeddings_provider="local")
    service = EmbeddingsHealthService(EmbeddingsGateway(settings), settings)

    snapshot = service.check()

    assert snapshot.provider == "local"
    assert snapshot.reachable is True


def test_embeddings_health_service_detects_a_dimension_mismatch() -> None:
    """A provider returning the wrong dimension should raise, not silently pass."""

    from backend.app.embeddings.errors import EmbeddingsResponseError
    from backend.app.embeddings.health import EmbeddingsHealthService

    class _WrongDimensionGateway:
        def embed_query(self, text: str) -> list[float]:
            return [0.0, 0.0, 0.0]  # deliberately not 768-dimensional

    settings = Settings(environment="test", embeddings_provider="ollama")
    service = EmbeddingsHealthService(_WrongDimensionGateway(), settings)

    try:
        service.check()
    except EmbeddingsResponseError as exc:
        assert "768" in str(exc)
        assert "3" in str(exc)
    else:
        raise AssertionError("Expected EmbeddingsResponseError to be raised")
