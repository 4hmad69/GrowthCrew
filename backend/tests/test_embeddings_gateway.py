"""Tests for the embeddings gateway."""

import pytest

from backend.app.config import Settings
from backend.app.embeddings.gateway import EmbeddingsGateway


class _TransientFailure(Exception):
    """Stand-in for a provider error that looks transient."""


class _FakeEmbeddingsClient:
    """A .embed_query()/.embed_documents()-able test double."""

    def __init__(
        self,
        *,
        query_result: list[float] | None = None,
        documents_result: list[list[float]] | None = None,
        fail_times: int = 0,
        error: Exception | None = None,
    ) -> None:
        self._query_result = query_result
        self._documents_result = documents_result
        self._fail_times = fail_times
        self._error = error or _TransientFailure("connection timed out")
        self.query_calls = 0
        self.documents_calls = 0

    def embed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        if self.query_calls <= self._fail_times:
            raise self._error
        return self._query_result or [0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.documents_calls += 1
        if self.documents_calls <= self._fail_times:
            raise self._error
        return self._documents_result or [[0.0] for _ in texts]


def _gateway_with_fake_client(
    fake_client: _FakeEmbeddingsClient,
    **settings_overrides: object,
) -> EmbeddingsGateway:
    """Build a real EmbeddingsGateway, then swap in a fake client (no network)."""

    settings = Settings(
        environment="test",
        embeddings_provider="ollama",
        llm_retry_initial_delay_seconds=0.1,
        **settings_overrides,
    )
    gateway = EmbeddingsGateway.__new__(EmbeddingsGateway)
    gateway._settings = settings
    gateway._client = fake_client
    return gateway


# --- local provider -----------------------------------------------------------


def test_embed_query_local_provider_returns_correct_dimension() -> None:
    gateway = EmbeddingsGateway(Settings(environment="test", embeddings_provider="local"))

    vector = gateway.embed_query("pricing information")

    assert len(vector) == 768


def test_embed_query_local_provider_is_deterministic() -> None:
    gateway = EmbeddingsGateway(Settings(environment="test", embeddings_provider="local"))

    first = gateway.embed_query("pricing information")
    second = gateway.embed_query("pricing information")

    assert first == second


def test_embed_documents_local_provider_returns_one_vector_per_text() -> None:
    gateway = EmbeddingsGateway(Settings(environment="test", embeddings_provider="local"))

    vectors = gateway.embed_documents(["a", "b", "c"])

    assert len(vectors) == 3
    assert all(len(vector) == 768 for vector in vectors)


# --- ollama provider ------------------------------------------------------------


def test_build_client_constructs_a_real_ollama_embeddings_client() -> None:
    gateway = EmbeddingsGateway(Settings(environment="test", embeddings_provider="ollama"))

    assert type(gateway._client).__name__ == "OllamaEmbeddings"
    assert gateway._client.model == "nomic-embed-text"


def test_embed_query_ollama_provider_returns_client_result() -> None:
    fake_client = _FakeEmbeddingsClient(query_result=[0.1, 0.2, 0.3])
    gateway = _gateway_with_fake_client(fake_client)

    result = gateway.embed_query("pricing information")

    assert result == [0.1, 0.2, 0.3]
    assert fake_client.query_calls == 1


def test_embed_documents_ollama_provider_returns_client_result() -> None:
    fake_client = _FakeEmbeddingsClient(documents_result=[[0.1], [0.2]])
    gateway = _gateway_with_fake_client(fake_client)

    result = gateway.embed_documents(["a", "b"])

    assert result == [[0.1], [0.2]]
    assert fake_client.documents_calls == 1


def test_embed_query_retries_transient_failures_then_succeeds() -> None:
    fake_client = _FakeEmbeddingsClient(query_result=[0.5], fail_times=2)
    gateway = _gateway_with_fake_client(fake_client, llm_retry_attempts=5)

    result = gateway.embed_query("pricing information")

    assert result == [0.5]
    assert fake_client.query_calls == 3


def test_embed_query_raises_immediately_on_non_transient_error() -> None:
    fake_client = _FakeEmbeddingsClient(fail_times=1, error=ValueError("bad request"))
    gateway = _gateway_with_fake_client(fake_client, llm_retry_attempts=5)

    with pytest.raises(ValueError, match="bad request"):
        gateway.embed_query("pricing information")

    assert fake_client.query_calls == 1
