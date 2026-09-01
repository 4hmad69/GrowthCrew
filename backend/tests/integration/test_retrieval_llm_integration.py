"""Real integration tests for retrieval against Postgres AND real Ollama Cloud.

Gated behind both integration flags at once, same as the Business
Understanding LLM integration test - this is the one place in the suite
that can actually prove retrieval *quality*, not just mechanics.
test_retrieval_integration.py (the Postgres-only file) uses the
deterministic local provider, which has zero real semantic
understanding - "the pricing chunk ranked first for a pricing query"
there is coincidental, not meaningful. Here, with real nomic-embed-text
embeddings, a query that shares no exact keywords with the right chunk
still has to rank it first for these tests to pass - that's the actual
thing worth proving.
"""

import os

import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.db.database import Database
from backend.app.db.models.workspace import Workspace
from backend.app.embeddings.gateway import EmbeddingsGateway
from backend.app.main import create_application
from backend.app.services.retrieval import RetrievalService

RUN_INTEGRATION_TESTS = os.getenv("GROWTHCREW_RUN_INTEGRATION_TESTS") == "1"
RUN_LLM_INTEGRATION_TESTS = os.getenv("GROWTHCREW_RUN_LLM_INTEGRATION_TESTS") == "1"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (RUN_INTEGRATION_TESTS and RUN_LLM_INTEGRATION_TESTS),
        reason=(
            "Set both GROWTHCREW_RUN_INTEGRATION_TESTS=1 and "
            "GROWTHCREW_RUN_LLM_INTEGRATION_TESTS=1 to run this test "
            "(needs real Postgres AND real Ollama Cloud)."
        ),
    ),
]


@pytest.fixture
def real_settings() -> Settings:
    """Settings pointed at real Postgres and real Ollama Cloud embeddings."""

    return Settings(environment="test", embeddings_provider="ollama")


def test_embeddings_health_endpoint_reports_real_ollama_cloud(real_settings: Settings) -> None:
    """The health endpoint should confirm real embeddings connectivity end to end."""

    application = create_application(real_settings)

    with TestClient(application) as client:
        response = client.get("/api/v1/health/embeddings")

    assert response.status_code == 200
    body = response.json()
    assert body["provider"] == "ollama"
    assert body["dimension"] == 768
    assert body["reachable"] is True


def test_search_finds_semantically_relevant_chunk_without_keyword_overlap(
    real_settings: Settings,
) -> None:
    """A query sharing no exact keywords with the right chunk should still rank it first.

    This is the actual proof of retrieval quality: "How much does this
    cost?" shares zero words with "Plans start at twenty-nine dollars a
    month" - a keyword-matching system would fail this outright. Only a
    model that genuinely understands the query and the content
    semantically can get this right.
    """

    if real_settings.database_url is None:
        pytest.fail("GROWTHCREW_DATABASE_URL is required for integration tests.")

    database = Database.from_settings(real_settings)
    embeddings = EmbeddingsGateway(real_settings)

    with database.session() as session:
        workspace = Workspace(name="Semantic Search Integration Co")
        session.add(workspace)
        session.commit()
        workspace_id = workspace.id

    try:
        with database.session() as session:
            service = RetrievalService(session, embeddings)
            service.add_chunks(
                workspace_id,
                [
                    ("test:pricing", "Plans start at twenty-nine dollars a month."),
                    (
                        "test:competitors",
                        "We are often compared to HelloFresh and Factor in the meal-kit space.",
                    ),
                    (
                        "test:customers",
                        "Our subscribers are mostly busy professionals in their "
                        "late twenties and thirties.",
                    ),
                    (
                        "test:seasonality",
                        "Signups spike every January as people commit to new year health goals.",
                    ),
                ],
            )

        with database.session() as session:
            service = RetrievalService(session, embeddings)
            results = service.search(workspace_id, "How much does this cost?", limit=1)

        assert len(results) == 1
        assert results[0].source == "test:pricing"
    finally:
        with database.session() as session:
            workspace = session.get(Workspace, workspace_id)
            if workspace is not None:
                session.delete(workspace)
                session.commit()
        database.dispose()
