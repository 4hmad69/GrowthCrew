"""Real integration tests for Competitor Analysis against Postgres AND Ollama Cloud.

Gated behind both integration flags at once, same as the Market
Research, Business Understanding, and retrieval LLM integration tests -
this is the one place in this agent's suite that can prove the whole
thing actually works: real generation, real embeddings, and (if
TAVILY_API_KEY happens to be configured in your environment) real web
search, all the way through persistence.

Note on force_regenerate: this file does NOT assert that regenerated
content differs from the original, or that the version number bumps.
LLMGateway fixes temperature at 0 for determinism, so a real model given
an unchanged prompt may legitimately reproduce very similar or identical
output for some or all of the four sections - asserting a content
difference here would be a flaky test failing for correct code. Commit
5's Postgres-only test already covers the mechanical guarantee
(force_regenerate reuses the same row, never creates a duplicate) with
the fully deterministic local provider, where that assertion is
reliable.

Note on sources: this file does NOT assert that `sources` is non-empty
for a fresh workspace with no seeded knowledge base, since whether any
section routes to the web (and therefore whether it finds anything)
depends on whether TAVILY_API_KEY happens to be configured wherever this
runs. What's provable regardless of that is covered by
test_generate_finds_and_cites_a_seeded_chunk_with_real_embeddings below,
which seeds a real chunk so there's always something a vectorstore-routed
section can find.
"""

import os
from collections.abc import Iterator
from uuid import UUID

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
def integration_context() -> Iterator[tuple[TestClient, Settings]]:
    """Create an API client backed by real PostgreSQL and real Ollama Cloud."""

    settings = Settings(environment="test", llm_provider="ollama", embeddings_provider="ollama")

    if settings.database_url is None:
        pytest.fail("GROWTHCREW_DATABASE_URL is required for integration tests.")

    application = create_application(settings)

    with TestClient(application) as client:
        yield client, settings


def cleanup_workspace(settings: Settings, workspace_id: UUID) -> None:
    """Remove only test-created data after an integration test."""

    database = Database.from_settings(settings)

    try:
        with database.session() as session:
            workspace = session.get(Workspace, workspace_id)
            if workspace is not None:
                session.delete(workspace)
                session.commit()
    finally:
        database.dispose()


def test_generate_produces_real_analysis_with_real_token_usage(
    integration_context: tuple[TestClient, Settings],
) -> None:
    """A real generate call should produce non-trivial content and real usage."""

    client, settings = integration_context

    workspace_response = client.post(
        "/api/v1/workspaces",
        json={"name": "FitMeal Analysis LLM Integration"},
    )
    assert workspace_response.status_code == 201
    workspace_id = UUID(workspace_response.json()["id"])

    profile_response = client.post(
        f"/api/v1/workspaces/{workspace_id}/business-profile",
        json={
            "business_name": "FitMeal",
            "product_or_service": "Weekly meal-prep subscription boxes",
            "industry": "Healthy Food",
            "target_customer": "Busy professionals who want to eat healthy",
            "price_range": "$60-$90 per week",
            "known_competitors": ["HelloFresh", "Factor"],
        },
    )
    assert profile_response.status_code == 201

    try:
        response = client.post(
            f"/api/v1/workspaces/{workspace_id}/competitor-analysis",
            json={},
        )
        assert response.status_code == 200

        body = response.json()
        assert len(body["competitor_overview"]) > 20
        assert len(body["strengths_and_weaknesses"]) > 20
        assert len(body["pricing_and_positioning"]) > 20
        assert len(body["differentiation_opportunities"]) > 20
        assert body["model_used"]
        assert body["input_tokens"] > 0
        assert body["output_tokens"] > 0

        # confirm real usage actually persisted, not just present in the response
        get_response = client.get(f"/api/v1/workspaces/{workspace_id}/competitor-analysis")
        assert get_response.status_code == 200
        persisted = get_response.json()
        assert persisted["input_tokens"] == body["input_tokens"]
        assert persisted["output_tokens"] == body["output_tokens"]
    finally:
        cleanup_workspace(settings, workspace_id)


def test_force_regenerate_succeeds_against_real_ollama_cloud(
    integration_context: tuple[TestClient, Settings],
) -> None:
    """force_regenerate should succeed end to end and keep the same row."""

    client, settings = integration_context

    workspace_response = client.post(
        "/api/v1/workspaces",
        json={"name": "Regenerate Analysis LLM Integration"},
    )
    assert workspace_response.status_code == 201
    workspace_id = UUID(workspace_response.json()["id"])

    profile_response = client.post(
        f"/api/v1/workspaces/{workspace_id}/business-profile",
        json={"business_name": "Regenerate Analysis Co", "industry": "Consulting"},
    )
    assert profile_response.status_code == 201

    url = f"/api/v1/workspaces/{workspace_id}/competitor-analysis"

    try:
        first = client.post(url, json={})
        assert first.status_code == 200

        second = client.post(url, json={"force_regenerate": True})
        assert second.status_code == 200
        assert second.json()["id"] == first.json()["id"]
    finally:
        cleanup_workspace(settings, workspace_id)


def test_generate_finds_and_cites_a_seeded_chunk_with_real_embeddings(
    integration_context: tuple[TestClient, Settings],
) -> None:
    """A section that routes to the vectorstore should find and cite real content.

    Mirrors test_retrieval_llm_integration.py's proof of genuine semantic
    understanding: the seeded chunk shares no exact keywords with the
    business profile's industry or product description, so a real
    embedding + real relevance grading has to do real work to surface and
    cite it - not just coincidentally match on shared words.
    """

    client, settings = integration_context

    workspace_response = client.post(
        "/api/v1/workspaces",
        json={"name": "Seeded Knowledge Analysis LLM Integration"},
    )
    assert workspace_response.status_code == 201
    workspace_id = UUID(workspace_response.json()["id"])

    profile_response = client.post(
        f"/api/v1/workspaces/{workspace_id}/business-profile",
        json={
            "business_name": "Acme Rockets",
            "industry": "Aerospace",
            "known_competitors": ["Orbital Dynamics"],
        },
    )
    assert profile_response.status_code == 201

    try:
        database = Database.from_settings(settings)
        embeddings = EmbeddingsGateway(settings)
        with database.session() as session:
            RetrievalService(session, embeddings).add_chunk(
                workspace_id,
                "internal:competitor-pricing",
                "Orbital Dynamics charges two million dollars for a single "
                "reusable booster launch.",
            )
        database.dispose()

        response = client.post(
            f"/api/v1/workspaces/{workspace_id}/competitor-analysis",
            json={},
        )
        assert response.status_code == 200

        sources = response.json()["sources"]
        assert any(source["source"] == "internal:competitor-pricing" for source in sources)
    finally:
        cleanup_workspace(settings, workspace_id)
