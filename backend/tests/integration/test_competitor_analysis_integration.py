"""Real PostgreSQL integration tests for the Competitor Analysis API.

Uses the deterministic "local" LLM and embeddings providers, and no
Tavily key (web search always resolves to no results) - this file is
about proving persistence, the API contract, the zero-known-competitors
fallback, and the "report survives even if the profile it was built
from later disappears" design decision are correct, none of which need
a real model or a real web search. Whether the real LLM produces
genuinely useful, well-grounded competitor analysis, and whether
force_regenerate produces genuinely different content against a real
model, is covered separately in
test_competitor_analysis_llm_integration.py against real Ollama Cloud,
gated behind GROWTHCREW_RUN_LLM_INTEGRATION_TESTS.
"""

import os
from collections.abc import Iterator
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, select

from backend.app.config import Settings
from backend.app.db.database import Database
from backend.app.db.models.business_profile import BusinessProfile
from backend.app.db.models.workspace import Workspace
from backend.app.embeddings.gateway import EmbeddingsGateway
from backend.app.main import create_application
from backend.app.services.retrieval import RetrievalService

RUN_INTEGRATION_TESTS = os.getenv("GROWTHCREW_RUN_INTEGRATION_TESTS") == "1"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not RUN_INTEGRATION_TESTS,
        reason=("Set GROWTHCREW_RUN_INTEGRATION_TESTS=1 to run PostgreSQL integration tests."),
    ),
]


@pytest.fixture
def integration_context() -> Iterator[tuple[TestClient, Settings]]:
    """Create an API client backed by real PostgreSQL and the local stubs."""

    settings = Settings(environment="test", llm_provider="local", embeddings_provider="local")

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


def _create_workspace_with_profile(
    client: TestClient,
    name: str,
    *,
    known_competitors: list[str] | None = None,
) -> UUID:
    """Create a workspace and a minimal business profile, return the workspace ID."""

    workspace_response = client.post("/api/v1/workspaces", json={"name": name})
    assert workspace_response.status_code == 201
    workspace_id = UUID(workspace_response.json()["id"])

    profile_payload = {"business_name": name, "industry": "Healthy Food"}
    if known_competitors is not None:
        profile_payload["known_competitors"] = known_competitors

    profile_response = client.post(
        f"/api/v1/workspaces/{workspace_id}/business-profile",
        json=profile_payload,
    )
    assert profile_response.status_code == 201

    return workspace_id


def test_get_before_generation_returns_404(
    integration_context: tuple[TestClient, Settings],
) -> None:
    """No report exists yet, so GET should 404 with a clear message."""

    client, settings = integration_context
    workspace_id = _create_workspace_with_profile(client, "Pre-Generation Analysis Co")

    try:
        response = client.get(f"/api/v1/workspaces/{workspace_id}/competitor-analysis")
        assert response.status_code == 404
    finally:
        cleanup_workspace(settings, workspace_id)


def test_generate_creates_and_persists_a_report(
    integration_context: tuple[TestClient, Settings],
) -> None:
    """Generating should create a record that a subsequent GET can retrieve."""

    client, settings = integration_context
    workspace_id = _create_workspace_with_profile(
        client, "Generate Analysis Co", known_competitors=["RivalCo", "AltBrand"]
    )
    url = f"/api/v1/workspaces/{workspace_id}/competitor-analysis"

    try:
        generate_response = client.post(url, json={})
        assert generate_response.status_code == 200

        generated = generate_response.json()
        assert generated["version"] == 1
        assert generated["competitor_overview"]
        assert generated["strengths_and_weaknesses"]
        assert generated["pricing_and_positioning"]
        assert generated["differentiation_opportunities"]
        assert generated["sources"] == []  # nothing indexed, no Tavily key -> nothing to cite
        assert generated["model_used"]

        get_response = client.get(url)
        assert get_response.status_code == 200
        assert get_response.json()["id"] == generated["id"]
    finally:
        cleanup_workspace(settings, workspace_id)


def test_generate_without_known_competitors_still_produces_all_sections(
    integration_context: tuple[TestClient, Settings],
) -> None:
    """With no named competitors, the report should still generate fully.

    Proves the discovery-fallback design decision: _build_queries() asks
    the graph to identify likely competitors itself rather than leaving
    a section empty or failing outright when known_competitors is [].
    """

    client, settings = integration_context
    workspace_id = _create_workspace_with_profile(client, "No Competitors Named Co")
    url = f"/api/v1/workspaces/{workspace_id}/competitor-analysis"

    try:
        response = client.post(url, json={})
        assert response.status_code == 200

        generated = response.json()
        assert generated["competitor_overview"]
        assert generated["strengths_and_weaknesses"]
        assert generated["pricing_and_positioning"]
        assert generated["differentiation_opportunities"]
    finally:
        cleanup_workspace(settings, workspace_id)


def test_generate_without_force_regenerate_returns_existing_record(
    integration_context: tuple[TestClient, Settings],
) -> None:
    """A second generate call should return the same record, not a new one."""

    client, settings = integration_context
    workspace_id = _create_workspace_with_profile(client, "Idempotent Analysis Co")
    url = f"/api/v1/workspaces/{workspace_id}/competitor-analysis"

    try:
        first = client.post(url, json={}).json()
        second = client.post(url, json={}).json()

        assert second["id"] == first["id"]
        assert second["version"] == first["version"]
    finally:
        cleanup_workspace(settings, workspace_id)


def test_force_regenerate_reuses_the_same_row(
    integration_context: tuple[TestClient, Settings],
) -> None:
    """force_regenerate should update the existing row, not create a duplicate."""

    client, settings = integration_context
    workspace_id = _create_workspace_with_profile(client, "Regenerate Analysis Co")
    url = f"/api/v1/workspaces/{workspace_id}/competitor-analysis"

    try:
        first = client.post(url, json={}).json()
        second = client.post(url, json={"force_regenerate": True}).json()

        assert second["id"] == first["id"]  # same row - the unique constraint holds
    finally:
        cleanup_workspace(settings, workspace_id)


def test_generate_without_a_business_profile_returns_404(
    integration_context: tuple[TestClient, Settings],
) -> None:
    """A workspace with no business profile yet should 404, not 500."""

    client, settings = integration_context

    workspace_response = client.post(
        "/api/v1/workspaces",
        json={"name": "No Profile Analysis Co"},
    )
    assert workspace_response.status_code == 201
    workspace_id = UUID(workspace_response.json()["id"])

    try:
        response = client.post(
            f"/api/v1/workspaces/{workspace_id}/competitor-analysis",
            json={},
        )
        assert response.status_code == 404
        assert response.json() == {"detail": "Business profile not found."}
    finally:
        cleanup_workspace(settings, workspace_id)


def test_generate_for_missing_workspace_returns_404(
    integration_context: tuple[TestClient, Settings],
) -> None:
    """A nonexistent workspace should 404, not 500."""

    client, _settings = integration_context

    response = client.post(
        f"/api/v1/workspaces/{UUID(int=0)}/competitor-analysis",
        json={},
    )
    assert response.status_code == 404
    assert response.json() == {"detail": "Workspace not found."}


def test_get_survives_the_business_profile_being_deleted(
    integration_context: tuple[TestClient, Settings],
) -> None:
    """A generated report stays readable even after its source profile is gone.

    Deliberately different from Business Understanding: CompetitorAnalysis
    is keyed by workspace_id, not profile_id, and generate()/get() were
    written to treat a finished report as a self-contained document, not
    a live view over the profile - this proves that design decision
    actually holds against real Postgres foreign-key behavior.
    """

    client, settings = integration_context
    workspace_id = _create_workspace_with_profile(client, "Orphaned Profile Analysis Co")
    url = f"/api/v1/workspaces/{workspace_id}/competitor-analysis"

    try:
        generated = client.post(url, json={}).json()

        database = Database.from_settings(settings)
        with database.session() as session:
            profile = session.scalar(
                select(BusinessProfile).where(BusinessProfile.workspace_id == workspace_id)
            )
            session.execute(
                delete(BusinessProfile).where(BusinessProfile.workspace_id == workspace_id)
            )
            session.commit()
        database.dispose()
        assert profile is not None  # sanity: there really was a profile to delete

        response = client.get(url)
        assert response.status_code == 200
        assert response.json()["id"] == generated["id"]
    finally:
        cleanup_workspace(settings, workspace_id)


def test_generate_populates_sources_from_seeded_knowledge_base(
    integration_context: tuple[TestClient, Settings],
) -> None:
    """Sources should reflect what was actually retrieved, not be fabricated."""

    client, settings = integration_context
    workspace_id = _create_workspace_with_profile(client, "Seeded Knowledge Analysis Co")
    url = f"/api/v1/workspaces/{workspace_id}/competitor-analysis"

    try:
        database = Database.from_settings(settings)
        embeddings = EmbeddingsGateway(settings)
        with database.session() as session:
            RetrievalService(session, embeddings).add_chunk(
                workspace_id,
                "internal:competitor-notes",
                "RivalCo charges $2M per launch and targets enterprise customers.",
            )
        database.dispose()

        generated = client.post(url, json={}).json()

        assert generated["sources"]
        for source in generated["sources"]:
            assert set(source) == {"section", "source", "snippet"}
            assert source["source"] == "internal:competitor-notes"
    finally:
        cleanup_workspace(settings, workspace_id)
