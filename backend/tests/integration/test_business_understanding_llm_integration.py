"""Real integration tests for Business Understanding against Postgres AND Ollama Cloud.

Gated behind both integration flags at once - this is the one test in the
suite that genuinely needs both real dependencies simultaneously (a real
row to persist to, and a real model to generate real content).

Note on force_regenerate: this file does NOT assert that regenerated
content differs from the original, or that the version number bumps.
The gateway fixes temperature at 0 for determinism, so a real model given
an unchanged prompt may legitimately reproduce very similar or identical
output - asserting a content difference here would be a flaky test
failing for correct code, not a real bug. Commit 5's Postgres-only test
already covers the mechanical guarantee (force_regenerate reuses the same
row, never creates a duplicate) with the fully deterministic local
provider, where that assertion is reliable. What's unique to real Ollama
Cloud, and what this file actually verifies, is that generation produces
genuine, non-trivial content and that real (non-zero) token usage
persists correctly through the full service + API + database stack.
"""

import os
from collections.abc import Iterator
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.db.database import Database
from backend.app.db.models.workspace import Workspace
from backend.app.main import create_application

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

    settings = Settings(environment="test", llm_provider="ollama")

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


def test_generate_produces_real_synthesis_with_real_token_usage(
    integration_context: tuple[TestClient, Settings],
) -> None:
    """A real generate call should produce non-trivial content and real usage."""

    client, settings = integration_context

    workspace_response = client.post(
        "/api/v1/workspaces",
        json={"name": "FitMeal LLM Integration"},
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
            "main_marketing_goal": "Increase monthly subscriptions",
            "brand_tone": "Friendly and energetic",
            "known_competitors": ["HelloFresh", "Factor"],
        },
    )
    assert profile_response.status_code == 201

    try:
        response = client.post(
            f"/api/v1/workspaces/{workspace_id}/business-profile/understanding",
            json={},
        )
        assert response.status_code == 200

        body = response.json()
        assert len(body["summary"]) > 20
        assert body["inferred_business_stage"]
        assert body["competitive_category"]
        assert body["input_tokens"] > 0
        assert body["output_tokens"] > 0

        # confirm real usage actually persisted, not just present in the response
        get_response = client.get(
            f"/api/v1/workspaces/{workspace_id}/business-profile/understanding"
        )
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
        json={"name": "Regenerate LLM Integration"},
    )
    assert workspace_response.status_code == 201
    workspace_id = UUID(workspace_response.json()["id"])

    profile_response = client.post(
        f"/api/v1/workspaces/{workspace_id}/business-profile",
        json={"business_name": "Regenerate Co", "industry": "Consulting"},
    )
    assert profile_response.status_code == 201

    url = f"/api/v1/workspaces/{workspace_id}/business-profile/understanding"

    try:
        first = client.post(url, json={})
        assert first.status_code == 200

        second = client.post(url, json={"force_regenerate": True})
        assert second.status_code == 200
        assert second.json()["id"] == first.json()["id"]
    finally:
        cleanup_workspace(settings, workspace_id)
