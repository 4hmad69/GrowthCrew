"""Real PostgreSQL integration tests for the Business Understanding API.

Uses the deterministic "local" LLM provider (see Settings.llm_provider),
not real Ollama Cloud - this file is about proving persistence and the
API contract are correct, which doesn't need a real model. Whether the
real LLM produces good synthesis, and whether force_regenerate produces
genuinely different content, is covered separately in
test_business_understanding_llm_integration.py against real Ollama Cloud
gated behind GROWTHCREW_RUN_LLM_INTEGRATION_TESTS.
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

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not RUN_INTEGRATION_TESTS,
        reason=("Set GROWTHCREW_RUN_INTEGRATION_TESTS=1 to run PostgreSQL integration tests."),
    ),
]


@pytest.fixture
def integration_context() -> Iterator[tuple[TestClient, Settings]]:
    """Create an API client backed by real PostgreSQL and the local LLM stub."""

    settings = Settings(environment="test", llm_provider="local")

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


def _create_workspace_with_profile(client: TestClient, name: str) -> UUID:
    """Create a workspace and a minimal business profile, return the workspace ID."""

    workspace_response = client.post("/api/v1/workspaces", json={"name": name})
    assert workspace_response.status_code == 201
    workspace_id = UUID(workspace_response.json()["id"])

    profile_response = client.post(
        f"/api/v1/workspaces/{workspace_id}/business-profile",
        json={"business_name": name, "industry": "Healthy Food"},
    )
    assert profile_response.status_code == 201

    return workspace_id


def test_get_before_generation_returns_404(
    integration_context: tuple[TestClient, Settings],
) -> None:
    """No understanding exists yet, so GET should 404 with a clear message."""

    client, settings = integration_context
    workspace_id = _create_workspace_with_profile(client, "Pre-Generation Co")

    try:
        response = client.get(f"/api/v1/workspaces/{workspace_id}/business-profile/understanding")
        assert response.status_code == 404
    finally:
        cleanup_workspace(settings, workspace_id)


def test_generate_creates_and_persists_an_understanding(
    integration_context: tuple[TestClient, Settings],
) -> None:
    """Generating should create a record that a subsequent GET can retrieve."""

    client, settings = integration_context
    workspace_id = _create_workspace_with_profile(client, "Generate Co")
    url = f"/api/v1/workspaces/{workspace_id}/business-profile/understanding"

    try:
        generate_response = client.post(url, json={})
        assert generate_response.status_code == 200

        generated = generate_response.json()
        assert generated["version"] == 1
        assert generated["summary"]
        assert generated["model_used"]

        get_response = client.get(url)
        assert get_response.status_code == 200
        assert get_response.json()["id"] == generated["id"]
    finally:
        cleanup_workspace(settings, workspace_id)


def test_generate_without_force_regenerate_returns_existing_record(
    integration_context: tuple[TestClient, Settings],
) -> None:
    """A second generate call should return the same record, not a new one."""

    client, settings = integration_context
    workspace_id = _create_workspace_with_profile(client, "Idempotent Co")
    url = f"/api/v1/workspaces/{workspace_id}/business-profile/understanding"

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
    workspace_id = _create_workspace_with_profile(client, "Regenerate Co")
    url = f"/api/v1/workspaces/{workspace_id}/business-profile/understanding"

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
        json={"name": "No Profile Co"},
    )
    assert workspace_response.status_code == 201
    workspace_id = UUID(workspace_response.json()["id"])

    try:
        response = client.post(
            f"/api/v1/workspaces/{workspace_id}/business-profile/understanding",
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
        f"/api/v1/workspaces/{UUID(int=0)}/business-profile/understanding",
        json={},
    )
    assert response.status_code == 404
    assert response.json() == {"detail": "Workspace not found."}
