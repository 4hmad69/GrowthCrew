"""Real PostgreSQL integration tests for onboarding APIs."""

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
    """Create an API client backed by the real local PostgreSQL service."""

    settings = Settings(environment="test")

    if settings.database_url is None:
        pytest.fail("GROWTHCREW_DATABASE_URL is required for integration tests.")

    application = create_application(settings)

    with TestClient(application) as client:
        yield client, settings


def cleanup_workspace(
    settings: Settings,
    workspace_id: UUID,
) -> None:
    """Remove only test-created data after an integration test."""

    database = Database.from_settings(settings)

    try:
        with database.session() as session:
            workspace = session.get(
                Workspace,
                workspace_id,
            )

            if workspace is not None:
                session.delete(workspace)
                session.commit()
    finally:
        database.dispose()


def test_workspace_lifecycle(
    integration_context: tuple[TestClient, Settings],
) -> None:
    """Create, update, detect stale state, and archive a workspace."""

    client, settings = integration_context

    create_response = client.post(
        "/api/v1/workspaces",
        json={"name": "Integration Workspace"},
    )
    assert create_response.status_code == 201

    created = create_response.json()
    workspace_id = UUID(created["id"])

    try:
        assert created["status"] == "active"
        assert created["version"] == 1

        read_response = client.get(f"/api/v1/workspaces/{workspace_id}")
        assert read_response.status_code == 200

        update_response = client.patch(
            f"/api/v1/workspaces/{workspace_id}",
            json={
                "name": "Updated Integration Workspace",
                "version": 1,
            },
        )
        assert update_response.status_code == 200

        updated = update_response.json()
        assert updated["name"] == "Updated Integration Workspace"
        assert updated["version"] == 2

        stale_response = client.patch(
            f"/api/v1/workspaces/{workspace_id}",
            json={
                "name": "Stale Update",
                "version": 1,
            },
        )
        assert stale_response.status_code == 409

        archive_response = client.delete(
            f"/api/v1/workspaces/{workspace_id}",
            params={"version": 2},
        )
        assert archive_response.status_code == 204

        archived_response = client.get(f"/api/v1/workspaces/{workspace_id}")
        assert archived_response.status_code == 200
        assert archived_response.json()["status"] == "archived"

        list_response = client.get("/api/v1/workspaces")
        assert list_response.status_code == 200

        listed_ids = {item["id"] for item in list_response.json()["items"]}
        assert str(workspace_id) not in listed_ids
    finally:
        cleanup_workspace(
            settings,
            workspace_id,
        )


def test_business_profile_lifecycle(
    integration_context: tuple[TestClient, Settings],
) -> None:
    """Create, update, reject stale state, and delete onboarding data."""

    client, settings = integration_context

    workspace_response = client.post(
        "/api/v1/workspaces",
        json={"name": "Profile Integration Workspace"},
    )
    assert workspace_response.status_code == 201

    workspace = workspace_response.json()
    workspace_id = UUID(workspace["id"])

    try:
        profile_response = client.post(
            (f"/api/v1/workspaces/{workspace_id}/business-profile"),
            json={
                "business_name": "Acme Growth",
                "website": "https://example.com",
                "product_or_service": "Marketing software",
                "industry": "Software",
                "country": "United States",
                "monthly_marketing_budget": "1200.00",
                "marketing_budget_currency": "usd",
                "existing_channels": [
                    "LinkedIn",
                    "Email",
                ],
                "known_competitors": [
                    "Competitor A",
                ],
            },
        )

        assert profile_response.status_code == 201

        profile = profile_response.json()
        assert profile["version"] == 1
        assert profile["marketing_budget_currency"] == "USD"

        update_response = client.patch(
            (f"/api/v1/workspaces/{workspace_id}/business-profile"),
            json={
                "version": 1,
                "brand_tone": "Confident and helpful",
                "existing_channels": [
                    "LinkedIn",
                    "Email",
                    "YouTube",
                ],
            },
        )

        assert update_response.status_code == 200
        updated = update_response.json()

        assert updated["version"] == 2
        assert updated["brand_tone"] == "Confident and helpful"

        stale_response = client.patch(
            (f"/api/v1/workspaces/{workspace_id}/business-profile"),
            json={
                "version": 1,
                "brand_tone": "Stale value",
            },
        )

        assert stale_response.status_code == 409

        delete_response = client.delete(
            (f"/api/v1/workspaces/{workspace_id}/business-profile"),
            params={"version": 2},
        )

        assert delete_response.status_code == 204

        missing_response = client.get(f"/api/v1/workspaces/{workspace_id}/business-profile")
        assert missing_response.status_code == 404
    finally:
        cleanup_workspace(
            settings,
            workspace_id,
        )


def test_archived_workspace_rejects_profile_creation(
    integration_context: tuple[TestClient, Settings],
) -> None:
    """Archived workspaces must be immutable."""

    client, settings = integration_context

    workspace_response = client.post(
        "/api/v1/workspaces",
        json={"name": "Archived Integration Workspace"},
    )
    assert workspace_response.status_code == 201

    workspace = workspace_response.json()
    workspace_id = UUID(workspace["id"])

    try:
        archive_response = client.delete(
            f"/api/v1/workspaces/{workspace_id}",
            params={"version": 1},
        )
        assert archive_response.status_code == 204

        profile_response = client.post(
            (f"/api/v1/workspaces/{workspace_id}/business-profile"),
            json={"business_name": "Should Not Persist"},
        )

        assert profile_response.status_code == 409
    finally:
        cleanup_workspace(
            settings,
            workspace_id,
        )
