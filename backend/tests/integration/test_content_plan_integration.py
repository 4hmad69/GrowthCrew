"""Real PostgreSQL integration tests for the Content Planning API.

Uses the deterministic "local" LLM provider (see Settings.llm_provider),
not real Ollama Cloud - this file is about proving persistence, the API
contract, the single hard-required prerequisite, and the "plan survives
even if the strategy it was built from is later deleted" design decision
are correct, none of which need a real model. Whether the real LLM
produces a genuinely useful 30-entry calendar, and whether
force_regenerate produces genuinely different content, is covered
separately in test_content_plan_llm_integration.py against real Ollama
Cloud, gated behind GROWTHCREW_RUN_LLM_INTEGRATION_TESTS.

The marketing strategy each test needs is seeded directly via the
repository rather than generated through the real five-step chain
(profile -> understanding -> market research -> competitor analysis ->
marketing strategy) - this suite is about Content Planning's own logic,
which only cares that a MarketingStrategy row exists, not how it got
there. Marketing Strategy's own generation is already covered by
test_marketing_strategy_integration.py. This mirrors how that file
itself seeds knowledge chunks directly via RetrievalService rather than
through a public endpoint.
"""

import os
from collections.abc import Iterator
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete

from backend.app.config import Settings
from backend.app.db.database import Database
from backend.app.db.models.marketing_strategy import MarketingStrategy
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


def _create_workspace(client: TestClient, name: str) -> UUID:
    """Create a bare workspace - Content Planning has no business profile dependency."""

    response = client.post("/api/v1/workspaces", json={"name": name})
    assert response.status_code == 201
    return UUID(response.json()["id"])


def _seed_marketing_strategy(settings: Settings, workspace_id: UUID) -> None:
    """Seed a marketing strategy directly, bypassing the real five-step chain."""

    database = Database.from_settings(settings)
    try:
        with database.session() as session:
            session.add(
                MarketingStrategy(
                    workspace_id=workspace_id,
                    recommended_channels_and_tactics="Focus on Instagram and email.",
                    content_and_messaging_pillars="Education, trust, and social proof.",
                    ninety_day_roadmap=(
                        "Month 1: awareness. Month 2: conversion. Month 3: retention."
                    ),
                    budget_allocation_and_kpis=("60% paid social, 40% email; track CTR and CAC."),
                    model_used="local",
                )
            )
            session.commit()
    finally:
        database.dispose()


def _create_workspace_with_strategy(
    client: TestClient,
    settings: Settings,
    name: str,
) -> UUID:
    """Create a workspace with a marketing strategy already seeded."""

    workspace_id = _create_workspace(client, name)
    _seed_marketing_strategy(settings, workspace_id)
    return workspace_id


def test_get_before_generation_returns_404(
    integration_context: tuple[TestClient, Settings],
) -> None:
    """No plan exists yet, so GET should 404 with a clear message."""

    client, settings = integration_context
    workspace_id = _create_workspace_with_strategy(client, settings, "Pre-Generation Plan Co")

    try:
        response = client.get(f"/api/v1/workspaces/{workspace_id}/content-plan")
        assert response.status_code == 404
        assert response.json() == {"detail": "Content plan has not been generated yet."}
    finally:
        cleanup_workspace(settings, workspace_id)


def test_generate_creates_and_persists_a_plan(
    integration_context: tuple[TestClient, Settings],
) -> None:
    """Generating should create a record that a subsequent GET can retrieve."""

    client, settings = integration_context
    workspace_id = _create_workspace_with_strategy(client, settings, "Generate Plan Co")
    url = f"/api/v1/workspaces/{workspace_id}/content-plan"

    try:
        generate_response = client.post(url, json={})
        assert generate_response.status_code == 200

        generated = generate_response.json()
        assert generated["version"] == 1
        assert generated["overview"]
        assert generated["entries"] == []  # local stub fills every list field with []
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
    workspace_id = _create_workspace_with_strategy(client, settings, "Idempotent Plan Co")
    url = f"/api/v1/workspaces/{workspace_id}/content-plan"

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
    """force_regenerate should update the existing row, not create a duplicate.

    Does not assert a version bump: the local stub is deterministic, so an
    unchanged prompt produces byte-identical output, SQLAlchemy sees no
    net attribute change, and version_id_col only increments on an actual
    UPDATE. Confirmed in this project's own commit history - see
    ContentPlanService's docstring.
    """

    client, settings = integration_context
    workspace_id = _create_workspace_with_strategy(client, settings, "Regenerate Plan Co")
    url = f"/api/v1/workspaces/{workspace_id}/content-plan"

    try:
        first = client.post(url, json={}).json()
        second = client.post(url, json={"force_regenerate": True}).json()

        assert second["id"] == first["id"]  # same row - the unique constraint holds
    finally:
        cleanup_workspace(settings, workspace_id)


def test_generate_without_marketing_strategy_returns_404(
    integration_context: tuple[TestClient, Settings],
) -> None:
    """A workspace with no marketing strategy yet should 404, not 500."""

    client, settings = integration_context
    workspace_id = _create_workspace(client, "No Strategy Plan Co")

    try:
        response = client.post(
            f"/api/v1/workspaces/{workspace_id}/content-plan",
            json={},
        )
        assert response.status_code == 404
        assert response.json() == {"detail": "Marketing strategy has not been generated yet."}
    finally:
        cleanup_workspace(settings, workspace_id)


def test_generate_for_missing_workspace_returns_404(
    integration_context: tuple[TestClient, Settings],
) -> None:
    """A nonexistent workspace should 404, not 500."""

    client, _settings = integration_context

    response = client.post(
        f"/api/v1/workspaces/{UUID(int=0)}/content-plan",
        json={},
    )
    assert response.status_code == 404
    assert response.json() == {"detail": "Workspace not found."}


def test_get_survives_marketing_strategy_being_deleted(
    integration_context: tuple[TestClient, Settings],
) -> None:
    """A generated plan stays readable even after the strategy it was built from is gone.

    ContentPlan carries no foreign key to MarketingStrategy at all - only
    to the workspace - so this confirms generate()/get() were written to
    treat a finished plan as a self-contained document, the same design
    decision MarketingStrategyService already made for its own three
    prerequisite reports.
    """

    client, settings = integration_context
    workspace_id = _create_workspace_with_strategy(client, settings, "Orphaned Plan Co")
    url = f"/api/v1/workspaces/{workspace_id}/content-plan"

    try:
        generated = client.post(url, json={}).json()

        database = Database.from_settings(settings)
        try:
            with database.session() as session:
                session.execute(
                    delete(MarketingStrategy).where(MarketingStrategy.workspace_id == workspace_id)
                )
                session.commit()
        finally:
            database.dispose()

        response = client.get(url)
        assert response.status_code == 200
        assert response.json()["id"] == generated["id"]
    finally:
        cleanup_workspace(settings, workspace_id)
