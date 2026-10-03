"""Real integration test for full strategy orchestration: Postgres AND Ollama Cloud.

Gated behind both integration flags at once, same as every other
*_llm_integration file here. This is the one place that proves the
whole seven-agent chain works end to end with real models: one
generate-full-strategy call, starting from nothing but a business
profile, drives real generation through Business Understanding, Market
Research, Competitor Analysis, Customer Personas, Brand Strategy,
Marketing Strategy, and Content Planning, with real embeddings and (if
TAVILY_API_KEY happens to be configured in your environment) real web
search, all the way through persistence.

A full run is expensive - seven real generations, three of them running
the CRAG graph per section - so this file is a single test that
asserts as much as one run can prove. It deliberately does NOT cover
force_regenerate: that would pay for a second full run, and the
mechanical guarantee (every stage reruns, each record keeps its
identity) is already proven exactly, with the deterministic local
provider, in test_strategy_orchestration_integration.py, while each
agent's real-model regeneration is covered by its own *_llm_integration
file. For the same reason as there, nothing here asserts that any
generated text contains specific phrases: real model output is not
deterministic enough for that to be anything but a flaky test.
"""

import os
from collections.abc import Iterator
from typing import Any
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

# Each stage's own read endpoint, in chain order, relative to the workspace.
STAGE_PATHS: list[tuple[str, str]] = [
    ("business_understanding", "business-profile/understanding"),
    ("market_research", "market-research"),
    ("competitor_analysis", "competitor-analysis"),
    ("customer_personas", "personas"),
    ("brand_strategy", "brand-strategy"),
    ("marketing_strategy", "marketing-strategy"),
    ("content_plan", "content-plan"),
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


def _create_workspace_with_profile(client: TestClient, name: str) -> UUID:
    """Create a workspace and a realistic business profile, return the workspace ID."""

    workspace_response = client.post("/api/v1/workspaces", json={"name": name})
    assert workspace_response.status_code == 201
    workspace_id = UUID(workspace_response.json()["id"])

    profile_response = client.post(
        f"/api/v1/workspaces/{workspace_id}/business-profile",
        json={
            "business_name": name,
            "product_or_service": "Weekly meal-prep subscription boxes",
            "industry": "Healthy Food",
            "target_customer": "Busy professionals who want to eat healthy",
            "price_range": "$60-$90 per week",
            "main_marketing_goal": "grow direct-to-consumer subscriptions",
            "existing_channels": ["instagram", "email"],
            "monthly_marketing_budget": 8000,
            "marketing_budget_currency": "USD",
            "current_challenges": "high customer acquisition cost",
            "known_competitors": ["HelloFresh", "Factor"],
        },
    )
    assert profile_response.status_code == 201

    return workspace_id


def test_full_strategy_runs_end_to_end_against_real_ollama_cloud(
    integration_context: tuple[TestClient, Settings],
) -> None:
    """One call takes a bare profile to a complete, persisted seven-stage strategy."""

    client, settings = integration_context
    workspace_id = _create_workspace_with_profile(client, "FitMeal Full Strategy LLM Integration")
    base = f"/api/v1/workspaces/{workspace_id}"

    try:
        before = client.get(f"{base}/strategy-status")
        assert before.status_code == 200
        assert before.json()["complete"] is False
        assert before.json()["next_stage"] == "business_understanding"

        response = client.post(f"{base}/generate-full-strategy", json={})
        assert response.status_code == 200

        run = response.json()
        # Show the whole per-stage report if a real provider hiccups mid-chain.
        assert run["complete"] is True, run
        assert run["failed_stage"] is None, run
        assert [item["stage"] for item in run["stages"]] == [name for name, _ in STAGE_PATHS]
        assert [item["outcome"] for item in run["stages"]] == ["generated"] * 7, run
        assert [item["version"] for item in run["stages"]] == [1] * 7

        # Every stage must really have been produced by the real model and
        # persisted: read each one back through its own endpoint.
        records: dict[str, dict[str, Any]] = {}
        for name, path in STAGE_PATHS:
            get_response = client.get(f"{base}/{path}")
            assert get_response.status_code == 200, name
            record = get_response.json()
            records[name] = record

            assert record["model_used"], name
            assert record["input_tokens"] > 0, name
            assert record["output_tokens"] > 0, name

        # The last two stages build on everything before them, so non-trivial
        # content there shows the whole chain fed through.
        assert len(records["marketing_strategy"]["ninety_day_roadmap"]) > 20
        assert len(records["content_plan"]["overview"]) > 20

        after = client.get(f"{base}/strategy-status")
        assert after.status_code == 200
        assert after.json()["complete"] is True
        assert after.json()["next_stage"] is None

        # A second run is free: nothing is skipped by accident and nothing
        # is regenerated, so every record keeps its identity and usage.
        second = client.post(f"{base}/generate-full-strategy", json={})
        assert second.status_code == 200
        assert [item["outcome"] for item in second.json()["stages"]] == ["skipped"] * 7

        for name, path in STAGE_PATHS:
            unchanged = client.get(f"{base}/{path}").json()
            assert unchanged["id"] == records[name]["id"], name
            assert unchanged["input_tokens"] == records[name]["input_tokens"], name
            assert unchanged["output_tokens"] == records[name]["output_tokens"], name
    finally:
        cleanup_workspace(settings, workspace_id)
