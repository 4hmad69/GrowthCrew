"""Real integration tests for Content Planning against Postgres AND Ollama Cloud.

Gated behind both integration flags at once, same as the Business
Understanding, Market Research, Competitor Analysis, and Marketing
Strategy LLM integration tests.

Note on the seeded marketing strategy: unlike
test_marketing_strategy_llm_integration.py, which regenerates all four
of its own prerequisite reports for real to prove the whole CRAG
retrieval chain works end to end, this file seeds a realistic but
static marketing strategy directly (same helper as commit 5's
Postgres-only test) rather than generating one through the real
five-step chain. That distinction matters there because Marketing
Strategy's own correctness depends on real retrieval and grading
behavior against a real vectorstore - there's nothing analogous to
re-prove here. Content Planning has no CRAG graph and no retrieval step
at all; it is one direct structured call over whatever strategy text it
is given. What this file needs to prove is specific to that one call:
does a real model, given a real strategy, produce genuine, well-formed
calendar entries. A fixed, realistic input isolates that question
without the cost of five chained real generations (each of which is
already proven correct by its own LLM integration test) just to
produce the input text.

Note on entry count: the prompt asks for exactly 30 entries, but
_ContentPlanDraft.entries deliberately has no min_length (see
ContentPlanService's docstring - the deterministic "local" provider
fills every list field with [], so a min_length would break every
Postgres-only test). That means a real model that under-delivers is a
product-quality question, not something this test can safely treat as
a hard failure without risking flakiness against a live, non-batch
model. This file asserts entries are non-empty and every entry present
is well-formed (all fields populated, day/week already schema-bounded)
rather than asserting an exact or minimum count.

Note on force_regenerate: this file does NOT assert that regenerated
content differs from the original, or that the version number bumps.
LLMGateway fixes temperature at 0 for determinism, so a real model
given an unchanged prompt may legitimately reproduce very similar or
identical output - asserting a content difference here would be a
flaky test failing for correct code. Commit 5's Postgres-only test
already covers the mechanical guarantee (force_regenerate reuses the
same row, never creates a duplicate) with the fully deterministic local
provider, where that assertion is reliable.
"""

import os
from collections.abc import Iterator
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.db.database import Database
from backend.app.db.models.marketing_strategy import MarketingStrategy
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


def _seed_marketing_strategy(settings: Settings, workspace_id: UUID) -> None:
    """Seed a realistic marketing strategy directly - see module docstring."""

    database = Database.from_settings(settings)
    try:
        with database.session() as session:
            session.add(
                MarketingStrategy(
                    workspace_id=workspace_id,
                    recommended_channels_and_tactics=(
                        "Prioritize Instagram Reels and a weekly email newsletter. "
                        "Use short-form video to show meal prep in under 60 seconds, "
                        "and email to nurture subscribers with recipes and tips."
                    ),
                    content_and_messaging_pillars=(
                        "Education (how our meals are made), Trust (real customer "
                        "results and reviews), and Convenience (fitting healthy "
                        "eating into a busy week)."
                    ),
                    ninety_day_roadmap=(
                        "Month 1: build awareness through Reels and influencer "
                        "partnerships. Month 2: convert with limited-time offers "
                        "and email nurture sequences. Month 3: retain subscribers "
                        "with loyalty perks and user-generated content."
                    ),
                    budget_allocation_and_kpis=(
                        "60% of budget to paid social, 40% to email and content "
                        "production. Track click-through rate, cost per "
                        "acquisition, and subscriber retention rate."
                    ),
                    model_used="local",
                )
            )
            session.commit()
    finally:
        database.dispose()


def test_generate_produces_real_calendar_with_real_token_usage(
    integration_context: tuple[TestClient, Settings],
) -> None:
    """A real generate call should produce well-formed entries and real usage."""

    client, settings = integration_context

    workspace_response = client.post(
        "/api/v1/workspaces",
        json={"name": "FitMeal Content Plan LLM Integration"},
    )
    assert workspace_response.status_code == 201
    workspace_id = UUID(workspace_response.json()["id"])
    _seed_marketing_strategy(settings, workspace_id)

    try:
        response = client.post(f"/api/v1/workspaces/{workspace_id}/content-plan", json={})
        assert response.status_code == 200

        body = response.json()
        assert len(body["overview"]) > 20
        assert body["input_tokens"] > 0
        assert body["output_tokens"] > 0

        assert body["entries"], "expected a real model to produce at least some calendar entries"
        assert len(body["entries"]) <= 30
        for entry in body["entries"]:
            assert entry["channel"]
            assert entry["content_type"]
            assert entry["pillar"]
            assert entry["topic"]
            assert entry["cta"]

        # confirm real usage and entries actually persisted, not just present in the response
        get_response = client.get(f"/api/v1/workspaces/{workspace_id}/content-plan")
        assert get_response.status_code == 200
        persisted = get_response.json()
        assert persisted["input_tokens"] == body["input_tokens"]
        assert persisted["output_tokens"] == body["output_tokens"]
        assert persisted["entries"] == body["entries"]
    finally:
        cleanup_workspace(settings, workspace_id)


def test_force_regenerate_succeeds_against_real_ollama_cloud(
    integration_context: tuple[TestClient, Settings],
) -> None:
    """force_regenerate should succeed end to end and keep the same row."""

    client, settings = integration_context

    workspace_response = client.post(
        "/api/v1/workspaces",
        json={"name": "Regenerate Content Plan LLM Integration"},
    )
    assert workspace_response.status_code == 201
    workspace_id = UUID(workspace_response.json()["id"])
    _seed_marketing_strategy(settings, workspace_id)

    url = f"/api/v1/workspaces/{workspace_id}/content-plan"

    try:
        first = client.post(url, json={})
        assert first.status_code == 200

        second = client.post(url, json={"force_regenerate": True})
        assert second.status_code == 200
        assert second.json()["id"] == first.json()["id"]
    finally:
        cleanup_workspace(settings, workspace_id)
