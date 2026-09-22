"""Real integration tests for Customer Personas against Postgres AND Ollama Cloud.

Gated behind both integration flags at once, same as the Business
Understanding, Market Research, Competitor Analysis, Marketing Strategy,
and Content Planning LLM integration tests.

Note on the seeded reports: like test_content_plan_llm_integration.py,
this file seeds a realistic but static business profile, Business
Understanding, and Market Research directly rather than generating them
through the real chain. Customer Personas has no CRAG graph and no
retrieval step - it is one direct structured call over whatever report
text it is given - so what this file must prove is specific to that one
call: does a real model, given real researched segments, produce
well-formed personas that are actually grounded in those segments? A
fixed, realistic input isolates that question without paying for chained
real generations that each have their own LLM test.

Note on grounding: the seeded segments are deliberately distinctive
(night-shift nurses, amateur marathon runners, new parents) rather than
generic ("busy professionals"). A model that ignored the research and
wrote stock personas would not mention any of them, so their presence is
real evidence the segments reached the model and shaped the output. This
is the one thing the deterministic "local" provider can never show,
since it returns personas == [] whatever it is given.

Note on persona count: the prompt asks for exactly 3 personas, but the
draft schema deliberately has no min_length (see
CustomerPersonasService's docstring - the "local" provider fills every
list with [], so a min_length would break every Postgres-only test), and
a live, non-batch model may legitimately under- or over-deliver. So, as
in the Content Planning LLM test, this file asserts personas are
non-empty and every persona present is well-formed rather than asserting
an exact count. The count is printed instead (run pytest with -s to see
it) so a real-world drift away from 3 is visible and can be acted on by
tightening the prompt, after which a hard assertion becomes safe.

Note on token budget: the test asserts output_tokens stays below
Settings.llm_num_predict. A generation that hit that cap would be
truncated mid-JSON; Content Planning shipped exactly that bug once (fixed
by raising the cap), so this makes the budget an explicit, checked
property rather than something inferred from a downstream parse failure.

Note on force_regenerate: this file does NOT assert that regenerated
content differs from the original, or that the version number bumps.
LLMGateway fixes temperature at 0 for determinism, so a real model given
an unchanged prompt may legitimately reproduce identical output - a
content-difference assertion would be a flaky test failing for correct
code. The Postgres-only tests already cover the mechanical guarantee
(force_regenerate reuses the same row) with the deterministic provider.
"""

import os
from collections.abc import Iterator
from typing import Any
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.db.database import Database
from backend.app.db.models.business_profile import BusinessProfile
from backend.app.db.models.business_understanding import BusinessUnderstanding
from backend.app.db.models.market_research import MarketResearch
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

_TEXT_FIELDS = ("name", "segment", "summary", "demographics")
_LIST_FIELDS = (
    "goals",
    "pain_points",
    "buying_triggers",
    "objections",
    "preferred_channels",
)

# Distinctive words per seeded segment. A persona counts as representing a
# segment if its name/segment/summary mentions any of them.
_SEGMENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "night-shift nurses": ("nurse", "night shift", "night-shift", "shift work", "hospital"),
    "amateur marathon runners": (
        "marathon",
        "runner",
        "running",
        "endurance",
        "athlete",
        "training",
    ),
    "new parents": ("parent", "newborn", "baby", "mother", "father", "mom", "dad"),
}


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


def _seed_prerequisites(settings: Settings, workspace_id: UUID) -> None:
    """Seed a realistic profile, Business Understanding, and Market Research.

    See the module docstring for why these are seeded rather than generated.
    """

    database = Database.from_settings(settings)
    try:
        with database.session() as session:
            profile = BusinessProfile(
                workspace_id=workspace_id,
                business_name="FitMeal Prep",
                product_or_service="Weekly healthy meal-prep subscription with home delivery",
                industry="Food and Nutrition",
                country="Pakistan",
                target_market="Urban Lahore",
                target_customer="Busy people who want healthy meals without cooking",
                price_range="Mid-range",
            )
            session.add(profile)
            session.flush()  # the understanding below needs profile.id

            session.add(
                BusinessUnderstanding(
                    profile_id=profile.id,
                    summary=(
                        "A meal-prep delivery startup offering portioned, "
                        "macro-balanced meals on a weekly subscription."
                    ),
                    inferred_business_stage="early",
                    competitive_category="Meal-prep delivery",
                    key_differentiators=[
                        "Macro-balanced portions",
                        "Flexible weekly subscription",
                        "Locally sourced ingredients",
                    ],
                    likely_customer_pain_points=[
                        "No time to cook after long or irregular hours",
                        "Difficulty eating enough protein while training",
                        "Healthy takeaway is expensive and inconsistent",
                    ],
                    model_used="local",
                )
            )
            session.add(
                MarketResearch(
                    workspace_id=workspace_id,
                    market_overview=(
                        "Demand for healthy, convenient prepared meals is growing "
                        "among urban professionals and fitness-minded consumers."
                    ),
                    target_customer_segments=(
                        "Three segments matter most. (1) Night-shift nurses and "
                        "hospital staff who finish work at odd hours, have no time "
                        "to cook, and want filling, healthy meals ready when they "
                        "get home. (2) Amateur marathon runners in structured "
                        "training who need high-protein, carb-timed meals and track "
                        "their nutrition closely. (3) New parents with a newborn "
                        "who have almost no time or energy to cook and worry about "
                        "eating well while sleep-deprived."
                    ),
                    competitive_landscape=(
                        "A few gym-affiliated meal plans and generic delivery apps."
                    ),
                    opportunities_and_risks="Opportunity: underserved niches. Risk: thin margins.",
                    sources=[],
                    model_used="local",
                )
            )
            session.commit()
    finally:
        database.dispose()


def _create_workspace(client: TestClient, name: str) -> UUID:
    response = client.post("/api/v1/workspaces", json={"name": name})
    assert response.status_code == 201
    return UUID(response.json()["id"])


def _persona_text(persona: dict[str, Any]) -> str:
    """Return the lowercase text used to decide which segment a persona represents."""

    return " ".join(str(persona[field]) for field in ("name", "segment", "summary")).lower()


def test_generate_produces_real_grounded_personas_with_real_token_usage(
    integration_context: tuple[TestClient, Settings],
) -> None:
    """A real generate call should produce well-formed, segment-grounded personas."""

    client, settings = integration_context
    workspace_id = _create_workspace(client, "FitMeal Personas LLM Integration")

    try:
        _seed_prerequisites(settings, workspace_id)

        response = client.post(f"/api/v1/workspaces/{workspace_id}/personas", json={})
        assert response.status_code == 200

        body = response.json()
        personas = body["personas"]

        # Visible with `pytest -s`: the numbers the architecture doc records.
        print(
            f"\n[personas llm] count={len(personas)} "
            f"input_tokens={body['input_tokens']} output_tokens={body['output_tokens']} "
            f"num_predict_cap={settings.llm_num_predict}"
        )
        for persona in personas:
            print(f"[personas llm]   - {persona['name']} | {persona['segment']}")

        assert len(body["overview"]) > 20
        assert body["model_used"] == settings.llm_model
        assert body["input_tokens"] > 0
        assert body["output_tokens"] > 0
        assert body["output_tokens"] < settings.llm_num_predict, (
            "output hit the token cap - the generation was likely truncated; "
            "raise llm_num_predict or shrink the schema"
        )

        assert personas, "expected a real model to produce at least some personas"
        assert len(personas) <= 5
        for persona in personas:
            for field in _TEXT_FIELDS:
                assert persona[field], f"persona {persona.get('name')!r} has an empty {field}"
            for field in _LIST_FIELDS:
                assert persona[field], f"persona {persona.get('name')!r} has an empty {field}"

        represented = [
            segment
            for segment, keywords in _SEGMENT_KEYWORDS.items()
            if any(keyword in _persona_text(p) for p in personas for keyword in keywords)
        ]
        assert len(represented) >= 2, (
            "personas do not look grounded in the seeded market research segments; "
            f"represented={represented}, "
            f"segments={[p['segment'] for p in personas]}"
        )

        # confirm real usage and personas actually persisted, not just present in the response
        get_response = client.get(f"/api/v1/workspaces/{workspace_id}/personas")
        assert get_response.status_code == 200
        persisted = get_response.json()
        assert persisted["input_tokens"] == body["input_tokens"]
        assert persisted["output_tokens"] == body["output_tokens"]
        assert persisted["personas"] == personas
    finally:
        cleanup_workspace(settings, workspace_id)


def test_force_regenerate_succeeds_against_real_ollama_cloud(
    integration_context: tuple[TestClient, Settings],
) -> None:
    """force_regenerate should succeed end to end and keep the same row."""

    client, settings = integration_context
    workspace_id = _create_workspace(client, "Regenerate Personas LLM Integration")
    url = f"/api/v1/workspaces/{workspace_id}/personas"

    try:
        _seed_prerequisites(settings, workspace_id)

        first = client.post(url, json={})
        assert first.status_code == 200

        second = client.post(url, json={"force_regenerate": True})
        assert second.status_code == 200
        assert second.json()["id"] == first.json()["id"]
    finally:
        cleanup_workspace(settings, workspace_id)
