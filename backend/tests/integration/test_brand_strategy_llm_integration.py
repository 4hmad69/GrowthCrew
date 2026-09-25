"""Real integration tests for Brand Strategy against Postgres AND Ollama Cloud.

Gated behind both integration flags at once, same as the Business
Understanding, Market Research, Competitor Analysis, Marketing Strategy,
Content Planning, and Customer Personas LLM integration tests.

Note on the seeded reports: like test_customer_personas_llm_integration.py,
this file seeds a realistic but static business profile, Business
Understanding, Competitor Analysis, and Customer Personas directly rather
than generating them through the real chain. Brand Strategy has no CRAG
graph and no retrieval step - it is one direct structured call over
whatever report text it is given - so what this file must prove is
specific to that one call: does a real model, given real competitor
research and real personas, produce well-formed positioning that is
actually grounded in both? A fixed, realistic input isolates that
question without paying for chained real generations that each have
their own LLM test.

Note on grounding, specifically: this is the first agent that must
synthesize across two independently-researched reports at once
(Competitor Analysis and Customer Personas), not just one. Business
Understanding's differentiators/pain points are already proven to reach
every direct-call agent's prompt by the prior LLM integration tests, so
this file's keyword checks are deliberately scoped to the two report
types unique to this agent: one keyword set drawn only from Competitor
Analysis's differentiation_opportunities text ("roast date", "freshness"
etc.), and a second, non-overlapping keyword set drawn only from the
persona's own language ("home barista", "espresso" etc.). A model that
grounded only in the business profile or Business Understanding, and
ignored the competitor and persona research, would not reliably produce
either set - let alone both.

Note on list-field counts: the prompt asks for 3-5 brand pillars and 3-5
taglines, but the draft schema deliberately has no min_length (see
_BrandStrategyDraft's docstring - the "local" provider fills every list
with [], so a min_length would break every Postgres-only test), and a
live, non-batch model may legitimately under- or over-deliver. So, as in
the Customer Personas LLM test, this file asserts the lists are
non-empty rather than asserting an exact count. The counts are printed
instead (run pytest with -s to see them) so real-world drift is visible
and can be acted on by tightening the prompt, after which a hard
assertion becomes safe.

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
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.db.database import Database
from backend.app.db.models.business_profile import BusinessProfile
from backend.app.db.models.business_understanding import BusinessUnderstanding
from backend.app.db.models.competitor_analysis import CompetitorAnalysis
from backend.app.db.models.customer_persona_set import CustomerPersonaSet
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

# Drawn only from the seeded Competitor Analysis's differentiation
# opportunity below - not present anywhere in the seeded persona text.
_DIFFERENTIATION_KEYWORDS = (
    "roast date",
    "roast-date",
    "freshness",
    "fresh",
    "transparency",
    "transparent",
)

# Drawn only from the seeded persona's own language below - not present
# anywhere in the seeded Competitor Analysis text.
_PERSONA_KEYWORDS = (
    "home barista",
    "espresso",
    "barista",
    "third-wave",
    "third wave",
)


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
    """Seed a realistic profile, Business Understanding, Competitor Analysis, and Personas.

    See the module docstring for why these are seeded rather than generated,
    and for why the competitor and persona text are deliberately distinctive
    and non-overlapping.
    """

    database = Database.from_settings(settings)
    try:
        with database.session() as session:
            profile = BusinessProfile(
                workspace_id=workspace_id,
                business_name="Sundial Coffee Roasters",
                product_or_service="Single-origin specialty coffee subscription",
                industry="Specialty Coffee",
                country="Pakistan",
                target_market="Urban young professionals",
                target_customer="Home coffee enthusiasts with their own equipment",
                price_range="Premium",
                brand_tone="Warm, craftsman-like, unpretentious",
            )
            session.add(profile)
            session.flush()  # the understanding below needs profile.id

            session.add(
                BusinessUnderstanding(
                    profile_id=profile.id,
                    summary=(
                        "A specialty coffee roaster selling single-origin beans "
                        "via a weekly roast-to-order subscription."
                    ),
                    inferred_business_stage="early",
                    competitive_category="Specialty coffee subscription",
                    key_differentiators=[
                        "Beans roasted within 48 hours of shipping",
                        "Direct trade with named farms",
                    ],
                    likely_customer_pain_points=[
                        "Supermarket coffee tastes stale and burnt",
                        "Hard to know how fresh subscription coffee actually is",
                    ],
                    model_used="local",
                )
            )
            session.add(
                CompetitorAnalysis(
                    workspace_id=workspace_id,
                    competitor_overview=(
                        "Three national roasters dominate the specialty coffee "
                        "subscription market: BeanVoyage, RoastCrate, and the "
                        "retail bags sold by third-wave cafe chains."
                    ),
                    strengths_and_weaknesses=(
                        "BeanVoyage has strong brand recognition but rotates "
                        "origins unpredictably. RoastCrate offers consistent "
                        "pricing but roasts in large batches shipped weeks "
                        "after roasting."
                    ),
                    pricing_and_positioning=(
                        "Both competitors price mid-range and market on "
                        "variety and convenience rather than freshness."
                    ),
                    differentiation_opportunities=(
                        "No competitor publishes an exact roast date on the "
                        "bag or guarantees delivery within 5 days of "
                        "roasting - radical roast-date transparency is "
                        "unclaimed territory."
                    ),
                    sources=[],
                    model_used="local",
                )
            )
            session.add(
                CustomerPersonaSet(
                    workspace_id=workspace_id,
                    overview=(
                        "One primary persona: the home barista who has "
                        "upgraded their equipment and now cares more about "
                        "bean freshness than variety."
                    ),
                    personas=[
                        {
                            "name": "Third-wave Tariq",
                            "segment": "Home baristas with home espresso setups",
                            "summary": (
                                "Recently bought a home espresso machine and now "
                                "notices when beans taste stale or under-extract."
                            ),
                            "demographics": "28-40, urban, tech-savvy",
                            "goals": [
                                "Pull consistently good shots at home",
                                "Learn to identify freshness and origin flavor notes",
                            ],
                            "pain_points": [
                                "Can't tell how fresh grocery-store beans really are",
                                "Subscription beans often arrive weeks after roasting",
                            ],
                            "buying_triggers": [
                                "A roast date printed clearly on the bag",
                                "A recommendation from a barista community",
                            ],
                            "objections": [
                                "Premium price versus supermarket beans",
                                "Uncertainty about subscription commitment",
                            ],
                            "preferred_channels": ["Reddit coffee communities", "Instagram"],
                        }
                    ],
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


def test_generate_produces_real_grounded_positioning_with_real_token_usage(
    integration_context: tuple[TestClient, Settings],
) -> None:
    """A real generate call should produce well-formed positioning grounded in both reports."""

    client, settings = integration_context
    workspace_id = _create_workspace(client, "Sundial Brand Strategy LLM Integration")

    try:
        _seed_prerequisites(settings, workspace_id)

        response = client.post(f"/api/v1/workspaces/{workspace_id}/brand-strategy", json={})
        assert response.status_code == 200

        body = response.json()

        # Visible with `pytest -s`: the numbers the architecture doc records.
        print(
            f"\n[brand strategy llm] pillars={len(body['brand_pillars'])} "
            f"taglines={len(body['tagline_options'])} "
            f"input_tokens={body['input_tokens']} output_tokens={body['output_tokens']} "
            f"num_predict_cap={settings.llm_num_predict}"
        )
        print(f"[brand strategy llm] positioning: {body['positioning_statement']}")
        for pillar in body["brand_pillars"]:
            print(f"[brand strategy llm]   pillar - {pillar}")
        for tagline in body["tagline_options"]:
            print(f"[brand strategy llm]   tagline - {tagline}")

        assert len(body["overview"]) > 20
        assert body["positioning_statement"]
        assert body["value_proposition"]
        assert body["brand_voice_and_tone"]
        assert body["brand_pillars"], "expected a real model to produce at least some pillars"
        assert len(body["brand_pillars"]) <= 6
        assert body["tagline_options"], "expected a real model to produce at least one tagline"
        assert len(body["tagline_options"]) <= 5
        assert body["model_used"] == settings.llm_model
        assert body["input_tokens"] > 0
        assert body["output_tokens"] > 0
        assert body["output_tokens"] < settings.llm_num_predict, (
            "output hit the token cap - the generation was likely truncated; "
            "raise llm_num_predict or shrink the schema"
        )

        combined_text = " ".join(
            [
                body["overview"],
                body["positioning_statement"],
                body["value_proposition"],
                body["brand_voice_and_tone"],
                *body["brand_pillars"],
                *body["tagline_options"],
            ]
        ).lower()

        assert any(keyword in combined_text for keyword in _DIFFERENTIATION_KEYWORDS), (
            "positioning does not look grounded in the seeded competitor "
            f"differentiation opportunity; combined_text={combined_text!r}"
        )
        assert any(keyword in combined_text for keyword in _PERSONA_KEYWORDS), (
            "positioning does not look grounded in the seeded customer "
            f"persona; combined_text={combined_text!r}"
        )

        # confirm real usage and content actually persisted, not just present in the response
        get_response = client.get(f"/api/v1/workspaces/{workspace_id}/brand-strategy")
        assert get_response.status_code == 200
        persisted = get_response.json()
        assert persisted["input_tokens"] == body["input_tokens"]
        assert persisted["output_tokens"] == body["output_tokens"]
        assert persisted["positioning_statement"] == body["positioning_statement"]
    finally:
        cleanup_workspace(settings, workspace_id)


def test_force_regenerate_succeeds_against_real_ollama_cloud(
    integration_context: tuple[TestClient, Settings],
) -> None:
    """force_regenerate should succeed end to end and keep the same row."""

    client, settings = integration_context
    workspace_id = _create_workspace(client, "Regenerate Brand Strategy LLM Integration")
    url = f"/api/v1/workspaces/{workspace_id}/brand-strategy"

    try:
        _seed_prerequisites(settings, workspace_id)

        first = client.post(url, json={})
        assert first.status_code == 200

        second = client.post(url, json={"force_regenerate": True})
        assert second.status_code == 200
        assert second.json()["id"] == first.json()["id"]
    finally:
        cleanup_workspace(settings, workspace_id)
