"""Real PostgreSQL integration tests for the Customer Personas API.

Uses the deterministic "local" LLM provider (see Settings.llm_provider),
not real Ollama Cloud - this file is about proving persistence, the API
contract, the three hard-required prerequisites, and the "persona set
survives even if the reports it was built from are later deleted" design
decision are correct, none of which need a real model. Whether the real
LLM produces genuinely useful, grounded personas is covered separately in
test_customer_personas_llm_integration.py against real Ollama Cloud,
gated behind GROWTHCREW_RUN_LLM_INTEGRATION_TESTS.

The local stub fills every list-typed field with [], so every persona set
generated here has personas == []. That means these API-level tests cannot
prove the researched customer segments actually reach the model - the two
_build_prompt tests at the bottom exist for exactly that reason, and check
the prompt directly against known report text.

Like test_marketing_strategy_integration.py, the prerequisite reports are
produced through the real endpoints (profile -> understanding -> market
research) rather than seeded, so the tests exercise the same chain a real
client would call.
"""

import os
from collections.abc import Iterator
from typing import Any
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, select

from backend.app.config import Settings
from backend.app.db.database import Database
from backend.app.db.models.business_profile import BusinessProfile
from backend.app.db.models.business_understanding import BusinessUnderstanding
from backend.app.db.models.customer_persona_set import CustomerPersonaSet
from backend.app.db.models.market_research import MarketResearch
from backend.app.db.models.workspace import Workspace
from backend.app.llm.gateway import LLMGateway
from backend.app.main import create_application
from backend.app.schemas.customer_persona_set import CustomerPersonaSetResponse
from backend.app.services.customer_personas import CustomerPersonasService

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
    """Create an API client backed by real PostgreSQL and the local stubs.

    Customer Personas itself needs only the LLM gateway, but the tests
    build the Market Research prerequisite through its real endpoint, and
    that runs the CRAG graph, which also calls the embeddings gateway - so
    embeddings must be stubbed too or every test would try to reach Ollama.
    """

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


def _create_workspace(client: TestClient, name: str) -> UUID:
    """Create a bare workspace with no profile or reports."""

    response = client.post("/api/v1/workspaces", json={"name": name})
    assert response.status_code == 201
    return UUID(response.json()["id"])


def _create_workspace_with_profile(client: TestClient, name: str) -> UUID:
    """Create a workspace and a minimal business profile, return the workspace ID."""

    workspace_id = _create_workspace(client, name)

    profile_response = client.post(
        f"/api/v1/workspaces/{workspace_id}/business-profile",
        json={"business_name": name, "industry": "Healthy Snacks"},
    )
    assert profile_response.status_code == 201

    return workspace_id


def _generate_business_understanding(client: TestClient, workspace_id: UUID) -> None:
    response = client.post(
        f"/api/v1/workspaces/{workspace_id}/business-profile/understanding", json={}
    )
    assert response.status_code == 200


def _generate_market_research(client: TestClient, workspace_id: UUID) -> None:
    response = client.post(f"/api/v1/workspaces/{workspace_id}/market-research", json={})
    assert response.status_code == 200


def _create_workspace_with_full_prerequisites(client: TestClient, name: str) -> UUID:
    """Create a workspace with a profile, Business Understanding, and Market Research."""

    workspace_id = _create_workspace_with_profile(client, name)
    _generate_business_understanding(client, workspace_id)
    _generate_market_research(client, workspace_id)
    return workspace_id


def _delete_prerequisite_reports(settings: Settings, workspace_id: UUID) -> None:
    """Delete Market Research and Business Understanding for a workspace."""

    database = Database.from_settings(settings)
    try:
        with database.session() as session:
            profile = session.scalar(
                select(BusinessProfile).where(BusinessProfile.workspace_id == workspace_id)
            )
            assert profile is not None
            session.execute(
                delete(MarketResearch).where(MarketResearch.workspace_id == workspace_id)
            )
            session.execute(
                delete(BusinessUnderstanding).where(BusinessUnderstanding.profile_id == profile.id)
            )
            session.commit()
    finally:
        database.dispose()


def test_get_before_generation_returns_404(
    integration_context: tuple[TestClient, Settings],
) -> None:
    """No persona set exists yet, so GET should 404 with a clear message."""

    client, settings = integration_context
    workspace_id = _create_workspace_with_full_prerequisites(client, "Pre-Generation Persona Co")

    try:
        response = client.get(f"/api/v1/workspaces/{workspace_id}/personas")
        assert response.status_code == 404
        assert response.json() == {"detail": "Customer personas have not been generated yet."}
    finally:
        cleanup_workspace(settings, workspace_id)


def test_generate_creates_and_persists_a_persona_set(
    integration_context: tuple[TestClient, Settings],
) -> None:
    """Generating should create a record that a subsequent GET can retrieve."""

    client, settings = integration_context
    workspace_id = _create_workspace_with_full_prerequisites(client, "Generate Persona Co")
    url = f"/api/v1/workspaces/{workspace_id}/personas"

    try:
        generate_response = client.post(url, json={})
        assert generate_response.status_code == 200

        generated = generate_response.json()
        assert set(generated) == set(CustomerPersonaSetResponse.model_fields)
        assert generated["workspace_id"] == str(workspace_id)
        assert generated["version"] == 1
        assert generated["overview"]
        assert generated["personas"] == []  # local stub fills every list field with []
        assert generated["model_used"]

        get_response = client.get(url)
        assert get_response.status_code == 200
        assert get_response.json() == generated
    finally:
        cleanup_workspace(settings, workspace_id)


def test_generate_without_force_regenerate_returns_existing_record(
    integration_context: tuple[TestClient, Settings],
) -> None:
    """A second generate call should return the same record, not a new one."""

    client, settings = integration_context
    workspace_id = _create_workspace_with_full_prerequisites(client, "Idempotent Persona Co")
    url = f"/api/v1/workspaces/{workspace_id}/personas"

    try:
        first = client.post(url, json={}).json()
        second = client.post(url, json={}).json()

        assert second["id"] == first["id"]
        assert second["version"] == first["version"]
    finally:
        cleanup_workspace(settings, workspace_id)


def test_only_force_regenerate_calls_the_llm_again(
    integration_context: tuple[TestClient, Settings],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generation costs real tokens, so a repeat request must not spend any.

    Counts real calls to the application's shared LLM gateway rather than
    inferring from the response: an unchanged response would look
    identical whether or not the model had been called again.
    """

    client, settings = integration_context
    workspace_id = _create_workspace_with_full_prerequisites(client, "LLM Call Count Persona Co")
    url = f"/api/v1/workspaces/{workspace_id}/personas"

    gateway = client.app.state.llm_gateway  # type: ignore[attr-defined]
    original = gateway.structured_with_usage
    calls: list[str] = []

    def counting(prompt: str, schema: Any) -> Any:
        calls.append(prompt)
        return original(prompt, schema)

    monkeypatch.setattr(gateway, "structured_with_usage", counting)

    try:
        assert client.post(url, json={}).status_code == 200
        assert len(calls) == 1

        assert client.post(url, json={}).status_code == 200
        assert len(calls) == 1  # idempotent repeat: no new LLM call

        assert client.post(url, json={"force_regenerate": True}).status_code == 200
        assert len(calls) == 2  # only an explicit force_regenerate spends tokens again
    finally:
        cleanup_workspace(settings, workspace_id)


def test_force_regenerate_reuses_the_same_row(
    integration_context: tuple[TestClient, Settings],
) -> None:
    """force_regenerate should update the existing row, not create a duplicate.

    Does not assert a version bump: the local stub is deterministic, so an
    unchanged prompt produces byte-identical output, SQLAlchemy sees no
    net attribute change, and version_id_col only increments on an actual
    UPDATE. Same behaviour, and same reasoning, as the Content Planning
    test of the same name.
    """

    client, settings = integration_context
    workspace_id = _create_workspace_with_full_prerequisites(client, "Regenerate Persona Co")
    url = f"/api/v1/workspaces/{workspace_id}/personas"

    try:
        first = client.post(url, json={}).json()
        second = client.post(url, json={"force_regenerate": True}).json()

        assert second["id"] == first["id"]  # same row - the unique constraint holds
    finally:
        cleanup_workspace(settings, workspace_id)


def test_generate_without_a_business_profile_returns_404(
    integration_context: tuple[TestClient, Settings],
) -> None:
    """A workspace with no business profile should 404, not 500."""

    client, settings = integration_context
    workspace_id = _create_workspace(client, "No Profile Persona Co")

    try:
        response = client.post(f"/api/v1/workspaces/{workspace_id}/personas", json={})
        assert response.status_code == 404
        assert response.json() == {"detail": "Business profile not found."}
    finally:
        cleanup_workspace(settings, workspace_id)


def test_generate_without_business_understanding_returns_404(
    integration_context: tuple[TestClient, Settings],
) -> None:
    """A profile with no Business Understanding yet should 404, not 500."""

    client, settings = integration_context
    workspace_id = _create_workspace_with_profile(client, "No Understanding Persona Co")

    try:
        response = client.post(f"/api/v1/workspaces/{workspace_id}/personas", json={})
        assert response.status_code == 404
        assert response.json() == {"detail": "Business understanding has not been generated yet."}
    finally:
        cleanup_workspace(settings, workspace_id)


def test_generate_without_market_research_returns_404(
    integration_context: tuple[TestClient, Settings],
) -> None:
    """A workspace with no Market Research yet should 404, not 500."""

    client, settings = integration_context
    workspace_id = _create_workspace_with_profile(client, "No Research Persona Co")
    _generate_business_understanding(client, workspace_id)

    try:
        response = client.post(f"/api/v1/workspaces/{workspace_id}/personas", json={})
        assert response.status_code == 404
        assert response.json() == {"detail": "Market research has not been generated yet."}
    finally:
        cleanup_workspace(settings, workspace_id)


def test_generate_for_missing_workspace_returns_404(
    integration_context: tuple[TestClient, Settings],
) -> None:
    """A nonexistent workspace should 404, not 500."""

    client, _settings = integration_context

    response = client.post(
        f"/api/v1/workspaces/{UUID(int=0)}/personas",
        json={},
    )
    assert response.status_code == 404
    assert response.json() == {"detail": "Workspace not found."}


def test_persona_set_survives_prerequisite_reports_being_deleted(
    integration_context: tuple[TestClient, Settings],
) -> None:
    """A generated persona set stays readable after the reports it came from are gone.

    CustomerPersonaSet carries no foreign key to MarketResearch or
    BusinessUnderstanding at all - only to the workspace - so this confirms
    generate()/get() treat a finished persona set as a self-contained
    document. Both GET and a plain (non-forced) POST must keep working: the
    existing-record check runs before any prerequisite is required.
    """

    client, settings = integration_context
    workspace_id = _create_workspace_with_full_prerequisites(client, "Orphaned Persona Co")
    url = f"/api/v1/workspaces/{workspace_id}/personas"

    try:
        generated = client.post(url, json={}).json()

        _delete_prerequisite_reports(settings, workspace_id)

        get_response = client.get(url)
        assert get_response.status_code == 200
        assert get_response.json()["id"] == generated["id"]

        repeat_response = client.post(url, json={})
        assert repeat_response.status_code == 200
        assert repeat_response.json()["id"] == generated["id"]
    finally:
        cleanup_workspace(settings, workspace_id)


def test_force_regenerate_after_prerequisites_deleted_keeps_the_existing_set(
    integration_context: tuple[TestClient, Settings],
) -> None:
    """A forced regeneration needs the prerequisites again, and must not damage what exists.

    Asking for a fresh generation is the one path that genuinely needs the
    source reports, so it 404s once they are gone - but the failure must
    happen before anything is written, leaving the old persona set intact.
    Both reports are deleted here, so the first missing prerequisite the
    service reports is Business Understanding (checked before Market
    Research).
    """

    client, settings = integration_context
    workspace_id = _create_workspace_with_full_prerequisites(client, "Forced Orphan Persona Co")
    url = f"/api/v1/workspaces/{workspace_id}/personas"

    try:
        generated = client.post(url, json={}).json()

        _delete_prerequisite_reports(settings, workspace_id)

        response = client.post(url, json={"force_regenerate": True})
        assert response.status_code == 404
        assert response.json() == {"detail": "Business understanding has not been generated yet."}

        after = client.get(url)
        assert after.status_code == 200
        assert after.json() == generated
    finally:
        cleanup_workspace(settings, workspace_id)


def test_deleting_a_workspace_cascades_to_its_persona_set(
    integration_context: tuple[TestClient, Settings],
) -> None:
    """Removing a workspace must not leave an orphaned persona set behind."""

    client, settings = integration_context
    workspace_id = _create_workspace_with_full_prerequisites(client, "Cascade Persona Co")
    url = f"/api/v1/workspaces/{workspace_id}/personas"

    try:
        assert client.post(url, json={}).status_code == 200

        cleanup_workspace(settings, workspace_id)

        database = Database.from_settings(settings)
        try:
            with database.session() as session:
                remaining = session.scalar(
                    select(CustomerPersonaSet).where(
                        CustomerPersonaSet.workspace_id == workspace_id
                    )
                )
                assert remaining is None
        finally:
            database.dispose()
    finally:
        cleanup_workspace(settings, workspace_id)


def _build_prompt_for(
    settings: Settings,
    profile: BusinessProfile,
    understanding: BusinessUnderstanding,
    research: MarketResearch,
) -> str:
    """Call the service's prompt builder with in-memory report objects.

    Nothing is persisted and no query runs; a real session is passed only
    because the service constructs its repositories eagerly.
    """

    database = Database.from_settings(settings)
    try:
        with database.session() as session:
            service = CustomerPersonasService(session, LLMGateway(settings), settings)
            return service._build_prompt(profile, understanding, research)
    finally:
        database.dispose()


def test_build_prompt_includes_the_report_text_the_model_must_ground_on() -> None:
    """The researched segments and pain points must actually reach the model.

    The local stub returns personas == [] whatever it is given, so no
    API-level test can tell a prompt that carries the market research
    from one that silently drops it.
    """

    settings = Settings(environment="test", llm_provider="local")
    segments = (
        "Segment A: young urban professionals who value convenience. "
        "Segment B: parents buying for the household."
    )

    prompt = _build_prompt_for(
        settings,
        BusinessProfile(
            business_name="Acme Bakes",
            product_or_service="Sourdough subscriptions",
            industry="Food",
            country="Pakistan",
            target_market="Urban Lahore",
            target_customer="busy home cooks",
            price_range="mid",
        ),
        BusinessUnderstanding(
            summary="Artisan bakery going online.",
            key_differentiators=["slow-fermented"],
            likely_customer_pain_points=["stale supermarket bread"],
        ),
        MarketResearch(target_customer_segments=segments),
    )

    assert segments in prompt  # full text, not an excerpt
    assert "stale supermarket bread" in prompt
    assert "slow-fermented" in prompt
    assert "Artisan bakery going online." in prompt
    for line in (
        "- Business name: Acme Bakes",
        "- Product or service: Sourdough subscriptions",
        "- Industry: Food",
        "- Country: Pakistan",
        "- Target market: Urban Lahore",
        "- Target customer: busy home cooks",
        "- Price range: mid",
    ):
        assert line in prompt
    assert "exactly 3 personas" in prompt


def test_build_prompt_omits_unset_profile_fields_instead_of_printing_none() -> None:
    """An optional profile field left blank must not show up as a literal 'None'."""

    settings = Settings(environment="test", llm_provider="local")

    prompt = _build_prompt_for(
        settings,
        BusinessProfile(business_name="Minimal Co"),
        BusinessUnderstanding(
            summary="A small business.",
            key_differentiators=[],
            likely_customer_pain_points=[],
        ),
        MarketResearch(target_customer_segments="One segment."),
    )

    assert "- Business name: Minimal Co" in prompt
    assert "None" not in prompt
    for label in ("Country", "Industry", "Target customer", "Price range", "Target market"):
        assert f"- {label}:" not in prompt
