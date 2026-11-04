"""Real PostgreSQL integration tests: staleness and generation in strategy-status.

Drives the real agents through the API with the deterministic "local"
providers and no Tavily key. The headline requirement is that staleness
is information only: forcing an upstream stage must change what the
status says about history and nothing it says about gating or approval.
The remaining tests cover the cases where history cannot honestly be
matched to a record - it is reported as unknown, never guessed.
"""

import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from backend.app.config import Settings
from backend.app.db.database import Database
from backend.app.db.models.business_profile import BusinessProfile
from backend.app.db.models.market_research import MarketResearch
from backend.app.db.models.stage_snapshot import StageSnapshot
from backend.app.db.models.workspace import Workspace
from backend.app.main import create_application
from backend.app.schemas.strategy_orchestration import StrategyStage

RUN_INTEGRATION_TESTS = os.getenv("GROWTHCREW_RUN_INTEGRATION_TESTS") == "1"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not RUN_INTEGRATION_TESTS,
        reason=("Set GROWTHCREW_RUN_INTEGRATION_TESTS=1 to run PostgreSQL integration tests."),
    ),
]

S = StrategyStage
STAGES = list(StrategyStage)

GATING_FIELDS = (
    "generated",
    "version",
    "approval",
    "can_generate",
    "missing_prerequisites",
    "unapproved_prerequisites",
)


@dataclass
class Context:
    """An API client and its settings."""

    client: TestClient
    settings: Settings

    def url(self, workspace_id: UUID, path: str) -> str:
        return f"/api/v1/workspaces/{workspace_id}/{path}"


@pytest.fixture
def context() -> Iterator[Context]:
    """Create an API client backed by real PostgreSQL and the local stubs."""

    settings = Settings(
        environment="test",
        llm_provider="local",
        embeddings_provider="local",
        tavily_api_key=None,
    )

    if settings.database_url is None:
        pytest.fail("GROWTHCREW_DATABASE_URL is required for integration tests.")

    with TestClient(create_application(settings)) as client:
        yield Context(client=client, settings=settings)


@pytest.fixture
def workspace(context: Context) -> Iterator[UUID]:
    """A workspace with a business profile and no generated stages."""

    response = context.client.post("/api/v1/workspaces", json={"name": "Status History Co"})
    assert response.status_code == 201
    workspace_id = UUID(response.json()["id"])
    profile = context.client.post(
        context.url(workspace_id, "business-profile"),
        json={"business_name": "Status History Co", "industry": "Healthy Snacks"},
    )
    assert profile.status_code == 201

    yield workspace_id

    database = Database.from_settings(context.settings)
    try:
        with database.session() as session:
            row = session.get(Workspace, workspace_id)
            if row is not None:
                session.delete(row)
                session.commit()
    finally:
        database.dispose()


def _status(ctx: Context, workspace_id: UUID) -> dict[str, Any]:
    response = ctx.client.get(ctx.url(workspace_id, "strategy-status"))
    assert response.status_code == 200
    return response.json()


def _by_stage(body: dict[str, Any]) -> dict[StrategyStage, dict[str, Any]]:
    return {StrategyStage(item["stage"]): item for item in body["stages"]}


def _gating(body: dict[str, Any]) -> dict[str, Any]:
    """Everything the status says about gating and approval, and nothing about history."""

    return {
        "stages": [{field: item[field] for field in GATING_FIELDS} for item in body["stages"]],
        "complete": body["complete"],
        "next_stage": body["next_stage"],
        "approved": body["approved"],
        "next_to_approve": body["next_to_approve"],
    }


def _build_chain(ctx: Context, workspace_id: UUID) -> None:
    response = ctx.client.post(
        ctx.url(workspace_id, "generate-full-strategy"),
        json={"force_regenerate": False, "auto_approve": True},
    )
    assert response.status_code == 200
    assert [item["outcome"] for item in response.json()["stages"]] == ["generated"] * 7


def _force(ctx: Context, workspace_id: UUID, path: str) -> None:
    response = ctx.client.post(ctx.url(workspace_id, path), json={"force_regenerate": True})
    assert response.status_code in {200, 201}


def _with_session(ctx: Context, action: Any) -> Any:
    database = Database.from_settings(ctx.settings)
    try:
        with database.session() as session:
            result = action(session)
            session.commit()
            return result
    finally:
        database.dispose()


def _edit_profile(ctx: Context, workspace_id: UUID) -> None:
    def action(session: Any) -> None:
        profile = session.scalars(
            select(BusinessProfile).where(BusinessProfile.workspace_id == workspace_id)
        ).one()
        profile.business_name = "Status History Co (edited)"

    _with_session(ctx, action)


def _market_research_record(workspace_id: UUID) -> MarketResearch:
    return MarketResearch(
        workspace_id=workspace_id,
        market_overview="Overview.",
        target_customer_segments="Segments.",
        competitive_landscape="Landscape.",
        opportunities_and_risks="Risks.",
        sources=[],
        model_used="test-model",
    )


# --- reporting -----------------------------------------------------------------------


def test_a_workspace_with_nothing_generated_reports_no_history(
    context: Context, workspace: UUID
) -> None:
    """No stage is generated, so none reports a generation or staleness."""

    body = _status(context, workspace)

    for item in body["stages"]:
        assert item["generation"] is None
        assert item["staleness"] is None
    assert body["stale_stages"] == []


def test_a_freshly_built_chain_reports_generation_one_tracked_and_not_stale(
    context: Context, workspace: UUID
) -> None:
    """Every stage built by the agents has a generation and known provenance."""

    _build_chain(context, workspace)

    body = _status(context, workspace)

    for item in body["stages"]:
        assert item["generation"] == 1, item["stage"]
        assert item["staleness"] == {"provenance": "tracked", "stale_inputs": [], "stale": False}
    assert body["stale_stages"] == []


def test_forcing_an_upstream_stage_flags_exactly_what_was_built_from_it(
    context: Context, workspace: UUID
) -> None:
    """Regenerating market research advances its generation and flags its dependants."""

    _build_chain(context, workspace)

    _force(context, workspace, "market-research")

    body = _status(context, workspace)
    stages = _by_stage(body)
    assert stages[S.MARKET_RESEARCH]["generation"] == 2
    assert stages[S.MARKET_RESEARCH]["staleness"]["stale"] is False
    assert body["stale_stages"] == ["customer_personas", "marketing_strategy"]
    for stage in (S.CUSTOMER_PERSONAS, S.MARKETING_STRATEGY):
        stale_inputs = stages[stage]["staleness"]["stale_inputs"]
        assert {
            "prerequisite": "market_research",
            "built_from": 1,
            "current": 2,
        } in stale_inputs
    for stage in (S.BUSINESS_UNDERSTANDING, S.COMPETITOR_ANALYSIS, S.BRAND_STRATEGY):
        assert stages[stage]["staleness"]["stale"] is False, stage


def test_staleness_changes_nothing_about_gating_or_approval(
    context: Context, workspace: UUID
) -> None:
    """Information only: forcing an upstream stage leaves every gating field as it was.

    The stage's own version does not move either - the local providers
    regenerate identical content - which is exactly the case where only
    the generation, not the version, shows that it happened.
    """

    _build_chain(context, workspace)
    before = _gating(_status(context, workspace))

    _force(context, workspace, "market-research")

    after = _status(context, workspace)
    assert _gating(after) == before
    assert after["stale_stages"] == ["customer_personas", "marketing_strategy"]
    assert after["approved"] is True


def test_a_stale_stage_can_still_be_regenerated_and_that_clears_its_staleness(
    context: Context, workspace: UUID
) -> None:
    """Stale never blocks generation, and rebuilding it from the newest inputs fixes it."""

    _build_chain(context, workspace)
    _force(context, workspace, "market-research")

    _force(context, workspace, "personas")

    body = _status(context, workspace)
    stages = _by_stage(body)
    assert stages[S.CUSTOMER_PERSONAS]["generation"] == 2
    assert stages[S.CUSTOMER_PERSONAS]["staleness"]["stale"] is False
    # Rebuilding personas fixed personas, but brand strategy was built from
    # the older personas, and marketing strategy from the older personas
    # *and* the older market research, so both are stale on what they used.
    assert body["stale_stages"] == ["brand_strategy", "marketing_strategy"]
    assert [
        item["prerequisite"] for item in stages[S.BRAND_STRATEGY]["staleness"]["stale_inputs"]
    ] == ["customer_personas"]
    assert [
        item["prerequisite"] for item in stages[S.MARKETING_STRATEGY]["staleness"]["stale_inputs"]
    ] == ["market_research", "customer_personas"]


def test_editing_the_business_profile_flags_every_stage_built_from_it(
    context: Context, workspace: UUID
) -> None:
    """The profile is an input too; the content plan, which does not use it, is unaffected."""

    _build_chain(context, workspace)
    before = _gating(_status(context, workspace))

    _edit_profile(context, workspace)

    body = _status(context, workspace)
    assert body["stale_stages"] == [
        "business_understanding",
        "market_research",
        "competitor_analysis",
        "customer_personas",
        "brand_strategy",
        "marketing_strategy",
    ]
    for stage in (S.BUSINESS_UNDERSTANDING, S.MARKET_RESEARCH):
        [stale_input] = _by_stage(body)[stage]["staleness"]["stale_inputs"]
        assert stale_input == {"prerequisite": "business_profile", "built_from": 1, "current": 2}
    assert _by_stage(body)[S.CONTENT_PLAN]["staleness"]["stale"] is False
    assert _gating(body) == before


# --- history that cannot be matched to a record ----------------------------------------


def test_a_stage_with_no_history_at_all_reports_unknown_provenance(
    context: Context, workspace: UUID
) -> None:
    """A record that never went through history has no generation and nothing to claim."""

    _with_session(context, lambda session: session.add(_market_research_record(workspace)))

    item = _by_stage(_status(context, workspace))[S.MARKET_RESEARCH]

    assert item["generated"] is True
    assert item["generation"] is None
    assert item["staleness"] == {"provenance": "unknown", "stale_inputs": [], "stale": False}


def test_a_stage_from_before_history_has_a_generation_but_unknown_provenance(
    context: Context, workspace: UUID
) -> None:
    """A backfilled generation 1 is reported, and is never stale however much has moved."""

    def action(session: Any) -> None:
        record = _market_research_record(workspace)
        session.add(record)
        session.flush()
        session.add(
            StageSnapshot(
                workspace_id=workspace,
                stage=S.MARKET_RESEARCH.value,
                generation=1,
                record_id=record.id,
                record_version=record.version,
                content={"market_overview": "Overview."},
                inputs=None,
                model_used="test-model",
            )
        )

    _with_session(context, action)
    _edit_profile(context, workspace)

    body = _status(context, workspace)
    item = _by_stage(body)[S.MARKET_RESEARCH]

    assert item["generation"] == 1
    assert item["staleness"] == {"provenance": "unknown", "stale_inputs": [], "stale": False}
    assert body["stale_stages"] == []


def test_a_record_changed_behind_historys_back_reports_no_generation(
    context: Context, workspace: UUID
) -> None:
    """If the live record no longer matches its newest snapshot, the snapshot is not claimed."""

    _build_chain(context, workspace)

    def action(session: Any) -> None:
        record = session.scalars(
            select(MarketResearch).where(MarketResearch.workspace_id == workspace)
        ).one()
        record.market_overview = "Edited without going through history."

    _with_session(context, action)

    stages = _by_stage(_status(context, workspace))

    assert stages[S.MARKET_RESEARCH]["generated"] is True
    assert stages[S.MARKET_RESEARCH]["generation"] is None
    assert stages[S.MARKET_RESEARCH]["staleness"]["provenance"] == "unknown"
    assert stages[S.COMPETITOR_ANALYSIS]["generation"] == 1


def test_a_deleted_stage_reports_no_history_even_though_its_snapshots_remain(
    context: Context, workspace: UUID
) -> None:
    """History describes the current record; with no record there is nothing to describe."""

    _build_chain(context, workspace)

    def action(session: Any) -> None:
        record = session.scalars(
            select(MarketResearch).where(MarketResearch.workspace_id == workspace)
        ).one()
        session.delete(record)

    _with_session(context, action)

    item = _by_stage(_status(context, workspace))[S.MARKET_RESEARCH]

    assert item["generated"] is False
    assert item["generation"] is None
    assert item["staleness"] is None
    history = context.client.get(context.url(workspace, "history/market_research")).json()
    assert history["latest_generation"] == 1
