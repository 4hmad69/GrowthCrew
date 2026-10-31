"""Real PostgreSQL integration tests: every agent writes history as it generates.

Drives the seven real agent services through the API with the
deterministic "local" providers and no Tavily key, so this file is about
how generation and history interact - not about model quality. What it
proves: each agent leaves a snapshot that matches its live record and
names exactly the inputs it was built from; a generation and its
snapshot succeed or fail together; and forcing an upstream stage marks
precisely the stages built from it as stale, without touching approvals.
"""

import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from backend.app.config import Settings
from backend.app.db.database import Database
from backend.app.db.models.brand_strategy import BrandStrategy
from backend.app.db.models.business_profile import BusinessProfile
from backend.app.db.models.business_understanding import BusinessUnderstanding
from backend.app.db.models.competitor_analysis import CompetitorAnalysis
from backend.app.db.models.content_plan import ContentPlan
from backend.app.db.models.customer_persona_set import CustomerPersonaSet
from backend.app.db.models.market_research import MarketResearch
from backend.app.db.models.marketing_strategy import MarketingStrategy
from backend.app.db.models.stage_snapshot import StageSnapshot
from backend.app.db.models.workspace import Workspace
from backend.app.exceptions import StaleResourceError
from backend.app.llm.errors import LLMProviderUnavailableError
from backend.app.llm.gateway import LLMGateway
from backend.app.main import create_application
from backend.app.schemas.stage_history import STAGE_CONTENT_FIELDS
from backend.app.schemas.strategy_orchestration import (
    ProvenanceState,
    StrategyPrerequisite,
    StrategyStage,
)
from backend.app.services.stage_history import StageHistoryService
from backend.app.services.stage_staleness import StageStalenessService
from backend.app.services.strategy_status import STAGE_PREREQUISITES

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

STAGE_PATHS: dict[StrategyStage, str] = {
    S.BUSINESS_UNDERSTANDING: "business-profile/understanding",
    S.MARKET_RESEARCH: "market-research",
    S.COMPETITOR_ANALYSIS: "competitor-analysis",
    S.CUSTOMER_PERSONAS: "personas",
    S.BRAND_STRATEGY: "brand-strategy",
    S.MARKETING_STRATEGY: "marketing-strategy",
    S.CONTENT_PLAN: "content-plan",
}

MODELS: dict[StrategyStage, Any] = {
    S.BUSINESS_UNDERSTANDING: BusinessUnderstanding,
    S.MARKET_RESEARCH: MarketResearch,
    S.COMPETITOR_ANALYSIS: CompetitorAnalysis,
    S.CUSTOMER_PERSONAS: CustomerPersonaSet,
    S.BRAND_STRATEGY: BrandStrategy,
    S.MARKETING_STRATEGY: MarketingStrategy,
    S.CONTENT_PLAN: ContentPlan,
}


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

    response = context.client.post("/api/v1/workspaces", json={"name": "History Agents Co"})
    assert response.status_code == 201
    workspace_id = UUID(response.json()["id"])
    profile = context.client.post(
        context.url(workspace_id, "business-profile"),
        json={"business_name": "History Agents Co", "industry": "Healthy Snacks"},
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


def _run(ctx: Context, workspace_id: UUID, *, force: bool = False, auto: bool = False) -> Any:
    response = ctx.client.post(
        ctx.url(workspace_id, "generate-full-strategy"),
        json={"force_regenerate": force, "auto_approve": auto},
    )
    assert response.status_code == 200
    return response.json()


def _status(ctx: Context, workspace_id: UUID) -> dict[str, Any]:
    response = ctx.client.get(ctx.url(workspace_id, "strategy-status"))
    assert response.status_code == 200
    return response.json()


def _approvals(ctx: Context, workspace_id: UUID) -> dict[str, Any]:
    return {item["stage"]: item["approval"] for item in _status(ctx, workspace_id)["stages"]}


def _snapshots(ctx: Context, workspace_id: UUID) -> dict[StrategyStage, list[StageSnapshot]]:
    """Every snapshot of the workspace, per stage, oldest generation first."""

    database = Database.from_settings(ctx.settings)
    try:
        with database.session() as session:
            rows = session.scalars(
                select(StageSnapshot)
                .where(StageSnapshot.workspace_id == workspace_id)
                .order_by(StageSnapshot.generation)
            )
            grouped: dict[StrategyStage, list[StageSnapshot]] = {stage: [] for stage in STAGES}
            for row in rows:
                grouped[StrategyStage(row.stage)].append(row)
            return grouped
    finally:
        database.dispose()


def _live_record_content(ctx: Context, workspace_id: UUID, stage: StrategyStage) -> dict[str, Any]:
    """Read a stage's live record and return its generated fields plus identity."""

    model = MODELS[stage]
    database = Database.from_settings(ctx.settings)
    try:
        with database.session() as session:
            if stage is S.BUSINESS_UNDERSTANDING:
                statement = (
                    select(model)
                    .join(BusinessProfile, BusinessProfile.id == model.profile_id)
                    .where(BusinessProfile.workspace_id == workspace_id)
                )
            else:
                statement = select(model).where(model.workspace_id == workspace_id)
            record = session.scalars(statement).one()
            return {
                "id": record.id,
                "version": record.version,
                "model_used": record.model_used,
                "fields": {field: getattr(record, field) for field in STAGE_CONTENT_FIELDS[stage]},
            }
    finally:
        database.dispose()


def _staleness(ctx: Context, workspace_id: UUID) -> dict[StrategyStage, Any]:
    database = Database.from_settings(ctx.settings)
    try:
        with database.session() as session:
            return StageStalenessService(session).staleness_by_stage(workspace_id)
    finally:
        database.dispose()


def _build_chain(ctx: Context, workspace_id: UUID) -> None:
    """Generate all seven stages in one hands-off run."""

    body = _run(ctx, workspace_id, auto=True)
    assert [item["outcome"] for item in body["stages"]] == ["generated"] * 7


# --- what each agent records ---------------------------------------------------------


def test_every_agent_records_generation_one_matching_its_live_record(
    context: Context, workspace: UUID
) -> None:
    """After a full run, each stage has exactly one snapshot, equal to its record."""

    _build_chain(context, workspace)

    snapshots = _snapshots(context, workspace)
    for stage in STAGES:
        [snapshot] = snapshots[stage]
        live = _live_record_content(context, workspace, stage)
        assert snapshot.generation == 1, stage
        assert snapshot.record_id == live["id"], stage
        assert snapshot.record_version == live["version"], stage
        assert snapshot.model_used == live["model_used"], stage
        assert snapshot.content == live["fields"], stage


def test_every_agent_records_exactly_the_inputs_it_depends_on(
    context: Context, workspace: UUID
) -> None:
    """Provenance names the profile and upstream generations each stage was built from."""

    _build_chain(context, workspace)

    snapshots = _snapshots(context, workspace)
    for stage in STAGES:
        [snapshot] = snapshots[stage]
        prerequisites = STAGE_PREREQUISITES[stage]
        assert snapshot.inputs is not None, stage

        uses_profile = StrategyPrerequisite.BUSINESS_PROFILE in prerequisites
        assert (snapshot.inputs["profile"] is not None) is uses_profile, stage

        upstream = {
            StrategyStage(item.value)
            for item in prerequisites
            if item is not StrategyPrerequisite.BUSINESS_PROFILE
        }
        assert {StrategyStage(key) for key in snapshot.inputs["stages"]} == upstream, stage
        for up in upstream:
            [up_snapshot] = snapshots[up]
            assert snapshot.inputs["stages"][up.value] == {
                "generation": 1,
                "record_id": str(up_snapshot.record_id),
                "record_version": up_snapshot.record_version,
            }, (stage, up)


def test_a_freshly_built_chain_is_tracked_and_not_stale(context: Context, workspace: UUID) -> None:
    """Nothing built by the agents themselves ever has unknown provenance."""

    _build_chain(context, workspace)

    result = _staleness(context, workspace)

    assert list(result) == STAGES
    assert all(item.provenance is ProvenanceState.TRACKED for item in result.values())
    assert not any(item.stale for item in result.values())


def test_asking_again_without_force_adds_no_history(context: Context, workspace: UUID) -> None:
    """Skipping an existing stage costs nothing and records nothing."""

    _build_chain(context, workspace)

    body = _run(context, workspace)

    assert [item["outcome"] for item in body["stages"]] == ["skipped"] * 7
    assert [len(items) for items in _snapshots(context, workspace).values()] == [1] * 7


# --- regenerating --------------------------------------------------------------------


def test_forcing_one_stage_adds_a_generation_and_flags_only_what_was_built_from_it(
    context: Context, workspace: UUID
) -> None:
    """Regenerating market research makes exactly the stages that use it stale."""

    _build_chain(context, workspace)
    approvals_before = _approvals(context, workspace)

    response = context.client.post(
        context.url(workspace, STAGE_PATHS[S.MARKET_RESEARCH]), json={"force_regenerate": True}
    )
    assert response.status_code in {200, 201}

    snapshots = _snapshots(context, workspace)
    assert [item.generation for item in snapshots[S.MARKET_RESEARCH]] == [1, 2]
    assert [len(snapshots[stage]) for stage in STAGES if stage is not S.MARKET_RESEARCH] == [1] * 6

    result = _staleness(context, workspace)
    expected = {
        stage
        for stage, prerequisites in STAGE_PREREQUISITES.items()
        if StrategyPrerequisite.MARKET_RESEARCH in prerequisites
    }
    assert expected == {S.CUSTOMER_PERSONAS, S.MARKETING_STRATEGY}
    assert {stage for stage, item in result.items() if item.stale} == expected
    for stage in expected:
        [stale_input] = result[stage].stale_inputs
        assert (stale_input.prerequisite.value, stale_input.built_from, stale_input.current) == (
            "market_research",
            1,
            2,
        )

    # Staleness is information only: no other stage's approval moved.
    approvals_after = _approvals(context, workspace)
    for stage in STAGES:
        if stage is not S.MARKET_RESEARCH:
            assert approvals_after[stage.value] == approvals_before[stage.value], stage


def test_a_full_forced_run_rebuilds_everything_from_the_newest_inputs(
    context: Context, workspace: UUID
) -> None:
    """Forcing the whole chain gives every stage generation 2, built from generation 2."""

    _build_chain(context, workspace)

    body = _run(context, workspace, force=True, auto=True)

    assert [item["outcome"] for item in body["stages"]] == ["generated"] * 7
    snapshots = _snapshots(context, workspace)
    for stage in STAGES:
        assert [item.generation for item in snapshots[stage]] == [1, 2], stage
        newest = snapshots[stage][-1]
        for up_key, reference in newest.inputs["stages"].items():
            assert reference["generation"] == 2, (stage, up_key)
    assert not any(item.stale for item in _staleness(context, workspace).values())


# --- failure paths -------------------------------------------------------------------


def test_a_generation_blocked_by_the_approval_gate_writes_no_history(
    context: Context, workspace: UUID
) -> None:
    """A 409 from the gate happens before any work, so it leaves nothing behind."""

    body = _run(context, workspace)
    assert [item["outcome"] for item in body["stages"]][:4] == [
        "generated",
        "generated",
        "generated",
        "awaiting_approval",
    ]

    blocked = context.client.post(context.url(workspace, STAGE_PATHS[S.CUSTOMER_PERSONAS]), json={})

    assert blocked.status_code == 409
    snapshots = _snapshots(context, workspace)
    assert snapshots[S.CUSTOMER_PERSONAS] == []
    assert sum(len(items) for items in snapshots.values()) == 3


def test_a_failed_model_call_writes_no_history(
    context: Context, workspace: UUID, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the model fails, there is no record and no snapshot - and no leaked error text."""

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise LLMProviderUnavailableError("upstream rejected key=sk-secret-token")

    monkeypatch.setattr(LLMGateway, "structured_with_usage", refuse)

    response = context.client.post(
        context.url(workspace, STAGE_PATHS[S.BUSINESS_UNDERSTANDING]), json={}
    )

    assert response.status_code >= 500
    assert "sk-secret-token" not in response.text
    assert sum(len(items) for items in _snapshots(context, workspace).values()) == 0
    missing = context.client.get(context.url(workspace, STAGE_PATHS[S.BUSINESS_UNDERSTANDING]))
    assert missing.status_code == 404


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (SQLAlchemyError("boom: password=hunter2"), 503),
        (StaleResourceError("The market research stage was regenerated by another request."), 409),
    ],
    ids=["database-failure", "generation-race"],
)
def test_a_history_failure_aborts_the_whole_generation(
    context: Context,
    workspace: UUID,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    expected_status: int,
) -> None:
    """The record and its snapshot succeed or fail together; nothing half-saved remains."""

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise error

    monkeypatch.setattr(StageHistoryService, "record", fail)

    response = context.client.post(context.url(workspace, STAGE_PATHS[S.MARKET_RESEARCH]), json={})

    assert response.status_code == expected_status
    assert "hunter2" not in response.text
    assert sum(len(items) for items in _snapshots(context, workspace).values()) == 0
    missing = context.client.get(context.url(workspace, STAGE_PATHS[S.MARKET_RESEARCH]))
    assert missing.status_code == 404

    # The failure left no stuck state: once history works again, the same
    # request succeeds and becomes generation 1.
    monkeypatch.undo()
    retry = context.client.post(context.url(workspace, STAGE_PATHS[S.MARKET_RESEARCH]), json={})
    assert retry.status_code in {200, 201}
    assert [item.generation for item in _snapshots(context, workspace)[S.MARKET_RESEARCH]] == [1]
