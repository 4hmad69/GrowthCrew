"""Real PostgreSQL integration tests: the report content resolver.

Drives the real agents through the API with the deterministic "local"
providers and no Tavily key, then builds the report straight from the
service. The headline requirements are that a report names a specific
generation for every stage it contains, carries each stage's approval
and staleness exactly as the rest of the system reports them, never
calls unknown provenance fresh, and leaves out - with a reason - any
stage it cannot honestly tie to a generation.
"""

import os
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from backend.app.config import Settings
from backend.app.db.database import Database
from backend.app.db.models.business_profile import BusinessProfile
from backend.app.db.models.market_research import MarketResearch
from backend.app.db.models.stage_approval import StageApproval
from backend.app.db.models.stage_snapshot import StageSnapshot
from backend.app.db.models.workspace import Workspace
from backend.app.exceptions import ResourceNotFoundError
from backend.app.main import create_application
from backend.app.schemas.report import (
    STAGE_ORDER,
    OmissionReason,
    ReportDocument,
    ReportStageFlag,
)
from backend.app.schemas.stage_history import STAGE_CONTENT_FIELDS
from backend.app.schemas.strategy_orchestration import (
    ProvenanceState,
    StageApprovalState,
    StrategyStage,
)
from backend.app.services.report_content import ReportContentService
from backend.app.services.stage_history import StageHistoryService

RUN_INTEGRATION_TESTS = os.getenv("GROWTHCREW_RUN_INTEGRATION_TESTS") == "1"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not RUN_INTEGRATION_TESTS,
        reason=("Set GROWTHCREW_RUN_INTEGRATION_TESTS=1 to run PostgreSQL integration tests."),
    ),
]

S = StrategyStage
BUSINESS_NAME = "Report Content Co"
FIXED_NOW = datetime(2026, 11, 4, 12, 0, tzinfo=UTC)


@dataclass
class Context:
    """An API client and its settings."""

    client: TestClient
    settings: Settings

    def url(self, workspace_id: UUID, path: str) -> str:
        return f"/api/v1/workspaces/{workspace_id}/{path}"


@pytest.fixture
def context() -> Iterator[Context]:
    """An API client backed by real PostgreSQL and the local stubs."""

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


def _delete_workspace(ctx: Context, workspace_id: UUID) -> None:
    database = Database.from_settings(ctx.settings)
    try:
        with database.session() as session:
            row = session.get(Workspace, workspace_id)
            if row is not None:
                session.delete(row)
                session.commit()
    finally:
        database.dispose()


@pytest.fixture
def workspace(context: Context) -> Iterator[UUID]:
    """A workspace with a business profile and no generated stages."""

    response = context.client.post("/api/v1/workspaces", json={"name": "Report Workspace"})
    assert response.status_code == 201
    workspace_id = UUID(response.json()["id"])
    profile = context.client.post(
        context.url(workspace_id, "business-profile"),
        json={"business_name": BUSINESS_NAME, "industry": "Healthy Snacks"},
    )
    assert profile.status_code == 201

    yield workspace_id

    _delete_workspace(context, workspace_id)


@pytest.fixture
def workspace_without_profile(context: Context) -> Iterator[UUID]:
    """A workspace that never had a business profile."""

    response = context.client.post("/api/v1/workspaces", json={"name": "Profile-less Workspace"})
    assert response.status_code == 201
    workspace_id = UUID(response.json()["id"])

    yield workspace_id

    _delete_workspace(context, workspace_id)


def _with_session(ctx: Context, action: Any) -> Any:
    database = Database.from_settings(ctx.settings)
    try:
        with database.session() as session:
            result = action(session)
            session.commit()
            return result
    finally:
        database.dispose()


def _report(ctx: Context, workspace_id: UUID) -> ReportDocument:
    def action(session: Any) -> ReportDocument:
        return ReportContentService(session, clock=lambda: FIXED_NOW).build(workspace_id)

    return _with_session(ctx, action)


def _by_stage(report: ReportDocument) -> dict[StrategyStage, Any]:
    return {section.stage: section for section in report.sections}


def _omitted(report: ReportDocument) -> dict[StrategyStage, OmissionReason]:
    return {item.stage: item.reason for item in report.omitted}


def _build_chain(ctx: Context, workspace_id: UUID, *, auto_approve: bool = True) -> None:
    response = ctx.client.post(
        ctx.url(workspace_id, "generate-full-strategy"),
        json={"force_regenerate": False, "auto_approve": auto_approve},
    )
    assert response.status_code == 200
    if auto_approve:
        assert [item["outcome"] for item in response.json()["stages"]] == ["generated"] * 7


def _revoke(ctx: Context, workspace_id: UUID, stage: StrategyStage) -> None:
    response = ctx.client.delete(ctx.url(workspace_id, f"approvals/{stage.value}"))
    assert response.status_code == 204


def _force(ctx: Context, workspace_id: UUID, path: str) -> None:
    response = ctx.client.post(ctx.url(workspace_id, path), json={"force_regenerate": True})
    assert response.status_code in {200, 201}


def _detail(ctx: Context, workspace_id: UUID, stage: StrategyStage, generation: int) -> Any:
    response = ctx.client.get(ctx.url(workspace_id, f"history/{stage.value}/{generation}"))
    assert response.status_code == 200
    return response.json()


def _edit_profile(ctx: Context, workspace_id: UUID) -> None:
    def action(session: Any) -> None:
        profile = session.scalars(
            select(BusinessProfile).where(BusinessProfile.workspace_id == workspace_id)
        ).one()
        profile.business_name = f"{BUSINESS_NAME} (edited)"

    _with_session(ctx, action)


def _change_market_research(ctx: Context, workspace_id: UUID, **changes: Any) -> None:
    """Overwrite market research with new content, the way an agent regenerating it would."""

    def action(session: Any) -> None:
        record = session.scalars(
            select(MarketResearch).where(MarketResearch.workspace_id == workspace_id)
        ).one()
        for name, value in changes.items():
            setattr(record, name, value)
        StageHistoryService(session).record(workspace_id, S.MARKET_RESEARCH, record)

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


# --- an empty workspace ----------------------------------------------------------------


def test_a_workspace_with_nothing_generated_is_an_empty_but_valid_report(
    context: Context, workspace: UUID
) -> None:
    report = _report(context, workspace)

    assert report.workspace_id == workspace
    assert report.business_name == BUSINESS_NAME
    assert report.generated_at == FIXED_NOW
    assert report.sections == []
    assert [item.stage for item in report.omitted] == list(STAGE_ORDER)
    assert set(_omitted(report).values()) == {OmissionReason.NOT_GENERATED}
    assert report.is_complete is False
    assert report.is_clean is False


def test_a_workspace_without_a_profile_is_titled_by_the_workspace_name(
    context: Context, workspace_without_profile: UUID
) -> None:
    report = _report(context, workspace_without_profile)

    assert report.business_name == "Profile-less Workspace"
    assert len(report.omitted) == len(STAGE_ORDER)


def test_an_unknown_workspace_is_not_found(context: Context) -> None:
    with pytest.raises(ResourceNotFoundError):
        _report(context, uuid4())


# --- a freshly built, approved chain ---------------------------------------------------


def test_a_fresh_approved_chain_is_a_complete_clean_report(
    context: Context, workspace: UUID
) -> None:
    _build_chain(context, workspace)

    report = _report(context, workspace)

    assert report.is_complete is True
    assert report.is_clean is True
    assert report.flagged_stages == []
    assert [section.stage for section in report.sections] == list(STAGE_ORDER)
    for section in report.sections:
        assert section.generation == 1, section.stage
        assert section.approval is StageApprovalState.APPROVED, section.stage
        assert section.staleness.provenance is ProvenanceState.TRACKED, section.stage
        assert section.flags == [], section.stage


def test_each_section_is_exactly_the_generation_history_holds(
    context: Context, workspace: UUID
) -> None:
    """Content, record identity and model all come from the named generation."""

    _build_chain(context, workspace)

    report = _report(context, workspace)

    for section in report.sections:
        detail = _detail(context, workspace, section.stage, section.generation)
        assert str(section.record_id) == detail["record_id"], section.stage
        assert section.record_version == detail["record_version"], section.stage
        assert section.model_used == detail["model_used"], section.stage
        assert section.generated_at == datetime.fromisoformat(detail["created_at"]), section.stage
        assert section.content == detail["content"], section.stage
        assert set(section.content) == set(STAGE_CONTENT_FIELDS[section.stage]), section.stage


def test_each_section_names_the_live_record_its_approval_is_pinned_to(
    context: Context, workspace: UUID
) -> None:
    _build_chain(context, workspace)

    report = _report(context, workspace)

    for section in report.sections:
        approval = context.client.get(
            context.url(workspace, f"approvals/{section.stage.value}")
        ).json()
        assert approval["record_id"] == str(section.record_id), section.stage
        assert approval["approved_version"] == section.record_version, section.stage


def test_building_a_report_is_read_only_and_repeatable(context: Context, workspace: UUID) -> None:
    _build_chain(context, workspace)

    def counts(session: Any) -> tuple[int, int]:
        return (
            session.scalar(select(func.count()).select_from(StageSnapshot)),
            session.scalar(select(func.count()).select_from(StageApproval)),
        )

    before = _with_session(context, counts)
    first = _report(context, workspace)
    second = _report(context, workspace)
    after = _with_session(context, counts)

    assert before == after
    assert first == second


def test_a_section_does_not_share_its_content_with_the_stored_snapshot(
    context: Context, workspace: UUID
) -> None:
    """Editing a built report cannot reach into history held by the session."""

    _build_chain(context, workspace)

    def action(session: Any) -> None:
        service = ReportContentService(session, clock=lambda: FIXED_NOW)
        first = service.build(workspace)
        original = {section.stage: dict(section.content) for section in first.sections}
        for section in first.sections:
            section.content["added_by_a_caller"] = "x"
        second = service.build(workspace)

        for section in second.sections:
            assert "added_by_a_caller" not in section.content, section.stage
            assert section.content == original[section.stage], section.stage

    _with_session(context, action)


# --- approval ---------------------------------------------------------------------------


def test_stages_nobody_has_approved_are_flagged_draft(context: Context, workspace: UUID) -> None:
    _build_chain(context, workspace)
    for stage in (S.CUSTOMER_PERSONAS, S.CONTENT_PLAN):
        _revoke(context, workspace, stage)

    report = _report(context, workspace)

    stages = _by_stage(report)
    assert stages[S.CUSTOMER_PERSONAS].approval is StageApprovalState.DRAFT
    assert stages[S.CUSTOMER_PERSONAS].flags == [ReportStageFlag.DRAFT]
    assert stages[S.CONTENT_PLAN].flags == [ReportStageFlag.DRAFT]
    assert stages[S.MARKET_RESEARCH].flags == []
    assert report.flagged_stages == [S.CUSTOMER_PERSONAS, S.CONTENT_PLAN]
    assert report.is_clean is False
    assert report.is_complete is True


def test_a_chain_with_every_approval_revoked_is_all_draft(
    context: Context, workspace: UUID
) -> None:
    _build_chain(context, workspace)
    for stage in STAGE_ORDER:
        _revoke(context, workspace, stage)

    report = _report(context, workspace)

    assert all(section.approval is StageApprovalState.DRAFT for section in report.sections)
    assert report.flagged_stages == list(STAGE_ORDER)


def test_a_content_change_retires_the_approval_and_reports_the_new_generation(
    context: Context, workspace: UUID
) -> None:
    """The report shows what is live now, as a draft - never the approved earlier content."""

    _build_chain(context, workspace)

    _change_market_research(context, workspace, market_overview="A rewritten overview.")

    section = _by_stage(_report(context, workspace))[S.MARKET_RESEARCH]
    assert section.generation == 2
    assert section.record_version == 2
    assert section.approval is StageApprovalState.DRAFT
    assert ReportStageFlag.DRAFT in section.flags
    assert section.content["market_overview"] == "A rewritten overview."


def test_an_identical_forced_regeneration_keeps_its_approval(
    context: Context, workspace: UUID
) -> None:
    """Version is unchanged, so the approval still matches: the newest generation is approved."""

    _build_chain(context, workspace)
    before = _by_stage(_report(context, workspace))[S.MARKET_RESEARCH]

    _force(context, workspace, "market-research")

    after = _by_stage(_report(context, workspace))[S.MARKET_RESEARCH]
    assert after.generation == 2
    assert after.record_id == before.record_id
    assert after.record_version == before.record_version
    assert after.approval is StageApprovalState.APPROVED
    assert after.flags == []


# --- staleness --------------------------------------------------------------------------


def test_forcing_an_upstream_stage_flags_exactly_what_was_built_from_it(
    context: Context, workspace: UUID
) -> None:
    _build_chain(context, workspace)

    _force(context, workspace, "market-research")

    report = _report(context, workspace)
    stages = _by_stage(report)
    assert report.flagged_stages == [S.CUSTOMER_PERSONAS, S.MARKETING_STRATEGY]
    for stage in (S.CUSTOMER_PERSONAS, S.MARKETING_STRATEGY):
        assert stages[stage].flags == [ReportStageFlag.STALE], stage
        assert stages[stage].approval is StageApprovalState.APPROVED, stage
        assert stages[stage].staleness.stale is True, stage
    assert stages[S.MARKET_RESEARCH].generation == 2
    assert stages[S.MARKET_RESEARCH].flags == []
    assert report.is_clean is False


def test_editing_the_business_profile_flags_every_stage_built_from_it(
    context: Context, workspace: UUID
) -> None:
    _build_chain(context, workspace)

    _edit_profile(context, workspace)

    report = _report(context, workspace)
    assert report.business_name == f"{BUSINESS_NAME} (edited)"
    assert report.flagged_stages == [
        S.BUSINESS_UNDERSTANDING,
        S.MARKET_RESEARCH,
        S.COMPETITOR_ANALYSIS,
        S.CUSTOMER_PERSONAS,
        S.BRAND_STRATEGY,
        S.MARKETING_STRATEGY,
    ]
    assert _by_stage(report)[S.CONTENT_PLAN].flags == []


def test_a_stage_can_be_both_draft_and_stale(context: Context, workspace: UUID) -> None:
    _build_chain(context, workspace)
    _force(context, workspace, "market-research")
    _revoke(context, workspace, S.CUSTOMER_PERSONAS)

    section = _by_stage(_report(context, workspace))[S.CUSTOMER_PERSONAS]

    assert section.flags == [ReportStageFlag.DRAFT, ReportStageFlag.STALE]


# --- unknown provenance -----------------------------------------------------------------


def test_a_stage_from_before_history_is_unknown_never_fresh_and_never_stale(
    context: Context, workspace: UUID
) -> None:
    """A backfilled generation is reported, flagged unknown, and not stale however much moved."""

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

    report = _report(context, workspace)

    section = _by_stage(report)[S.MARKET_RESEARCH]
    assert section.generation == 1
    assert section.staleness.provenance is ProvenanceState.UNKNOWN
    assert section.staleness.stale is False
    assert section.flags == [ReportStageFlag.DRAFT, ReportStageFlag.UNKNOWN_PROVENANCE]
    assert section.is_clean is False
    assert section.content == {"market_overview": "Overview."}
    assert report.flagged_stages == [S.MARKET_RESEARCH]
    assert set(_omitted(report).values()) == {OmissionReason.NOT_GENERATED}


def test_an_approved_unknown_provenance_stage_is_still_not_clean(
    context: Context, workspace: UUID
) -> None:
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
    approved = context.client.post(
        context.url(workspace, "approvals/market_research"), json={"version": 1}
    )
    assert approved.status_code == 200

    section = _by_stage(_report(context, workspace))[S.MARKET_RESEARCH]

    assert section.approval is StageApprovalState.APPROVED
    assert section.flags == [ReportStageFlag.UNKNOWN_PROVENANCE]
    assert section.is_clean is False


# --- stages the report cannot tie to a generation ---------------------------------------


def test_a_record_with_no_history_is_omitted_as_out_of_sync(
    context: Context, workspace: UUID
) -> None:
    _with_session(context, lambda session: session.add(_market_research_record(workspace)))

    report = _report(context, workspace)

    assert _omitted(report)[S.MARKET_RESEARCH] is OmissionReason.HISTORY_OUT_OF_SYNC
    assert S.MARKET_RESEARCH not in _by_stage(report)


def test_a_record_changed_behind_historys_back_is_omitted_as_out_of_sync(
    context: Context, workspace: UUID
) -> None:
    """Only that stage is lost; the rest of the chain is still reported."""

    _build_chain(context, workspace)

    def action(session: Any) -> None:
        record = session.scalars(
            select(MarketResearch).where(MarketResearch.workspace_id == workspace)
        ).one()
        record.market_overview = "Edited without going through history."

    _with_session(context, action)

    report = _report(context, workspace)

    assert _omitted(report) == {S.MARKET_RESEARCH: OmissionReason.HISTORY_OUT_OF_SYNC}
    assert [section.stage for section in report.sections] == [
        stage for stage in STAGE_ORDER if stage is not S.MARKET_RESEARCH
    ]
    assert report.is_complete is False
    assert report.is_clean is False


def test_a_recreated_record_at_the_same_version_is_not_matched_to_the_old_history(
    context: Context, workspace: UUID
) -> None:
    """Version alone is not identity: only the record id tells a recreated record apart."""

    _build_chain(context, workspace)

    def action(session: Any) -> None:
        old = session.scalars(
            select(MarketResearch).where(MarketResearch.workspace_id == workspace)
        ).one()
        old_version = old.version
        session.delete(old)
        session.flush()
        replacement = _market_research_record(workspace)
        session.add(replacement)
        session.flush()
        assert replacement.version == old_version

    _with_session(context, action)

    report = _report(context, workspace)

    assert _omitted(report) == {S.MARKET_RESEARCH: OmissionReason.HISTORY_OUT_OF_SYNC}


def test_a_deleted_stage_is_omitted_as_not_generated_though_its_history_remains(
    context: Context, workspace: UUID
) -> None:
    _build_chain(context, workspace)

    def action(session: Any) -> None:
        record = session.scalars(
            select(MarketResearch).where(MarketResearch.workspace_id == workspace)
        ).one()
        session.delete(record)

    _with_session(context, action)

    report = _report(context, workspace)

    assert _omitted(report) == {S.MARKET_RESEARCH: OmissionReason.NOT_GENERATED}
    history = context.client.get(context.url(workspace, "history/market_research")).json()
    assert history["latest_generation"] == 1
