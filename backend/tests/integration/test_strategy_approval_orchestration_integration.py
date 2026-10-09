"""Real PostgreSQL integration tests for auto-approval in full-strategy runs.

Uses the deterministic "local" LLM and embeddings providers and no
Tavily key, so this file is about how a full-strategy run and the stage
approvals interact against the real agent services and a real database -
not about model quality.

The rules under test: a run only auto-approves what it generated itself
(never a stage it merely skipped, since a person may not have reviewed
it); an automatic approval never overwrites a human one; an edited stage
is never rubber-stamped; and a run that fails partway keeps the
approvals it already made, so a retry resumes with a consistent state.

The approval gate on generation does not exist yet at this point in the
step, so these tests cover the run's own behavior; how a run stops when a
prerequisite is unapproved is covered once the gate lands.
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
from backend.app.db.models.market_research import MarketResearch
from backend.app.db.models.stage_approval import StageApproval
from backend.app.db.models.workspace import Workspace
from backend.app.llm.errors import LLMProviderUnavailableError
from backend.app.llm.gateway import LLMGateway
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
MR = S.MARKET_RESEARCH

# Each stage's own single-agent endpoint, relative to the workspace.
STAGE_PATHS: dict[StrategyStage, str] = {
    S.BUSINESS_UNDERSTANDING: "business-profile/understanding",
    S.MARKET_RESEARCH: "market-research",
    S.COMPETITOR_ANALYSIS: "competitor-analysis",
    S.CUSTOMER_PERSONAS: "personas",
    S.BRAND_STRATEGY: "brand-strategy",
    S.MARKETING_STRATEGY: "marketing-strategy",
    S.CONTENT_PLAN: "content-plan",
}


class CountingLLM:
    """Proxy a real LLMGateway, counting calls and optionally failing.

    When fail_after is set, every call after that many successful ones
    raises a controlled LLM error carrying text that must never reach a
    client.
    """

    _TRACKED = frozenset({"chat", "chat_with_usage", "structured", "structured_with_usage"})

    def __init__(self, inner: LLMGateway) -> None:
        self._inner = inner
        self.calls = 0
        self.fail_after: int | None = None

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._inner, name)
        if name not in self._TRACKED:
            return attribute

        def tracked(*args: Any, **kwargs: Any) -> Any:
            if self.fail_after is not None and self.calls >= self.fail_after:
                raise LLMProviderUnavailableError("upstream rejected key=sk-secret-token")
            self.calls += 1
            return attribute(*args, **kwargs)

        return tracked


@dataclass
class Context:
    """An API client, its settings, and the LLM proxy."""

    client: TestClient
    settings: Settings
    llm: CountingLLM

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

    llm = CountingLLM(LLMGateway(settings))

    with TestClient(create_application(settings, llm_gateway=llm)) as client:
        yield Context(client=client, settings=settings, llm=llm)


def _cleanup(settings: Settings, workspace_id: UUID) -> None:
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


@pytest.fixture
def workspace(context: Context) -> Iterator[UUID]:
    """A workspace with a business profile and no generated stages."""

    response = context.client.post("/api/v1/workspaces", json={"name": "Auto Approval Co"})
    assert response.status_code == 201
    workspace_id = UUID(response.json()["id"])

    profile = context.client.post(
        context.url(workspace_id, "business-profile"),
        json={"business_name": "Auto Approval Co", "industry": "Healthy Snacks"},
    )
    assert profile.status_code == 201

    yield workspace_id
    _cleanup(context.settings, workspace_id)


def _run(
    ctx: Context, workspace_id: UUID, *, force: bool = False, auto: bool = False
) -> dict[str, Any]:
    response = ctx.client.post(
        ctx.url(workspace_id, "generate-full-strategy"),
        json={"force_regenerate": force, "auto_approve": auto},
    )
    assert response.status_code == 200
    return response.json()


def _outcomes(body: dict[str, Any]) -> list[str]:
    return [item["outcome"] for item in body["stages"]]


def _status(ctx: Context, workspace_id: UUID) -> dict[str, Any]:
    response = ctx.client.get(ctx.url(workspace_id, "strategy-status"))
    assert response.status_code == 200
    return response.json()


def _approvals(status: dict[str, Any]) -> list[str | None]:
    return [item["approval"] for item in status["stages"]]


def _rows(settings: Settings, workspace_id: UUID) -> dict[str, StageApproval]:
    database = Database.from_settings(settings)
    try:
        with database.session() as session:
            return {
                row.stage: row
                for row in session.scalars(
                    select(StageApproval).where(StageApproval.workspace_id == workspace_id)
                )
            }
    finally:
        database.dispose()


def _generate_single(ctx: Context, workspace_id: UUID, stage: StrategyStage) -> None:
    response = ctx.client.post(ctx.url(workspace_id, STAGE_PATHS[stage]), json={})
    assert response.status_code in {200, 201}


def _approve_over_http(ctx: Context, workspace_id: UUID, stage: StrategyStage) -> None:
    """Approve a stage as a reviewer would: at the version the status reports."""

    version = next(
        item["version"]
        for item in _status(ctx, workspace_id)["stages"]
        if item["stage"] == stage.value
    )
    response = ctx.client.post(
        ctx.url(workspace_id, f"approvals/{stage.value}"), json={"version": version}
    )
    assert response.status_code == 200


def _generate_all_as_drafts(ctx: Context, workspace_id: UUID) -> None:
    """Build all seven stages, then leave every one of them unreviewed.

    Approval gates generation, so the chain is built hands-off and the
    approvals are then revoked through the real route.
    """

    assert _outcomes(_run(ctx, workspace_id, auto=True)) == ["generated"] * 7
    for stage in STAGES:
        revoked = ctx.client.delete(ctx.url(workspace_id, f"approvals/{stage.value}"))
        assert revoked.status_code == 204


def _change_market_research(settings: Settings, workspace_id: UUID) -> None:
    """Edit the market research record in place, bumping its version."""

    database = Database.from_settings(settings)
    try:
        with database.session() as session:
            record = session.scalars(
                select(MarketResearch).where(MarketResearch.workspace_id == workspace_id)
            ).one()
            record.model_used = "edited-after-run"
            session.commit()
    finally:
        database.dispose()


# --- the default is human review -----------------------------------------------------


def test_a_run_without_auto_approve_pauses_at_the_first_stage_that_needs_a_review(
    context: Context, workspace: UUID
) -> None:
    """Human review is the default: nothing is approved, so the run stops for a person.

    The three stages that need only the profile are generated, and the
    run then reports that customer personas is waiting on the other two
    reviews - as data with HTTP 200, not as a failure.
    """

    body = _run(context, workspace)

    assert _outcomes(body) == ["generated"] * 3 + ["awaiting_approval"] + ["not_attempted"] * 3
    assert body["auto_approve"] is False
    assert body["complete"] is False
    assert body["failed_stage"] is None
    assert body["awaiting_approval_stage"] == "customer_personas"
    assert body["stages"][3]["unapproved_prerequisites"] == [
        "business_understanding",
        "market_research",
    ]
    assert body["stages"][3]["error"] is None
    assert body["stages"][3]["version"] is None
    assert [item["auto_approved"] for item in body["stages"]] == [False] * 7
    assert _approvals(_status(context, workspace)) == ["draft"] * 3 + [None] * 4
    assert _rows(context.settings, workspace) == {}
    assert context.client.get(context.url(workspace, "personas")).status_code == 404


# --- auto-approve --------------------------------------------------------------------


def test_auto_approve_approves_the_whole_strategy_in_one_run(
    context: Context, workspace: UUID
) -> None:
    """One hands-off call yields a fully approved chain, every approval marked automatic."""

    body = _run(context, workspace, auto=True)
    status = _status(context, workspace)
    rows = _rows(context.settings, workspace)

    assert _outcomes(body) == ["generated"] * 7
    assert body["auto_approve"] is True
    assert [item["auto_approved"] for item in body["stages"]] == [True] * 7
    assert _approvals(status) == ["approved"] * 7
    assert status["approved"] is True
    assert set(rows) == {stage.value for stage in STAGES}
    assert {row.source for row in rows.values()} == {"auto"}


def test_each_auto_approval_is_pinned_to_the_version_the_run_saved(
    context: Context, workspace: UUID
) -> None:
    """The recorded approval names the very version the response reported."""

    body = _run(context, workspace, auto=True)
    rows = _rows(context.settings, workspace)

    for item in body["stages"]:
        assert rows[item["stage"]].approved_version == item["version"]


def test_an_auto_approval_can_be_read_back_as_automatic_over_http(
    context: Context, workspace: UUID
) -> None:
    """The approval routes show who approved: a run, not a person."""

    _run(context, workspace, auto=True)

    response = context.client.get(context.url(workspace, "approvals/market_research"))

    assert response.status_code == 200
    assert response.json()["source"] == "auto"


def test_a_person_approving_afterwards_takes_over_an_auto_approval(
    context: Context, workspace: UUID
) -> None:
    """Human review supersedes an automatic approval of the same version."""

    _run(context, workspace, auto=True)
    version = _status(context, workspace)["stages"][1]["version"]

    response = context.client.post(
        context.url(workspace, "approvals/market_research"), json={"version": version}
    )

    assert response.status_code == 200
    assert response.json()["source"] == "human"
    assert _rows(context.settings, workspace)["market_research"].source == "human"
    assert len(_rows(context.settings, workspace)) == 7


# --- skipped stages are never approved -----------------------------------------------


def test_a_run_never_approves_stages_it_only_skipped_so_a_person_must_review_them(
    context: Context, workspace: UUID
) -> None:
    """Hand-generated stages stay drafts and hold the run at a checkpoint until reviewed.

    The run generates and approves what it built itself, then stops at
    customer personas because the two skipped stages it builds on were
    never reviewed. After a person approves them, the same call finishes
    the strategy.
    """

    _generate_single(context, workspace, S.BUSINESS_UNDERSTANDING)
    _generate_single(context, workspace, MR)

    paused = _run(context, workspace, auto=True)

    assert _outcomes(paused) == [
        "skipped",
        "skipped",
        "generated",
        "awaiting_approval",
        "not_attempted",
        "not_attempted",
        "not_attempted",
    ]
    assert [item["auto_approved"] for item in paused["stages"]] == [False, False, True] + [
        False
    ] * 4
    assert paused["awaiting_approval_stage"] == "customer_personas"
    assert paused["stages"][3]["unapproved_prerequisites"] == [
        "business_understanding",
        "market_research",
    ]
    assert _approvals(_status(context, workspace)) == ["draft", "draft", "approved"] + [None] * 4

    _approve_over_http(context, workspace, S.BUSINESS_UNDERSTANDING)
    _approve_over_http(context, workspace, MR)

    resumed = _run(context, workspace, auto=True)
    rows = _rows(context.settings, workspace)

    assert _outcomes(resumed) == ["skipped"] * 3 + ["generated"] * 4
    assert [item["auto_approved"] for item in resumed["stages"]] == [False] * 3 + [True] * 4
    assert _approvals(_status(context, workspace)) == ["approved"] * 7
    assert {stage: row.source for stage, row in rows.items()} == {
        "business_understanding": "human",
        "market_research": "human",
        "competitor_analysis": "auto",
        "customer_personas": "auto",
        "brand_strategy": "auto",
        "marketing_strategy": "auto",
        "content_plan": "auto",
    }


def test_rerunning_with_auto_approve_does_not_rubber_stamp_existing_drafts(
    context: Context, workspace: UUID
) -> None:
    """Turning auto_approve on later cannot retroactively approve unreviewed stages."""

    _run(context, workspace)
    _generate_all_as_drafts(context, workspace)
    calls_before = context.llm.calls

    body = _run(context, workspace, auto=True)

    assert _outcomes(body) == ["skipped"] * 7
    assert [item["auto_approved"] for item in body["stages"]] == [False] * 7
    assert _approvals(_status(context, workspace)) == ["draft"] * 7
    assert _rows(context.settings, workspace) == {}
    assert context.llm.calls == calls_before


def test_an_edited_stage_is_not_rubber_stamped_by_a_later_run(
    context: Context, workspace: UUID
) -> None:
    """A stage changed after approval goes back to draft and a run leaves it that way."""

    _run(context, workspace, auto=True)
    _change_market_research(context.settings, workspace)
    assert _status(context, workspace)["stages"][1]["approval"] == "draft"

    body = _run(context, workspace, auto=True)

    assert _outcomes(body) == ["skipped"] * 7
    stages = _status(context, workspace)["stages"]
    assert [item["approval"] for item in stages] == [
        "approved",
        "draft",
        "approved",
        "approved",
        "approved",
        "approved",
        "approved",
    ]


# --- human approvals survive ----------------------------------------------------------


def test_a_forced_auto_run_never_downgrades_a_human_approval(
    context: Context, workspace: UUID
) -> None:
    """If regeneration leaves a stage unchanged, its human approval stays human."""

    _run(context, workspace, auto=True)
    _approve_over_http(context, workspace, MR)
    assert _rows(context.settings, workspace)["market_research"].source == "human"

    body = _run(context, workspace, force=True, auto=True)
    rows = _rows(context.settings, workspace)

    assert _outcomes(body) == ["generated"] * 7
    assert rows["market_research"].source == "human"
    assert {row.source for stage, row in rows.items() if stage != "market_research"} == {"auto"}
    assert _approvals(_status(context, workspace)) == ["approved"] * 7


# --- failure and resume ----------------------------------------------------------------


def test_a_failed_run_keeps_the_approvals_it_made_and_a_retry_finishes_the_job(
    context: Context, workspace: UUID
) -> None:
    """The failing stage and everything after it are unapproved; a retry completes the chain."""

    context.llm.fail_after = 1
    failed = _run(context, workspace, auto=True)

    assert _outcomes(failed) == ["generated", "failed"] + ["not_attempted"] * 5
    assert failed["failed_stage"] == "market_research"
    assert "sk-secret-token" not in str(failed)
    assert [item["auto_approved"] for item in failed["stages"]] == [True] + [False] * 6
    assert _approvals(_status(context, workspace))[0] == "approved"
    assert set(_rows(context.settings, workspace)) == {"business_understanding"}

    context.llm.fail_after = None
    resumed = _run(context, workspace, auto=True)
    status = _status(context, workspace)

    assert _outcomes(resumed) == ["skipped"] + ["generated"] * 6
    assert [item["auto_approved"] for item in resumed["stages"]] == [False] + [True] * 6
    assert _approvals(status) == ["approved"] * 7
    assert status["approved"] is True


def test_a_failed_run_without_auto_approve_approves_nothing(
    context: Context, workspace: UUID
) -> None:
    """A mid-chain failure must not leave behind approvals nobody asked for."""

    context.llm.fail_after = 1

    failed = _run(context, workspace)

    assert _outcomes(failed)[:2] == ["generated", "failed"]
    assert _rows(context.settings, workspace) == {}
