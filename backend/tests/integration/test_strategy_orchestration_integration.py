"""Real PostgreSQL integration tests for strategy orchestration.

Uses the deterministic "local" LLM and embeddings providers and no
Tavily key (web search always resolves to no results), so this file is
about the orchestration contract against the real agent services and a
real database: dependency ordering, skip/force/resume behavior, failure
reporting, and - most importantly - that the prerequisite map the status
endpoint reports matches what each agent service actually enforces.
Whether a real model produces a good full strategy end to end is covered
separately in test_strategy_orchestration_llm_integration.py against
real Ollama Cloud, gated behind GROWTHCREW_RUN_LLM_INTEGRATION_TESTS.

The LLM gateway is wrapped in a counting proxy. A skipped stage must
cost zero LLM calls and a forced run must pay for every stage again, and
because the local provider is deterministic the call counts are exact -
which also makes them a reliable stand-in for "did this stage actually
run", since forced regeneration of identical content deliberately does
not change a record's version.
"""

import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete

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
from backend.app.db.models.workspace import Workspace
from backend.app.llm.errors import LLMProviderUnavailableError
from backend.app.llm.gateway import LLMGateway
from backend.app.main import create_application
from backend.app.schemas.strategy_orchestration import StrategyStage
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
ALL_GENERATED = ["generated"] * 7

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

# Models with a workspace_id column. Business Understanding hangs off the
# business profile instead, so it is handled separately.
WORKSPACE_MODELS: dict[StrategyStage, Any] = {
    S.MARKET_RESEARCH: MarketResearch,
    S.COMPETITOR_ANALYSIS: CompetitorAnalysis,
    S.CUSTOMER_PERSONAS: CustomerPersonaSet,
    S.BRAND_STRATEGY: BrandStrategy,
    S.MARKETING_STRATEGY: MarketingStrategy,
    S.CONTENT_PLAN: ContentPlan,
}

LLM_FAILURE_MESSAGE = "A required LLM operation could not be completed."


class CountingLLM:
    """Proxy a real LLMGateway, counting calls and optionally failing.

    Counts the four public generation methods. When fail_after is set,
    every call after that many successful ones raises a controlled LLM
    error carrying text that must never reach a client.
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
    """Everything a test needs: API client, settings, and the LLM proxy."""

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
    application = create_application(settings, llm_gateway=llm)

    with TestClient(application) as client:
        yield Context(client=client, settings=settings, llm=llm)


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


def _create_workspace(ctx: Context, name: str, *, with_profile: bool = True) -> UUID:
    """Create a workspace, optionally with a minimal business profile."""

    response = ctx.client.post("/api/v1/workspaces", json={"name": name})
    assert response.status_code == 201
    workspace_id = UUID(response.json()["id"])

    if with_profile:
        profile = ctx.client.post(
            ctx.url(workspace_id, "business-profile"),
            json={"business_name": name, "industry": "Healthy Snacks"},
        )
        assert profile.status_code == 201

    return workspace_id


def _status(ctx: Context, workspace_id: UUID) -> dict[str, Any]:
    response = ctx.client.get(ctx.url(workspace_id, "strategy-status"))
    assert response.status_code == 200
    return response.json()


def _run(
    ctx: Context, workspace_id: UUID, *, force: bool = False, auto: bool = True
) -> dict[str, Any]:
    """Run the full strategy.

    These tests are about how the chain is walked, not about human
    review, so a run auto-approves what it generates by default - without
    that, generation would stop at the first stage whose prerequisites
    await approval. How runs behave at approval checkpoints is covered in
    the approval integration tests.
    """

    response = ctx.client.post(
        ctx.url(workspace_id, "generate-full-strategy"),
        json={"force_regenerate": force, "auto_approve": auto},
    )
    assert response.status_code == 200
    return response.json()


def _outcomes(body: dict[str, Any]) -> list[str]:
    return [item["outcome"] for item in body["stages"]]


def _stage_record(ctx: Context, workspace_id: UUID, stage: StrategyStage) -> dict[str, Any]:
    response = ctx.client.get(ctx.url(workspace_id, STAGE_PATHS[stage]))
    assert response.status_code == 200
    return response.json()


def _generate_single(ctx: Context, workspace_id: UUID, stage: StrategyStage) -> Any:
    """Generate one stage through its own endpoint, then approve it if it was built.

    Models a person who generated a stage by hand and reviewed it, so the
    stages that build on it are not held back by the approval gate. A
    stage that could not be generated (a missing prerequisite, say) is
    returned as-is, with nothing approved.
    """

    response = ctx.client.post(ctx.url(workspace_id, STAGE_PATHS[stage]), json={})

    if response.status_code == 200:
        version = next(
            item["version"]
            for item in _status(ctx, workspace_id)["stages"]
            if item["stage"] == stage.value
        )
        approved = ctx.client.post(
            ctx.url(workspace_id, f"approvals/{stage.value}"), json={"version": version}
        )
        assert approved.status_code == 200

    return response


def _delete_stage_records(
    settings: Settings,
    workspace_id: UUID,
    stages: list[StrategyStage],
) -> None:
    """Delete stage records directly, leaving everything else untouched.

    Stage records deliberately carry no foreign key to each other, so any
    subset can be removed - which is exactly what lets these tests build
    states the API would never produce and ask each agent what it needs.
    """

    database = Database.from_settings(settings)

    try:
        with database.session() as session:
            for stage in stages:
                if stage is S.BUSINESS_UNDERSTANDING:
                    profile_ids = session.query(BusinessProfile.id).filter(
                        BusinessProfile.workspace_id == workspace_id
                    )
                    session.execute(
                        delete(BusinessUnderstanding).where(
                            BusinessUnderstanding.profile_id.in_(profile_ids)
                        )
                    )
                else:
                    model = WORKSPACE_MODELS[stage]
                    session.execute(delete(model).where(model.workspace_id == workspace_id))
            session.commit()
    finally:
        database.dispose()


def _delete_profile(settings: Settings, workspace_id: UUID) -> None:
    """Delete the business profile directly (cascades to Business Understanding)."""

    database = Database.from_settings(settings)

    try:
        with database.session() as session:
            session.execute(
                delete(BusinessProfile).where(BusinessProfile.workspace_id == workspace_id)
            )
            session.commit()
    finally:
        database.dispose()


def _with_workspace(
    ctx: Context, name: str, body: Callable[[UUID], None], *, with_profile: bool = True
) -> None:
    """Run a test body against a fresh workspace and always clean up."""

    workspace_id = _create_workspace(ctx, name, with_profile=with_profile)
    try:
        body(workspace_id)
    finally:
        cleanup_workspace(ctx.settings, workspace_id)


def test_status_for_a_workspace_without_a_profile(context: Context) -> None:
    """Nothing exists yet: everything is blocked, starting with the profile."""

    def body(workspace_id: UUID) -> None:
        status = _status(context, workspace_id)
        by_stage = {item["stage"]: item for item in status["stages"]}

        assert [item["stage"] for item in status["stages"]] == [s.value for s in STAGES]
        assert status["complete"] is False
        assert status["next_stage"] == "business_understanding"
        assert all(item["generated"] is False for item in status["stages"])
        assert all(item["version"] is None for item in status["stages"])
        assert all(item["can_generate"] is False for item in status["stages"])
        assert by_stage["business_understanding"]["missing_prerequisites"] == ["business_profile"]
        assert by_stage["brand_strategy"]["missing_prerequisites"] == [
            "business_profile",
            "business_understanding",
            "competitor_analysis",
            "customer_personas",
        ]
        assert by_stage["content_plan"]["missing_prerequisites"] == ["marketing_strategy"]

    _with_workspace(context, "Status Bare Co", body, with_profile=False)


def test_status_with_only_a_profile_unblocks_the_three_independent_stages(
    context: Context,
) -> None:
    """The profile alone is enough for exactly the three independent agents."""

    def body(workspace_id: UUID) -> None:
        status = _status(context, workspace_id)

        assert [item["can_generate"] for item in status["stages"]] == [True] * 3 + [False] * 4
        assert status["next_stage"] == "business_understanding"

    _with_workspace(context, "Status Profile Co", body)


def test_unknown_workspace_is_404_on_both_routes(context: Context) -> None:
    """A workspace that does not exist is the one error case."""

    unknown = UUID("00000000-0000-4000-8000-000000000000")

    status_response = context.client.get(context.url(unknown, "strategy-status"))
    run_response = context.client.post(context.url(unknown, "generate-full-strategy"), json={})

    assert status_response.status_code == 404
    assert run_response.status_code == 404
    assert status_response.json() == {"detail": "Workspace not found."}
    assert run_response.json() == {"detail": "Workspace not found."}


def test_full_run_builds_every_stage_from_the_profile_alone(context: Context) -> None:
    """One call takes a bare profile to a complete, readable seven-stage chain."""

    def body(workspace_id: UUID) -> None:
        run = _run(context, workspace_id)

        assert [item["stage"] for item in run["stages"]] == [s.value for s in STAGES]
        assert _outcomes(run) == ALL_GENERATED
        assert [item["version"] for item in run["stages"]] == [1] * 7
        assert run["complete"] is True
        assert run["failed_stage"] is None
        assert run["force_regenerate"] is False

        for stage in STAGES:
            assert _stage_record(context, workspace_id, stage)["version"] == 1

        status = _status(context, workspace_id)
        assert status["complete"] is True
        assert status["next_stage"] is None
        assert all(item["generated"] and item["can_generate"] for item in status["stages"])

    _with_workspace(context, "Full Run Co", body)


def test_second_run_skips_everything_without_calling_the_llm(context: Context) -> None:
    """A finished workspace costs nothing and changes nothing."""

    def body(workspace_id: UUID) -> None:
        _run(context, workspace_id)
        calls_after_first_run = context.llm.calls
        ids_before = {stage: _stage_record(context, workspace_id, stage)["id"] for stage in STAGES}

        second = _run(context, workspace_id)

        assert calls_after_first_run > 0
        assert context.llm.calls == calls_after_first_run
        assert _outcomes(second) == ["skipped"] * 7
        assert [item["version"] for item in second["stages"]] == [1] * 7
        assert second["complete"] is True
        assert {
            stage: _stage_record(context, workspace_id, stage)["id"] for stage in STAGES
        } == ids_before

    _with_workspace(context, "Idempotent Run Co", body)


def test_run_resumes_after_partial_generation(context: Context) -> None:
    """Stages built earlier by hand are skipped; the rest are generated."""

    def body(workspace_id: UUID) -> None:
        assert _generate_single(context, workspace_id, S.BUSINESS_UNDERSTANDING).status_code == 200
        assert _generate_single(context, workspace_id, S.MARKET_RESEARCH).status_code == 200
        ids_before = {
            stage: _stage_record(context, workspace_id, stage)["id"]
            for stage in (S.BUSINESS_UNDERSTANDING, S.MARKET_RESEARCH)
        }

        run = _run(context, workspace_id)

        assert _outcomes(run) == ["skipped", "skipped", *["generated"] * 5]
        assert run["complete"] is True
        for stage, record_id in ids_before.items():
            assert _stage_record(context, workspace_id, stage)["id"] == record_id

    _with_workspace(context, "Resume Run Co", body)


def test_run_fills_a_gap_in_the_middle_of_the_chain(context: Context) -> None:
    """Skipping is by stage, not position: a deleted middle stage is rebuilt."""

    def body(workspace_id: UUID) -> None:
        _run(context, workspace_id)
        _delete_stage_records(context.settings, workspace_id, [S.BRAND_STRATEGY])
        assert _status(context, workspace_id)["next_stage"] == "brand_strategy"

        run = _run(context, workspace_id)

        assert _outcomes(run) == ["skipped"] * 4 + ["generated"] + ["skipped"] * 2
        assert run["complete"] is True

    _with_workspace(context, "Gap Run Co", body)


def test_force_regenerate_pays_for_every_stage_again(context: Context) -> None:
    """Forcing reruns all seven agents, and keeps each record's identity."""

    def body(workspace_id: UUID) -> None:
        _run(context, workspace_id)
        first_run_calls = context.llm.calls
        ids_before = {stage: _stage_record(context, workspace_id, stage)["id"] for stage in STAGES}

        forced = _run(context, workspace_id, force=True)

        assert _outcomes(forced) == ALL_GENERATED
        assert forced["force_regenerate"] is True
        # The local provider is deterministic, so a forced run costs exactly
        # what the first run did: every stage really ran.
        assert context.llm.calls == first_run_calls * 2
        assert {
            stage: _stage_record(context, workspace_id, stage)["id"] for stage in STAGES
        } == ids_before

    _with_workspace(context, "Force Run Co", body)


def test_mid_chain_llm_failure_is_reported_safely_and_resumable(context: Context) -> None:
    """A real LLM failure partway through stops the run, then a retry resumes."""

    # Learn how many LLM calls the first stage takes, so the failure can be
    # placed exactly at the start of the second. Deterministic provider.
    probe_id = _create_workspace(context, "Failure Probe Co")
    try:
        before = context.llm.calls
        assert _generate_single(context, probe_id, S.BUSINESS_UNDERSTANDING).status_code == 200
        first_stage_calls = context.llm.calls - before
    finally:
        cleanup_workspace(context.settings, probe_id)

    assert first_stage_calls > 0

    def body(workspace_id: UUID) -> None:
        context.llm.fail_after = context.llm.calls + first_stage_calls
        try:
            response = context.client.post(
                context.url(workspace_id, "generate-full-strategy"),
                json={"auto_approve": True},
            )
        finally:
            context.llm.fail_after = None

        assert response.status_code == 200
        failed = response.json()
        assert _outcomes(failed) == ["generated", "failed", *["not_attempted"] * 5]
        assert failed["failed_stage"] == "market_research"
        assert failed["complete"] is False
        assert failed["stages"][1]["error"] == LLM_FAILURE_MESSAGE
        assert failed["stages"][1]["version"] is None
        assert "sk-secret-token" not in response.text

        status = _status(context, workspace_id)
        assert [item["generated"] for item in status["stages"]] == [True] + [False] * 6
        assert status["next_stage"] == "market_research"

        resumed = _run(context, workspace_id)

        assert _outcomes(resumed) == ["skipped", *["generated"] * 6]
        assert resumed["complete"] is True
        assert _status(context, workspace_id)["complete"] is True

    _with_workspace(context, "Failure Run Co", body)


def test_run_without_a_profile_fails_the_first_stage_with_http_200(context: Context) -> None:
    """A missing profile is a reported stage failure, not an error status."""

    def body(workspace_id: UUID) -> None:
        run = _run(context, workspace_id)

        assert _outcomes(run) == ["failed", *["not_attempted"] * 6]
        assert run["failed_stage"] == "business_understanding"
        assert run["stages"][0]["error"] == "Business profile not found."
        assert context.llm.calls == 0

    _with_workspace(context, "No Profile Run Co", body, with_profile=False)


def test_deleting_the_profile_blocks_regeneration_but_keeps_later_stages(
    context: Context,
) -> None:
    """Generated stages survive their profile; status says what can still run."""

    def body(workspace_id: UUID) -> None:
        _run(context, workspace_id)
        version = context.client.get(context.url(workspace_id, "business-profile")).json()[
            "version"
        ]
        deleted = context.client.delete(
            context.url(workspace_id, "business-profile"), params={"version": version}
        )
        assert deleted.status_code == 204

        status = _status(context, workspace_id)
        by_stage = {item["stage"]: item for item in status["stages"]}

        # Business Understanding hangs off the profile and went with it...
        assert by_stage["business_understanding"]["generated"] is False
        assert by_stage["business_understanding"]["missing_prerequisites"] == ["business_profile"]
        # ...the other six agents' records survive, but most can no longer be regenerated...
        assert [item["generated"] for item in status["stages"]] == [False] + [True] * 6
        assert by_stage["marketing_strategy"]["can_generate"] is False
        assert by_stage["marketing_strategy"]["missing_prerequisites"] == [
            "business_profile",
            "business_understanding",
        ]
        # ...except Content Planning, which only ever needed a Marketing Strategy.
        assert by_stage["content_plan"]["can_generate"] is True
        assert by_stage["content_plan"]["missing_prerequisites"] == []

        run = _run(context, workspace_id)

        assert _outcomes(run) == ["failed", *["not_attempted"] * 6]
        assert run["stages"][0]["error"] == "Business profile not found."

    _with_workspace(context, "Deleted Profile Co", body)


@pytest.mark.parametrize(
    ("stage", "missing"),
    [(stage, item) for stage in STAGES for item in STAGE_PREREQUISITES[stage]],
    ids=lambda value: getattr(value, "value", str(value)),
)
def test_every_mapped_prerequisite_is_really_required(
    context: Context, stage: StrategyStage, missing: Any
) -> None:
    """Removing any one prerequisite in the map makes the real agent 404.

    One limitation, inherent to the schema: Business Understanding is a
    child of the profile, so removing the profile also removes it, and for
    the stages that need both the profile case cannot isolate the two.
    """

    def body(workspace_id: UUID) -> None:
        _run(context, workspace_id)
        _delete_stage_records(context.settings, workspace_id, [stage])

        if missing.value == "business_profile":
            _delete_profile(context.settings, workspace_id)
        else:
            _delete_stage_records(context.settings, workspace_id, [StrategyStage(missing.value)])

        response = _generate_single(context, workspace_id, stage)

        assert response.status_code == 404
        status = _status(context, workspace_id)
        assert {item["stage"]: item for item in status["stages"]}[stage.value]["generated"] is False

    _with_workspace(context, f"Required {stage.value} {missing.value}", body)


@pytest.mark.parametrize("stage", STAGES, ids=lambda stage: stage.value)
def test_the_mapped_prerequisites_alone_are_enough(context: Context, stage: StrategyStage) -> None:
    """With only the mapped prerequisites present, the real agent succeeds.

    Everything else - later stages, earlier stages the map does not list,
    and for Content Planning the profile itself - is deleted first, so a
    prerequisite the map omits but the service secretly needs would show
    up as a 404 here.
    """

    mapped = {item.value for item in STAGE_PREREQUISITES[stage]}

    def body(workspace_id: UUID) -> None:
        _run(context, workspace_id)
        old_id = _stage_record(context, workspace_id, stage)["id"]

        unneeded = [other for other in STAGES if other is stage or other.value not in mapped]
        _delete_stage_records(context.settings, workspace_id, unneeded)
        if "business_profile" not in mapped:
            _delete_profile(context.settings, workspace_id)

        response = _generate_single(context, workspace_id, stage)

        assert response.status_code == 200
        assert response.json()["version"] == 1
        assert response.json()["id"] != old_id

    _with_workspace(context, f"Sufficient {stage.value}", body)
