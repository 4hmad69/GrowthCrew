"""Real PostgreSQL integration tests for the approval gate on generation.

Customer Personas, Brand Strategy, Marketing Strategy, and Content
Planning refuse to generate until every stage they build on has been
approved - not merely generated. These tests drive each of the four real
endpoints over HTTP, with the deterministic "local" LLM and embeddings
providers and no Tavily key, against real PostgreSQL.

What they pin down, per gated stage:

- unapproved prerequisites give a 409 that names exactly what to approve,
  creates nothing, and costs zero model calls;
- approving the prerequisites one at a time unblocks generation exactly
  when the last one is approved;
- force_regenerate is gated too, while a plain call still returns an
  existing record untouched, as it always has;
- a prerequisite that does not exist is still a 404, checked before
  approval is ever considered;
- an approval that has gone stale (its stage changed) blocks like none;
- and the strategy status' can_generate always agrees with what the real
  endpoints then do.

The three stages that need only the business profile are never gated.
"""

import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, select

from backend.app.config import Settings
from backend.app.db.database import Database
from backend.app.db.models.brand_strategy import BrandStrategy
from backend.app.db.models.content_plan import ContentPlan
from backend.app.db.models.customer_persona_set import CustomerPersonaSet
from backend.app.db.models.market_research import MarketResearch
from backend.app.db.models.marketing_strategy import MarketingStrategy
from backend.app.db.models.workspace import Workspace
from backend.app.llm.gateway import LLMGateway
from backend.app.main import create_application
from backend.app.schemas.strategy_orchestration import StrategyStage
from backend.app.services.stage_approval import approval_required_message
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
PROFILE_ONLY = [S.BUSINESS_UNDERSTANDING, S.MARKET_RESEARCH, S.COMPETITOR_ANALYSIS]
GATED = [S.CUSTOMER_PERSONAS, S.BRAND_STRATEGY, S.MARKETING_STRATEGY, S.CONTENT_PLAN]

STAGE_PATHS: dict[StrategyStage, str] = {
    S.BUSINESS_UNDERSTANDING: "business-profile/understanding",
    S.MARKET_RESEARCH: "market-research",
    S.COMPETITOR_ANALYSIS: "competitor-analysis",
    S.CUSTOMER_PERSONAS: "personas",
    S.BRAND_STRATEGY: "brand-strategy",
    S.MARKETING_STRATEGY: "marketing-strategy",
    S.CONTENT_PLAN: "content-plan",
}

# The record each gated stage keeps, so a test can remove just that stage.
GATED_MODELS: dict[StrategyStage, Any] = {
    S.CUSTOMER_PERSONAS: CustomerPersonaSet,
    S.BRAND_STRATEGY: BrandStrategy,
    S.MARKETING_STRATEGY: MarketingStrategy,
    S.CONTENT_PLAN: ContentPlan,
}

# What each service reports when a prerequisite does not exist at all.
MISSING_MESSAGES: dict[StrategyStage, str] = {
    S.MARKET_RESEARCH: "Market research has not been generated yet.",
    S.CUSTOMER_PERSONAS: "Customer personas have not been generated yet.",
    S.BRAND_STRATEGY: "Brand strategy has not been generated yet.",
    S.MARKETING_STRATEGY: "Marketing strategy has not been generated yet.",
}


def _prerequisites(stage: StrategyStage) -> list[StrategyStage]:
    """The stages a stage builds on, from the real prerequisite map (profile excluded)."""

    return [
        S(item.value) for item in STAGE_PREREQUISITES[stage] if item.value != "business_profile"
    ]


class CountingLLM:
    """Proxy a real LLMGateway, counting its four generation methods."""

    _TRACKED = frozenset({"chat", "chat_with_usage", "structured", "structured_with_usage"})

    def __init__(self, inner: LLMGateway) -> None:
        self._inner = inner
        self.calls = 0

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._inner, name)
        if name not in self._TRACKED:
            return attribute

        def tracked(*args: Any, **kwargs: Any) -> Any:
            self.calls += 1
            return attribute(*args, **kwargs)

        return tracked


@dataclass
class Context:
    """An API client, its settings, and the LLM call counter."""

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
    """A workspace with all seven stages generated and auto-approved."""

    response = context.client.post("/api/v1/workspaces", json={"name": "Approval Gate Co"})
    assert response.status_code == 201
    workspace_id = UUID(response.json()["id"])

    profile = context.client.post(
        context.url(workspace_id, "business-profile"),
        json={"business_name": "Approval Gate Co", "industry": "Healthy Snacks"},
    )
    assert profile.status_code == 201

    run = context.client.post(
        context.url(workspace_id, "generate-full-strategy"),
        json={"force_regenerate": False, "auto_approve": True},
    )
    assert run.status_code == 200
    assert [item["outcome"] for item in run.json()["stages"]] == ["generated"] * 7

    yield workspace_id
    _cleanup(context.settings, workspace_id)


def _status_by_stage(ctx: Context, workspace_id: UUID) -> dict[str, dict[str, Any]]:
    response = ctx.client.get(ctx.url(workspace_id, "strategy-status"))
    assert response.status_code == 200
    return {item["stage"]: item for item in response.json()["stages"]}


def _revoke(ctx: Context, workspace_id: UUID, stage: StrategyStage) -> None:
    response = ctx.client.delete(ctx.url(workspace_id, f"approvals/{stage.value}"))
    assert response.status_code == 204


def _approve(ctx: Context, workspace_id: UUID, stage: StrategyStage) -> None:
    version = _status_by_stage(ctx, workspace_id)[stage.value]["version"]
    response = ctx.client.post(
        ctx.url(workspace_id, f"approvals/{stage.value}"), json={"version": version}
    )
    assert response.status_code == 200


def _remove_stage(settings: Settings, workspace_id: UUID, stage: StrategyStage) -> None:
    """Delete a gated stage's record so it can be generated afresh."""

    model = GATED_MODELS[stage]
    database = Database.from_settings(settings)
    try:
        with database.session() as session:
            session.execute(delete(model).where(model.workspace_id == workspace_id))
            session.commit()
    finally:
        database.dispose()


def _remove_market_research(settings: Settings, workspace_id: UUID) -> None:
    database = Database.from_settings(settings)
    try:
        with database.session() as session:
            session.execute(
                delete(MarketResearch).where(MarketResearch.workspace_id == workspace_id)
            )
            session.commit()
    finally:
        database.dispose()


def _change_market_research(settings: Settings, workspace_id: UUID) -> None:
    """Edit the market research record in place, bumping its version."""

    database = Database.from_settings(settings)
    try:
        with database.session() as session:
            record = session.scalars(
                select(MarketResearch).where(MarketResearch.workspace_id == workspace_id)
            ).one()
            record.model_used = "edited-after-approval"
            session.commit()
    finally:
        database.dispose()


def _generate(ctx: Context, workspace_id: UUID, stage: StrategyStage, **body: Any) -> Any:
    return ctx.client.post(ctx.url(workspace_id, STAGE_PATHS[stage]), json=body)


def _unblocked_but_removed(ctx: Context, workspace_id: UUID, stage: StrategyStage) -> None:
    """Leave a gated stage ungenerated with every prerequisite generated but unapproved."""

    _remove_stage(ctx.settings, workspace_id, stage)
    for prerequisite in _prerequisites(stage):
        _revoke(ctx, workspace_id, prerequisite)


# --- the gate refuses ---------------------------------------------------------------------


@pytest.mark.parametrize("stage", GATED)
def test_unapproved_prerequisites_block_generation_and_cost_nothing(
    context: Context, workspace: UUID, stage: StrategyStage
) -> None:
    """A 409 names every stage to approve, creates nothing, and makes no model call."""

    _unblocked_but_removed(context, workspace, stage)
    outstanding = _prerequisites(stage)
    calls_before = context.llm.calls

    response = _generate(context, workspace, stage)

    assert response.status_code == 409
    assert response.json() == {"detail": approval_required_message(stage, outstanding)}
    assert context.llm.calls == calls_before
    assert context.client.get(context.url(workspace, STAGE_PATHS[stage])).status_code == 404


def test_the_409_message_reads_naturally_for_one_and_for_several_stages(
    context: Context, workspace: UUID
) -> None:
    """Pin the exact client-facing text for a single blocker and for three."""

    _unblocked_but_removed(context, workspace, S.CONTENT_PLAN)
    single = _generate(context, workspace, S.CONTENT_PLAN)

    _unblocked_but_removed(context, workspace, S.BRAND_STRATEGY)
    several = _generate(context, workspace, S.BRAND_STRATEGY)

    assert single.json()["detail"] == (
        "Approve the marketing strategy stage before generating the content plan stage."
    )
    assert several.json()["detail"] == (
        "Approve the business understanding, competitor analysis and customer personas "
        "stages before generating the brand strategy stage."
    )


@pytest.mark.parametrize("stage", GATED)
def test_approving_prerequisites_one_at_a_time_unblocks_exactly_at_the_last(
    context: Context, workspace: UUID, stage: StrategyStage
) -> None:
    """Each approval shrinks the list the 409 reports; the last one lets generation through."""

    _unblocked_but_removed(context, workspace, stage)
    prerequisites = _prerequisites(stage)

    for index, prerequisite in enumerate(prerequisites):
        _approve(context, workspace, prerequisite)
        remaining = prerequisites[index + 1 :]
        response = _generate(context, workspace, stage)

        if remaining:
            assert response.status_code == 409
            assert response.json() == {"detail": approval_required_message(stage, remaining)}
        else:
            assert response.status_code == 200

    assert context.client.get(context.url(workspace, STAGE_PATHS[stage])).status_code == 200


# --- force_regenerate is gated; plain calls still return what exists ------------------------


@pytest.mark.parametrize("stage", GATED)
def test_force_regenerate_is_gated_too_and_leaves_the_stage_untouched(
    context: Context, workspace: UUID, stage: StrategyStage
) -> None:
    """Regenerating needs approved prerequisites just like generating; refusal changes nothing."""

    before = context.client.get(context.url(workspace, STAGE_PATHS[stage])).json()
    _revoke(context, workspace, _prerequisites(stage)[0])
    calls_before = context.llm.calls

    response = _generate(context, workspace, stage, force_regenerate=True)

    assert response.status_code == 409
    assert context.llm.calls == calls_before
    assert context.client.get(context.url(workspace, STAGE_PATHS[stage])).json() == before


@pytest.mark.parametrize("stage", GATED)
def test_an_existing_record_is_still_returned_without_force_even_if_a_prerequisite_is_unapproved(
    context: Context, workspace: UUID, stage: StrategyStage
) -> None:
    """The long-standing rule survives: an existing stage is read back before any check."""

    before = context.client.get(context.url(workspace, STAGE_PATHS[stage])).json()
    _revoke(context, workspace, _prerequisites(stage)[0])
    calls_before = context.llm.calls

    response = _generate(context, workspace, stage)

    assert response.status_code == 200
    assert response.json() == before
    assert context.llm.calls == calls_before


# --- existence is checked before approval --------------------------------------------------


@pytest.mark.parametrize(
    ("stage", "missing"),
    [
        (S.CUSTOMER_PERSONAS, S.MARKET_RESEARCH),
        (S.BRAND_STRATEGY, S.CUSTOMER_PERSONAS),
        (S.MARKETING_STRATEGY, S.BRAND_STRATEGY),
        (S.CONTENT_PLAN, S.MARKETING_STRATEGY),
    ],
)
def test_a_prerequisite_that_does_not_exist_is_a_404_before_approval_is_considered(
    context: Context, workspace: UUID, stage: StrategyStage, missing: StrategyStage
) -> None:
    """With one prerequisite missing and the rest unapproved, the answer is still 404."""

    _unblocked_but_removed(context, workspace, stage)
    if missing is S.MARKET_RESEARCH:
        _remove_market_research(context.settings, workspace)
    else:
        _remove_stage(context.settings, workspace, missing)

    response = _generate(context, workspace, stage)

    assert response.status_code == 404
    assert response.json() == {"detail": MISSING_MESSAGES[missing]}


# --- stale approvals do not count -------------------------------------------------------------


def test_an_approval_that_went_stale_blocks_like_no_approval(
    context: Context, workspace: UUID
) -> None:
    """Editing an approved prerequisite retires its approval, so the stage after it is gated."""

    _remove_stage(context.settings, workspace, S.CUSTOMER_PERSONAS)
    _change_market_research(context.settings, workspace)

    response = _generate(context, workspace, S.CUSTOMER_PERSONAS)

    assert response.status_code == 409
    assert response.json() == {
        "detail": "Approve the market research stage before generating the customer personas stage."
    }


# --- the profile-only stages are never gated -----------------------------------------------------


@pytest.mark.parametrize("stage", PROFILE_ONLY)
def test_the_stages_that_need_only_the_profile_are_never_gated(
    context: Context, workspace: UUID, stage: StrategyStage
) -> None:
    """Nothing approved anywhere, and these three still regenerate freely."""

    for each in STAGES:
        _revoke(context, workspace, each)

    response = _generate(context, workspace, stage, force_regenerate=True)

    assert response.status_code == 200


# --- status tells the truth ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "approved",
    [
        pytest.param(set(STAGES), id="everything-approved"),
        pytest.param(set(), id="nothing-approved"),
        pytest.param({S.BUSINESS_UNDERSTANDING, S.MARKET_RESEARCH}, id="two-reviews-done"),
        pytest.param(
            {
                S.BUSINESS_UNDERSTANDING,
                S.MARKET_RESEARCH,
                S.COMPETITOR_ANALYSIS,
                S.CUSTOMER_PERSONAS,
            },
            id="up-to-personas",
        ),
    ],
)
def test_status_can_generate_agrees_with_what_each_endpoint_then_does(
    context: Context, workspace: UUID, approved: set[StrategyStage]
) -> None:
    """For every stage, can_generate is true exactly when a forced generate would succeed.

    Runs the real endpoints in each approval state and compares them to
    what the status promised beforehand - the check that keeps the status
    from claiming a stage is blocked (or free) when it is not.
    """

    for stage in STAGES:
        if stage not in approved:
            _revoke(context, workspace, stage)
    promised = _status_by_stage(context, workspace)

    for stage in STAGES:
        item = promised[stage.value]
        response = _generate(context, workspace, stage, force_regenerate=True)

        assert (response.status_code == 200) is item["can_generate"], (
            f"{stage.value}: status said can_generate={item['can_generate']} "
            f"but generating returned {response.status_code}"
        )
        if not item["can_generate"]:
            assert response.status_code == 409
            assert item["missing_prerequisites"] == []
            assert item["unapproved_prerequisites"] == [
                p.value for p in _prerequisites(stage) if p not in approved
            ]


def test_status_names_unapproved_prerequisites_separately_from_missing_ones(
    context: Context, workspace: UUID
) -> None:
    """A prerequisite is either missing or unapproved, never both, and each is named once."""

    _revoke(context, workspace, S.MARKET_RESEARCH)
    _remove_stage(context.settings, workspace, S.BRAND_STRATEGY)

    stages = _status_by_stage(context, workspace)

    assert stages["customer_personas"]["unapproved_prerequisites"] == ["market_research"]
    assert stages["customer_personas"]["missing_prerequisites"] == []
    assert stages["customer_personas"]["can_generate"] is False
    assert stages["marketing_strategy"]["missing_prerequisites"] == ["brand_strategy"]
    assert stages["marketing_strategy"]["unapproved_prerequisites"] == ["market_research"]
    assert stages["content_plan"]["can_generate"] is True
