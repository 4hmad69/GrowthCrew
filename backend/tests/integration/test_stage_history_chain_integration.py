"""End-to-end integration tests for stage history across the whole system.

Each earlier layer is tested on its own; this file checks the seams
between them, through the real API against real PostgreSQL with the
deterministic "local" providers and no Tavily key:

- an approval can always be traced to the exact content that was
  approved, even after the stage has been overwritten;
- a forced regeneration that changes nothing keeps its approval, and
  history shows both generations matching it;
- a stage deleted and generated again continues its history rather than
  restarting it;
- a workspace that existed before history was introduced, carried
  through the real migration, keeps working and gains tracked history
  one regeneration at a time without ever being wrongly called stale.
"""

import os
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, make_url, select, text
from sqlalchemy.orm import Session

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
from backend.app.main import create_application
from backend.app.schemas.strategy_orchestration import StrategyStage
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
REPO_ROOT = Path(__file__).resolve().parents[3]
ALEMBIC_INI = REPO_ROOT / "backend" / "alembic.ini"
REVISION_BEFORE_HISTORY = "20261003_0011"


@dataclass
class Context:
    """An API client and its settings."""

    client: TestClient
    settings: Settings

    def url(self, workspace_id: UUID, path: str) -> str:
        return f"/api/v1/workspaces/{workspace_id}/{path}"


def _settings(**overrides: Any) -> Settings:
    settings = Settings(
        environment="test",
        llm_provider="local",
        embeddings_provider="local",
        tavily_api_key=None,
        **overrides,
    )
    if settings.database_url is None:
        pytest.fail("GROWTHCREW_DATABASE_URL is required for integration tests.")
    return settings


@pytest.fixture
def context() -> Iterator[Context]:
    """An API client backed by real PostgreSQL and the local stubs."""

    settings = _settings()
    with TestClient(create_application(settings)) as client:
        yield Context(client=client, settings=settings)


@pytest.fixture
def workspace(context: Context) -> Iterator[UUID]:
    """A workspace with a business profile and no generated stages."""

    response = context.client.post("/api/v1/workspaces", json={"name": "History Chain Co"})
    assert response.status_code == 201
    workspace_id = UUID(response.json()["id"])
    profile = context.client.post(
        context.url(workspace_id, "business-profile"),
        json={"business_name": "History Chain Co", "industry": "Healthy Snacks"},
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


def _get(ctx: Context, workspace_id: UUID, path: str) -> Any:
    return ctx.client.get(ctx.url(workspace_id, path))


def _post(ctx: Context, workspace_id: UUID, path: str, body: dict[str, Any]) -> Any:
    return ctx.client.post(ctx.url(workspace_id, path), json=body)


def _stages(ctx: Context, workspace_id: UUID) -> dict[StrategyStage, dict[str, Any]]:
    response = _get(ctx, workspace_id, "strategy-status")
    assert response.status_code == 200
    return {StrategyStage(item["stage"]): item for item in response.json()["stages"]}


def _stale_stages(ctx: Context, workspace_id: UUID) -> list[str]:
    return _get(ctx, workspace_id, "strategy-status").json()["stale_stages"]


def _history(ctx: Context, workspace_id: UUID, stage: str) -> list[dict[str, Any]]:
    response = _get(ctx, workspace_id, f"history/{stage}")
    assert response.status_code == 200
    return response.json()["snapshots"]


def _detail(ctx: Context, workspace_id: UUID, stage: str, generation: int) -> dict[str, Any]:
    response = _get(ctx, workspace_id, f"history/{stage}/{generation}")
    assert response.status_code == 200
    return response.json()


def _build_chain(ctx: Context, workspace_id: UUID) -> None:
    response = _post(
        ctx,
        workspace_id,
        "generate-full-strategy",
        {"force_regenerate": False, "auto_approve": True},
    )
    assert response.status_code == 200
    assert [item["outcome"] for item in response.json()["stages"]] == ["generated"] * 7


def _change_market_research(ctx: Context, workspace_id: UUID, **changes: Any) -> None:
    """Overwrite market research with new content, the way an agent regenerating it would."""

    database = Database.from_settings(ctx.settings)
    try:
        with database.session() as session:
            record = session.scalars(
                select(MarketResearch).where(MarketResearch.workspace_id == workspace_id)
            ).one()
            for name, value in changes.items():
                setattr(record, name, value)
            StageHistoryService(session).record(workspace_id, S.MARKET_RESEARCH, record)
            session.commit()
    finally:
        database.dispose()


# --- approvals and history --------------------------------------------------------------


def test_the_approved_content_stays_recoverable_after_the_stage_is_overwritten(
    context: Context, workspace: UUID
) -> None:
    """An approval names a record and a version; history can still produce that content.

    This is what a report needs: not the stage as it is now, but the stage
    as it was when somebody signed it off.
    """

    generated = _post(context, workspace, "market-research", {})
    assert generated.status_code in {200, 201}
    reviewed = _get(context, workspace, "market-research").json()
    approved = _post(context, workspace, "approvals/market_research", {"version": 1})
    assert approved.status_code == 200
    approval = approved.json()

    _change_market_research(context, workspace, market_overview="A rewritten overview.")

    # The approval is retired by the change, exactly as before history existed...
    assert _get(context, workspace, "approvals/market_research").status_code == 404
    assert _stages(context, workspace)[S.MARKET_RESEARCH]["approval"] == "draft"
    live = _get(context, workspace, "market-research").json()
    assert live["market_overview"] == "A rewritten overview."
    # ...but the content that was approved is still there, found by the pair
    # the approval carries.
    matching = [
        item
        for item in _history(context, workspace, "market_research")
        if item["record_id"] == approval["record_id"]
        and item["record_version"] == approval["approved_version"]
    ]
    assert [item["generation"] for item in matching] == [1]
    approved_content = _detail(context, workspace, "market_research", 1)["content"]
    assert approved_content["market_overview"] == reviewed["market_overview"]
    assert approved_content["market_overview"] != live["market_overview"]


def test_an_identical_regeneration_keeps_its_approval_and_both_generations_match_it(
    context: Context, workspace: UUID
) -> None:
    """The approval pin stays meaningful: same record, same version, same content."""

    _post(context, workspace, "market-research", {})
    approval = _post(context, workspace, "approvals/market_research", {"version": 1}).json()

    again = _post(context, workspace, "market-research", {"force_regenerate": True})
    assert again.status_code in {200, 201}

    current = _get(context, workspace, "approvals/market_research")
    assert current.status_code == 200
    assert current.json()["record_id"] == approval["record_id"]
    assert current.json()["approved_version"] == approval["approved_version"]
    snapshots = _history(context, workspace, "market_research")
    assert [item["generation"] for item in snapshots] == [2, 1]
    for item in snapshots:
        assert item["record_id"] == approval["record_id"]
        assert item["record_version"] == approval["approved_version"]
    assert (
        _detail(context, workspace, "market_research", 1)["content"]
        == _detail(context, workspace, "market_research", 2)["content"]
    )


def test_a_stage_deleted_and_generated_again_continues_its_history(
    context: Context, workspace: UUID
) -> None:
    """The record is new, but the history is the same stage's: numbering carries on."""

    _build_chain(context, workspace)
    original_record = _detail(context, workspace, "market_research", 1)["record_id"]
    database = Database.from_settings(context.settings)
    try:
        with database.session() as session:
            record = session.scalars(
                select(MarketResearch).where(MarketResearch.workspace_id == workspace)
            ).one()
            session.delete(record)
            session.commit()
    finally:
        database.dispose()
    assert _stages(context, workspace)[S.MARKET_RESEARCH]["generated"] is False

    regenerated = _post(context, workspace, "market-research", {})
    assert regenerated.status_code in {200, 201}

    snapshots = _history(context, workspace, "market_research")
    assert [item["generation"] for item in snapshots] == [2, 1]
    assert snapshots[0]["record_id"] != original_record
    assert snapshots[1]["record_id"] == original_record
    stages = _stages(context, workspace)
    assert stages[S.MARKET_RESEARCH]["generation"] == 2
    # The approval was pinned to the deleted record, so the new one is a draft,
    # and what was built from the old research is stale.
    assert stages[S.MARKET_RESEARCH]["approval"] == "draft"
    assert _stale_stages(context, workspace) == ["customer_personas", "marketing_strategy"]


# --- a workspace from before history ------------------------------------------------------


def _alembic(url: str, *args: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "-c", str(ALEMBIC_INI), *args],
        cwd=REPO_ROOT,
        env={**os.environ, "GROWTHCREW_DATABASE_URL": url},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _seed_legacy_chain(session: Session) -> UUID:
    """Insert a workspace with all seven stages, as they existed before history."""

    workspace = Workspace(name="Legacy Chain Co")
    session.add(workspace)
    session.flush()
    profile = BusinessProfile(
        workspace_id=workspace.id, business_name="Legacy Chain Co", industry="Healthy Snacks"
    )
    session.add(profile)
    session.flush()
    common: dict[str, Any] = {"model_used": "legacy-model", "input_tokens": 1, "output_tokens": 2}
    session.add_all(
        [
            BusinessUnderstanding(
                profile_id=profile.id,
                summary="Legacy summary.",
                inferred_business_stage="seed",
                competitive_category="snacks",
                key_differentiators=["fast"],
                likely_customer_pain_points=["price"],
                confidence_notes=None,
                **common,
            ),
            MarketResearch(
                workspace_id=workspace.id,
                market_overview="Legacy overview.",
                target_customer_segments="Segments.",
                competitive_landscape="Landscape.",
                opportunities_and_risks="Risks.",
                sources=[],
                **common,
            ),
            CompetitorAnalysis(
                workspace_id=workspace.id,
                competitor_overview="Competitors.",
                strengths_and_weaknesses="S and W.",
                pricing_and_positioning="Pricing.",
                differentiation_opportunities="Gaps.",
                sources=[],
                **common,
            ),
            CustomerPersonaSet(
                workspace_id=workspace.id, overview="Personas.", personas=[], **common
            ),
            BrandStrategy(
                workspace_id=workspace.id,
                overview="Brand.",
                positioning_statement="Position.",
                value_proposition="Value.",
                brand_voice_and_tone="Warm.",
                brand_pillars=[],
                tagline_options=[],
                **common,
            ),
            MarketingStrategy(
                workspace_id=workspace.id,
                recommended_channels_and_tactics="Channels.",
                content_and_messaging_pillars="Pillars.",
                ninety_day_roadmap="Roadmap.",
                budget_allocation_and_kpis="Budget.",
                sources=[],
                **common,
            ),
            ContentPlan(workspace_id=workspace.id, overview="Plan.", entries=[], **common),
        ]
    )
    session.commit()
    return workspace.id


@pytest.fixture
def legacy_context() -> Iterator[tuple[Context, UUID]]:
    """An app on a throwaway database that was migrated *through* the history migration.

    The database is built at the revision before history, seeded with a
    full chain the old way, then upgraded - so every stage is backfilled
    exactly as it would be for a real workspace - and only then is the
    application pointed at it.
    """

    base_url = _settings().database_url_value
    assert base_url is not None
    name = f"gc_chain_{uuid4().hex[:12]}"
    admin = create_engine(base_url, isolation_level="AUTOCOMMIT")
    with admin.connect() as connection:
        connection.execute(text(f'CREATE DATABASE "{name}"'))
    url = make_url(base_url).set(database=name).render_as_string(hide_password=False)

    try:
        _alembic(url, "upgrade", REVISION_BEFORE_HISTORY)
        seed_engine = create_engine(url)
        try:
            with Session(seed_engine) as session:
                workspace_id = _seed_legacy_chain(session)
        finally:
            seed_engine.dispose()
        _alembic(url, "upgrade", "head")

        settings = _settings(database_url=url)
        with TestClient(create_application(settings)) as client:
            yield Context(client=client, settings=settings), workspace_id
    finally:
        with admin.connect() as connection:
            connection.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        admin.dispose()


def test_a_workspace_from_before_history_keeps_working_and_gains_tracked_history(
    legacy_context: tuple[Context, UUID],
) -> None:
    """Backfilled stages are known but unknowable, never stale, and regenerate into tracked ones."""

    context, workspace = legacy_context

    # The migration gave every existing stage a generation 1 with unknown provenance.
    stages = _stages(context, workspace)
    for stage, item in stages.items():
        assert item["generated"] is True, stage
        assert item["generation"] == 1, stage
        assert item["staleness"] == {"provenance": "unknown", "stale_inputs": [], "stale": False}
    assert _stale_stages(context, workspace) == []
    legacy = _detail(context, workspace, "market_research", 1)
    assert legacy["provenance"] == "unknown"
    assert legacy["inputs"] is None
    assert legacy["content"]["market_overview"] == "Legacy overview."

    # Regenerating personas needs its prerequisites approved, as ever.
    for stage in ("business_understanding", "market_research"):
        approved = _post(context, workspace, f"approvals/{stage}", {"version": 1})
        assert approved.status_code == 200, stage
    forced = _post(context, workspace, "personas", {"force_regenerate": True})
    assert forced.status_code in {200, 201}

    # The new generation is tracked, and names the legacy generations it was built from.
    personas = _detail(context, workspace, "customer_personas", 2)
    assert personas["provenance"] == "tracked"
    assert personas["inputs"]["profile"]["version"] == 1
    for upstream in ("business_understanding", "market_research"):
        reference = personas["inputs"]["stages"][upstream]
        original = _detail(context, workspace, upstream, 1)
        assert reference == {
            "generation": 1,
            "record_id": original["record_id"],
            "record_version": original["record_version"],
        }
    stages = _stages(context, workspace)
    assert stages[S.CUSTOMER_PERSONAS]["generation"] == 2
    assert stages[S.CUSTOMER_PERSONAS]["staleness"]["provenance"] == "tracked"
    # Legacy stages built from older personas can't be judged, so are never flagged.
    assert stages[S.BRAND_STRATEGY]["staleness"]["provenance"] == "unknown"
    assert _stale_stages(context, workspace) == []

    # Once an input of the *tracked* stage moves, it is flagged - and only it.
    regenerated = _post(
        context, workspace, "business-profile/understanding", {"force_regenerate": True}
    )
    assert regenerated.status_code in {200, 201}
    assert _stale_stages(context, workspace) == ["customer_personas"]
    [stale_input] = _stages(context, workspace)[S.CUSTOMER_PERSONAS]["staleness"]["stale_inputs"]
    assert stale_input == {"prerequisite": "business_understanding", "built_from": 1, "current": 2}
