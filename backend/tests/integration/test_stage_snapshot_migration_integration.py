"""Real PostgreSQL tests for migration 20261010_0012 (stage snapshots).

Each test builds a throwaway database, migrates it to the revision before
this one, seeds stage records the way they existed before history did,
then runs the migration - so the backfill is proven against real legacy
rows, and the shared development database is never downgraded or touched.
"""

import os
import subprocess
import sys
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import Engine, create_engine, inspect, make_url, select, text
from sqlalchemy.orm import Session

from backend.app.config import Settings
from backend.app.db.base import Base
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

RUN_INTEGRATION_TESTS = os.getenv("GROWTHCREW_RUN_INTEGRATION_TESTS") == "1"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not RUN_INTEGRATION_TESTS,
        reason=("Set GROWTHCREW_RUN_INTEGRATION_TESTS=1 to run PostgreSQL integration tests."),
    ),
]

REPO_ROOT = Path(__file__).resolve().parents[3]
ALEMBIC_INI = REPO_ROOT / "backend" / "alembic.ini"
PREVIOUS_REVISION = "20261003_0011"

# Independent restatement of what each stage's snapshot content must hold.
CONTENT_FIELDS: dict[str, tuple[str, ...]] = {
    "business_understanding": (
        "summary",
        "inferred_business_stage",
        "competitive_category",
        "key_differentiators",
        "likely_customer_pain_points",
        "confidence_notes",
    ),
    "market_research": (
        "market_overview",
        "target_customer_segments",
        "competitive_landscape",
        "opportunities_and_risks",
        "sources",
    ),
    "competitor_analysis": (
        "competitor_overview",
        "strengths_and_weaknesses",
        "pricing_and_positioning",
        "differentiation_opportunities",
        "sources",
    ),
    "customer_personas": ("overview", "personas"),
    "brand_strategy": (
        "overview",
        "positioning_statement",
        "value_proposition",
        "brand_voice_and_tone",
        "brand_pillars",
        "tagline_options",
    ),
    "marketing_strategy": (
        "recommended_channels_and_tactics",
        "content_and_messaging_pillars",
        "ninety_day_roadmap",
        "budget_allocation_and_kpis",
        "sources",
    ),
    "content_plan": ("overview", "entries"),
}

TABLES: dict[str, str] = {
    "business_understanding": "business_understandings",
    "market_research": "market_researches",
    "competitor_analysis": "competitor_analyses",
    "customer_personas": "customer_persona_sets",
    "brand_strategy": "brand_strategies",
    "marketing_strategy": "marketing_strategies",
    "content_plan": "content_plans",
}


class Scratch:
    """A throwaway database that tests migrate and seed freely."""

    def __init__(self, engine: Engine, url: str) -> None:
        self.engine = engine
        self.url = url

    def alembic(self, *args: str) -> None:
        """Run an Alembic command against this database only."""

        result = subprocess.run(
            [sys.executable, "-m", "alembic", "-c", str(ALEMBIC_INI), *args],
            cwd=REPO_ROOT,
            env={**os.environ, "GROWTHCREW_DATABASE_URL": self.url},
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture
def scratch() -> Iterator[Scratch]:
    """Create an empty database, and drop it afterwards."""

    settings = Settings(environment="test")
    base_url = settings.database_url_value
    if base_url is None:
        pytest.fail("GROWTHCREW_DATABASE_URL is required for integration tests.")

    name = f"gc_migration_{uuid4().hex[:12]}"
    admin = create_engine(base_url, isolation_level="AUTOCOMMIT")
    with admin.connect() as connection:
        connection.execute(text(f'CREATE DATABASE "{name}"'))

    url = make_url(base_url).set(database=name).render_as_string(hide_password=False)
    engine = create_engine(url)
    try:
        yield Scratch(engine, url)
    finally:
        engine.dispose()
        with admin.connect() as connection:
            connection.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        admin.dispose()


def _seed_full_chain(session: Session, name: str) -> dict[str, Any]:
    """Insert a workspace, a profile, and all seven stage records; return the records."""

    workspace = Workspace(name=name)
    session.add(workspace)
    session.flush()
    profile = BusinessProfile(workspace_id=workspace.id, business_name=name)
    session.add(profile)
    session.flush()

    records: dict[str, Any] = {
        "business_understanding": BusinessUnderstanding(
            profile_id=profile.id,
            summary="Summary.",
            inferred_business_stage="seed",
            competitive_category="snacks",
            key_differentiators=["fast", "local"],
            likely_customer_pain_points=["price"],
            confidence_notes=None,
            model_used="model-bu",
            input_tokens=11,
            output_tokens=12,
        ),
        "market_research": MarketResearch(
            workspace_id=workspace.id,
            market_overview="Overview.",
            target_customer_segments="Segments.",
            competitive_landscape="Landscape.",
            opportunities_and_risks="Risks.",
            sources=[{"url": "https://example.com", "title": "Café"}],
            model_used="model-mr",
            input_tokens=21,
            output_tokens=22,
        ),
        "competitor_analysis": CompetitorAnalysis(
            workspace_id=workspace.id,
            competitor_overview="Competitors.",
            strengths_and_weaknesses="S and W.",
            pricing_and_positioning="Pricing.",
            differentiation_opportunities="Gaps.",
            sources=[],
            model_used="model-ca",
            input_tokens=31,
            output_tokens=32,
        ),
        "customer_personas": CustomerPersonaSet(
            workspace_id=workspace.id,
            overview="Personas overview.",
            personas=[{"name": "Sam"}],
            model_used="model-cp",
            input_tokens=41,
            output_tokens=42,
        ),
        "brand_strategy": BrandStrategy(
            workspace_id=workspace.id,
            overview="Brand overview.",
            positioning_statement="Position.",
            value_proposition="Value.",
            brand_voice_and_tone="Warm.",
            brand_pillars=["one", "two"],
            tagline_options=["Eat well"],
            model_used="model-bs",
            input_tokens=51,
            output_tokens=52,
        ),
        "marketing_strategy": MarketingStrategy(
            workspace_id=workspace.id,
            recommended_channels_and_tactics="Channels.",
            content_and_messaging_pillars="Pillars.",
            ninety_day_roadmap="Roadmap.",
            budget_allocation_and_kpis="Budget.",
            sources=[{"url": "internal:notes"}],
            model_used="model-ms",
            input_tokens=61,
            output_tokens=62,
        ),
        "content_plan": ContentPlan(
            workspace_id=workspace.id,
            overview="Plan overview.",
            entries=[{"day": 1, "topic": "Launch"}],
            model_used="model-cpl",
            input_tokens=71,
            output_tokens=72,
        ),
    }
    session.add_all(records.values())
    session.commit()

    # Give each record a distinct version and last-written time, the way
    # a stage that was regenerated a few times would look.
    for offset, (stage, record) in enumerate(records.items(), start=2):
        session.execute(
            text(f"UPDATE {TABLES[stage]} SET version = :v, updated_at = :ts WHERE id = :id"),
            {
                "v": offset,
                "ts": datetime(2026, 9, offset, 12, 0, tzinfo=UTC),
                "id": record.id,
            },
        )
    session.commit()
    for record in records.values():
        session.refresh(record)
    records["workspace_id"] = workspace.id
    return records


def test_upgrade_backfills_generation_one_for_every_existing_stage(scratch: Scratch) -> None:
    """Pre-history stages each get one snapshot copying their content and identity."""

    scratch.alembic("upgrade", PREVIOUS_REVISION)
    with Session(scratch.engine) as session:
        seeded = _seed_full_chain(session, "legacy-full")
    workspace_id = seeded["workspace_id"]

    scratch.alembic("upgrade", "head")

    with Session(scratch.engine) as session:
        snapshots = {
            row.stage: row
            for row in session.scalars(
                select(StageSnapshot).where(StageSnapshot.workspace_id == workspace_id)
            )
        }
        assert set(snapshots) == set(CONTENT_FIELDS)

        for stage, fields in CONTENT_FIELDS.items():
            record = seeded[stage]
            snapshot = snapshots[stage]

            assert snapshot.generation == 1
            assert snapshot.record_id == record.id
            assert snapshot.record_version == record.version
            assert snapshot.model_used == record.model_used
            assert snapshot.input_tokens == record.input_tokens
            assert snapshot.output_tokens == record.output_tokens
            assert snapshot.created_at == record.updated_at
            assert set(snapshot.content) == set(fields)
            assert snapshot.content == {field: getattr(record, field) for field in fields}

        # Nothing recorded what these stages were built from, so provenance
        # is unknown - SQL NULL, not an empty object that would read as "none".
        unknown_inputs = session.scalar(
            text("SELECT count(*) FROM stage_snapshots WHERE inputs IS NULL")
        )
        assert unknown_inputs == 7


def test_backfill_covers_only_stages_that_exist_and_keeps_workspaces_apart(
    scratch: Scratch,
) -> None:
    """A workspace gets snapshots for exactly the stages it has, and none for others."""

    scratch.alembic("upgrade", PREVIOUS_REVISION)
    with Session(scratch.engine) as session:
        full = _seed_full_chain(session, "legacy-full")["workspace_id"]

        partial_workspace = Workspace(name="legacy-partial")
        empty_workspace = Workspace(name="legacy-empty")
        session.add_all([partial_workspace, empty_workspace])
        session.flush()
        session.add(
            MarketResearch(
                workspace_id=partial_workspace.id,
                market_overview="Only this.",
                target_customer_segments="x",
                competitive_landscape="x",
                opportunities_and_risks="x",
                model_used="model-mr",
            )
        )
        session.commit()
        partial, empty = partial_workspace.id, empty_workspace.id

    scratch.alembic("upgrade", "head")

    with Session(scratch.engine) as session:

        def stages_of(workspace_id: Any) -> set[str]:
            return set(
                session.scalars(
                    select(StageSnapshot.stage).where(StageSnapshot.workspace_id == workspace_id)
                )
            )

        assert stages_of(full) == set(CONTENT_FIELDS)
        assert stages_of(partial) == {"market_research"}
        assert stages_of(empty) == set()


def test_stage_records_are_untouched_by_upgrade_and_downgrade(scratch: Scratch) -> None:
    """The migration only adds history; it never edits or removes a stage record."""

    scratch.alembic("upgrade", PREVIOUS_REVISION)
    with Session(scratch.engine) as session:
        _seed_full_chain(session, "legacy-untouched")

    def snapshot_of_records() -> dict[str, tuple[Any, ...]]:
        with Session(scratch.engine) as session:
            return {
                stage: tuple(
                    session.execute(text(f"SELECT id, version, updated_at FROM {table}")).one()
                )
                for stage, table in TABLES.items()
            }

    before = snapshot_of_records()
    scratch.alembic("upgrade", "head")
    assert snapshot_of_records() == before
    scratch.alembic("downgrade", PREVIOUS_REVISION)
    assert snapshot_of_records() == before


def test_downgrade_removes_the_table_trigger_and_function_and_reupgrade_works(
    scratch: Scratch,
) -> None:
    """Downgrade leaves nothing behind, and the migration can be applied again."""

    scratch.alembic("upgrade", PREVIOUS_REVISION)
    with Session(scratch.engine) as session:
        _seed_full_chain(session, "legacy-roundtrip")

    scratch.alembic("upgrade", "head")
    scratch.alembic("downgrade", PREVIOUS_REVISION)

    assert "stage_snapshots" not in inspect(scratch.engine).get_table_names()
    with Session(scratch.engine) as session:
        leftover = session.scalar(
            text("SELECT count(*) FROM pg_proc WHERE proname = 'stage_snapshots_reject_update'")
        )
    assert leftover == 0

    scratch.alembic("upgrade", "head")
    with Session(scratch.engine) as session:
        assert session.scalar(text("SELECT count(*) FROM stage_snapshots")) == 7


def test_migrated_table_matches_the_model(scratch: Scratch) -> None:
    """Autogenerate finds no difference between the migration and the ORM model.

    Scoped to stage_snapshots on purpose: the repository has an older,
    unrelated difference on knowledge_chunks that this check must not
    be blamed for.
    """

    scratch.alembic("upgrade", "head")

    def only_stage_snapshots(
        obj: Any, name: str | None, type_: str, reflected: bool, compare_to: Any
    ) -> bool:
        if type_ == "table":
            return name == "stage_snapshots"
        table = getattr(obj, "table", None)
        return table is not None and table.name == "stage_snapshots"

    with scratch.engine.connect() as connection:
        context = MigrationContext.configure(
            connection,
            opts={
                "compare_type": True,
                "compare_server_default": True,
                "include_object": only_stage_snapshots,
            },
        )
        differences = compare_metadata(context, Base.metadata)

    assert differences == []