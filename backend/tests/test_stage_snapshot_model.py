"""Unit tests for the StageSnapshot model's static definition."""

from sqlalchemy import CheckConstraint

from backend.app.db import models
from backend.app.db.base import Base
from backend.app.db.models.stage_snapshot import STAGE_SNAPSHOT_STAGES, StageSnapshot
from backend.app.db.repositories.stage_snapshots import StageSnapshotRepository
from backend.app.schemas.strategy_orchestration import StrategyStage


def _check_constraint_sql(name: str) -> str:
    """Return the SQL text of one named CHECK constraint on the table."""

    for constraint in StageSnapshot.__table__.constraints:
        if isinstance(constraint, CheckConstraint) and constraint.name == name:
            return str(constraint.sqltext)
    raise AssertionError(f"Constraint {name} not found.")


def test_allowed_stages_match_strategy_stage_enum_exactly() -> None:
    """The db layer's literal stage list must never drift from StrategyStage."""

    assert STAGE_SNAPSHOT_STAGES == tuple(stage.value for stage in StrategyStage)


def test_stage_check_constraint_lists_every_stage_in_order() -> None:
    """The CHECK constraint is built from the same literals the tuple exposes."""

    sql = _check_constraint_sql("ck_stage_snapshots_stage")

    for stage in STAGE_SNAPSHOT_STAGES:
        assert f"'{stage}'" in sql
    positions = [sql.index(f"'{stage}'") for stage in STAGE_SNAPSHOT_STAGES]
    assert positions == sorted(positions)


def test_model_is_registered_for_alembic_autogenerate() -> None:
    """A model missing from the registry is silently skipped by autogenerate."""

    assert models.StageSnapshot is StageSnapshot
    assert "StageSnapshot" in models.__all__
    assert "stage_snapshots" in Base.metadata.tables


def test_one_snapshot_per_workspace_stage_and_generation() -> None:
    """A generation number can be used once per stage - forks fail loudly."""

    unique_columns = {
        tuple(column.name for column in constraint.columns)
        for constraint in StageSnapshot.__table__.constraints
        if constraint.__class__.__name__ == "UniqueConstraint"
    }

    assert ("workspace_id", "stage", "generation") in unique_columns


def test_record_id_is_not_a_foreign_key() -> None:
    """record_id points into seven different tables, so it cannot be a foreign key."""

    assert not StageSnapshot.__table__.c.record_id.foreign_keys


def test_inputs_are_nullable_but_content_is_not() -> None:
    """NULL inputs mean 'provenance unknown'; content is always present."""

    table = StageSnapshot.__table__

    assert table.c.inputs.nullable is True
    assert table.c.content.nullable is False


def test_repository_exposes_no_way_to_change_or_remove_history() -> None:
    """History is append-only: the repository can add and read, nothing else."""

    public = {name for name in vars(StageSnapshotRepository) if not name.startswith("_")}

    assert public == {
        "add",
        "get",
        "get_latest",
        "next_generation",
        "list_by_stage",
        "list_latest_by_workspace",
    }
