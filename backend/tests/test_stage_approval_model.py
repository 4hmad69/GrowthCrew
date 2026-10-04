"""Unit tests for the StageApproval model's static definition."""

from sqlalchemy import CheckConstraint

from backend.app.db import models
from backend.app.db.base import Base
from backend.app.db.models.stage_approval import (
    STAGE_APPROVAL_SOURCES,
    STAGE_APPROVAL_STAGES,
    StageApproval,
)
from backend.app.schemas.strategy_orchestration import StrategyStage


def _check_constraint_sql(name: str) -> str:
    """Return the SQL text of one named CHECK constraint on the table."""

    for constraint in StageApproval.__table__.constraints:
        if isinstance(constraint, CheckConstraint) and constraint.name == name:
            return str(constraint.sqltext)
    raise AssertionError(f"Constraint {name} not found.")


def test_allowed_stages_match_strategy_stage_enum_exactly() -> None:
    """The db layer's literal stage list must never drift from StrategyStage."""

    assert STAGE_APPROVAL_STAGES == tuple(stage.value for stage in StrategyStage)


def test_stage_check_constraint_lists_every_stage_in_order() -> None:
    """The CHECK constraint is built from the same literals the tuple exposes."""

    sql = _check_constraint_sql("ck_stage_approvals_stage")

    for stage in STAGE_APPROVAL_STAGES:
        assert f"'{stage}'" in sql
    positions = [sql.index(f"'{stage}'") for stage in STAGE_APPROVAL_STAGES]
    assert positions == sorted(positions)


def test_allowed_sources_are_human_and_auto() -> None:
    """Only a person or an explicitly auto-approving run can record approval."""

    assert STAGE_APPROVAL_SOURCES == ("human", "auto")
    sql = _check_constraint_sql("ck_stage_approvals_source")
    assert "'human'" in sql
    assert "'auto'" in sql


def test_model_is_registered_for_alembic_autogenerate() -> None:
    """A model missing from the registry is silently skipped by autogenerate."""

    assert models.StageApproval is StageApproval
    assert "StageApproval" in models.__all__
    assert "stage_approvals" in Base.metadata.tables


def test_one_approval_per_workspace_and_stage() -> None:
    """Approving again must replace, not add - enforced by a unique constraint."""

    unique_columns = {
        tuple(column.name for column in constraint.columns)
        for constraint in StageApproval.__table__.constraints
        if constraint.__class__.__name__ == "UniqueConstraint"
    }

    assert ("workspace_id", "stage") in unique_columns


def test_record_id_is_not_a_foreign_key() -> None:
    """record_id points into seven different tables, so it cannot be a foreign key."""

    column = StageApproval.__table__.c.record_id

    assert not column.foreign_keys
