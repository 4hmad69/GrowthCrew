"""Real PostgreSQL integration tests for the stage approval model and repository.

Proves what only a real database can: the CHECK and UNIQUE constraints the
migration actually created, the workspace cascade delete, and the
repository's queries. The approval service's own rules (version pinning,
staleness, gating) are covered by their own tests; this file is about the
persistence layer underneath them.
"""

import os
from collections.abc import Iterator
from uuid import UUID, uuid4

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from backend.app.config import Settings
from backend.app.db.database import Database
from backend.app.db.models.stage_approval import STAGE_APPROVAL_STAGES, StageApproval
from backend.app.db.models.workspace import Workspace
from backend.app.db.repositories.stage_approvals import StageApprovalRepository

RUN_INTEGRATION_TESTS = os.getenv("GROWTHCREW_RUN_INTEGRATION_TESTS") == "1"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not RUN_INTEGRATION_TESTS,
        reason=("Set GROWTHCREW_RUN_INTEGRATION_TESTS=1 to run PostgreSQL integration tests."),
    ),
]


@pytest.fixture
def database() -> Iterator[Database]:
    """Provide a real database handle and dispose it afterwards."""

    settings = Settings(environment="test")

    if settings.database_url is None:
        pytest.fail("GROWTHCREW_DATABASE_URL is required for integration tests.")

    db = Database.from_settings(settings)
    try:
        yield db
    finally:
        db.dispose()


@pytest.fixture
def workspace_ids(database: Database) -> Iterator[list[UUID]]:
    """Track created workspaces and remove only those after the test."""

    created: list[UUID] = []
    yield created

    with database.session() as session:
        for workspace_id in created:
            workspace = session.get(Workspace, workspace_id)
            if workspace is not None:
                session.delete(workspace)
        session.commit()


def _new_workspace(database: Database, workspace_ids: list[UUID], name: str) -> UUID:
    """Insert a bare workspace and register it for cleanup."""

    with database.session() as session:
        workspace = Workspace(name=name)
        session.add(workspace)
        session.commit()
        workspace_id = workspace.id

    workspace_ids.append(workspace_id)
    return workspace_id


def _count(database: Database, workspace_id: UUID) -> int:
    """Count approval rows for a workspace through a fresh session."""

    with database.session() as session:
        return (
            session.scalar(
                select(func.count())
                .select_from(StageApproval)
                .where(StageApproval.workspace_id == workspace_id)
            )
            or 0
        )


def test_round_trip_applies_defaults(database: Database, workspace_ids: list[UUID]) -> None:
    """An approval saves and reloads, defaulting to a human source with a timestamp."""

    workspace_id = _new_workspace(database, workspace_ids, "approval-roundtrip")
    record_id = uuid4()

    with database.session() as session:
        repository = StageApprovalRepository(session)
        repository.add(
            StageApproval(
                workspace_id=workspace_id,
                stage="market_research",
                record_id=record_id,
                approved_version=3,
            )
        )
        session.commit()

    with database.session() as session:
        loaded = StageApprovalRepository(session).get_by_workspace_and_stage(
            workspace_id, "market_research"
        )

        assert loaded is not None
        assert loaded.record_id == record_id
        assert loaded.approved_version == 3
        assert loaded.source == "human"
        assert loaded.approved_at is not None


def test_get_returns_none_when_not_approved(database: Database, workspace_ids: list[UUID]) -> None:
    """No row means no approval - callers rely on None, not an exception."""

    workspace_id = _new_workspace(database, workspace_ids, "approval-none")

    with database.session() as session:
        assert (
            StageApprovalRepository(session).get_by_workspace_and_stage(
                workspace_id, "brand_strategy"
            )
            is None
        )


def test_list_is_scoped_to_one_workspace(database: Database, workspace_ids: list[UUID]) -> None:
    """Listing must never leak another workspace's approvals."""

    first = _new_workspace(database, workspace_ids, "approval-list-a")
    second = _new_workspace(database, workspace_ids, "approval-list-b")

    with database.session() as session:
        repository = StageApprovalRepository(session)
        repository.add(
            StageApproval(
                workspace_id=first,
                stage="business_understanding",
                record_id=uuid4(),
                approved_version=1,
            )
        )
        repository.add(
            StageApproval(
                workspace_id=first,
                stage="market_research",
                record_id=uuid4(),
                approved_version=1,
            )
        )
        repository.add(
            StageApproval(
                workspace_id=second,
                stage="content_plan",
                record_id=uuid4(),
                approved_version=1,
            )
        )
        session.commit()

    with database.session() as session:
        repository = StageApprovalRepository(session)

        assert {item.stage for item in repository.list_by_workspace(first)} == {
            "business_understanding",
            "market_research",
        }
        assert [item.stage for item in repository.list_by_workspace(second)] == ["content_plan"]


@pytest.mark.parametrize("stage", STAGE_APPROVAL_STAGES)
def test_every_chain_stage_is_accepted(
    database: Database, workspace_ids: list[UUID], stage: str
) -> None:
    """The migration's CHECK constraint must accept each of the seven stages."""

    workspace_id = _new_workspace(database, workspace_ids, f"approval-stage-{stage}")

    with database.session() as session:
        StageApprovalRepository(session).add(
            StageApproval(
                workspace_id=workspace_id,
                stage=stage,
                record_id=uuid4(),
                approved_version=1,
            )
        )
        session.commit()

    assert _count(database, workspace_id) == 1


def test_duplicate_workspace_and_stage_is_rejected(
    database: Database, workspace_ids: list[UUID]
) -> None:
    """At most one approval per stage - the service must replace, never add."""

    workspace_id = _new_workspace(database, workspace_ids, "approval-duplicate")

    with database.session() as session:
        StageApprovalRepository(session).add(
            StageApproval(
                workspace_id=workspace_id,
                stage="competitor_analysis",
                record_id=uuid4(),
                approved_version=1,
            )
        )
        session.commit()

    with database.session() as session:
        StageApprovalRepository(session).add(
            StageApproval(
                workspace_id=workspace_id,
                stage="competitor_analysis",
                record_id=uuid4(),
                approved_version=2,
            )
        )
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()

    assert _count(database, workspace_id) == 1


@pytest.mark.parametrize(
    ("stage", "source", "approved_version"),
    [
        ("not_a_stage", "human", 1),
        ("market_research", "robot", 1),
        ("market_research", "human", 0),
        ("market_research", "human", -2),
    ],
)
def test_check_constraints_reject_invalid_values(
    database: Database,
    workspace_ids: list[UUID],
    stage: str,
    source: str,
    approved_version: int,
) -> None:
    """Bad stage, source, or version must be refused by the database itself."""

    workspace_id = _new_workspace(database, workspace_ids, "approval-check")

    with database.session() as session:
        StageApprovalRepository(session).add(
            StageApproval(
                workspace_id=workspace_id,
                stage=stage,
                source=source,
                record_id=uuid4(),
                approved_version=approved_version,
            )
        )
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()

    assert _count(database, workspace_id) == 0


def test_unknown_workspace_is_rejected(database: Database) -> None:
    """The workspace foreign key must refuse an approval with no workspace."""

    with database.session() as session:
        StageApprovalRepository(session).add(
            StageApproval(
                workspace_id=uuid4(),
                stage="market_research",
                record_id=uuid4(),
                approved_version=1,
            )
        )
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()


def test_record_id_needs_no_matching_record(database: Database, workspace_ids: list[UUID]) -> None:
    """record_id is polymorphic, so a record that no longer exists must not block a row."""

    workspace_id = _new_workspace(database, workspace_ids, "approval-orphan")

    with database.session() as session:
        StageApprovalRepository(session).add(
            StageApproval(
                workspace_id=workspace_id,
                stage="customer_personas",
                record_id=uuid4(),
                approved_version=1,
            )
        )
        session.commit()

    assert _count(database, workspace_id) == 1


def test_deleting_workspace_cascades_to_approvals(
    database: Database, workspace_ids: list[UUID]
) -> None:
    """Approvals must disappear with their workspace, leaving no orphan rows."""

    workspace_id = _new_workspace(database, workspace_ids, "approval-cascade")

    with database.session() as session:
        repository = StageApprovalRepository(session)
        for stage in ("business_understanding", "market_research", "content_plan"):
            repository.add(
                StageApproval(
                    workspace_id=workspace_id,
                    stage=stage,
                    record_id=uuid4(),
                    approved_version=1,
                )
            )
        session.commit()

    assert _count(database, workspace_id) == 3

    with database.session() as session:
        workspace = session.get(Workspace, workspace_id)
        assert workspace is not None
        session.delete(workspace)
        session.commit()

    assert _count(database, workspace_id) == 0


def test_delete_removes_only_the_given_approval(
    database: Database, workspace_ids: list[UUID]
) -> None:
    """Revoking one stage must leave the others untouched."""

    workspace_id = _new_workspace(database, workspace_ids, "approval-delete")

    with database.session() as session:
        repository = StageApprovalRepository(session)
        for stage in ("market_research", "brand_strategy"):
            repository.add(
                StageApproval(
                    workspace_id=workspace_id,
                    stage=stage,
                    record_id=uuid4(),
                    approved_version=1,
                )
            )
        session.commit()

    with database.session() as session:
        repository = StageApprovalRepository(session)
        target = repository.get_by_workspace_and_stage(workspace_id, "market_research")
        assert target is not None
        repository.delete(target)
        session.commit()

    with database.session() as session:
        remaining = StageApprovalRepository(session).list_by_workspace(workspace_id)

        assert [item.stage for item in remaining] == ["brand_strategy"]


def test_replacing_an_approval_refreshes_its_timestamp(
    database: Database, workspace_ids: list[UUID]
) -> None:
    """Re-approving (updating the row) must move approved_at forward."""

    workspace_id = _new_workspace(database, workspace_ids, "approval-timestamp")

    with database.session() as session:
        StageApprovalRepository(session).add(
            StageApproval(
                workspace_id=workspace_id,
                stage="market_research",
                record_id=uuid4(),
                approved_version=1,
            )
        )
        session.commit()

    with database.session() as session:
        first = StageApprovalRepository(session).get_by_workspace_and_stage(
            workspace_id, "market_research"
        )
        assert first is not None
        first_at = first.approved_at

        first.approved_version = 2
        first.source = "auto"
        session.commit()

    with database.session() as session:
        second = StageApprovalRepository(session).get_by_workspace_and_stage(
            workspace_id, "market_research"
        )
        assert second is not None

        assert second.approved_version == 2
        assert second.source == "auto"
        assert second.approved_at > first_at
