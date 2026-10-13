"""Real PostgreSQL integration tests for the stage snapshot model and repository.

Proves what only a real database can: the CHECK and UNIQUE constraints the
migration actually created, the append-only trigger, the workspace cascade
delete, and the repository's queries. How snapshots get recorded, and what
they mean for staleness, is covered by the history service's own tests.
"""

import os
from collections.abc import Iterator
from typing import Any
from uuid import UUID, uuid4

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError

from backend.app.config import Settings
from backend.app.db.database import Database
from backend.app.db.models.stage_snapshot import STAGE_SNAPSHOT_STAGES, StageSnapshot
from backend.app.db.models.workspace import Workspace
from backend.app.db.repositories.stage_snapshots import StageSnapshotRepository

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


def _snapshot(workspace_id: UUID, **overrides: Any) -> StageSnapshot:
    """Build a valid snapshot, with any field overridable."""

    values: dict[str, Any] = {
        "workspace_id": workspace_id,
        "stage": "market_research",
        "generation": 1,
        "record_id": uuid4(),
        "record_version": 1,
        "content": {"market_overview": "An overview."},
        "inputs": None,
        "model_used": "test-model",
    }
    values.update(overrides)
    return StageSnapshot(**values)


def _save(database: Database, *snapshots: StageSnapshot) -> None:
    """Persist snapshots through the repository in one transaction."""

    with database.session() as session:
        repository = StageSnapshotRepository(session)
        for snapshot in snapshots:
            repository.add(snapshot)
        session.commit()


def _count(database: Database, workspace_id: UUID) -> int:
    """Count snapshot rows for a workspace through a fresh session."""

    with database.session() as session:
        return (
            session.scalar(
                select(func.count())
                .select_from(StageSnapshot)
                .where(StageSnapshot.workspace_id == workspace_id)
            )
            or 0
        )


def test_round_trip_applies_defaults(database: Database, workspace_ids: list[UUID]) -> None:
    """A snapshot saves and reloads with token defaults, no inputs, and a timestamp."""

    workspace_id = _new_workspace(database, workspace_ids, "snapshot-roundtrip")
    record_id = uuid4()
    _save(database, _snapshot(workspace_id, record_id=record_id, record_version=4))

    with database.session() as session:
        loaded = StageSnapshotRepository(session).get(workspace_id, "market_research", 1)

        assert loaded is not None
        assert loaded.record_id == record_id
        assert loaded.record_version == 4
        assert loaded.model_used == "test-model"
        assert loaded.input_tokens == 0
        assert loaded.output_tokens == 0
        assert loaded.inputs is None
        assert loaded.created_at is not None


def test_content_and_inputs_round_trip_nested_json(
    database: Database, workspace_ids: list[UUID]
) -> None:
    """Nested objects, lists, and non-ASCII text survive a save and reload unchanged."""

    workspace_id = _new_workspace(database, workspace_ids, "snapshot-json")
    content = {
        "sources": [{"url": "https://example.com/a", "title": "Café guide"}],
        "key_differentiators": ["fast", "local"],
        "confidence_notes": None,
    }
    inputs = {
        "profile": {"id": str(uuid4()), "version": 2},
        "stages": {"business_understanding": {"generation": 3, "record_version": 2}},
    }
    _save(
        database,
        _snapshot(workspace_id, stage="customer_personas", content=content, inputs=inputs),
    )

    with database.session() as session:
        loaded = StageSnapshotRepository(session).get(workspace_id, "customer_personas", 1)

        assert loaded is not None
        assert loaded.content == content
        assert loaded.inputs == inputs


def test_unknown_inputs_are_stored_as_sql_null(
    database: Database, workspace_ids: list[UUID]
) -> None:
    """None means 'provenance unknown' and must be SQL NULL, not the JSON value null."""

    workspace_id = _new_workspace(database, workspace_ids, "snapshot-null-inputs")
    _save(database, _snapshot(workspace_id, inputs=None))

    with database.session() as session:
        stored_as_sql_null = session.scalar(
            text("SELECT inputs IS NULL FROM stage_snapshots WHERE workspace_id = :id"),
            {"id": workspace_id},
        )

    assert stored_as_sql_null is True


def test_get_returns_none_for_a_generation_that_does_not_exist(
    database: Database, workspace_ids: list[UUID]
) -> None:
    """A missing generation is None, and a different generation does not match."""

    workspace_id = _new_workspace(database, workspace_ids, "snapshot-get-missing")
    _save(database, _snapshot(workspace_id, generation=1))

    with database.session() as session:
        repository = StageSnapshotRepository(session)

        assert repository.get(workspace_id, "market_research", 2) is None
        assert repository.get(workspace_id, "brand_strategy", 1) is None


def test_get_latest_returns_the_newest_generation(
    database: Database, workspace_ids: list[UUID]
) -> None:
    """The latest snapshot is the highest generation, whatever order rows were added in."""

    workspace_id = _new_workspace(database, workspace_ids, "snapshot-latest")
    _save(
        database,
        _snapshot(workspace_id, generation=2),
        _snapshot(workspace_id, generation=3),
        _snapshot(workspace_id, generation=1),
    )

    with database.session() as session:
        repository = StageSnapshotRepository(session)
        latest = repository.get_latest(workspace_id, "market_research")

        assert latest is not None
        assert latest.generation == 3
        assert repository.get_latest(workspace_id, "content_plan") is None


def test_next_generation_starts_at_one_and_follows_the_highest(
    database: Database, workspace_ids: list[UUID]
) -> None:
    """The next number is one past the highest used, independently per stage."""

    workspace_id = _new_workspace(database, workspace_ids, "snapshot-next")
    other_id = _new_workspace(database, workspace_ids, "snapshot-next-other")
    _save(
        database,
        _snapshot(workspace_id, generation=1),
        _snapshot(workspace_id, generation=4),
        _snapshot(workspace_id, stage="brand_strategy", generation=2),
        _snapshot(other_id, generation=9),
    )

    with database.session() as session:
        repository = StageSnapshotRepository(session)

        assert repository.next_generation(workspace_id, "market_research") == 5
        assert repository.next_generation(workspace_id, "brand_strategy") == 3
        assert repository.next_generation(workspace_id, "content_plan") == 1


def test_list_by_stage_is_newest_first_and_scoped(
    database: Database, workspace_ids: list[UUID]
) -> None:
    """Listing returns one stage of one workspace, newest generation first."""

    workspace_id = _new_workspace(database, workspace_ids, "snapshot-list")
    other_id = _new_workspace(database, workspace_ids, "snapshot-list-other")
    _save(
        database,
        _snapshot(workspace_id, generation=1),
        _snapshot(workspace_id, generation=2),
        _snapshot(workspace_id, generation=3),
        _snapshot(workspace_id, stage="brand_strategy", generation=1),
        _snapshot(other_id, generation=1),
    )

    with database.session() as session:
        repository = StageSnapshotRepository(session)

        listed = repository.list_by_stage(workspace_id, "market_research")

        assert [snapshot.generation for snapshot in listed] == [3, 2, 1]
        assert repository.list_by_stage(workspace_id, "content_plan") == []


def test_list_latest_by_workspace_returns_one_newest_snapshot_per_stage(
    database: Database, workspace_ids: list[UUID]
) -> None:
    """Each stage appears once, at its newest generation, for this workspace only."""

    workspace_id = _new_workspace(database, workspace_ids, "snapshot-latest-all")
    other_id = _new_workspace(database, workspace_ids, "snapshot-latest-all-other")
    _save(
        database,
        _snapshot(workspace_id, generation=1),
        _snapshot(workspace_id, generation=2),
        _snapshot(workspace_id, stage="brand_strategy", generation=1),
        _snapshot(workspace_id, stage="content_plan", generation=5),
        _snapshot(other_id, stage="market_research", generation=7),
    )

    with database.session() as session:
        latest = StageSnapshotRepository(session).list_latest_by_workspace(workspace_id)

        assert {snapshot.stage: snapshot.generation for snapshot in latest} == {
            "brand_strategy": 1,
            "content_plan": 5,
            "market_research": 2,
        }
        assert StageSnapshotRepository(session).list_latest_by_workspace(uuid4()) == []


@pytest.mark.parametrize("stage", STAGE_SNAPSHOT_STAGES)
def test_every_chain_stage_is_accepted(
    database: Database, workspace_ids: list[UUID], stage: str
) -> None:
    """The CHECK constraint the migration created allows each of the seven stages."""

    workspace_id = _new_workspace(database, workspace_ids, f"snapshot-{stage}")

    _save(database, _snapshot(workspace_id, stage=stage))

    assert _count(database, workspace_id) == 1


def test_duplicate_generation_is_rejected_but_other_scopes_are_not(
    database: Database, workspace_ids: list[UUID]
) -> None:
    """A generation number is unique per workspace and stage, and only there."""

    workspace_id = _new_workspace(database, workspace_ids, "snapshot-unique")
    other_id = _new_workspace(database, workspace_ids, "snapshot-unique-other")
    _save(database, _snapshot(workspace_id, generation=1))

    with pytest.raises(IntegrityError):
        _save(database, _snapshot(workspace_id, generation=1))

    _save(
        database,
        _snapshot(workspace_id, generation=2),
        _snapshot(workspace_id, stage="brand_strategy", generation=1),
        _snapshot(other_id, generation=1),
    )
    assert _count(database, workspace_id) == 3
    assert _count(database, other_id) == 1


@pytest.mark.parametrize(
    "overrides",
    [
        {"stage": "not_a_stage"},
        {"generation": 0},
        {"record_version": 0},
        {"content": []},
        {"content": "just text"},
        {"inputs": []},
        {"inputs": "just text"},
    ],
    ids=[
        "unknown-stage",
        "generation-zero",
        "record-version-zero",
        "content-array",
        "content-string",
        "inputs-array",
        "inputs-string",
    ],
)
def test_check_constraints_reject_invalid_values(
    database: Database, workspace_ids: list[UUID], overrides: dict[str, Any]
) -> None:
    """The database itself refuses malformed snapshots, not just the application."""

    workspace_id = _new_workspace(database, workspace_ids, "snapshot-checks")

    with pytest.raises(IntegrityError):
        _save(database, _snapshot(workspace_id, **overrides))

    assert _count(database, workspace_id) == 0


def test_unknown_workspace_is_rejected(database: Database) -> None:
    """A snapshot cannot exist without its workspace."""

    with pytest.raises(IntegrityError):
        _save(database, _snapshot(uuid4()))


def test_record_id_needs_no_matching_record(database: Database, workspace_ids: list[UUID]) -> None:
    """History outlives the record it came from, so record_id is not a foreign key."""

    workspace_id = _new_workspace(database, workspace_ids, "snapshot-no-record")

    _save(database, _snapshot(workspace_id, record_id=uuid4()))

    assert _count(database, workspace_id) == 1


def test_deleting_workspace_cascades_to_snapshots(
    database: Database, workspace_ids: list[UUID]
) -> None:
    """History goes away with its workspace and with nothing else."""

    workspace_id = _new_workspace(database, workspace_ids, "snapshot-cascade")
    other_id = _new_workspace(database, workspace_ids, "snapshot-cascade-other")
    _save(
        database,
        _snapshot(workspace_id, generation=1),
        _snapshot(workspace_id, generation=2),
        _snapshot(other_id, generation=1),
    )

    with database.session() as session:
        workspace = session.get(Workspace, workspace_id)
        assert workspace is not None
        session.delete(workspace)
        session.commit()

    assert _count(database, workspace_id) == 0
    assert _count(database, other_id) == 1


def test_updating_a_snapshot_through_the_orm_is_rejected(
    database: Database, workspace_ids: list[UUID]
) -> None:
    """The append-only trigger stops an ORM edit, and the stored row is unchanged."""

    workspace_id = _new_workspace(database, workspace_ids, "snapshot-orm-update")
    _save(database, _snapshot(workspace_id, content={"market_overview": "Original."}))

    with database.session() as session:
        loaded = StageSnapshotRepository(session).get(workspace_id, "market_research", 1)
        assert loaded is not None
        loaded.content = {"market_overview": "Rewritten history."}
        with pytest.raises(IntegrityError, match="append-only"):
            session.commit()

    with database.session() as session:
        reloaded = StageSnapshotRepository(session).get(workspace_id, "market_research", 1)
        assert reloaded is not None
        assert reloaded.content == {"market_overview": "Original."}


def test_updating_a_snapshot_with_raw_sql_is_rejected(
    database: Database, workspace_ids: list[UUID]
) -> None:
    """The guarantee is in the database, so it holds for any client, not just this app."""

    workspace_id = _new_workspace(database, workspace_ids, "snapshot-sql-update")
    _save(database, _snapshot(workspace_id))

    with database.session() as session:
        with pytest.raises(IntegrityError, match="append-only"):
            session.execute(
                text("UPDATE stage_snapshots SET model_used = 'forged' WHERE workspace_id = :id"),
                {"id": workspace_id},
            )
        session.rollback()

    with database.session() as session:
        reloaded = StageSnapshotRepository(session).get(workspace_id, "market_research", 1)
        assert reloaded is not None
        assert reloaded.model_used == "test-model"
