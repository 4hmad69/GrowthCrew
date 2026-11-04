"""Real PostgreSQL integration tests for stage history and staleness.

These drive the history service the way a generating agent does: load or
create a stage record, change it, hand it to ``record``, commit. What
they prove is what only a real database can: that history is written in
the same transaction as the generation, that it survives the record
being overwritten, that two requests cannot claim one generation, and
that staleness follows real regenerations and profile edits.
"""

import os
import threading
import time
from collections.abc import Iterator
from typing import Any
from uuid import UUID

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError

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
from backend.app.db.models.stage_snapshot import StageSnapshot
from backend.app.db.models.workspace import Workspace
from backend.app.exceptions import ResourceNotFoundError, StaleResourceError
from backend.app.schemas.stage_history import STAGE_CONTENT_FIELDS
from backend.app.schemas.strategy_orchestration import ProvenanceState, StrategyStage
from backend.app.services.stage_history import StageHistoryService
from backend.app.services.stage_staleness import StageStalenessService

RUN_INTEGRATION_TESTS = os.getenv("GROWTHCREW_RUN_INTEGRATION_TESTS") == "1"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not RUN_INTEGRATION_TESTS,
        reason=("Set GROWTHCREW_RUN_INTEGRATION_TESTS=1 to run PostgreSQL integration tests."),
    ),
]

S = StrategyStage

MODELS: dict[StrategyStage, Any] = {
    S.BUSINESS_UNDERSTANDING: BusinessUnderstanding,
    S.MARKET_RESEARCH: MarketResearch,
    S.COMPETITOR_ANALYSIS: CompetitorAnalysis,
    S.CUSTOMER_PERSONAS: CustomerPersonaSet,
    S.BRAND_STRATEGY: BrandStrategy,
    S.MARKETING_STRATEGY: MarketingStrategy,
    S.CONTENT_PLAN: ContentPlan,
}


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


def _workspace(
    database: Database,
    workspace_ids: list[UUID],
    name: str,
    *,
    profile: bool = True,
) -> tuple[UUID, UUID | None]:
    """Create a workspace, optionally with a business profile; return their ids."""

    with database.session() as session:
        workspace = Workspace(name=name)
        session.add(workspace)
        session.flush()
        profile_row = BusinessProfile(workspace_id=workspace.id, business_name=name)
        if profile:
            session.add(profile_row)
        session.commit()
        workspace_id = workspace.id
        profile_id = profile_row.id if profile else None

    workspace_ids.append(workspace_id)
    return workspace_id, profile_id


def _new_record(stage: StrategyStage, workspace_id: UUID, profile_id: UUID | None) -> Any:
    """Build a valid, unsaved stage record with recognizable content."""

    common: dict[str, Any] = {"model_used": "test-model", "input_tokens": 5, "output_tokens": 6}
    if stage is S.BUSINESS_UNDERSTANDING:
        return BusinessUnderstanding(
            profile_id=profile_id,
            summary="Summary.",
            inferred_business_stage="seed",
            competitive_category="snacks",
            key_differentiators=["fast"],
            likely_customer_pain_points=["price"],
            confidence_notes=None,
            **common,
        )
    if stage is S.MARKET_RESEARCH:
        return MarketResearch(
            workspace_id=workspace_id,
            market_overview="Overview.",
            target_customer_segments="Segments.",
            competitive_landscape="Landscape.",
            opportunities_and_risks="Risks.",
            sources=[{"url": "https://example.com", "title": "Café"}],
            **common,
        )
    if stage is S.COMPETITOR_ANALYSIS:
        return CompetitorAnalysis(
            workspace_id=workspace_id,
            competitor_overview="Competitors.",
            strengths_and_weaknesses="S and W.",
            pricing_and_positioning="Pricing.",
            differentiation_opportunities="Gaps.",
            sources=[],
            **common,
        )
    if stage is S.CUSTOMER_PERSONAS:
        return CustomerPersonaSet(
            workspace_id=workspace_id,
            overview="Personas.",
            personas=[{"name": "Sam"}],
            **common,
        )
    if stage is S.BRAND_STRATEGY:
        return BrandStrategy(
            workspace_id=workspace_id,
            overview="Brand.",
            positioning_statement="Position.",
            value_proposition="Value.",
            brand_voice_and_tone="Warm.",
            brand_pillars=["one"],
            tagline_options=["Eat well"],
            **common,
        )
    if stage is S.MARKETING_STRATEGY:
        return MarketingStrategy(
            workspace_id=workspace_id,
            recommended_channels_and_tactics="Channels.",
            content_and_messaging_pillars="Pillars.",
            ninety_day_roadmap="Roadmap.",
            budget_allocation_and_kpis="Budget.",
            sources=[],
            **common,
        )
    return ContentPlan(
        workspace_id=workspace_id,
        overview="Plan.",
        entries=[{"day": 1, "topic": "Launch"}],
        **common,
    )


def _existing(
    session: Any, stage: StrategyStage, workspace_id: UUID, profile_id: UUID | None
) -> Any:
    """Load the current record of a stage, if there is one."""

    model = MODELS[stage]
    if stage is S.BUSINESS_UNDERSTANDING:
        return session.scalar(select(model).where(model.profile_id == profile_id))
    return session.scalar(select(model).where(model.workspace_id == workspace_id))


def _generate(
    database: Database,
    workspace_id: UUID,
    profile_id: UUID | None,
    stage: StrategyStage,
    **changes: Any,
) -> StageSnapshot:
    """Do what a generating agent does: create or overwrite the record, record, commit."""

    with database.session() as session:
        record = _existing(session, stage, workspace_id, profile_id)
        if record is None:
            record = _new_record(stage, workspace_id, profile_id)
            session.add(record)
        for name, value in changes.items():
            setattr(record, name, value)
        snapshot = StageHistoryService(session).record(workspace_id, stage, record)
        session.commit()
        return snapshot


def _generate_chain(database: Database, workspace_id: UUID, profile_id: UUID | None) -> None:
    """Generate all seven stages in chain order, as a full run would."""

    for stage in StrategyStage:
        _generate(database, workspace_id, profile_id, stage)


def _snapshot_count(
    database: Database, workspace_id: UUID, stage: StrategyStage | None = None
) -> int:
    """Count snapshots for a workspace, optionally for one stage."""

    statement = (
        select(func.count())
        .select_from(StageSnapshot)
        .where(StageSnapshot.workspace_id == workspace_id)
    )
    if stage is not None:
        statement = statement.where(StageSnapshot.stage == stage.value)
    with database.session() as session:
        return session.scalar(statement) or 0


def _staleness(database: Database, workspace_id: UUID) -> dict[StrategyStage, Any]:
    """Return the staleness of every stage that has history."""

    with database.session() as session:
        return StageStalenessService(session).staleness_by_stage(workspace_id)


def _edit_profile(database: Database, profile_id: UUID) -> None:
    """Change the business profile, which advances its version."""

    with database.session() as session:
        profile = session.get(BusinessProfile, profile_id)
        assert profile is not None
        profile.business_name = f"{profile.business_name} (edited)"
        session.commit()


# --- recording ---------------------------------------------------------------


def test_first_generation_records_generation_one_with_the_records_identity(
    database: Database, workspace_ids: list[UUID]
) -> None:
    """A new stage gets generation 1, copying its content, identity and usage."""

    workspace_id, profile_id = _workspace(database, workspace_ids, "history-first")

    with database.session() as session:
        record = _new_record(S.MARKET_RESEARCH, workspace_id, profile_id)
        session.add(record)
        snapshot = StageHistoryService(session).record(workspace_id, S.MARKET_RESEARCH, record)

        # The record was flushed first, so its id and final version are known.
        assert record.id is not None
        assert snapshot.record_id == record.id
        assert snapshot.record_version == record.version == 1
        session.commit()

    with database.session() as session:
        stored = StageHistoryService(session).get_snapshot(workspace_id, S.MARKET_RESEARCH, 1)
        assert stored.generation == 1
        assert stored.model_used == "test-model"
        assert (stored.input_tokens, stored.output_tokens) == (5, 6)
        assert stored.content["market_overview"] == "Overview."
        assert stored.content["sources"] == [{"url": "https://example.com", "title": "Café"}]


def test_record_does_not_commit_so_a_rollback_loses_both_record_and_history(
    database: Database, workspace_ids: list[UUID]
) -> None:
    """History is written in the generation's own transaction, never separately."""

    workspace_id, profile_id = _workspace(database, workspace_ids, "history-rollback")

    with database.session() as session:
        record = _new_record(S.MARKET_RESEARCH, workspace_id, profile_id)
        session.add(record)
        StageHistoryService(session).record(workspace_id, S.MARKET_RESEARCH, record)
        session.rollback()

    assert _snapshot_count(database, workspace_id) == 0
    with database.session() as session:
        assert _existing(session, S.MARKET_RESEARCH, workspace_id, profile_id) is None


def test_regenerating_keeps_the_previous_generation_intact(
    database: Database, workspace_ids: list[UUID]
) -> None:
    """The whole point: the record is overwritten, but what it was is still there."""

    workspace_id, profile_id = _workspace(database, workspace_ids, "history-overwrite")
    first = _generate(database, workspace_id, profile_id, S.MARKET_RESEARCH)

    second = _generate(
        database, workspace_id, profile_id, S.MARKET_RESEARCH, market_overview="A new overview."
    )

    assert (first.generation, second.generation) == (1, 2)
    assert (first.record_version, second.record_version) == (1, 2)
    assert first.record_id == second.record_id
    with database.session() as session:
        service = StageHistoryService(session)
        assert service.get_snapshot(workspace_id, S.MARKET_RESEARCH, 1).content[
            "market_overview"
        ] == ("Overview.")
        assert service.get_snapshot(workspace_id, S.MARKET_RESEARCH, 2).content[
            "market_overview"
        ] == ("A new overview.")
        live = _existing(session, S.MARKET_RESEARCH, workspace_id, profile_id)
        assert live.market_overview == "A new overview."


def test_regenerating_identical_content_still_adds_a_generation(
    database: Database, workspace_ids: list[UUID]
) -> None:
    """The record's version stays put, so only the generation proves it happened.

    This is also what keeps approvals meaningful: two generations that
    share a record id and version hold identical content, so an approval
    pinned to that pair cannot be pointing at the wrong one.
    """

    workspace_id, profile_id = _workspace(database, workspace_ids, "history-identical")
    first = _generate(database, workspace_id, profile_id, S.MARKET_RESEARCH)

    second = _generate(database, workspace_id, profile_id, S.MARKET_RESEARCH)

    assert (first.generation, second.generation) == (1, 2)
    assert first.record_version == second.record_version == 1
    assert first.record_id == second.record_id
    assert first.content == second.content


def test_snapshot_content_is_a_copy_not_the_records_own_lists(
    database: Database, workspace_ids: list[UUID]
) -> None:
    """Later edits to the record's lists must not reach back into history."""

    workspace_id, profile_id = _workspace(database, workspace_ids, "history-copy")

    with database.session() as session:
        record = _new_record(S.MARKET_RESEARCH, workspace_id, profile_id)
        session.add(record)
        snapshot = StageHistoryService(session).record(workspace_id, S.MARKET_RESEARCH, record)

        assert snapshot.content["sources"] is not record.sources
        record.sources.append({"url": "https://later.example"})
        assert len(snapshot.content["sources"]) == 1


@pytest.mark.parametrize("stage", list(StrategyStage))
def test_every_stage_snapshots_exactly_its_registered_fields(
    database: Database, workspace_ids: list[UUID], stage: StrategyStage
) -> None:
    """Content holds each registered field, with the record's own value, and nothing else."""

    workspace_id, profile_id = _workspace(database, workspace_ids, f"history-{stage.value}")

    snapshot = _generate(database, workspace_id, profile_id, stage)

    with database.session() as session:
        record = _existing(session, stage, workspace_id, profile_id)
        assert set(snapshot.content) == set(STAGE_CONTENT_FIELDS[stage])
        assert snapshot.content == {
            field: getattr(record, field) for field in STAGE_CONTENT_FIELDS[stage]
        }
        assert snapshot.stage == stage.value


def test_a_conflicting_foreign_key_error_is_not_mistaken_for_a_generation_race(
    database: Database, workspace_ids: list[UUID]
) -> None:
    """Only a claimed generation number is a conflict; other failures pass through."""

    workspace_id, profile_id = _workspace(database, workspace_ids, "history-other-error")
    _generate(database, workspace_id, profile_id, S.MARKET_RESEARCH)

    with database.session() as session:
        record = _existing(session, S.MARKET_RESEARCH, workspace_id, profile_id)
        with pytest.raises(IntegrityError):
            StageHistoryService(session).record(
                UUID("00000000-0000-0000-0000-000000000001"), S.MARKET_RESEARCH, record
            )


def _wait_for_a_blocked_session(database: Database, timeout: float = 15.0) -> None:
    """Wait until some database session is stuck waiting on another one's lock."""

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with database.engine.connect() as connection:
            blocked = connection.scalar(
                text(
                    "SELECT count(*) FROM pg_stat_activity "
                    "WHERE wait_event_type = 'Lock' AND datname = current_database()"
                )
            )
        if blocked:
            return
        time.sleep(0.05)
    pytest.fail("No session ever blocked; the race was not set up.")


def test_two_requests_claiming_one_generation_one_wins_and_one_conflicts(
    database: Database, workspace_ids: list[UUID]
) -> None:
    """Two concurrent forced regenerations: the loser is told, and leaves nothing behind.

    Both regenerate identical content, so neither touches the record's
    row and the only thing they contend over is the generation number.
    """

    workspace_id, profile_id = _workspace(database, workspace_ids, "history-race")
    _generate(database, workspace_id, profile_id, S.CONTENT_PLAN)
    outcome: dict[str, Any] = {}

    def loser() -> None:
        with database.session() as session:
            record = _existing(session, S.CONTENT_PLAN, workspace_id, profile_id)
            try:
                StageHistoryService(session).record(workspace_id, S.CONTENT_PLAN, record)
                session.commit()
            except Exception as exc:
                outcome["error"] = exc
                outcome["session_usable"] = session.is_active

    with database.session() as winner_session:
        record = _existing(winner_session, S.CONTENT_PLAN, workspace_id, profile_id)
        winner = StageHistoryService(winner_session).record(workspace_id, S.CONTENT_PLAN, record)
        assert winner.generation == 2

        thread = threading.Thread(target=loser)
        thread.start()
        _wait_for_a_blocked_session(database)
        winner_session.commit()

    thread.join(timeout=15)
    assert not thread.is_alive()
    assert isinstance(outcome.get("error"), StaleResourceError)
    assert "regenerated by another request" in str(outcome["error"])
    assert outcome["session_usable"] is True
    assert _snapshot_count(database, workspace_id, S.CONTENT_PLAN) == 2


# --- provenance --------------------------------------------------------------


def test_profile_only_stage_records_the_profile_version_it_was_built_from(
    database: Database, workspace_ids: list[UUID]
) -> None:
    """The first three stages build from the profile alone."""

    workspace_id, profile_id = _workspace(database, workspace_ids, "inputs-profile")
    _edit_profile(database, profile_id)

    snapshot = _generate(database, workspace_id, profile_id, S.MARKET_RESEARCH)

    assert snapshot.inputs == {
        "profile": {"id": str(profile_id), "version": 2},
        "stages": {},
    }


def test_downstream_stage_records_each_upstream_generation_it_was_built_from(
    database: Database, workspace_ids: list[UUID]
) -> None:
    """Personas build from the profile, business understanding and market research."""

    workspace_id, profile_id = _workspace(database, workspace_ids, "inputs-chain")
    understanding = _generate(database, workspace_id, profile_id, S.BUSINESS_UNDERSTANDING)
    research = _generate(database, workspace_id, profile_id, S.MARKET_RESEARCH)

    personas = _generate(database, workspace_id, profile_id, S.CUSTOMER_PERSONAS)

    assert personas.inputs == {
        "profile": {"id": str(profile_id), "version": 1},
        "stages": {
            "business_understanding": {
                "generation": 1,
                "record_id": str(understanding.record_id),
                "record_version": 1,
            },
            "market_research": {
                "generation": 1,
                "record_id": str(research.record_id),
                "record_version": 1,
            },
        },
    }


def test_provenance_follows_the_upstream_generation_current_at_the_time(
    database: Database, workspace_ids: list[UUID]
) -> None:
    """After an upstream stage is regenerated, new work records the newer generation."""

    workspace_id, profile_id = _workspace(database, workspace_ids, "inputs-newer")
    _generate(database, workspace_id, profile_id, S.BUSINESS_UNDERSTANDING)
    _generate(database, workspace_id, profile_id, S.MARKET_RESEARCH)
    _generate(database, workspace_id, profile_id, S.MARKET_RESEARCH, market_overview="Changed.")

    personas = _generate(database, workspace_id, profile_id, S.CUSTOMER_PERSONAS)

    research_ref = personas.inputs["stages"]["market_research"]
    assert (research_ref["generation"], research_ref["record_version"]) == (2, 2)


def test_content_plan_records_only_the_marketing_strategy(
    database: Database, workspace_ids: list[UUID]
) -> None:
    """The content plan does not build from the profile, so it does not record one."""

    workspace_id, profile_id = _workspace(database, workspace_ids, "inputs-content-plan")
    strategy = _generate(database, workspace_id, profile_id, S.MARKETING_STRATEGY)

    plan = _generate(database, workspace_id, profile_id, S.CONTENT_PLAN)

    assert plan.inputs == {
        "profile": None,
        "stages": {
            "marketing_strategy": {
                "generation": 1,
                "record_id": str(strategy.record_id),
                "record_version": 1,
            }
        },
    }


def test_unidentifiable_upstream_input_makes_provenance_unknown(
    database: Database, workspace_ids: list[UUID]
) -> None:
    """An upstream record with no snapshot cannot be named, so nothing is guessed."""

    workspace_id, profile_id = _workspace(database, workspace_ids, "inputs-no-snapshot")
    with database.session() as session:
        session.add(_new_record(S.MARKET_RESEARCH, workspace_id, profile_id))
        session.commit()
    _generate(database, workspace_id, profile_id, S.BUSINESS_UNDERSTANDING)

    personas = _generate(database, workspace_id, profile_id, S.CUSTOMER_PERSONAS)

    assert personas.inputs is None


def test_upstream_changed_outside_history_makes_provenance_unknown(
    database: Database, workspace_ids: list[UUID]
) -> None:
    """If a record moved on without a snapshot, its latest snapshot no longer describes it."""

    workspace_id, profile_id = _workspace(database, workspace_ids, "inputs-bypassed")
    _generate(database, workspace_id, profile_id, S.BUSINESS_UNDERSTANDING)
    _generate(database, workspace_id, profile_id, S.MARKET_RESEARCH)
    with database.session() as session:
        record = _existing(session, S.MARKET_RESEARCH, workspace_id, profile_id)
        record.market_overview = "Edited behind history's back."
        session.commit()

    personas = _generate(database, workspace_id, profile_id, S.CUSTOMER_PERSONAS)

    assert personas.inputs is None


def test_missing_profile_makes_provenance_unknown(
    database: Database, workspace_ids: list[UUID]
) -> None:
    """A stage that should name the profile but has none cannot state its inputs."""

    workspace_id, _ = _workspace(database, workspace_ids, "inputs-no-profile", profile=False)

    snapshot = _generate(database, workspace_id, None, S.MARKET_RESEARCH)

    assert snapshot.inputs is None


# --- reading -----------------------------------------------------------------


def test_history_lists_newest_first_and_only_the_requested_stage(
    database: Database, workspace_ids: list[UUID]
) -> None:
    """Each stage has its own history, returned newest generation first."""

    workspace_id, profile_id = _workspace(database, workspace_ids, "read-list")
    for _ in range(3):
        _generate(database, workspace_id, profile_id, S.MARKET_RESEARCH)
    _generate(database, workspace_id, profile_id, S.COMPETITOR_ANALYSIS)

    with database.session() as session:
        service = StageHistoryService(session)

        listed = service.list_history(workspace_id, S.MARKET_RESEARCH)

        assert [snapshot.generation for snapshot in listed] == [3, 2, 1]
        assert service.list_history(workspace_id, S.CONTENT_PLAN) == []


def test_reading_history_of_an_unknown_workspace_or_generation_is_not_found(
    database: Database, workspace_ids: list[UUID]
) -> None:
    """Both a missing workspace and a missing generation are reported as not found."""

    workspace_id, profile_id = _workspace(database, workspace_ids, "read-missing")
    _generate(database, workspace_id, profile_id, S.MARKET_RESEARCH)
    nowhere = UUID("00000000-0000-0000-0000-000000000002")

    with database.session() as session:
        service = StageHistoryService(session)

        with pytest.raises(ResourceNotFoundError, match="Workspace"):
            service.list_history(nowhere, S.MARKET_RESEARCH)
        with pytest.raises(ResourceNotFoundError, match="Workspace"):
            service.get_snapshot(nowhere, S.MARKET_RESEARCH, 1)
        with pytest.raises(ResourceNotFoundError, match="generation"):
            service.get_snapshot(workspace_id, S.MARKET_RESEARCH, 2)


# --- staleness ---------------------------------------------------------------


def test_a_freshly_generated_chain_is_not_stale(
    database: Database, workspace_ids: list[UUID]
) -> None:
    """Everything built from the newest of everything is fresh, and tracked."""

    workspace_id, profile_id = _workspace(database, workspace_ids, "stale-fresh")
    _generate_chain(database, workspace_id, profile_id)

    result = _staleness(database, workspace_id)

    assert list(result) == list(StrategyStage)
    assert all(item.provenance is ProvenanceState.TRACKED for item in result.values())
    assert not any(item.stale for item in result.values())


def test_regenerating_an_upstream_stage_makes_what_was_built_from_it_stale(
    database: Database, workspace_ids: list[UUID]
) -> None:
    """Personas built from market research go stale when research is regenerated."""

    workspace_id, profile_id = _workspace(database, workspace_ids, "stale-upstream")
    _generate_chain(database, workspace_id, profile_id)

    _generate(database, workspace_id, profile_id, S.MARKET_RESEARCH, market_overview="New.")

    result = _staleness(database, workspace_id)
    personas = result[S.CUSTOMER_PERSONAS]
    assert personas.stale is True
    assert [
        (item.prerequisite.value, item.built_from, item.current) for item in personas.stale_inputs
    ] == [("market_research", 1, 2)]
    assert result[S.MARKET_RESEARCH].stale is False
    assert result[S.COMPETITOR_ANALYSIS].stale is False


def test_regenerating_identical_content_still_makes_dependants_stale(
    database: Database, workspace_ids: list[UUID]
) -> None:
    """Staleness follows generations, not content: a forced regeneration counts.

    The record's version does not move here, which is exactly why a
    version comparison could not have seen it.
    """

    workspace_id, profile_id = _workspace(database, workspace_ids, "stale-identical")
    _generate_chain(database, workspace_id, profile_id)

    regenerated = _generate(database, workspace_id, profile_id, S.MARKET_RESEARCH)

    assert regenerated.record_version == 1
    assert _staleness(database, workspace_id)[S.CUSTOMER_PERSONAS].stale is True


def test_regenerating_the_dependant_clears_its_staleness(
    database: Database, workspace_ids: list[UUID]
) -> None:
    """Rebuilding personas from the newer research makes them fresh again."""

    workspace_id, profile_id = _workspace(database, workspace_ids, "stale-cleared")
    _generate_chain(database, workspace_id, profile_id)
    _generate(database, workspace_id, profile_id, S.MARKET_RESEARCH, market_overview="New.")
    assert _staleness(database, workspace_id)[S.CUSTOMER_PERSONAS].stale is True

    _generate(database, workspace_id, profile_id, S.CUSTOMER_PERSONAS)

    assert _staleness(database, workspace_id)[S.CUSTOMER_PERSONAS].stale is False


def test_editing_the_profile_makes_profile_dependent_stages_stale(
    database: Database, workspace_ids: list[UUID]
) -> None:
    """Every stage built from the profile is stale after it changes; the content plan is not."""

    workspace_id, profile_id = _workspace(database, workspace_ids, "stale-profile")
    _generate_chain(database, workspace_id, profile_id)

    _edit_profile(database, profile_id)

    result = _staleness(database, workspace_id)
    for stage in (
        S.BUSINESS_UNDERSTANDING,
        S.MARKET_RESEARCH,
        S.COMPETITOR_ANALYSIS,
        S.CUSTOMER_PERSONAS,
        S.BRAND_STRATEGY,
        S.MARKETING_STRATEGY,
    ):
        assert result[stage].stale is True, stage
        profile_input = result[stage].stale_inputs[0]
        assert profile_input.prerequisite.value == "business_profile"
        assert (profile_input.built_from, profile_input.current) == (1, 2)
    assert result[S.CONTENT_PLAN].stale is False


def test_staleness_is_judged_by_direct_inputs_only(
    database: Database, workspace_ids: list[UUID]
) -> None:
    """Each stage is judged by what it was directly built from; nothing propagates.

    Regenerating business understanding makes the marketing strategy
    (which uses it directly) stale, but not the content plan, which was
    built only from the strategy and has no way to know. It becomes stale
    when the strategy itself is regenerated.
    """

    workspace_id, profile_id = _workspace(database, workspace_ids, "stale-direct")
    _generate_chain(database, workspace_id, profile_id)

    _generate(database, workspace_id, profile_id, S.BUSINESS_UNDERSTANDING, summary="New.")

    result = _staleness(database, workspace_id)
    assert result[S.MARKETING_STRATEGY].stale is True
    assert result[S.CONTENT_PLAN].stale is False

    _generate(database, workspace_id, profile_id, S.MARKETING_STRATEGY)

    assert _staleness(database, workspace_id)[S.CONTENT_PLAN].stale is True


def test_a_stage_from_before_history_is_unknown_and_never_stale(
    database: Database, workspace_ids: list[UUID]
) -> None:
    """Backfilled generations have no recorded inputs, so nothing can be said about them."""

    workspace_id, profile_id = _workspace(database, workspace_ids, "stale-legacy")
    _generate(database, workspace_id, profile_id, S.MARKET_RESEARCH)
    # A legacy generation, as the migration's backfill would have written it:
    # content with no recorded inputs.
    with database.session() as session:
        session.add(
            StageSnapshot(
                workspace_id=workspace_id,
                stage=S.COMPETITOR_ANALYSIS.value,
                generation=1,
                record_id=UUID("00000000-0000-0000-0000-000000000003"),
                record_version=1,
                content={"competitor_overview": "Legacy."},
                inputs=None,
                model_used="legacy-model",
            )
        )
        session.commit()
    _edit_profile(database, profile_id)

    result = _staleness(database, workspace_id)

    assert result[S.COMPETITOR_ANALYSIS].provenance is ProvenanceState.UNKNOWN
    assert result[S.COMPETITOR_ANALYSIS].stale is False
    assert result[S.MARKET_RESEARCH].stale is True


def test_stages_without_history_are_absent_from_staleness(
    database: Database, workspace_ids: list[UUID]
) -> None:
    """Only stages that have been generated are reported."""

    workspace_id, profile_id = _workspace(database, workspace_ids, "stale-absent")
    _generate(database, workspace_id, profile_id, S.COMPETITOR_ANALYSIS)
    _generate(database, workspace_id, profile_id, S.BUSINESS_UNDERSTANDING)

    result = _staleness(database, workspace_id)

    assert list(result) == [S.BUSINESS_UNDERSTANDING, S.COMPETITOR_ANALYSIS]
    assert _staleness(database, UUID("00000000-0000-0000-0000-000000000004")) == {}
