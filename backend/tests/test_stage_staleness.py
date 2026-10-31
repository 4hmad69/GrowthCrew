"""Unit tests for the staleness rules, which are pure and need no database."""

from uuid import uuid4

from backend.app.schemas.stage_history import ProfileInputRef, StageInputRef, StageInputs
from backend.app.schemas.strategy_orchestration import (
    ProvenanceState,
    StrategyPrerequisite,
    StrategyStage,
)
from backend.app.services.stage_staleness import assess_staleness

S = StrategyStage


def _stage_ref(generation: int) -> StageInputRef:
    """Build a reference to an upstream generation."""

    return StageInputRef(generation=generation, record_id=uuid4(), record_version=generation)


def _inputs(
    *,
    profile_version: int | None = None,
    stages: dict[StrategyStage, int] | None = None,
) -> StageInputs:
    """Build recorded inputs from a profile version and upstream generations."""

    return StageInputs(
        profile=(
            ProfileInputRef(id=uuid4(), version=profile_version)
            if profile_version is not None
            else None
        ),
        stages={stage: _stage_ref(generation) for stage, generation in (stages or {}).items()},
    )


def test_unknown_inputs_are_unknown_and_never_stale() -> None:
    """Nothing was recorded, so nothing can be claimed - however much has moved."""

    result = assess_staleness(
        None,
        current_generations={S.MARKET_RESEARCH: 9},
        current_profile_version=9,
    )

    assert result.provenance is ProvenanceState.UNKNOWN
    assert result.stale is False


def test_inputs_that_have_not_moved_are_fresh() -> None:
    """Built from the newest of everything means not stale."""

    result = assess_staleness(
        _inputs(profile_version=2, stages={S.BUSINESS_UNDERSTANDING: 3, S.MARKET_RESEARCH: 1}),
        current_generations={S.BUSINESS_UNDERSTANDING: 3, S.MARKET_RESEARCH: 1},
        current_profile_version=2,
    )

    assert result.provenance is ProvenanceState.TRACKED
    assert result.stale is False
    assert result.stale_inputs == []


def test_a_newer_profile_version_makes_a_stage_stale() -> None:
    """Editing the business profile after generating makes the stages built from it stale."""

    result = assess_staleness(
        _inputs(profile_version=1),
        current_generations={},
        current_profile_version=3,
    )

    assert result.stale is True
    [item] = result.stale_inputs
    assert item.prerequisite is StrategyPrerequisite.BUSINESS_PROFILE
    assert (item.built_from, item.current) == (1, 3)


def test_a_newer_upstream_generation_makes_a_stage_stale() -> None:
    """Regenerating an upstream stage makes the stages built from its old output stale."""

    result = assess_staleness(
        _inputs(stages={S.MARKET_RESEARCH: 1}),
        current_generations={S.MARKET_RESEARCH: 2},
        current_profile_version=None,
    )

    assert result.stale is True
    [item] = result.stale_inputs
    assert item.prerequisite is StrategyPrerequisite.MARKET_RESEARCH
    assert (item.built_from, item.current) == (1, 2)


def test_only_the_inputs_that_moved_are_reported_in_canonical_order() -> None:
    """The profile comes first, then stages in chain order, whatever order they were given."""

    result = assess_staleness(
        _inputs(
            profile_version=1,
            stages={
                S.BRAND_STRATEGY: 1,
                S.BUSINESS_UNDERSTANDING: 1,
                S.CUSTOMER_PERSONAS: 4,
                S.MARKET_RESEARCH: 2,
            },
        ),
        current_generations={
            S.BRAND_STRATEGY: 2,
            S.BUSINESS_UNDERSTANDING: 1,
            S.CUSTOMER_PERSONAS: 4,
            S.MARKET_RESEARCH: 3,
        },
        current_profile_version=2,
    )

    assert [item.prerequisite.value for item in result.stale_inputs] == [
        "business_profile",
        "market_research",
        "brand_strategy",
    ]


def test_an_input_with_no_history_now_is_not_reported() -> None:
    """If an upstream stage has no history at all, it cannot be said to have moved."""

    result = assess_staleness(
        _inputs(stages={S.MARKET_RESEARCH: 1}),
        current_generations={},
        current_profile_version=None,
    )

    assert result.stale is False


def test_a_missing_profile_now_is_not_reported() -> None:
    """A profile that no longer exists has not 'moved forward'."""

    result = assess_staleness(
        _inputs(profile_version=3),
        current_generations={},
        current_profile_version=None,
    )

    assert result.stale is False


def test_an_input_that_went_backwards_is_not_stale() -> None:
    """Only newer counts: a lower number now (a recreated profile) is not a moved-on input."""

    result = assess_staleness(
        _inputs(profile_version=5, stages={S.MARKET_RESEARCH: 4}),
        current_generations={S.MARKET_RESEARCH: 1},
        current_profile_version=1,
    )

    assert result.stale is False


def test_an_upstream_stage_the_stage_was_not_built_from_is_ignored() -> None:
    """Regenerating a stage this one does not use cannot make it stale."""

    result = assess_staleness(
        _inputs(stages={S.MARKET_RESEARCH: 1}),
        current_generations={S.MARKET_RESEARCH: 1, S.COMPETITOR_ANALYSIS: 7},
        current_profile_version=None,
    )

    assert result.stale is False
