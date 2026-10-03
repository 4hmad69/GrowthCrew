"""Unit tests for the strategy prerequisite map and its pure helper.

The service's database reads are covered against real PostgreSQL in the
orchestration integration tests, per this project's convention of not
mocking SQLAlchemy behavior. What is unit-testable without a database is
the dependency graph itself, so it is pinned here explicitly: changing
the graph must be a deliberate edit to both the map and this file.
"""

import pytest

from backend.app.schemas.strategy_orchestration import StrategyPrerequisite, StrategyStage
from backend.app.services.strategy_status import STAGE_PREREQUISITES, missing_prerequisites

P = StrategyPrerequisite
S = StrategyStage
CHAIN = list(StrategyStage)
CANONICAL = list(StrategyPrerequisite)


def test_prerequisite_map_matches_the_verified_dependency_graph() -> None:
    """Pinned against what each agent service actually requires in generate()."""

    assert dict(STAGE_PREREQUISITES) == {
        S.BUSINESS_UNDERSTANDING: (P.BUSINESS_PROFILE,),
        S.MARKET_RESEARCH: (P.BUSINESS_PROFILE,),
        S.COMPETITOR_ANALYSIS: (P.BUSINESS_PROFILE,),
        S.CUSTOMER_PERSONAS: (P.BUSINESS_PROFILE, P.BUSINESS_UNDERSTANDING, P.MARKET_RESEARCH),
        S.BRAND_STRATEGY: (
            P.BUSINESS_PROFILE,
            P.BUSINESS_UNDERSTANDING,
            P.COMPETITOR_ANALYSIS,
            P.CUSTOMER_PERSONAS,
        ),
        S.MARKETING_STRATEGY: (
            P.BUSINESS_PROFILE,
            P.BUSINESS_UNDERSTANDING,
            P.MARKET_RESEARCH,
            P.COMPETITOR_ANALYSIS,
            P.CUSTOMER_PERSONAS,
            P.BRAND_STRATEGY,
        ),
        S.CONTENT_PLAN: (P.MARKETING_STRATEGY,),
    }


def test_every_stage_has_an_entry() -> None:
    """A stage missing from the map would raise KeyError at request time."""

    assert set(STAGE_PREREQUISITES) == set(StrategyStage)


@pytest.mark.parametrize("stage", CHAIN)
def test_prerequisites_only_reference_earlier_stages(stage: StrategyStage) -> None:
    """Chain order must be a valid generation order for the whole graph."""

    earlier = {item.value for item in CHAIN[: CHAIN.index(stage)]}

    for prerequisite in STAGE_PREREQUISITES[stage]:
        assert prerequisite is P.BUSINESS_PROFILE or prerequisite.value in earlier


@pytest.mark.parametrize("stage", CHAIN)
def test_prerequisites_are_unique_and_in_canonical_order(stage: StrategyStage) -> None:
    """Clients see missing_prerequisites in this order, so it must be stable."""

    items = list(STAGE_PREREQUISITES[stage])

    assert len(set(items)) == len(items)
    assert items == sorted(items, key=CANONICAL.index)


def test_content_plan_does_not_require_the_business_profile() -> None:
    """Content Planning needs only a Marketing Strategy, exactly as its service does."""

    assert STAGE_PREREQUISITES[S.CONTENT_PLAN] == (P.MARKETING_STRATEGY,)


def test_map_is_read_only() -> None:
    """The shared graph must not be mutable at runtime."""

    with pytest.raises(TypeError):
        STAGE_PREREQUISITES[S.CONTENT_PLAN] = ()


def test_missing_prerequisites_is_empty_when_everything_exists() -> None:
    """Nothing missing means the stage can be generated."""

    assert missing_prerequisites(S.MARKETING_STRATEGY, set(P)) == []


def test_missing_prerequisites_reports_everything_for_an_empty_workspace() -> None:
    """With nothing existing, all of a stage's prerequisites are missing."""

    assert missing_prerequisites(S.BRAND_STRATEGY, set()) == list(
        STAGE_PREREQUISITES[S.BRAND_STRATEGY]
    )


def test_missing_prerequisites_reports_only_what_is_absent_in_canonical_order() -> None:
    """Partially satisfied stages list just the gaps."""

    existing = {P.BUSINESS_PROFILE, P.BUSINESS_UNDERSTANDING, P.COMPETITOR_ANALYSIS}

    assert missing_prerequisites(S.MARKETING_STRATEGY, existing) == [
        P.MARKET_RESEARCH,
        P.CUSTOMER_PERSONAS,
        P.BRAND_STRATEGY,
    ]


def test_missing_prerequisites_ignores_unrelated_existing_items() -> None:
    """Extra existing records do not affect a stage that does not need them."""

    assert missing_prerequisites(S.BUSINESS_UNDERSTANDING, {P.CONTENT_PLAN}) == [P.BUSINESS_PROFILE]
