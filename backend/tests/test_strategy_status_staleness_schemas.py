"""Unit tests for the history fields on the strategy status schemas."""

from typing import Any
from uuid import uuid4

import pytest
from pydantic import ValidationError

from backend.app.schemas.strategy_orchestration import (
    ProvenanceState,
    StageApprovalState,
    StageStaleness,
    StaleInput,
    StrategyPrerequisite,
    StrategyStage,
    StrategyStageStatus,
    StrategyStatusResponse,
)

CHAIN = list(StrategyStage)


def _stale(*prerequisites: StrategyPrerequisite) -> StageStaleness:
    """Build tracked staleness listing the given inputs as moved from 1 to 2."""

    return StageStaleness(
        provenance=ProvenanceState.TRACKED,
        stale_inputs=[
            StaleInput(prerequisite=prerequisite, built_from=1, current=2)
            for prerequisite in prerequisites
        ],
    )


FRESH = StageStaleness(provenance=ProvenanceState.TRACKED)
UNKNOWN = StageStaleness(provenance=ProvenanceState.UNKNOWN)


def _status(stage: StrategyStage, *, generated: bool = True, **fields: Any) -> StrategyStageStatus:
    """Build one stage's status, generated and draft unless told otherwise."""

    return StrategyStageStatus(
        stage=stage,
        generated=generated,
        version=1 if generated else None,
        approval=StageApprovalState.DRAFT if generated else None,
        can_generate=True,
        **fields,
    )


def _response(**per_stage: dict[str, Any]) -> StrategyStatusResponse:
    """Build a full-chain response, applying history fields to named stages."""

    return StrategyStatusResponse(
        workspace_id=uuid4(),
        stages=[_status(stage, **per_stage.get(stage.value, {})) for stage in CHAIN],
    )


# --- per-stage fields ---------------------------------------------------------------


def test_history_fields_are_optional_so_existing_constructions_stay_valid() -> None:
    """A generated stage may omit both; nothing that built a status before breaks."""

    item = _status(StrategyStage.MARKET_RESEARCH)

    assert item.generation is None
    assert item.staleness is None


def test_a_generated_stage_reports_its_generation_and_staleness() -> None:
    """Both fields are serialized alongside the stage's existing ones."""

    item = _status(
        StrategyStage.CUSTOMER_PERSONAS,
        generation=3,
        staleness=_stale(StrategyPrerequisite.MARKET_RESEARCH),
    )

    dumped = item.model_dump(mode="json")

    assert dumped["generation"] == 3
    assert dumped["staleness"] == {
        "provenance": "tracked",
        "stale_inputs": [{"prerequisite": "market_research", "built_from": 1, "current": 2}],
        "stale": True,
    }


@pytest.mark.parametrize(
    "fields",
    [{"generation": 1}, {"staleness": FRESH}, {"generation": 1, "staleness": FRESH}],
    ids=["generation", "staleness", "both"],
)
def test_a_stage_that_is_not_generated_cannot_report_history(fields: dict[str, Any]) -> None:
    """History describes the current record, and there is none."""

    with pytest.raises(ValidationError, match="not generated cannot report a generation"):
        _status(StrategyStage.MARKET_RESEARCH, generated=False, **fields)


def test_tracked_provenance_requires_a_known_generation() -> None:
    """You cannot know what a stage was built from without knowing which generation it is."""

    with pytest.raises(ValidationError, match="requires a known generation"):
        _status(StrategyStage.MARKET_RESEARCH, staleness=FRESH)


def test_a_generated_stage_without_matching_history_can_be_unknown() -> None:
    """No generation and unknown provenance is how an unrecorded stage is reported."""

    item = _status(StrategyStage.MARKET_RESEARCH, staleness=UNKNOWN)

    assert item.generation is None
    assert item.staleness is not None
    assert item.staleness.stale is False


def test_a_stage_before_history_has_a_generation_but_unknown_provenance() -> None:
    """A backfilled generation 1 is known, even though what it was built from is not."""

    item = _status(StrategyStage.MARKET_RESEARCH, generation=1, staleness=UNKNOWN)

    assert item.generation == 1


def test_generation_starts_at_one() -> None:
    """Generations are numbered from one."""

    with pytest.raises(ValidationError):
        _status(StrategyStage.MARKET_RESEARCH, generation=0, staleness=FRESH)


def test_staleness_never_constrains_whether_a_stage_can_be_generated() -> None:
    """A stale stage can be generatable or blocked exactly as if it were fresh."""

    stale = _stale(StrategyPrerequisite.BUSINESS_PROFILE)

    generatable = _status(StrategyStage.BRAND_STRATEGY, generation=2, staleness=stale)
    blocked = StrategyStageStatus(
        stage=StrategyStage.BRAND_STRATEGY,
        generated=True,
        version=1,
        approval=StageApprovalState.APPROVED,
        can_generate=False,
        unapproved_prerequisites=[StrategyStage.CUSTOMER_PERSONAS],
        generation=2,
        staleness=stale,
    )

    assert generatable.can_generate is True
    assert blocked.can_generate is False
    assert generatable.staleness is not None and generatable.staleness.stale
    assert blocked.staleness is not None and blocked.staleness.stale


# --- stale_stages -------------------------------------------------------------------


def test_stale_stages_lists_stale_generated_stages_in_chain_order() -> None:
    """The top-level list gives clients every stage to look at, in chain order."""

    response = _response(
        marketing_strategy={
            "generation": 1,
            "staleness": _stale(StrategyPrerequisite.BRAND_STRATEGY),
        },
        market_research={"generation": 1, "staleness": FRESH},
        customer_personas={
            "generation": 1,
            "staleness": _stale(StrategyPrerequisite.MARKET_RESEARCH),
        },
    )

    assert response.stale_stages == [
        StrategyStage.CUSTOMER_PERSONAS,
        StrategyStage.MARKETING_STRATEGY,
    ]
    assert response.model_dump(mode="json")["stale_stages"] == [
        "customer_personas",
        "marketing_strategy",
    ]


def test_stale_stages_is_empty_when_nothing_is_stale() -> None:
    """Fresh, unknown, and unreported stages are all not stale."""

    response = _response(
        business_understanding={"generation": 1, "staleness": FRESH},
        market_research={"staleness": UNKNOWN},
    )

    assert response.stale_stages == []


def test_a_response_with_no_history_fields_has_no_stale_stages() -> None:
    """The shape every existing caller already produces."""

    assert _response().stale_stages == []
