"""Unit tests for strategy orchestration request and response schemas."""

from typing import Any
from uuid import uuid4

import pytest
from pydantic import ValidationError

from backend.app.schemas.strategy_orchestration import (
    StageApprovalState,
    StrategyGenerateRequest,
    StrategyGenerateResponse,
    StrategyPrerequisite,
    StrategyStage,
    StrategyStageOutcome,
    StrategyStageResult,
    StrategyStageStatus,
    StrategyStatusResponse,
)

ALL_STAGES = list(StrategyStage)


def _status(
    stage: StrategyStage,
    *,
    generated: bool = False,
    version: int | None = None,
    approved: bool = False,
    missing: list[StrategyPrerequisite] | None = None,
) -> StrategyStageStatus:
    """Build one internally consistent stage status.

    A generated stage is a draft unless ``approved`` is set; a stage that
    is not generated has no approval state.
    """

    missing = missing or []
    approval = None
    if generated:
        approval = StageApprovalState.APPROVED if approved else StageApprovalState.DRAFT
    return StrategyStageStatus(
        stage=stage,
        generated=generated,
        version=version if version is not None else (1 if generated else None),
        approval=approval,
        can_generate=not missing,
        missing_prerequisites=missing,
    )


def _status_response(generated_count: int, approved_count: int = 0) -> StrategyStatusResponse:
    """Build a response where the first N stages are generated and the first M approved."""

    return StrategyStatusResponse(
        workspace_id=uuid4(),
        stages=[
            _status(
                stage,
                generated=index < generated_count,
                approved=index < approved_count,
            )
            for index, stage in enumerate(ALL_STAGES)
        ],
    )


def _result(stage: StrategyStage, outcome: StrategyStageOutcome) -> StrategyStageResult:
    """Build one internally consistent stage result."""

    if outcome in {StrategyStageOutcome.GENERATED, StrategyStageOutcome.SKIPPED}:
        return StrategyStageResult(stage=stage, outcome=outcome, version=1)
    if outcome is StrategyStageOutcome.FAILED:
        return StrategyStageResult(stage=stage, outcome=outcome, error="Generation failed.")
    return StrategyStageResult(stage=stage, outcome=outcome)


def _run(outcomes: list[StrategyStageOutcome], *, force: bool = False) -> StrategyGenerateResponse:
    """Build a run result from one outcome per stage, in chain order."""

    return StrategyGenerateResponse(
        workspace_id=uuid4(),
        force_regenerate=force,
        stages=[
            _result(stage, outcome) for stage, outcome in zip(ALL_STAGES, outcomes, strict=True)
        ],
    )


GEN = StrategyStageOutcome.GENERATED
SKIP = StrategyStageOutcome.SKIPPED
FAIL = StrategyStageOutcome.FAILED
NOPE = StrategyStageOutcome.NOT_ATTEMPTED


def test_stage_order_is_the_generation_order() -> None:
    """Enum order is the contract every response and the orchestrator relies on."""

    assert [stage.value for stage in ALL_STAGES] == [
        "business_understanding",
        "market_research",
        "competitor_analysis",
        "customer_personas",
        "brand_strategy",
        "marketing_strategy",
        "content_plan",
    ]


def test_prerequisite_enum_is_business_profile_plus_every_stage() -> None:
    """A stage name must always be usable as a prerequisite name."""

    assert {item.value for item in StrategyPrerequisite} == {
        "business_profile",
        *(stage.value for stage in StrategyStage),
    }


def test_generate_request_defaults_to_not_forcing_regeneration() -> None:
    """A full run is resumable and idempotent unless the caller opts out."""

    assert StrategyGenerateRequest().force_regenerate is False


def test_generate_request_rejects_unknown_fields() -> None:
    """Typos in the request body should fail loudly, not be ignored."""

    with pytest.raises(ValidationError):
        StrategyGenerateRequest.model_validate({"force_regenerat": True})


def test_stage_status_accepts_generated_stage() -> None:
    """A generated stage reports its version and can still be regenerated."""

    status = _status(StrategyStage.MARKET_RESEARCH, generated=True, version=3)

    assert status.version == 3
    assert status.can_generate is True


def test_stage_status_accepts_blocked_stage() -> None:
    """A stage with missing prerequisites lists them and cannot be generated."""

    status = _status(
        StrategyStage.BRAND_STRATEGY,
        missing=[StrategyPrerequisite.CUSTOMER_PERSONAS],
    )

    assert status.can_generate is False
    assert status.missing_prerequisites == [StrategyPrerequisite.CUSTOMER_PERSONAS]


@pytest.mark.parametrize(
    "overrides",
    [
        {"generated": True, "version": None},
        {"generated": False, "version": 1},
        {"generated": True, "version": 0},
        {"generated": True, "version": 1, "approval": None},
        {"generated": False, "version": None, "approval": "draft"},
        {"generated": False, "version": None, "approval": "approved"},
        {"generated": True, "version": 1, "approval": "pending_review"},
        {"can_generate": False, "missing_prerequisites": []},
        {"can_generate": True, "missing_prerequisites": [StrategyPrerequisite.BUSINESS_PROFILE]},
        {
            "can_generate": False,
            "missing_prerequisites": [
                StrategyPrerequisite.BUSINESS_PROFILE,
                StrategyPrerequisite.BUSINESS_PROFILE,
            ],
        },
        {
            "can_generate": False,
            "missing_prerequisites": [StrategyPrerequisite.CONTENT_PLAN],
        },
    ],
)
def test_stage_status_rejects_inconsistent_state(overrides: dict[str, Any]) -> None:
    """Contradictory status fields should never reach a client."""

    fields: dict[str, Any] = {
        "stage": StrategyStage.CONTENT_PLAN,
        "generated": False,
        "version": None,
        "can_generate": True,
        "missing_prerequisites": [],
    }
    fields.update(overrides)

    with pytest.raises(ValidationError):
        StrategyStageStatus.model_validate(fields)


def test_stage_status_rejects_unknown_stage_and_prerequisite() -> None:
    """Only known stage and prerequisite names are valid."""

    with pytest.raises(ValidationError):
        StrategyStageStatus.model_validate(
            {"stage": "seo_audit", "generated": False, "can_generate": True}
        )
    with pytest.raises(ValidationError):
        StrategyStageStatus.model_validate(
            {
                "stage": "content_plan",
                "generated": False,
                "can_generate": False,
                "missing_prerequisites": ["seo_audit"],
            }
        )


def test_stage_status_reports_draft_and_approved_generated_stages() -> None:
    """A generated stage is either awaiting review or approved - never unreported."""

    draft = _status(StrategyStage.MARKET_RESEARCH, generated=True)
    approved = _status(StrategyStage.MARKET_RESEARCH, generated=True, approved=True)

    assert draft.approval is StageApprovalState.DRAFT
    assert approved.approval is StageApprovalState.APPROVED


def test_stage_status_has_no_approval_before_generation() -> None:
    """There is nothing to approve until a stage has been generated."""

    assert _status(StrategyStage.MARKET_RESEARCH).approval is None


def test_approval_states_are_draft_and_approved() -> None:
    """The two states clients see are fixed values."""

    assert [state.value for state in StageApprovalState] == ["draft", "approved"]


def test_status_response_derives_progress_for_empty_chain() -> None:
    """Nothing generated: not complete, and the first stage is next."""

    response = _status_response(0)

    assert response.complete is False
    assert response.next_stage is StrategyStage.BUSINESS_UNDERSTANDING


def test_status_response_derives_progress_for_partial_chain() -> None:
    """The next stage is the first ungenerated one in chain order."""

    response = _status_response(4)

    assert response.complete is False
    assert response.next_stage is StrategyStage.BRAND_STRATEGY


def test_status_response_derives_progress_for_full_chain() -> None:
    """Everything generated: complete, and nothing is next."""

    response = _status_response(len(ALL_STAGES))

    assert response.complete is True
    assert response.next_stage is None


def test_status_response_nothing_to_approve_when_nothing_generated() -> None:
    """With no generated stages there is nothing awaiting review and nothing approved."""

    response = _status_response(0)

    assert response.approved is False
    assert response.next_to_approve is None


def test_status_response_next_to_approve_is_first_draft_in_chain_order() -> None:
    """Review proceeds in chain order, so the first draft stage is the one to review."""

    response = _status_response(5, approved_count=2)

    assert response.approved is False
    assert response.next_to_approve is StrategyStage.COMPETITOR_ANALYSIS


def test_status_response_next_to_approve_skips_ungenerated_stages() -> None:
    """A stage that does not exist yet is never offered for approval."""

    response = _status_response(3, approved_count=3)

    assert response.next_to_approve is None
    assert response.approved is False
    assert response.next_stage is StrategyStage.CUSTOMER_PERSONAS


def test_status_response_approved_only_when_every_stage_is_approved() -> None:
    """Fully approved means all seven stages exist and none is a draft."""

    assert _status_response(len(ALL_STAGES), approved_count=len(ALL_STAGES) - 1).approved is False
    full = _status_response(len(ALL_STAGES), approved_count=len(ALL_STAGES))

    assert full.approved is True
    assert full.next_to_approve is None


def test_status_response_serializes_approval_fields() -> None:
    """Clients receive each stage's approval state and the derived review fields."""

    payload = _status_response(3, approved_count=1).model_dump(mode="json")

    assert [item["approval"] for item in payload["stages"]] == [
        "approved",
        "draft",
        "draft",
        None,
        None,
        None,
        None,
    ]
    assert payload["approved"] is False
    assert payload["next_to_approve"] == "market_research"


def test_status_response_serializes_derived_fields() -> None:
    """Clients receive complete and next_stage without recomputing them."""

    payload = _status_response(2).model_dump(mode="json")

    assert payload["complete"] is False
    assert payload["next_stage"] == "competitor_analysis"
    assert [item["stage"] for item in payload["stages"]] == [s.value for s in ALL_STAGES]


@pytest.mark.parametrize(
    "stages",
    [
        ALL_STAGES[:-1],
        [*ALL_STAGES, StrategyStage.CONTENT_PLAN],
        list(reversed(ALL_STAGES)),
        [ALL_STAGES[1], ALL_STAGES[0], *ALL_STAGES[2:]],
        [],
    ],
)
def test_status_response_requires_every_stage_once_in_order(stages: list[StrategyStage]) -> None:
    """A dropped, duplicated, or reordered stage is a service bug, not a response."""

    with pytest.raises(ValidationError):
        StrategyStatusResponse(workspace_id=uuid4(), stages=[_status(stage) for stage in stages])


@pytest.mark.parametrize(
    ("outcome", "fields"),
    [
        (GEN, {"version": 2}),
        (SKIP, {"version": 1}),
        (FAIL, {"error": "The language model call failed."}),
        (NOPE, {}),
    ],
)
def test_stage_result_accepts_each_valid_outcome(
    outcome: StrategyStageOutcome, fields: dict[str, Any]
) -> None:
    """Each outcome carries exactly the fields that make sense for it."""

    result = StrategyStageResult(stage=StrategyStage.MARKET_RESEARCH, outcome=outcome, **fields)

    assert result.outcome is outcome


@pytest.mark.parametrize(
    ("outcome", "fields"),
    [
        (GEN, {}),
        (SKIP, {}),
        (GEN, {"version": 1, "error": "unexpected"}),
        (FAIL, {}),
        (FAIL, {"error": "boom", "version": 1}),
        (NOPE, {"version": 1}),
        (NOPE, {"error": "boom"}),
        (GEN, {"version": 0}),
        (FAIL, {"error": ""}),
        (FAIL, {"error": "x" * 501}),
    ],
)
def test_stage_result_rejects_inconsistent_state(
    outcome: StrategyStageOutcome, fields: dict[str, Any]
) -> None:
    """Version and error must match the outcome."""

    with pytest.raises(ValidationError):
        StrategyStageResult(stage=StrategyStage.MARKET_RESEARCH, outcome=outcome, **fields)


def test_generate_response_accepts_full_success() -> None:
    """Every stage generated: complete, no failed stage."""

    response = _run([GEN] * 7)

    assert response.complete is True
    assert response.failed_stage is None


def test_generate_response_accepts_resumed_run_with_skips() -> None:
    """A retry skips what already exists and generates the rest."""

    response = _run([SKIP, SKIP, SKIP, GEN, GEN, GEN, GEN])

    assert response.complete is True


def test_generate_response_accepts_forced_run() -> None:
    """A forced run regenerates every stage."""

    assert _run([GEN] * 7, force=True).complete is True


def test_generate_response_accepts_mid_chain_failure() -> None:
    """Prefix of successes, one failure, then everything else not attempted."""

    response = _run([SKIP, GEN, GEN, FAIL, NOPE, NOPE, NOPE])

    assert response.complete is False
    assert response.failed_stage is StrategyStage.CUSTOMER_PERSONAS


def test_generate_response_accepts_failure_on_first_and_last_stage() -> None:
    """The boundaries of the chain are valid failure points."""

    assert _run([FAIL, NOPE, NOPE, NOPE, NOPE, NOPE, NOPE]).failed_stage is (
        StrategyStage.BUSINESS_UNDERSTANDING
    )
    assert _run([GEN, GEN, GEN, GEN, GEN, GEN, FAIL]).failed_stage is StrategyStage.CONTENT_PLAN


def test_generate_response_serializes_derived_fields() -> None:
    """Clients receive complete and failed_stage without recomputing them."""

    payload = _run([GEN, GEN, FAIL, NOPE, NOPE, NOPE, NOPE]).model_dump(mode="json")

    assert payload["complete"] is False
    assert payload["failed_stage"] == "competitor_analysis"
    assert payload["stages"][2] == {
        "stage": "competitor_analysis",
        "outcome": "failed",
        "version": None,
        "error": "Generation failed.",
    }


@pytest.mark.parametrize(
    "outcomes",
    [
        [GEN, GEN, NOPE, NOPE, NOPE, NOPE, NOPE],
        [GEN, GEN, GEN, FAIL, GEN, NOPE, NOPE],
        [GEN, GEN, GEN, FAIL, NOPE, FAIL, NOPE],
        [GEN, FAIL, GEN, NOPE, NOPE, NOPE, NOPE],
        [NOPE, GEN, GEN, GEN, GEN, GEN, FAIL],
        [GEN, GEN, GEN, GEN, GEN, GEN, NOPE],
    ],
)
def test_generate_response_rejects_impossible_run_shapes(
    outcomes: list[StrategyStageOutcome],
) -> None:
    """A run cannot skip past a failure or leave stages unattempted without one."""

    with pytest.raises(ValidationError):
        _run(outcomes)


def test_generate_response_rejects_skipped_stages_in_a_forced_run() -> None:
    """Force regenerates everything, so nothing can have been skipped."""

    with pytest.raises(ValidationError):
        _run([SKIP, GEN, GEN, GEN, GEN, GEN, GEN], force=True)


@pytest.mark.parametrize(
    "stages",
    [
        ALL_STAGES[:-1],
        list(reversed(ALL_STAGES)),
        [ALL_STAGES[0], ALL_STAGES[0], *ALL_STAGES[2:]],
    ],
)
def test_generate_response_requires_every_stage_once_in_order(
    stages: list[StrategyStage],
) -> None:
    """The result must cover the whole chain in order, even for a short run."""

    with pytest.raises(ValidationError):
        StrategyGenerateResponse(
            workspace_id=uuid4(),
            force_regenerate=False,
            stages=[_result(stage, GEN) for stage in stages],
        )
