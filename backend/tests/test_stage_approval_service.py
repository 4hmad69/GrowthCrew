"""Unit tests for the pure rules behind stage approval.

No database here: the prerequisite-approval check, the client-facing
messages, the "is this approval still current" rule, and how the new
exception reaches a client. The service's behavior against real records
is covered in tests/integration/test_stage_approval_service_integration.py.
"""

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.db.models.stage_approval import StageApproval
from backend.app.errors import handle_domain_error
from backend.app.exceptions import (
    ApprovalRequiredError,
    DomainError,
    ResourceConflictError,
)
from backend.app.schemas.strategy_orchestration import StrategyPrerequisite, StrategyStage
from backend.app.services.stage_approval import (
    STAGE_LABELS,
    approval_required_message,
)
from backend.app.services.strategy_records import (
    StageRecordRef,
    StrategyRecords,
    is_current,
)
from backend.app.services.strategy_status import (
    STAGE_PREREQUISITES,
    unapproved_prerequisites,
)

S = StrategyStage
ALL_STAGES = list(StrategyStage)


def _approval(record_id: object, version: int) -> StageApproval:
    """Build an approval row without touching a database."""

    return StageApproval(
        workspace_id=uuid4(),
        stage="market_research",
        record_id=record_id,
        approved_version=version,
        source="human",
        approved_at=datetime(2026, 10, 3, tzinfo=UTC),
    )


# --- unapproved_prerequisites ------------------------------------------------


@pytest.mark.parametrize(
    "stage", [S.BUSINESS_UNDERSTANDING, S.MARKET_RESEARCH, S.COMPETITOR_ANALYSIS]
)
def test_stages_that_only_need_the_profile_never_wait_for_approval(stage: S) -> None:
    """The profile is not a generated stage, so these three are never approval-gated."""

    assert unapproved_prerequisites(stage, set()) == []


@pytest.mark.parametrize("stage", ALL_STAGES)
def test_with_nothing_approved_every_stage_prerequisite_is_unapproved(stage: S) -> None:
    """Nothing approved means the whole stage-prerequisite list is outstanding, in order."""

    expected = [
        StrategyStage(item.value)
        for item in STAGE_PREREQUISITES[stage]
        if item is not StrategyPrerequisite.BUSINESS_PROFILE
    ]

    assert unapproved_prerequisites(stage, set()) == expected


@pytest.mark.parametrize("stage", ALL_STAGES)
def test_with_everything_approved_nothing_is_outstanding(stage: S) -> None:
    """Approving every stage clears every gate."""

    assert unapproved_prerequisites(stage, set(ALL_STAGES)) == []


def test_only_the_missing_approvals_are_reported() -> None:
    """Marketing strategy needs five stages approved; report just the ones still draft."""

    approved = {S.BUSINESS_UNDERSTANDING, S.MARKET_RESEARCH, S.COMPETITOR_ANALYSIS}

    assert unapproved_prerequisites(S.MARKETING_STRATEGY, approved) == [
        S.CUSTOMER_PERSONAS,
        S.BRAND_STRATEGY,
    ]


def test_approving_an_unrelated_stage_does_not_unblock_anything() -> None:
    """Content plan builds on marketing strategy alone; other approvals are irrelevant."""

    approved = set(ALL_STAGES) - {S.MARKETING_STRATEGY}

    assert unapproved_prerequisites(S.CONTENT_PLAN, approved) == [S.MARKETING_STRATEGY]


def test_a_stage_is_never_its_own_prerequisite() -> None:
    """Approving a stage must not be a precondition of generating that same stage."""

    for stage in ALL_STAGES:
        assert stage not in unapproved_prerequisites(stage, set())


# --- messages ----------------------------------------------------------------


def test_every_stage_has_a_label() -> None:
    """A stage without a label would crash the error message that needs it."""

    assert set(STAGE_LABELS) == set(StrategyStage)


def test_message_for_a_single_unapproved_stage_is_singular() -> None:
    """One blocker reads as 'the X stage', not 'stages'."""

    message = approval_required_message(S.CONTENT_PLAN, [S.MARKETING_STRATEGY])

    assert message == (
        "Approve the marketing strategy stage before generating the content plan stage."
    )


def test_message_for_two_unapproved_stages_uses_and() -> None:
    """Two blockers are joined with 'and' and the noun becomes plural."""

    message = approval_required_message(
        S.CUSTOMER_PERSONAS,
        [S.BUSINESS_UNDERSTANDING, S.MARKET_RESEARCH],
    )

    assert message == (
        "Approve the business understanding and market research stages "
        "before generating the customer personas stage."
    )


def test_message_for_three_unapproved_stages_uses_commas_then_and() -> None:
    """Three blockers read as a natural list."""

    message = approval_required_message(
        S.BRAND_STRATEGY,
        [S.BUSINESS_UNDERSTANDING, S.COMPETITOR_ANALYSIS, S.CUSTOMER_PERSONAS],
    )

    assert message == (
        "Approve the business understanding, competitor analysis and customer personas "
        "stages before generating the brand strategy stage."
    )


# --- is_current and StrategyRecords ------------------------------------------


def test_approval_is_current_only_for_the_same_record_and_version() -> None:
    """Both the record id and the version must match."""

    record_id = uuid4()
    approval = _approval(record_id, 3)

    assert is_current(approval, StageRecordRef(id=record_id, version=3)) is True
    assert is_current(approval, StageRecordRef(id=record_id, version=4)) is False
    assert is_current(approval, StageRecordRef(id=uuid4(), version=3)) is False
    assert is_current(approval, StageRecordRef(id=uuid4(), version=4)) is False


def test_a_recreated_record_at_the_same_version_is_not_approved() -> None:
    """Delete-and-regenerate restarts at version 1; only the id can tell it apart."""

    approved_record = uuid4()
    recreated = StageRecordRef(id=uuid4(), version=1)

    assert is_current(_approval(approved_record, 1), recreated) is False


def test_approved_stages_lists_the_stages_with_a_current_approval() -> None:
    """approved_stages is exactly the keys of the current approvals."""

    records = StrategyRecords(
        has_profile=True,
        refs={},
        approvals={S.MARKET_RESEARCH: _approval(uuid4(), 1)},
    )

    assert records.approved_stages == frozenset({S.MARKET_RESEARCH})


# --- ApprovalRequiredError ---------------------------------------------------


def test_approval_required_error_is_a_conflict_carrying_what_blocks() -> None:
    """It must be a conflict (so it maps to 409) and expose the blocking stages."""

    error = ApprovalRequiredError(
        "Approve the market research stage before generating the customer personas stage.",
        stage="customer_personas",
        unapproved=["market_research"],
    )

    assert isinstance(error, ResourceConflictError)
    assert isinstance(error, DomainError)
    assert error.stage == "customer_personas"
    assert error.unapproved == ("market_research",)
    assert str(error).startswith("Approve the market research stage")


def test_approval_required_error_reaches_clients_as_409_with_its_message() -> None:
    """Through the real domain-error handler it becomes a 409 with a safe detail."""

    app = FastAPI()
    app.add_exception_handler(DomainError, handle_domain_error)

    @app.get("/blocked")
    def blocked() -> None:
        raise ApprovalRequiredError(
            "Approve the market research stage before generating the customer personas stage.",
            stage="customer_personas",
            unapproved=["market_research"],
        )

    response = TestClient(app).get("/blocked")

    assert response.status_code == 409
    assert response.json() == {
        "detail": "Approve the market research stage before generating the customer personas stage."
    }
