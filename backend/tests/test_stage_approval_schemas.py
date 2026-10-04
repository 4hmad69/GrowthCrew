"""Unit tests for stage approval request and response schemas."""

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest
from pydantic import ValidationError

from backend.app.db.models.stage_approval import STAGE_APPROVAL_SOURCES, StageApproval
from backend.app.schemas.stage_approval import (
    ApprovalSource,
    StageApprovalRequest,
    StageApprovalResponse,
)
from backend.app.schemas.strategy_orchestration import StrategyStage


def _response_payload(**overrides: Any) -> dict[str, Any]:
    """Build a valid response payload, with optional field overrides."""

    payload: dict[str, Any] = {
        "workspace_id": uuid4(),
        "stage": StrategyStage.MARKET_RESEARCH,
        "record_id": uuid4(),
        "approved_version": 2,
        "source": ApprovalSource.HUMAN,
        "approved_at": datetime(2026, 10, 3, 12, 0, tzinfo=UTC),
    }
    payload.update(overrides)
    return payload


def test_approval_sources_match_the_database_constraint() -> None:
    """The enum and the table's source CHECK values must never drift apart."""

    assert tuple(source.value for source in ApprovalSource) == STAGE_APPROVAL_SOURCES


def test_request_accepts_a_positive_integer_version() -> None:
    """A real integer version is the one thing an approval request needs."""

    assert StageApprovalRequest(version=1).version == 1
    assert StageApprovalRequest.model_validate({"version": 7}).version == 7


@pytest.mark.parametrize("version", [0, -1, -100])
def test_request_rejects_non_positive_versions(version: int) -> None:
    """Record versions start at 1, so anything lower can never match."""

    with pytest.raises(ValidationError):
        StageApprovalRequest(version=version)


@pytest.mark.parametrize("version", [True, False, "3", 2.0, None])
def test_request_rejects_values_that_are_not_real_integers(version: Any) -> None:
    """The version is a concurrency token - no coercion from bools, strings, or floats."""

    with pytest.raises(ValidationError):
        StageApprovalRequest.model_validate({"version": version})


def test_request_requires_a_version() -> None:
    """An empty body must not be read as an approval of some default version."""

    with pytest.raises(ValidationError):
        StageApprovalRequest.model_validate({})


def test_request_rejects_unknown_fields() -> None:
    """A client cannot smuggle in fields such as the approval source."""

    with pytest.raises(ValidationError):
        StageApprovalRequest.model_validate({"version": 1, "source": "auto"})


def test_response_accepts_a_valid_payload() -> None:
    """A fully specified approval round-trips through the schema."""

    payload = _response_payload()

    response = StageApprovalResponse(**payload)

    assert response.stage is StrategyStage.MARKET_RESEARCH
    assert response.source is ApprovalSource.HUMAN
    assert response.approved_version == 2


def test_response_validates_from_an_orm_object() -> None:
    """Responses are built straight from the ORM row, with strings coerced to enums."""

    row = StageApproval(
        workspace_id=uuid4(),
        stage="brand_strategy",
        record_id=uuid4(),
        approved_version=4,
        source="auto",
        approved_at=datetime(2026, 10, 3, 12, 0, tzinfo=UTC),
    )

    response = StageApprovalResponse.model_validate(row)

    assert response.stage is StrategyStage.BRAND_STRATEGY
    assert response.source is ApprovalSource.AUTO
    assert response.record_id == row.record_id
    assert response.approved_version == 4


def test_response_serializes_enums_as_plain_strings() -> None:
    """The JSON a client sees uses the stage and source values, not enum reprs."""

    data = StageApprovalResponse(**_response_payload()).model_dump(mode="json")

    assert data["stage"] == "market_research"
    assert data["source"] == "human"
    assert isinstance(data["workspace_id"], str)
    assert isinstance(data["record_id"], str)
    assert isinstance(data["approved_at"], str)


@pytest.mark.parametrize(
    "override",
    [
        {"stage": "business_profile"},
        {"stage": "not_a_stage"},
        {"source": "robot"},
        {"approved_version": 0},
        {"approved_version": -1},
        {"record_id": "not-a-uuid"},
    ],
)
def test_response_rejects_invalid_values(override: dict[str, Any]) -> None:
    """Anything outside the stage, source, or version rules must be refused."""

    with pytest.raises(ValidationError):
        StageApprovalResponse(**_response_payload(**override))


def test_response_rejects_unknown_fields() -> None:
    """Responses are strict too, so a stray field cannot leak through unnoticed."""

    with pytest.raises(ValidationError):
        StageApprovalResponse(**_response_payload(extra_field="nope"))


@pytest.mark.parametrize("stage", list(StrategyStage))
def test_response_accepts_every_chain_stage(stage: StrategyStage) -> None:
    """Each of the seven stages is a valid approval target."""

    assert StageApprovalResponse(**_response_payload(stage=stage)).stage is stage
