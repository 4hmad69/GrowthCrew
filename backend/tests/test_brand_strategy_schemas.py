"""Unit tests for Brand Strategy request and response schemas."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from backend.app.db.models.brand_strategy import BrandStrategy
from backend.app.schemas.brand_strategy import (
    BrandStrategyGenerateRequest,
    BrandStrategyResponse,
)


def _brand_strategy(
    brand_pillars: list[str],
    tagline_options: list[str],
) -> BrandStrategy:
    """Build an in-memory ORM record - no session or database involved."""

    now = datetime.now(UTC)
    return BrandStrategy(
        id=uuid4(),
        workspace_id=uuid4(),
        overview="Positioned as the reliable, budget-friendly choice.",
        positioning_statement=(
            "For price-sensitive first-time buyers, Acme is the booking platform "
            "that guarantees the lowest fare, because it price-matches daily."
        ),
        value_proposition="Book with confidence: the lowest fare, guaranteed, every time.",
        brand_voice_and_tone="Warm, plain-spoken, and reassuring - never salesy.",
        brand_pillars=brand_pillars,
        tagline_options=tagline_options,
        model_used="test-model",
        input_tokens=11,
        output_tokens=22,
        version=1,
        created_at=now,
        updated_at=now,
    )


def test_generate_request_defaults_to_not_forcing_regeneration() -> None:
    """Generation is idempotent unless the caller explicitly opts out."""

    assert BrandStrategyGenerateRequest().force_regenerate is False


def test_generate_request_rejects_unknown_fields() -> None:
    """Typos in the request body should fail loudly, not be ignored."""

    with pytest.raises(ValidationError):
        BrandStrategyGenerateRequest.model_validate({"force_regenerat": True})


def test_response_validates_from_orm_record() -> None:
    """Stored fields should come back out through the response schema unchanged."""

    response = BrandStrategyResponse.model_validate(
        _brand_strategy(
            brand_pillars=["Reliability", "Transparency", "Value"],
            tagline_options=["Fly sure.", "The fare you see is the fare you pay."],
        )
    )

    assert response.brand_pillars == ["Reliability", "Transparency", "Value"]
    assert response.tagline_options == ["Fly sure.", "The fare you see is the fare you pay."]
    assert response.input_tokens == 11
    assert response.output_tokens == 22


def test_response_accepts_empty_pillars_and_taglines() -> None:
    """The deterministic local provider produces [] - that must stay valid."""

    response = BrandStrategyResponse.model_validate(
        _brand_strategy(brand_pillars=[], tagline_options=[])
    )

    assert response.brand_pillars == []
    assert response.tagline_options == []


def test_response_rejects_missing_required_field() -> None:
    """A response missing a required field is corrupt data, not a soft default."""

    payload = BrandStrategyResponse.model_validate(
        _brand_strategy(brand_pillars=["Trust"], tagline_options=["Fly sure."])
    ).model_dump()
    del payload["positioning_statement"]

    with pytest.raises(ValidationError):
        BrandStrategyResponse.model_validate(payload)


def test_response_rejects_unexpected_field() -> None:
    """Drift between the ORM record and the schema should surface immediately."""

    payload = BrandStrategyResponse.model_validate(
        _brand_strategy(brand_pillars=["Trust"], tagline_options=["Fly sure."])
    ).model_dump()
    payload["unexpected"] = "value"

    with pytest.raises(ValidationError):
        BrandStrategyResponse.model_validate(payload)
