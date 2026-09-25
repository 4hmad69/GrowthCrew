"""Unit tests for the Brand Strategy LLM-facing draft schema.

These need no database: they pin the contract between the schema the LLM
is asked to fill (private to the service) and the schema the API returns.
"""

import pytest
from pydantic import ValidationError

from backend.app.config import Settings
from backend.app.llm.gateway import LLMGateway
from backend.app.schemas.brand_strategy import BrandStrategyResponse
from backend.app.services.brand_strategy import _BrandStrategyDraft

# Fields that only ever come from the LLM - excludes id/workspace_id/
# model_used/token counts/version/timestamps, which BrandStrategyResponse
# carries but the draft never produces.
_LLM_CONTENT_FIELDS = {
    "overview",
    "positioning_statement",
    "value_proposition",
    "brand_voice_and_tone",
    "brand_pillars",
    "tagline_options",
}


def _draft() -> _BrandStrategyDraft:
    """Return a fully populated brand strategy draft."""

    return _BrandStrategyDraft(
        overview="Positioned as the reliable, budget-friendly choice.",
        positioning_statement=(
            "For price-sensitive first-time buyers, Acme is the booking "
            "platform that guarantees the lowest fare, because it "
            "price-matches daily."
        ),
        value_proposition="Book with confidence: the lowest fare, guaranteed, every time.",
        brand_voice_and_tone="Warm, plain-spoken, and reassuring - never salesy.",
        brand_pillars=["Reliability", "Transparency", "Value"],
        tagline_options=["Fly sure.", "The fare you see is the fare you pay."],
    )


def test_draft_content_fields_match_the_response_schema() -> None:
    """The LLM-facing draft and the API schema must never drift apart."""

    assert set(_BrandStrategyDraft.model_fields) == _LLM_CONTENT_FIELDS
    assert _LLM_CONTENT_FIELDS.issubset(set(BrandStrategyResponse.model_fields))


def test_draft_fields_validate_as_the_response_schema() -> None:
    """What the service stores must be readable through the API schema."""

    draft = _draft().model_dump()

    stored = BrandStrategyResponse.model_validate(
        {
            **draft,
            "id": "00000000-0000-0000-0000-000000000000",
            "workspace_id": "00000000-0000-0000-0000-000000000000",
            "model_used": "test-model",
            "input_tokens": 1,
            "output_tokens": 1,
            "version": 1,
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
        }
    )

    assert stored.positioning_statement == draft["positioning_statement"]
    assert stored.brand_pillars == ["Reliability", "Transparency", "Value"]


def test_local_provider_output_validates_against_the_draft() -> None:
    """The deterministic local stub must be able to fill the draft schema.

    LocalStructuredRunnable fills list fields with [] and strings with a
    short snippet of the prompt. A min_length on a list, or a max_length
    too small for that snippet, would make every local-backed Postgres
    test fail before reaching the service.
    """

    gateway = LLMGateway(Settings(environment="test", llm_provider="local"))

    draft, _usage = gateway.structured_with_usage(
        "Produce a brand strategy for a long enough prompt to give the stub real words to use.",
        _BrandStrategyDraft,
    )

    assert draft.overview
    assert draft.positioning_statement
    assert draft.brand_pillars == []
    assert draft.tagline_options == []


def test_draft_caps_the_number_of_pillars() -> None:
    """The schema only bounds the worst case; the prompt asks for 3-5."""

    payload = _draft().model_dump()
    payload["brand_pillars"] = [f"Pillar {i}" for i in range(10)]

    with pytest.raises(ValidationError):
        _BrandStrategyDraft(**payload)


def test_draft_rejects_an_empty_positioning_statement() -> None:
    """An empty positioning statement is a failed generation, not a valid one."""

    payload = _draft().model_dump()
    payload["positioning_statement"] = ""

    with pytest.raises(ValidationError):
        _BrandStrategyDraft(**payload)
