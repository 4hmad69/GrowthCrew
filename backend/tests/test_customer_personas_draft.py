"""Unit tests for the Customer Personas LLM-facing draft schemas.

These need no database: they pin the contract between the schema the LLM
is asked to fill (private to the service) and the schema the API returns.
"""

import pytest
from pydantic import ValidationError

from backend.app.config import Settings
from backend.app.llm.gateway import LLMGateway
from backend.app.schemas.customer_persona_set import PersonaSchema
from backend.app.services.customer_personas import (
    _CustomerPersonasDraft,
    _PersonaDraft,
)


def _persona_draft() -> _PersonaDraft:
    """Return a fully populated persona draft."""

    return _PersonaDraft(
        name="Budget-conscious Bilal",
        segment="Price-sensitive first-time buyers",
        summary="Compares three options before committing.",
        demographics="25-34, urban, mid-income",
        goals=["Save money"],
        pain_points=["Hidden fees"],
        buying_triggers=["Seasonal discount"],
        objections=["Unproven brand"],
        preferred_channels=["Instagram"],
    )


def test_persona_draft_and_response_schema_have_identical_fields() -> None:
    """The LLM-facing draft and the API schema must never drift apart."""

    assert set(_PersonaDraft.model_fields) == set(PersonaSchema.model_fields)


def test_persona_draft_dump_validates_as_the_response_schema() -> None:
    """What the service stores must be readable through the API schema."""

    stored = _persona_draft().model_dump()

    persona = PersonaSchema.model_validate(stored)

    assert persona.name == "Budget-conscious Bilal"
    assert persona.preferred_channels == ["Instagram"]


def test_local_provider_output_validates_against_the_draft() -> None:
    """The deterministic local stub must be able to fill the draft schema.

    LocalStructuredRunnable fills list fields with [] and strings with a
    short snippet of the prompt. A min_length on a list, or a max_length
    too small for that snippet, would make every local-backed Postgres
    test fail before reaching the service.
    """

    gateway = LLMGateway(Settings(environment="test", llm_provider="local"))

    draft, _usage = gateway.structured_with_usage(
        "Produce personas for a long enough prompt to give the stub real words to use.",
        _CustomerPersonasDraft,
    )

    assert draft.overview
    assert draft.personas == []


def test_draft_caps_the_number_of_personas() -> None:
    """The schema only bounds the worst case; the prompt asks for exactly 3."""

    with pytest.raises(ValidationError):
        _CustomerPersonasDraft(
            overview="Too many personas.",
            personas=[_persona_draft() for _ in range(6)],
        )


def test_draft_rejects_an_empty_overview() -> None:
    """An empty overview is a failed generation, not a valid one."""

    with pytest.raises(ValidationError):
        _CustomerPersonasDraft(overview="", personas=[])
