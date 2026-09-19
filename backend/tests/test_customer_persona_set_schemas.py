"""Unit tests for Customer Personas request and response schemas."""

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest
from pydantic import ValidationError

from backend.app.db.models.customer_persona_set import CustomerPersonaSet
from backend.app.schemas.customer_persona_set import (
    CustomerPersonaSetGenerateRequest,
    CustomerPersonaSetResponse,
    PersonaSchema,
)


def _persona_payload() -> dict[str, Any]:
    """Return a complete, valid persona as it is stored in the JSONB column."""

    return {
        "name": "Budget-conscious Bilal",
        "segment": "Price-sensitive first-time buyers",
        "summary": "Compares three options before committing.",
        "demographics": "25-34, urban, mid-income",
        "goals": ["Save money", "Avoid a bad purchase"],
        "pain_points": ["Hidden fees"],
        "buying_triggers": ["Seasonal discount"],
        "objections": ["Unproven brand"],
        "preferred_channels": ["Instagram", "WhatsApp"],
    }


def _persona_set(personas: list[dict[str, Any]]) -> CustomerPersonaSet:
    """Build an in-memory ORM record - no session or database involved."""

    now = datetime.now(UTC)
    return CustomerPersonaSet(
        id=uuid4(),
        workspace_id=uuid4(),
        overview="Three personas, primary first.",
        personas=personas,
        model_used="test-model",
        input_tokens=11,
        output_tokens=22,
        version=1,
        created_at=now,
        updated_at=now,
    )


def test_generate_request_defaults_to_not_forcing_regeneration() -> None:
    """Generation is idempotent unless the caller explicitly opts out."""

    assert CustomerPersonaSetGenerateRequest().force_regenerate is False


def test_generate_request_rejects_unknown_fields() -> None:
    """Typos in the request body should fail loudly, not be ignored."""

    with pytest.raises(ValidationError):
        CustomerPersonaSetGenerateRequest.model_validate({"force_regenerat": True})


def test_response_validates_from_orm_record_with_jsonb_personas() -> None:
    """Stored JSONB dicts should come back out as typed PersonaSchema objects."""

    response = CustomerPersonaSetResponse.model_validate(_persona_set([_persona_payload()]))

    assert len(response.personas) == 1
    assert isinstance(response.personas[0], PersonaSchema)
    assert response.personas[0].name == "Budget-conscious Bilal"
    assert response.personas[0].preferred_channels == ["Instagram", "WhatsApp"]
    assert response.input_tokens == 11
    assert response.output_tokens == 22


def test_response_accepts_an_empty_persona_list() -> None:
    """The deterministic local provider produces [] - that must stay valid."""

    response = CustomerPersonaSetResponse.model_validate(_persona_set([]))

    assert response.personas == []


def test_persona_rejects_missing_required_field() -> None:
    """A stored persona missing a field is corrupt data, not a soft default."""

    payload = _persona_payload()
    del payload["name"]

    with pytest.raises(ValidationError):
        PersonaSchema.model_validate(payload)


def test_persona_rejects_unexpected_field() -> None:
    """Drift between stored JSON and the schema should surface immediately."""

    payload = _persona_payload()
    payload["unexpected"] = "value"

    with pytest.raises(ValidationError):
        PersonaSchema.model_validate(payload)
