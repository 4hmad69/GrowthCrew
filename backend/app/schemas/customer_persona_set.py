"""Customer Personas agent request and response schemas."""

from datetime import datetime
from uuid import UUID

from backend.app.schemas.common import OrmSchema, StrictSchema


class CustomerPersonaSetGenerateRequest(StrictSchema):
    """Optional parameters when triggering Customer Personas generation."""

    force_regenerate: bool = False


class PersonaSchema(StrictSchema):
    """One structured customer persona.

    Deliberately plain types with no length constraints, matching
    ContentCalendarEntrySchema: this is the read-side shape of data
    already validated by the LLM-facing draft schema in the service
    layer. The list fields here never carry min_length because the
    deterministic "local" LLM provider fills every list with [].
    """

    name: str
    segment: str
    summary: str
    demographics: str
    goals: list[str]
    pain_points: list[str]
    buying_triggers: list[str]
    objections: list[str]
    preferred_channels: list[str]


class CustomerPersonaSetResponse(OrmSchema):
    """Public representation of the Customer Personas agent's output."""

    id: UUID
    workspace_id: UUID

    overview: str
    personas: list[PersonaSchema]

    model_used: str
    input_tokens: int
    output_tokens: int

    version: int
    created_at: datetime
    updated_at: datetime
