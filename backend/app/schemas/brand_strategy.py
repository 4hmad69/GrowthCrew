"""Brand Strategy agent request and response schemas."""

from datetime import datetime
from uuid import UUID

from backend.app.schemas.common import OrmSchema, StrictSchema


class BrandStrategyGenerateRequest(StrictSchema):
    """Optional parameters when triggering Brand Strategy generation."""

    force_regenerate: bool = False


class BrandStrategyResponse(OrmSchema):
    """Public representation of the Brand Strategy agent's output.

    Flat string fields plus two short string lists - unlike Customer
    Personas, no nested object schema is needed here since brand_pillars
    and tagline_options are plain list[str], not list[dict]. Neither
    list field carries min_length, matching every other LLM-facing list
    field in this codebase: the deterministic "local" provider fills
    every list-typed field with [].
    """

    id: UUID
    workspace_id: UUID

    overview: str
    positioning_statement: str
    value_proposition: str
    brand_voice_and_tone: str
    brand_pillars: list[str]
    tagline_options: list[str]

    model_used: str
    input_tokens: int
    output_tokens: int

    version: int
    created_at: datetime
    updated_at: datetime
