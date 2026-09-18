"""Content Planning agent request and response schemas."""

from datetime import datetime
from uuid import UUID

from backend.app.schemas.common import OrmSchema, StrictSchema


class ContentPlanGenerateRequest(StrictSchema):
    """Optional parameters when triggering Content Planning generation."""

    force_regenerate: bool = False


class ContentCalendarEntrySchema(StrictSchema):
    """One scheduled item in the 30-day content calendar."""

    day: int
    week: int
    channel: str
    content_type: str
    pillar: str
    topic: str
    cta: str


class ContentPlanResponse(OrmSchema):
    """Public representation of the Content Planning agent's output."""

    id: UUID
    workspace_id: UUID

    overview: str
    entries: list[ContentCalendarEntrySchema]

    model_used: str
    input_tokens: int
    output_tokens: int

    version: int
    created_at: datetime
    updated_at: datetime
