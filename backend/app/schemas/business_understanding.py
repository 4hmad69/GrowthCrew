"""Business Understanding agent request and response schemas."""

from datetime import datetime
from uuid import UUID

from backend.app.schemas.common import OrmSchema, StrictSchema


class BusinessUnderstandingGenerateRequest(StrictSchema):
    """Optional parameters when triggering Business Understanding generation."""

    force_regenerate: bool = False


class BusinessUnderstandingResponse(OrmSchema):
    """Public representation of the Business Understanding agent's output."""

    id: UUID
    profile_id: UUID

    summary: str
    inferred_business_stage: str
    competitive_category: str
    key_differentiators: list[str]
    likely_customer_pain_points: list[str]
    confidence_notes: str | None

    model_used: str
    input_tokens: int
    output_tokens: int

    version: int
    created_at: datetime
    updated_at: datetime
