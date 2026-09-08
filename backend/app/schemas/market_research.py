"""Market Research agent request and response schemas."""

from datetime import datetime
from uuid import UUID

from backend.app.schemas.common import OrmSchema, StrictSchema


class MarketResearchGenerateRequest(StrictSchema):
    """Optional parameters when triggering Market Research generation."""

    force_regenerate: bool = False


class MarketResearchSourceSchema(StrictSchema):
    """One source actually used by a section of the report.

    Built server-side from the CRAG graph's own relevant_documents for
    that section - never asked of the model - so a citation appearing
    here always corresponds to something genuinely retrieved.
    """

    section: str
    source: str
    snippet: str


class MarketResearchResponse(OrmSchema):
    """Public representation of the Market Research agent's output."""

    id: UUID
    workspace_id: UUID

    market_overview: str
    target_customer_segments: str
    competitive_landscape: str
    opportunities_and_risks: str
    sources: list[MarketResearchSourceSchema]

    model_used: str
    input_tokens: int
    output_tokens: int

    version: int
    created_at: datetime
    updated_at: datetime
