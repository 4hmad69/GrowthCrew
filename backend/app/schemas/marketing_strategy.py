"""Marketing Strategy agent request and response schemas."""

from datetime import datetime
from uuid import UUID

from backend.app.schemas.common import OrmSchema, StrictSchema


class MarketingStrategyGenerateRequest(StrictSchema):
    """Optional parameters when triggering Marketing Strategy generation."""

    force_regenerate: bool = False


class MarketingStrategySourceSchema(StrictSchema):
    """One source actually used by a section of the report.

    Built server-side from the CRAG graph's own relevant_documents for
    that section - never asked of the model - so a citation appearing
    here always corresponds to something genuinely retrieved.
    """

    section: str
    source: str
    snippet: str


class MarketingStrategyResponse(OrmSchema):
    """Public representation of the Marketing Strategy agent's output."""

    id: UUID
    workspace_id: UUID

    recommended_channels_and_tactics: str
    content_and_messaging_pillars: str
    ninety_day_roadmap: str
    budget_allocation_and_kpis: str
    sources: list[MarketingStrategySourceSchema]

    model_used: str
    input_tokens: int
    output_tokens: int

    version: int
    created_at: datetime
    updated_at: datetime
