"""Competitor Analysis agent request and response schemas."""

from datetime import datetime
from uuid import UUID

from backend.app.schemas.common import OrmSchema, StrictSchema


class CompetitorAnalysisGenerateRequest(StrictSchema):
    """Optional parameters when triggering Competitor Analysis generation."""

    force_regenerate: bool = False


class CompetitorAnalysisSourceSchema(StrictSchema):
    """One source actually used by a section of the report.

    Built server-side from the CRAG graph's own relevant_documents for
    that section - never asked of the model - so a citation appearing
    here always corresponds to something genuinely retrieved.
    """

    section: str
    source: str
    snippet: str


class CompetitorAnalysisResponse(OrmSchema):
    """Public representation of the Competitor Analysis agent's output."""

    id: UUID
    workspace_id: UUID

    competitor_overview: str
    strengths_and_weaknesses: str
    pricing_and_positioning: str
    differentiation_opportunities: str
    sources: list[CompetitorAnalysisSourceSchema]

    model_used: str
    input_tokens: int
    output_tokens: int

    version: int
    created_at: datetime
    updated_at: datetime
