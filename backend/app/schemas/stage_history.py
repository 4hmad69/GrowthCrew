"""Stage history response schemas and the snapshot content registry.

A snapshot is one generation of one stage, kept after the stage record
has been overwritten by a later generation. These schemas describe what
the history API returns and the shape of the provenance a snapshot
carries; the staleness types that the strategy status reuses live in
``strategy_orchestration`` (they cannot live here, because that module
cannot import this one without a cycle).

Reading history must keep working however the stages evolve, so the
response schemas deliberately do not require a snapshot's content to
match today's field list. The registry below is enforced where it
matters instead: when a snapshot is written, and by a unit test that
fails the moment a stage model gains or loses a generated column.
"""

from datetime import datetime
from itertools import pairwise
from typing import Self
from uuid import UUID

from pydantic import Field, JsonValue, computed_field, model_validator

from backend.app.schemas.common import OrmSchema, StrictSchema
from backend.app.schemas.strategy_orchestration import ProvenanceState, StrategyStage

# The generated fields of each stage: what a snapshot's content holds.
# Everything else on a stage record is bookkeeping (identity, ownership,
# model and token counts, version, timestamps) and is stored in the
# snapshot's own columns or deliberately left out.
STAGE_CONTENT_FIELDS: dict[StrategyStage, tuple[str, ...]] = {
    StrategyStage.BUSINESS_UNDERSTANDING: (
        "summary",
        "inferred_business_stage",
        "competitive_category",
        "key_differentiators",
        "likely_customer_pain_points",
        "confidence_notes",
    ),
    StrategyStage.MARKET_RESEARCH: (
        "market_overview",
        "target_customer_segments",
        "competitive_landscape",
        "opportunities_and_risks",
        "sources",
    ),
    StrategyStage.COMPETITOR_ANALYSIS: (
        "competitor_overview",
        "strengths_and_weaknesses",
        "pricing_and_positioning",
        "differentiation_opportunities",
        "sources",
    ),
    StrategyStage.CUSTOMER_PERSONAS: (
        "overview",
        "personas",
    ),
    StrategyStage.BRAND_STRATEGY: (
        "overview",
        "positioning_statement",
        "value_proposition",
        "brand_voice_and_tone",
        "brand_pillars",
        "tagline_options",
    ),
    StrategyStage.MARKETING_STRATEGY: (
        "recommended_channels_and_tactics",
        "content_and_messaging_pillars",
        "ninety_day_roadmap",
        "budget_allocation_and_kpis",
        "sources",
    ),
    StrategyStage.CONTENT_PLAN: (
        "overview",
        "entries",
    ),
}


class ProfileInputRef(StrictSchema):
    """The business profile a stage was built from, at the version it had then."""

    id: UUID
    version: int = Field(ge=1)


class StageInputRef(StrictSchema):
    """An upstream stage a stage was built from.

    generation is what staleness is judged by. record_id and
    record_version say which stage record that generation left behind,
    which is what an approval of it was pinned to.
    """

    generation: int = Field(ge=1)
    record_id: UUID
    record_version: int = Field(ge=1)


class StageInputs(StrictSchema):
    """Everything a stage was built from, as recorded when it was generated.

    This is the shape stored in a snapshot's ``inputs`` column. Every
    stage is built from the business profile, upstream stages, or both,
    so recording nothing at all is rejected rather than read as "built
    from nothing" - a stage whose inputs were never recorded has no
    ``StageInputs``, which is what unknown provenance means.
    """

    profile: ProfileInputRef | None = None
    stages: dict[StrategyStage, StageInputRef] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_not_empty(self) -> Self:
        if self.profile is None and not self.stages:
            raise ValueError("Recorded inputs must include the profile or at least one stage.")
        return self


class StageSnapshotSummary(OrmSchema):
    """One generation of a stage, without its content.

    record_id and record_version identify the stage record as it stood
    after this generation. Two generations with the same pair hold the
    same content, because any content change bumps the record's version.

    inputs is read so provenance can be reported, but is not part of the
    summary's output; the detail view returns it in full.
    """

    stage: StrategyStage
    generation: int = Field(ge=1)
    record_id: UUID
    record_version: int = Field(ge=1)
    model_used: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    created_at: datetime
    inputs: StageInputs | None = Field(default=None, exclude=True)

    @computed_field
    @property
    def provenance(self) -> ProvenanceState:
        """Whether what this generation was built from is known."""

        if self.inputs is None:
            return ProvenanceState.UNKNOWN
        return ProvenanceState.TRACKED


class StageSnapshotDetail(StageSnapshotSummary):
    """One generation of a stage with its full content and provenance.

    content is the generated fields exactly as that generation produced
    them. It is not checked against the stage's current field list: a
    snapshot written before a stage gained a field must stay readable.
    """

    workspace_id: UUID
    content: dict[str, JsonValue]
    inputs: StageInputs | None = None


class StageHistoryResponse(StrictSchema):
    """Every recorded generation of one stage, newest first."""

    workspace_id: UUID
    stage: StrategyStage
    snapshots: list[StageSnapshotSummary]

    @model_validator(mode="after")
    def _check_snapshots(self) -> Self:
        if any(item.stage is not self.stage for item in self.snapshots):
            raise ValueError("Every snapshot must belong to the stage being listed.")
        generations = [item.generation for item in self.snapshots]
        if any(newer <= older for newer, older in pairwise(generations)):
            raise ValueError("Snapshots must be listed newest first, with no repeated generation.")
        return self

    @computed_field
    @property
    def latest_generation(self) -> int | None:
        """The newest generation recorded, or None when there is none."""

        return self.snapshots[0].generation if self.snapshots else None
