"""Report schemas: what an exported strategy report is made of.

A report is the whole strategy chain written out for a person to read or
hand on. Every stage in it points at one specific history generation and
says plainly how far that generation can be trusted:

- ``draft``: nobody has approved this record yet (or its approval was
  given for an earlier record).
- ``stale``: it was built from inputs that have since moved on.
- ``unknown_provenance``: nothing recorded what it was built from, so it
  can be neither called fresh nor called stale.

A stage with none of those flags is approved, built from inputs that
still stand, and its provenance is known. Nothing else is called clean:
unknown provenance is never reported as fresh.

A stage that cannot be tied to a generation is left out of the report
and listed in ``omitted`` with the reason, rather than guessed at. These
schemas only describe a report; building one (``report_content``) and
writing it out (the csv, docx and pdf exporters) live elsewhere.
"""

from datetime import datetime
from enum import StrEnum
from typing import Self
from uuid import UUID

from pydantic import ConfigDict, Field, JsonValue, computed_field, model_validator

from backend.app.schemas.common import StrictSchema
from backend.app.schemas.strategy_orchestration import (
    ProvenanceState,
    StageApprovalState,
    StageStaleness,
    StrategyStage,
)

# Stages are always reported in generation order, which is the order the
# StrategyStage enum declares them in.
STAGE_ORDER: tuple[StrategyStage, ...] = tuple(StrategyStage)


class ReportFormat(StrEnum):
    """A file format a report can be exported as."""

    PDF = "pdf"
    DOCX = "docx"
    CSV = "csv"


class ReportStageFlag(StrEnum):
    """A reason a stage's content should not be presented as settled."""

    DRAFT = "draft"
    STALE = "stale"
    UNKNOWN_PROVENANCE = "unknown_provenance"


class OmissionReason(StrEnum):
    """Why a stage has no section in a report."""

    # The stage has never been generated for this workspace.
    NOT_GENERATED = "not_generated"
    # The stage exists, but its newest history snapshot does not match
    # the live record, so no generation can be named for what it holds.
    HISTORY_OUT_OF_SYNC = "history_out_of_sync"


class OmittedStage(StrictSchema):
    """A stage that is not in the report, and why."""

    stage: StrategyStage
    reason: OmissionReason


class ReportStageSection(StrictSchema):
    """One stage of the report: a specific generation and how far to trust it.

    record_id and record_version are what an approval of this stage is
    pinned to, so they say which record the approval state refers to.
    content is the generation's generated fields exactly as stored. It is
    not checked against today's field list, so a stage that gains a field
    later cannot make an old report unreadable.

    The shared base schema trims surrounding whitespace from every string
    it validates, and that reaches strings (and keys) nested inside
    content. A report must not quietly change what a generation said, so
    trimming is switched off here.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)

    stage: StrategyStage
    generation: int = Field(ge=1)
    record_id: UUID
    record_version: int = Field(ge=1)
    approval: StageApprovalState
    staleness: StageStaleness
    model_used: str = Field(min_length=1)
    generated_at: datetime
    content: dict[str, JsonValue]

    @computed_field
    @property
    def flags(self) -> list[ReportStageFlag]:
        """Every reason this stage should not be presented as settled."""

        flags: list[ReportStageFlag] = []
        if self.approval is not StageApprovalState.APPROVED:
            flags.append(ReportStageFlag.DRAFT)
        if self.staleness.stale:
            flags.append(ReportStageFlag.STALE)
        if self.staleness.provenance is ProvenanceState.UNKNOWN:
            flags.append(ReportStageFlag.UNKNOWN_PROVENANCE)
        return flags

    @computed_field
    @property
    def is_clean(self) -> bool:
        """True only for an approved stage with known, current provenance."""

        return not self.flags


class ReportDocument(StrictSchema):
    """A whole strategy report, ready to be written out in any format.

    Every stage of the chain appears exactly once: as a section, in
    generation order, or in ``omitted`` with the reason it is missing.
    """

    workspace_id: UUID
    business_name: str = Field(min_length=1, max_length=160)
    generated_at: datetime
    sections: list[ReportStageSection] = Field(default_factory=list)
    omitted: list[OmittedStage] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_every_stage_once(self) -> Self:
        section_stages = [section.stage for section in self.sections]
        omitted_stages = [item.stage for item in self.omitted]
        if section_stages != sorted(section_stages, key=STAGE_ORDER.index):
            raise ValueError("Sections must be listed in generation order.")
        if omitted_stages != sorted(omitted_stages, key=STAGE_ORDER.index):
            raise ValueError("Omitted stages must be listed in generation order.")
        accounted = section_stages + omitted_stages
        if len(set(accounted)) != len(accounted):
            raise ValueError("A stage must appear only once, as a section or as omitted.")
        if set(accounted) != set(STAGE_ORDER):
            raise ValueError("Every stage must appear, as a section or as omitted.")
        return self

    @computed_field
    @property
    def is_complete(self) -> bool:
        """True when every stage of the chain has a section."""

        return not self.omitted

    @computed_field
    @property
    def is_clean(self) -> bool:
        """True only when the report is complete and every section is clean."""

        return self.is_complete and all(section.is_clean for section in self.sections)

    @computed_field
    @property
    def flagged_stages(self) -> list[StrategyStage]:
        """The stages with at least one flag, in generation order."""

        return [section.stage for section in self.sections if section.flags]
