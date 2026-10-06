"""Strategy orchestration request and response schemas.

Orchestration ties the seven agent stages into one workflow without
adding any persisted state of its own: every value here is derived from
the stages' existing records, the stages' existing records and their approvals, so nothing in this
module maps to a table.

The shape of the chain is defined once, by ``StrategyStage``'s member
order: it is simultaneously the display order, the generation order, and
a valid topological order of the real prerequisite graph (every stage's
prerequisites come earlier in the enum). Both responses enforce that all
seven stages are present exactly once, in that order, so a client can
rely on positional structure and a service bug cannot silently drop or
reorder a stage.
"""

from enum import StrEnum
from typing import Self
from uuid import UUID

from pydantic import Field, computed_field, model_validator

from backend.app.schemas.common import StrictSchema


class StrategyStage(StrEnum):
    """One agent stage of the strategy chain, in generation order."""

    BUSINESS_UNDERSTANDING = "business_understanding"
    MARKET_RESEARCH = "market_research"
    COMPETITOR_ANALYSIS = "competitor_analysis"
    CUSTOMER_PERSONAS = "customer_personas"
    BRAND_STRATEGY = "brand_strategy"
    MARKETING_STRATEGY = "marketing_strategy"
    CONTENT_PLAN = "content_plan"


class StrategyPrerequisite(StrEnum):
    """Something a stage needs to exist before it can be generated.

    Every stage's prerequisites are the onboarding business profile
    (which is not itself a generated stage) and/or earlier stages, so
    this enum is ``business_profile`` plus every ``StrategyStage`` value.
    The two must stay in sync; a unit test enforces it.
    """

    BUSINESS_PROFILE = "business_profile"
    BUSINESS_UNDERSTANDING = "business_understanding"
    MARKET_RESEARCH = "market_research"
    COMPETITOR_ANALYSIS = "competitor_analysis"
    CUSTOMER_PERSONAS = "customer_personas"
    BRAND_STRATEGY = "brand_strategy"
    MARKETING_STRATEGY = "marketing_strategy"
    CONTENT_PLAN = "content_plan"


class StageApprovalState(StrEnum):
    """Where a generated stage stands in human review.

    A stage is ``draft`` until a person approves it, and goes back to
    ``draft`` whenever it is regenerated, because its approval was given
    for the earlier record. A stage that has not been generated has no
    approval state at all.
    """

    DRAFT = "draft"
    APPROVED = "approved"


class StrategyStageOutcome(StrEnum):
    """What a full-strategy run did with one stage."""

    GENERATED = "generated"
    SKIPPED = "skipped"
    FAILED = "failed"
    NOT_ATTEMPTED = "not_attempted"


_EXPECTED_STAGES: tuple[StrategyStage, ...] = tuple(StrategyStage)


def _require_full_chain(stages: list[StrategyStage], *, label: str) -> None:
    """Reject anything other than every stage exactly once, in chain order."""

    if tuple(stages) != _EXPECTED_STAGES:
        expected = ", ".join(stage.value for stage in _EXPECTED_STAGES)
        raise ValueError(f"{label} must list every stage exactly once, in order: {expected}.")


class StrategyGenerateRequest(StrictSchema):
    """Optional parameters when triggering full strategy generation.

    force_regenerate applies to the whole run, all or nothing: every
    stage is regenerated in order. Without it, stages that already exist
    are skipped, which is also what makes a retry after a mid-chain
    failure resume where the previous run stopped.
    """

    force_regenerate: bool = False


class StrategyStageStatus(StrictSchema):
    """Where one stage stands, and whether it could be generated now.

    can_generate reports whether every prerequisite currently exists,
    independent of whether the stage itself has already been generated.
    approval is the stage's review state and is present exactly when the
    stage has been generated.
    """

    stage: StrategyStage
    generated: bool
    version: int | None = Field(default=None, ge=1)
    approval: StageApprovalState | None = None
    can_generate: bool
    missing_prerequisites: list[StrategyPrerequisite] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if self.generated and self.version is None:
            raise ValueError("A generated stage must report its version.")
        if not self.generated and self.version is not None:
            raise ValueError("A stage that is not generated cannot report a version.")
        if self.generated and self.approval is None:
            raise ValueError("A generated stage must report its approval state.")
        if not self.generated and self.approval is not None:
            raise ValueError("A stage that is not generated cannot report an approval state.")

        if self.can_generate == bool(self.missing_prerequisites):
            raise ValueError("can_generate must be true exactly when no prerequisites are missing.")
        if len(set(self.missing_prerequisites)) != len(self.missing_prerequisites):
            raise ValueError("missing_prerequisites must not contain duplicates.")
        if self.stage.value in {item.value for item in self.missing_prerequisites}:
            raise ValueError("A stage cannot be its own prerequisite.")

        return self


class StrategyStatusResponse(StrictSchema):
    """Status of the whole strategy chain for one workspace."""

    workspace_id: UUID
    stages: list[StrategyStageStatus]

    @model_validator(mode="after")
    def _check_stages(self) -> Self:
        _require_full_chain([item.stage for item in self.stages], label="stages")
        return self

    @computed_field
    @property
    def complete(self) -> bool:
        """True once every stage has been generated."""

        return all(item.generated for item in self.stages)

    @computed_field
    @property
    def next_stage(self) -> StrategyStage | None:
        """The first stage, in chain order, that has not been generated."""

        for item in self.stages:
            if not item.generated:
                return item.stage
        return None

    @computed_field
    @property
    def approved(self) -> bool:
        """True once every stage has been generated and approved."""

        return all(item.approval is StageApprovalState.APPROVED for item in self.stages)

    @computed_field
    @property
    def next_to_approve(self) -> StrategyStage | None:
        """The first generated stage, in chain order, still awaiting approval."""

        for item in self.stages:
            if item.approval is StageApprovalState.DRAFT:
                return item.stage
        return None


class StrategyStageResult(StrictSchema):
    """What a full-strategy run did with one stage.

    version is the stage record's version after the run, present exactly
    when the stage ended up with a record from this run (generated or
    skipped). error is present exactly when the stage failed, and is a
    message already safe to show a client - the service supplies it, this
    schema never sees a raw exception.
    """

    stage: StrategyStage
    outcome: StrategyStageOutcome
    version: int | None = Field(default=None, ge=1)
    error: str | None = Field(default=None, min_length=1, max_length=500)

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        has_record = self.outcome in {
            StrategyStageOutcome.GENERATED,
            StrategyStageOutcome.SKIPPED,
        }

        if has_record and self.version is None:
            raise ValueError(f"A {self.outcome.value} stage must report its version.")
        if not has_record and self.version is not None:
            raise ValueError(f"A {self.outcome.value} stage cannot report a version.")

        if self.outcome is StrategyStageOutcome.FAILED and self.error is None:
            raise ValueError("A failed stage must report an error.")
        if self.outcome is not StrategyStageOutcome.FAILED and self.error is not None:
            raise ValueError("Only a failed stage can report an error.")

        return self


class StrategyGenerateResponse(StrictSchema):
    """Result of one full-strategy run.

    A run always walks the chain in order and stops at the first failure,
    so a valid result is always: some stages generated or skipped, then
    optionally one failed stage, then every remaining stage not
    attempted. Anything else is rejected, which keeps the failure
    semantics honest for clients that only look at the first non-success.
    """

    workspace_id: UUID
    force_regenerate: bool
    stages: list[StrategyStageResult]

    @model_validator(mode="after")
    def _check_run_shape(self) -> Self:
        _require_full_chain([item.stage for item in self.stages], label="stages")

        outcomes = [item.outcome for item in self.stages]

        if self.force_regenerate and StrategyStageOutcome.SKIPPED in outcomes:
            raise ValueError("A forced run cannot skip stages.")

        if outcomes.count(StrategyStageOutcome.FAILED) > 1:
            raise ValueError("A run stops at its first failure, so at most one stage can fail.")

        failed_index = (
            outcomes.index(StrategyStageOutcome.FAILED)
            if StrategyStageOutcome.FAILED in outcomes
            else None
        )

        if failed_index is None:
            if StrategyStageOutcome.NOT_ATTEMPTED in outcomes:
                raise ValueError("Stages can only be not_attempted after a failed stage.")
            return self

        if any(
            outcome in {StrategyStageOutcome.FAILED, StrategyStageOutcome.NOT_ATTEMPTED}
            for outcome in outcomes[:failed_index]
        ):
            raise ValueError("Stages before the failed stage must be generated or skipped.")
        if any(
            outcome is not StrategyStageOutcome.NOT_ATTEMPTED
            for outcome in outcomes[failed_index + 1 :]
        ):
            raise ValueError("Every stage after the failed stage must be not_attempted.")

        return self

    @computed_field
    @property
    def complete(self) -> bool:
        """True when every stage now has a record (nothing failed)."""

        return all(
            item.outcome in {StrategyStageOutcome.GENERATED, StrategyStageOutcome.SKIPPED}
            for item in self.stages
        )

    @computed_field
    @property
    def failed_stage(self) -> StrategyStage | None:
        """The stage that stopped the run, if any."""

        for item in self.stages:
            if item.outcome is StrategyStageOutcome.FAILED:
                return item.stage
        return None
