"""Stage approval request and response schemas.

An approval records that a person reviewed one generated stage of the
strategy chain and accepted it as the basis for the stages that build on
it. These schemas describe a single approval; the per-stage approval
state shown in the strategy status lives with the status schemas in
``strategy_orchestration`` (it cannot live here, because this module
imports ``StrategyStage`` from that one).

A stage's approval exists only while it still matches the stage record it
was given for. Once the stage is regenerated the approval is no longer
current, and the API treats the stage as unapproved rather than returning
a stale approval, so ``StageApprovalResponse`` never describes one.
"""

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import Field

from backend.app.schemas.common import OrmSchema, StrictSchema
from backend.app.schemas.strategy_orchestration import StrategyStage


class ApprovalSource(StrEnum):
    """Who recorded an approval.

    The two values must stay in sync with the stage_approvals table's
    source CHECK constraint; a unit test enforces it.
    """

    HUMAN = "human"
    AUTO = "auto"


class StageApprovalRequest(StrictSchema):
    """Body of a request to approve one stage.

    version is the version of the stage record the reviewer actually
    looked at. The approval is refused if the stage has been regenerated
    since, so a person can never approve content they have not seen. It
    is deliberately strict (a real integer, no coercion from booleans or
    numeric strings): it is a concurrency token, not free-form input.
    """

    version: int = Field(strict=True, ge=1)


class StageApprovalResponse(OrmSchema):
    """A current approval of one stage.

    record_id and approved_version identify exactly which stage record
    was approved; source says whether a person or an explicitly
    auto-approving full-strategy run recorded it.
    """

    workspace_id: UUID
    stage: StrategyStage
    record_id: UUID
    approved_version: int = Field(ge=1)
    source: ApprovalSource
    approved_at: datetime
