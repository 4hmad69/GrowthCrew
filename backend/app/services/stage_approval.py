"""Approve, revoke, and require approval of strategy stages.

An approval is a person's recorded decision that a generated stage is
good enough for later stages to build on. It is pinned to the exact
stage record that was reviewed (see ``strategy_records``), so it
retires itself the moment that record changes - nothing here, and no
agent service, has to remember to reset it.

This service never calls an LLM, retrieval, or web search. The human
rules it owns:

- A stage can be approved only once it exists, and only at the version
  the reviewer actually looked at; a stale version is refused, so a
  person can never approve content they have not seen.
- Approving is idempotent. Approving what is already approved changes
  nothing, and a person approving what a run auto-approved upgrades the
  record to human - never the other way around.
- Revoking is idempotent too: it removes whatever approval row exists,
  current or stale, and succeeds even if there was none.
- ``require_prerequisites_approved`` is the single check the downstream
  agents use before generating, built on the same prerequisite map the
  status endpoint reports.
"""

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from uuid import UUID

from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from backend.app.db.errors import DatabaseOperationError
from backend.app.db.models.stage_approval import StageApproval
from backend.app.db.repositories.stage_approvals import StageApprovalRepository
from backend.app.db.repositories.workspaces import WorkspaceRepository
from backend.app.exceptions import (
    ApprovalRequiredError,
    ResourceConflictError,
    ResourceNotFoundError,
    StaleResourceError,
)
from backend.app.schemas.stage_approval import ApprovalSource
from backend.app.schemas.strategy_orchestration import StrategyStage
from backend.app.services.strategy_records import (
    StageRecordRef,
    StrategyRecordLoader,
    is_current,
)
from backend.app.services.strategy_status import unapproved_prerequisites

# How each stage is named inside error messages. Written as a noun
# phrase followed by the word "stage" so one sentence shape reads
# correctly for every stage, singular or plural ("customer personas").
STAGE_LABELS: Mapping[StrategyStage, str] = MappingProxyType(
    {
        StrategyStage.BUSINESS_UNDERSTANDING: "business understanding",
        StrategyStage.MARKET_RESEARCH: "market research",
        StrategyStage.COMPETITOR_ANALYSIS: "competitor analysis",
        StrategyStage.CUSTOMER_PERSONAS: "customer personas",
        StrategyStage.BRAND_STRATEGY: "brand strategy",
        StrategyStage.MARKETING_STRATEGY: "marketing strategy",
        StrategyStage.CONTENT_PLAN: "content plan",
    }
)


def _join_labels(stages: Sequence[StrategyStage]) -> str:
    """Join stage labels as 'a', 'a and b', or 'a, b and c'."""

    labels = [STAGE_LABELS[stage] for stage in stages]
    if len(labels) <= 1:
        return "".join(labels)
    return f"{', '.join(labels[:-1])} and {labels[-1]}"


def approval_required_message(
    stage: StrategyStage,
    unapproved: Sequence[StrategyStage],
) -> str:
    """Build the client-safe message for a stage blocked on unapproved prerequisites."""

    noun = "stage" if len(unapproved) == 1 else "stages"
    return (
        f"Approve the {_join_labels(unapproved)} {noun} "
        f"before generating the {STAGE_LABELS[stage]} stage."
    )


class StageApprovalService:
    """Record and check approvals of the strategy chain's stages."""

    def __init__(self, session: Session) -> None:
        self._session = session
        self._workspaces = WorkspaceRepository(session)
        self._approvals = StageApprovalRepository(session)
        self._records = StrategyRecordLoader(session)

    def approve(
        self,
        workspace_id: UUID,
        stage: StrategyStage,
        version: int,
        *,
        source: ApprovalSource = ApprovalSource.HUMAN,
    ) -> StageApproval:
        """Approve the stage's current record, which the caller says is at ``version``.

        Raises ResourceNotFoundError if the workspace or the stage does
        not exist, and StaleResourceError if the stage is no longer at
        ``version`` (it was regenerated since the caller looked).
        """

        self._require_workspace(workspace_id)

        ref = self._records.load(workspace_id).refs.get(stage)
        if ref is None:
            raise ResourceNotFoundError(
                f"The {STAGE_LABELS[stage]} stage has not been generated yet."
            )
        if version != ref.version:
            raise StaleResourceError(
                f"The {STAGE_LABELS[stage]} stage is now at version {ref.version}, "
                f"not version {version}. Review the latest version before approving it."
            )

        existing = self._approvals.get_by_workspace_and_stage(workspace_id, stage.value)

        if existing is not None and is_current(existing, ref):
            # Already approved at this exact record and version. Never
            # downgrade a human approval, and do not churn the timestamp
            # for a repeat; only a person approving what a run
            # auto-approved changes anything (the source becomes human).
            if source is ApprovalSource.AUTO or existing.source == source.value:
                return existing
            existing.source = source.value
            approval = existing
        elif existing is not None:
            # An approval left over from an earlier record or version of
            # this stage: replace it in place (one row per stage).
            existing.record_id = ref.id
            existing.approved_version = ref.version
            existing.source = source.value
            approval = existing
        else:
            approval = StageApproval(
                workspace_id=workspace_id,
                stage=stage.value,
                record_id=ref.id,
                approved_version=ref.version,
                source=source.value,
            )
            self._approvals.add(approval)

        try:
            self._session.commit()
        except IntegrityError as exc:
            self._session.rollback()
            return self._settle_concurrent_approval(workspace_id, stage, ref, exc)
        except SQLAlchemyError as exc:
            self._session.rollback()
            raise DatabaseOperationError("Stage approval failed.") from exc

        self._session.refresh(approval)
        return approval

    def revoke(self, workspace_id: UUID, stage: StrategyStage) -> None:
        """Remove the stage's approval, if it has one. Idempotent."""

        self._require_workspace(workspace_id)

        existing = self._approvals.get_by_workspace_and_stage(workspace_id, stage.value)
        if existing is None:
            return

        self._approvals.delete(existing)

        try:
            self._session.commit()
        except SQLAlchemyError as exc:
            self._session.rollback()
            raise DatabaseOperationError("Stage approval revocation failed.") from exc

    def get(self, workspace_id: UUID, stage: StrategyStage) -> StageApproval:
        """Return the stage's current approval.

        A stage whose approval no longer matches its record counts as
        unapproved, exactly as the status endpoint reports it.
        """

        self._require_workspace(workspace_id)

        approval = self._records.load(workspace_id).approvals.get(stage)
        if approval is None:
            raise ResourceNotFoundError(f"The {STAGE_LABELS[stage]} stage is not approved.")

        return approval

    def require_prerequisites_approved(self, workspace_id: UUID, stage: StrategyStage) -> None:
        """Raise ApprovalRequiredError unless every stage this one builds on is approved.

        Callers run this after their own prerequisite-exists checks, so a
        prerequisite that does not exist has already been reported as a
        404; here a prerequisite that is missing would simply count as
        unapproved. The business profile is never checked - it is not a
        generated stage and has its own onboarding confirmation.
        """

        records = self._records.load(workspace_id)
        unapproved = unapproved_prerequisites(stage, records.approved_stages)

        if unapproved:
            raise ApprovalRequiredError(
                approval_required_message(stage, unapproved),
                stage=stage.value,
                unapproved=[item.value for item in unapproved],
            )

    def _settle_concurrent_approval(
        self,
        workspace_id: UUID,
        stage: StrategyStage,
        ref: StageRecordRef,
        cause: IntegrityError,
    ) -> StageApproval:
        """Resolve a unique-constraint race between two approvals of one stage.

        If the request that won the race approved the very same record
        and version, this one has nothing left to do and returns that
        approval. Anything else changed underneath us, so say so.
        """

        existing = self._approvals.get_by_workspace_and_stage(workspace_id, stage.value)
        if existing is not None and is_current(existing, ref):
            return existing

        raise ResourceConflictError(
            f"The {STAGE_LABELS[stage]} stage's approval was changed by another request. Try again."
        ) from cause

    def _require_workspace(self, workspace_id: UUID) -> None:
        """Ensure the parent workspace exists."""

        if self._workspaces.get(workspace_id) is None:
            raise ResourceNotFoundError("Workspace not found.")
