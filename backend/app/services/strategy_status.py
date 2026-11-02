"""Strategy status: which stages of the seven-agent chain exist, and what blocks the rest.

Read-only. Nothing here calls an LLM, retrieval, or web search, and
nothing is written - status is derived entirely from the stages' own
existing records and their approvals, so it can never disagree with
them.

STAGE_PREREQUISITES is the single, explicit description of the chain's
dependency graph. It mirrors the hard requirements each agent service
enforces in its own generate() (the source of truth for what actually
404s), and is deliberately spelled out here instead of inferred, so the
orchestrator and the status endpoint share one definition. Drift between
this map and the real services is caught by an integration test that
compares it against each service's actual 404 behavior.

Two consequences of the real graph are worth stating:

- Content Planning requires only a Marketing Strategy, not the business
  profile, exactly as ContentPlanService does. A workspace whose profile
  is later deleted can still generate a content plan.
- can_generate reports whether the stage could be generated right now:
  every prerequisite exists *and* every prerequisite stage is approved,
  matching what each agent's generate() enforces (existence first, then
  approval). It is independent of whether the stage itself is already
  generated, so a generated stage can be reported as not currently
  regenerable - each agent's generate() deliberately returns an
  existing record before re-checking its prerequisites. What holds a
  stage back is split into missing_prerequisites (they do not exist) and
  unapproved_prerequisites (they exist but are not approved).

Each generated stage also reports its history generation and whether it
is stale (built from inputs that have since changed). That is derived
from the stage history and is information only: it is computed after,
and never feeds into, can_generate or the prerequisite lists. A stage's
history is only reported when its newest snapshot still describes the
live record; if a record changed without going through history, the
stage reports no generation and unknown provenance rather than a guess.
"""

from collections.abc import Collection, Mapping
from types import MappingProxyType
from uuid import UUID

from sqlalchemy.orm import Session

from backend.app.db.repositories.stage_snapshots import StageSnapshotRepository
from backend.app.db.repositories.workspaces import WorkspaceRepository
from backend.app.exceptions import ResourceNotFoundError
from backend.app.schemas.strategy_orchestration import (
    ProvenanceState,
    StageApprovalState,
    StageStaleness,
    StrategyPrerequisite,
    StrategyStage,
    StrategyStageStatus,
    StrategyStatusResponse,
)
from backend.app.services.stage_staleness import StageStalenessService
from backend.app.services.strategy_records import StrategyRecordLoader

_P = StrategyPrerequisite

# Each tuple is in canonical order: business_profile first, then stages in
# chain order. That order is what clients see in missing_prerequisites.
STAGE_PREREQUISITES: Mapping[StrategyStage, tuple[StrategyPrerequisite, ...]] = MappingProxyType(
    {
        StrategyStage.BUSINESS_UNDERSTANDING: (_P.BUSINESS_PROFILE,),
        StrategyStage.MARKET_RESEARCH: (_P.BUSINESS_PROFILE,),
        StrategyStage.COMPETITOR_ANALYSIS: (_P.BUSINESS_PROFILE,),
        StrategyStage.CUSTOMER_PERSONAS: (
            _P.BUSINESS_PROFILE,
            _P.BUSINESS_UNDERSTANDING,
            _P.MARKET_RESEARCH,
        ),
        StrategyStage.BRAND_STRATEGY: (
            _P.BUSINESS_PROFILE,
            _P.BUSINESS_UNDERSTANDING,
            _P.COMPETITOR_ANALYSIS,
            _P.CUSTOMER_PERSONAS,
        ),
        StrategyStage.MARKETING_STRATEGY: (
            _P.BUSINESS_PROFILE,
            _P.BUSINESS_UNDERSTANDING,
            _P.MARKET_RESEARCH,
            _P.COMPETITOR_ANALYSIS,
            _P.CUSTOMER_PERSONAS,
            _P.BRAND_STRATEGY,
        ),
        StrategyStage.CONTENT_PLAN: (_P.MARKETING_STRATEGY,),
    }
)


def missing_prerequisites(
    stage: StrategyStage,
    existing: Collection[StrategyPrerequisite],
) -> list[StrategyPrerequisite]:
    """Return the stage's prerequisites that are not in ``existing``, in canonical order."""

    return [item for item in STAGE_PREREQUISITES[stage] if item not in existing]


def unapproved_prerequisites(
    stage: StrategyStage,
    approved: Collection[StrategyStage],
) -> list[StrategyStage]:
    """Return the stage's prerequisite stages that are not in ``approved``, in canonical order.

    The business profile is skipped: it is not a generated stage, so it
    has no approval to wait for (onboarding has its own confirmation).
    """

    return [
        StrategyStage(item.value)
        for item in STAGE_PREREQUISITES[stage]
        if item is not StrategyPrerequisite.BUSINESS_PROFILE
        and StrategyStage(item.value) not in approved
    ]


class StrategyStatusService:
    """Report the state of the strategy chain for one workspace."""

    def __init__(self, session: Session) -> None:
        self._workspaces = WorkspaceRepository(session)
        self._records = StrategyRecordLoader(session)
        self._snapshots = StageSnapshotRepository(session)
        self._staleness = StageStalenessService(session)

    def get_status(self, workspace_id: UUID) -> StrategyStatusResponse:
        """Return every stage's state, in chain order.

        Raises ResourceNotFoundError only if the workspace itself does
        not exist. A missing business profile is a normal, reportable
        state (it shows up as a missing prerequisite), not an error.
        """

        if self._workspaces.get(workspace_id) is None:
            raise ResourceNotFoundError("Workspace not found.")

        records = self._records.load(workspace_id)
        newest = {
            StrategyStage(snapshot.stage): snapshot
            for snapshot in self._snapshots.list_latest_by_workspace(workspace_id)
        }
        staleness = self._staleness.staleness_by_stage(workspace_id)

        existing = {StrategyPrerequisite(stage.value) for stage in records.refs}
        if records.has_profile:
            existing.add(StrategyPrerequisite.BUSINESS_PROFILE)

        stages: list[StrategyStageStatus] = []
        for stage in StrategyStage:
            ref = records.refs.get(stage)
            missing = missing_prerequisites(stage, existing)
            # An unapproved prerequisite must exist: one that is missing is
            # already reported above, and the two lists never overlap.
            unapproved = [
                item
                for item in unapproved_prerequisites(stage, records.approved_stages)
                if item in records.refs
            ]

            approval: StageApprovalState | None = None
            if ref is not None:
                approval = (
                    StageApprovalState.APPROVED
                    if stage in records.approvals
                    else StageApprovalState.DRAFT
                )

            generation: int | None = None
            stage_staleness: StageStaleness | None = None
            if ref is not None:
                snapshot = newest.get(stage)
                if (
                    snapshot is not None
                    and snapshot.record_id == ref.id
                    and snapshot.record_version == ref.version
                ):
                    generation = snapshot.generation
                    stage_staleness = staleness[stage]
                else:
                    # No snapshot, or one that no longer matches the live
                    # record: nothing recorded what this content was built
                    # from, so nothing can be claimed about it.
                    stage_staleness = StageStaleness(provenance=ProvenanceState.UNKNOWN)

            stages.append(
                StrategyStageStatus(
                    stage=stage,
                    generated=ref is not None,
                    version=ref.version if ref is not None else None,
                    approval=approval,
                    can_generate=not missing and not unapproved,
                    missing_prerequisites=missing,
                    unapproved_prerequisites=unapproved,
                    generation=generation,
                    staleness=stage_staleness,
                )
            )

        return StrategyStatusResponse(workspace_id=workspace_id, stages=stages)
