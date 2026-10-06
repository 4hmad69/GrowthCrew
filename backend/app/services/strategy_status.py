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
- can_generate reports whether prerequisites currently exist,
  independent of whether the stage itself is already generated. A
  generated stage can therefore be reported as not currently
  regenerable - each agent's generate() deliberately returns an
  existing record before re-checking its prerequisites.
"""

from collections.abc import Collection, Mapping
from types import MappingProxyType
from uuid import UUID

from sqlalchemy.orm import Session

from backend.app.db.repositories.workspaces import WorkspaceRepository
from backend.app.exceptions import ResourceNotFoundError
from backend.app.schemas.strategy_orchestration import (
    StageApprovalState,
    StrategyPrerequisite,
    StrategyStage,
    StrategyStageStatus,
    StrategyStatusResponse,
)
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

    def get_status(self, workspace_id: UUID) -> StrategyStatusResponse:
        """Return every stage's state, in chain order.

        Raises ResourceNotFoundError only if the workspace itself does
        not exist. A missing business profile is a normal, reportable
        state (it shows up as a missing prerequisite), not an error.
        """

        if self._workspaces.get(workspace_id) is None:
            raise ResourceNotFoundError("Workspace not found.")

        records = self._records.load(workspace_id)

        existing = {StrategyPrerequisite(stage.value) for stage in records.refs}
        if records.has_profile:
            existing.add(StrategyPrerequisite.BUSINESS_PROFILE)

        stages: list[StrategyStageStatus] = []
        for stage in StrategyStage:
            ref = records.refs.get(stage)
            missing = missing_prerequisites(stage, existing)

            approval: StageApprovalState | None = None
            if ref is not None:
                approval = (
                    StageApprovalState.APPROVED
                    if stage in records.approvals
                    else StageApprovalState.DRAFT
                )

            stages.append(
                StrategyStageStatus(
                    stage=stage,
                    generated=ref is not None,
                    version=ref.version if ref is not None else None,
                    approval=approval,
                    can_generate=not missing,
                    missing_prerequisites=missing,
                )
            )

        return StrategyStatusResponse(workspace_id=workspace_id, stages=stages)
