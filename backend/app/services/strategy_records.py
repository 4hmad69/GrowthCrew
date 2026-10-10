"""Load the current record behind each strategy stage, and which are approved.

This is the one place that knows how to find the seven stage records of
a workspace and how to decide whether a stage is currently approved, so
the status service and the approval service can never disagree about
either.

An approval counts only while it still matches the stage record it was
given for: same record id *and* same version. Matching on version alone
would be wrong, because deleting a stage and generating it again starts
a new record at version 1 - the id is what tells a recreated record
apart from the one that was approved.

Read-only. Callers are responsible for checking that the workspace
exists; this loader reports whatever records it finds (none, for a
workspace that does not exist).
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from uuid import UUID

from sqlalchemy.orm import Session

from backend.app.db.models.stage_approval import StageApproval
from backend.app.db.repositories.brand_strategy import BrandStrategyRepository
from backend.app.db.repositories.business_profiles import BusinessProfileRepository
from backend.app.db.repositories.business_understanding import (
    BusinessUnderstandingRepository,
)
from backend.app.db.repositories.competitor_analysis import CompetitorAnalysisRepository
from backend.app.db.repositories.content_plan import ContentPlanRepository
from backend.app.db.repositories.customer_persona_set import CustomerPersonaSetRepository
from backend.app.db.repositories.market_research import MarketResearchRepository
from backend.app.db.repositories.marketing_strategy import MarketingStrategyRepository
from backend.app.db.repositories.stage_approvals import StageApprovalRepository
from backend.app.schemas.strategy_orchestration import StrategyStage


@dataclass(frozen=True, slots=True)
class StageRecordRef:
    """Identity of one stage record: which row it is, and which version of it."""

    id: UUID
    version: int


def is_current(approval: StageApproval, ref: StageRecordRef) -> bool:
    """Return True if the approval was given for exactly this record and version."""

    return approval.record_id == ref.id and approval.approved_version == ref.version


@dataclass(frozen=True, slots=True)
class StrategyRecords:
    """A snapshot of a workspace's stage records and their current approvals.

    refs holds only the stages that exist. approvals holds only the
    approvals that are still current - an approval left over from a
    record that has since changed, or been deleted and recreated, is
    simply absent, which is what makes that stage read as unapproved.
    """

    has_profile: bool
    refs: Mapping[StrategyStage, StageRecordRef]
    approvals: Mapping[StrategyStage, StageApproval]

    @property
    def approved_stages(self) -> frozenset[StrategyStage]:
        """The stages whose current record is approved."""

        return frozenset(self.approvals)


class StrategyRecordLoader:
    """Read stage records and approvals for one workspace."""

    def __init__(self, session: Session) -> None:
        self._profiles = BusinessProfileRepository(session)
        self._understandings = BusinessUnderstandingRepository(session)
        self._research = MarketResearchRepository(session)
        self._analyses = CompetitorAnalysisRepository(session)
        self._persona_sets = CustomerPersonaSetRepository(session)
        self._brand_strategies = BrandStrategyRepository(session)
        self._strategies = MarketingStrategyRepository(session)
        self._plans = ContentPlanRepository(session)
        self._approvals = StageApprovalRepository(session)

    def load(self, workspace_id: UUID) -> StrategyRecords:
        """Return the workspace's stage records and which are currently approved."""

        profile = self._profiles.get_by_workspace(workspace_id)
        understanding = (
            self._understandings.get_by_business_profile(profile.id)
            if profile is not None
            else None
        )

        records = {
            StrategyStage.BUSINESS_UNDERSTANDING: understanding,
            StrategyStage.MARKET_RESEARCH: self._research.get_by_workspace(workspace_id),
            StrategyStage.COMPETITOR_ANALYSIS: self._analyses.get_by_workspace(workspace_id),
            StrategyStage.CUSTOMER_PERSONAS: self._persona_sets.get_by_workspace(workspace_id),
            StrategyStage.BRAND_STRATEGY: self._brand_strategies.get_by_workspace(workspace_id),
            StrategyStage.MARKETING_STRATEGY: self._strategies.get_by_workspace(workspace_id),
            StrategyStage.CONTENT_PLAN: self._plans.get_by_workspace(workspace_id),
        }
        refs = {
            stage: StageRecordRef(id=record.id, version=record.version)
            for stage, record in records.items()
            if record is not None
        }

        approvals: dict[StrategyStage, StageApproval] = {}
        for approval in self._approvals.list_by_workspace(workspace_id):
            stage = StrategyStage(approval.stage)
            ref = refs.get(stage)
            if ref is not None and is_current(approval, ref):
                approvals[stage] = approval

        return StrategyRecords(
            has_profile=profile is not None,
            refs=MappingProxyType(refs),
            approvals=MappingProxyType(approvals),
        )
