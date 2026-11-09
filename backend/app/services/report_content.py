"""Resolve a workspace's strategy chain into a report document.

Read-only. Nothing here calls an LLM, retrieval or web search, and
nothing is written: a report is derived from the stage records, their
approvals and the stage history, so it can never disagree with them.

Each stage's section is built from its **newest history snapshot**, and
only when that snapshot still describes the live record by ``id`` and
``version`` (the same rule the strategy status uses before it reports a
generation). That is what lets a report name a specific generation:

- The section's approval state is the live record's current approval,
  matched by ``record_id`` + ``approved_version`` exactly as everywhere
  else. A forced regeneration that produced identical content leaves the
  record's version, and so its approval, unchanged; the newest of those
  generations is the one reported.
- Staleness is the history's own assessment of that generation. Unknown
  provenance is carried through as unknown: it is never reported as
  fresh and never as stale.
- A stage that has never been generated, or whose record was deleted, is
  omitted as ``not_generated``. A stage whose newest snapshot no longer
  matches its live record (the record was changed outside history) is
  omitted as ``history_out_of_sync`` rather than reported with content
  that cannot be tied to a generation.

The reads happen in one request but not in one database snapshot. If a
stage is regenerated while a report is being built, the record and its
snapshot can disagree; that shows up as ``history_out_of_sync`` for that
stage, never as content from one generation labelled as another.
"""

from collections.abc import Callable
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy.orm import Session

from backend.app.db.models.stage_snapshot import StageSnapshot
from backend.app.db.repositories.business_profiles import BusinessProfileRepository
from backend.app.db.repositories.stage_snapshots import StageSnapshotRepository
from backend.app.db.repositories.workspaces import WorkspaceRepository
from backend.app.exceptions import ResourceNotFoundError
from backend.app.schemas.report import (
    STAGE_ORDER,
    OmissionReason,
    OmittedStage,
    ReportDocument,
    ReportStageSection,
)
from backend.app.schemas.strategy_orchestration import StageApprovalState
from backend.app.services.stage_staleness import StageStalenessService
from backend.app.services.strategy_records import StrategyRecordLoader


def _utc_now() -> datetime:
    return datetime.now(UTC)


class ReportContentService:
    """Build the report document for one workspace."""

    def __init__(
        self,
        session: Session,
        *,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._workspaces = WorkspaceRepository(session)
        self._profiles = BusinessProfileRepository(session)
        self._records = StrategyRecordLoader(session)
        self._snapshots = StageSnapshotRepository(session)
        self._staleness = StageStalenessService(session)
        self._clock = clock

    def build(self, workspace_id: UUID) -> ReportDocument:
        """Return the report for the workspace, with every stage accounted for.

        Raises ResourceNotFoundError only if the workspace does not
        exist. A workspace with nothing generated is a valid, empty
        report.
        """

        workspace = self._workspaces.get(workspace_id)
        if workspace is None:
            raise ResourceNotFoundError("Workspace not found.")

        profile = self._profiles.get_by_workspace(workspace_id)
        business_name = profile.business_name if profile is not None else workspace.name

        records = self._records.load(workspace_id)
        newest = {
            snapshot.stage: snapshot
            for snapshot in self._snapshots.list_latest_by_workspace(workspace_id)
        }
        staleness = self._staleness.staleness_by_stage(workspace_id)

        sections: list[ReportStageSection] = []
        omitted: list[OmittedStage] = []
        for stage in STAGE_ORDER:
            ref = records.refs.get(stage)
            if ref is None:
                omitted.append(OmittedStage(stage=stage, reason=OmissionReason.NOT_GENERATED))
                continue

            snapshot: StageSnapshot | None = newest.get(stage.value)
            if snapshot is None or (snapshot.record_id, snapshot.record_version) != (
                ref.id,
                ref.version,
            ):
                omitted.append(OmittedStage(stage=stage, reason=OmissionReason.HISTORY_OUT_OF_SYNC))
                continue

            sections.append(
                ReportStageSection(
                    stage=stage,
                    generation=snapshot.generation,
                    record_id=snapshot.record_id,
                    record_version=snapshot.record_version,
                    approval=(
                        StageApprovalState.APPROVED
                        if stage in records.approvals
                        else StageApprovalState.DRAFT
                    ),
                    staleness=staleness[stage],
                    model_used=snapshot.model_used,
                    generated_at=snapshot.created_at,
                    content=snapshot.content,
                )
            )

        return ReportDocument(
            workspace_id=workspace_id,
            business_name=business_name,
            generated_at=self._clock(),
            sections=sections,
            omitted=omitted,
        )
