"""Record and read the history of generated stages.

Every stage record is overwritten in place when it is regenerated. This
service is what keeps the past: each time a stage is generated, the
agent hands its record here and a snapshot is written in the *same
transaction*, so a generation and its history either both exist or
neither does. Nothing is ever updated or deleted afterwards.

``record`` deliberately never commits. The calling agent owns the
transaction and its error handling; this service only stages the
snapshot and flushes it, so a conflict surfaces immediately and the
whole generation rolls back with it.

Each snapshot carries its provenance: which business profile version and
which upstream generations the stage was built from. That is captured
only when it can be stated exactly. If any input cannot be identified -
an upstream stage with no snapshot, an upstream record that has changed
since its latest snapshot (so something bypassed history), or a missing
profile - the snapshot records unknown provenance rather than a partial
or guessed answer. Unknown is never reported as stale.

Whether a stage is stale is decided elsewhere (``stage_staleness``);
this module has no opinion about it.
"""

import copy
from typing import Protocol
from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.db.models.stage_snapshot import StageSnapshot
from backend.app.db.repositories.business_profiles import BusinessProfileRepository
from backend.app.db.repositories.stage_snapshots import StageSnapshotRepository
from backend.app.db.repositories.workspaces import WorkspaceRepository
from backend.app.exceptions import ResourceNotFoundError, StaleResourceError
from backend.app.schemas.stage_history import (
    STAGE_CONTENT_FIELDS,
    ProfileInputRef,
    StageInputRef,
    StageInputs,
)
from backend.app.schemas.strategy_orchestration import StrategyPrerequisite, StrategyStage
from backend.app.services.stage_approval import STAGE_LABELS
from backend.app.services.strategy_records import StrategyRecordLoader
from backend.app.services.strategy_status import STAGE_PREREQUISITES

_GENERATION_CONSTRAINT = "uq_stage_snapshots_workspace_id_stage_generation"


class StageRecord(Protocol):
    """The bookkeeping every stage record carries.

    The generated content fields differ per stage and are read by name
    from the registry in ``schemas.stage_history``.
    """

    id: UUID
    version: int
    model_used: str
    input_tokens: int
    output_tokens: int


def _violates_generation_uniqueness(error: IntegrityError) -> bool:
    """Return True if the error is two writers claiming the same generation."""

    diagnostics = getattr(error.orig, "diag", None)
    return getattr(diagnostics, "constraint_name", None) == _GENERATION_CONSTRAINT


class StageHistoryService:
    """Write snapshots as stages are generated, and read them back."""

    def __init__(self, session: Session) -> None:
        self._session = session
        self._snapshots = StageSnapshotRepository(session)
        self._profiles = BusinessProfileRepository(session)
        self._workspaces = WorkspaceRepository(session)
        self._loader = StrategyRecordLoader(session)

    def record(
        self,
        workspace_id: UUID,
        stage: StrategyStage,
        record: StageRecord,
    ) -> StageSnapshot:
        """Stage a snapshot of a stage record that was just generated.

        The record is flushed first, so a new record has its id and an
        updated one its final version before either is copied. The
        snapshot is then flushed too, so that if another request claimed
        the same generation number in the meantime the conflict is
        raised here, as ``StaleResourceError``, and everything the
        caller staged is rolled back. Never commits.

        A generation is recorded even when regenerating produced content
        identical to the last time: the record's version does not move
        then, but the generation still happened and is what staleness
        is judged by.
        """

        self._session.flush()

        content = {
            field: copy.deepcopy(getattr(record, field)) for field in STAGE_CONTENT_FIELDS[stage]
        }
        inputs = self._capture_inputs(workspace_id, stage)
        snapshot = StageSnapshot(
            workspace_id=workspace_id,
            stage=stage.value,
            generation=self._snapshots.next_generation(workspace_id, stage.value),
            record_id=record.id,
            record_version=record.version,
            content=content,
            inputs=inputs.model_dump(mode="json") if inputs is not None else None,
            model_used=record.model_used,
            input_tokens=record.input_tokens,
            output_tokens=record.output_tokens,
        )
        self._snapshots.add(snapshot)

        try:
            self._session.flush()
        except IntegrityError as exc:
            if not _violates_generation_uniqueness(exc):
                raise
            self._session.rollback()
            raise StaleResourceError(
                f"The {STAGE_LABELS[stage]} stage was regenerated by another request "
                "at the same time."
            ) from exc

        return snapshot

    def list_history(self, workspace_id: UUID, stage: StrategyStage) -> list[StageSnapshot]:
        """Return every generation of a stage, newest first (empty if none)."""

        self._require_workspace(workspace_id)
        return self._snapshots.list_by_stage(workspace_id, stage.value)

    def get_snapshot(
        self,
        workspace_id: UUID,
        stage: StrategyStage,
        generation: int,
    ) -> StageSnapshot:
        """Return one generation of a stage."""

        self._require_workspace(workspace_id)
        snapshot = self._snapshots.get(workspace_id, stage.value, generation)
        if snapshot is None:
            raise ResourceNotFoundError("That generation of this stage does not exist.")
        return snapshot

    def _require_workspace(self, workspace_id: UUID) -> None:
        """Ensure the workspace exists."""

        if self._workspaces.get(workspace_id) is None:
            raise ResourceNotFoundError("Workspace not found.")

    def _capture_inputs(self, workspace_id: UUID, stage: StrategyStage) -> StageInputs | None:
        """Describe exactly what a stage is being built from, or None if unsure."""

        records = self._loader.load(workspace_id)
        latest = {
            StrategyStage(snapshot.stage): snapshot
            for snapshot in self._snapshots.list_latest_by_workspace(workspace_id)
        }

        profile_ref: ProfileInputRef | None = None
        stage_refs: dict[StrategyStage, StageInputRef] = {}

        for prerequisite in STAGE_PREREQUISITES[stage]:
            if prerequisite is StrategyPrerequisite.BUSINESS_PROFILE:
                profile = self._profiles.get_by_workspace(workspace_id)
                if profile is None:
                    return None
                profile_ref = ProfileInputRef(id=profile.id, version=profile.version)
                continue

            upstream = StrategyStage(prerequisite.value)
            live = records.refs.get(upstream)
            newest = latest.get(upstream)
            if live is None or newest is None:
                return None
            # The newest snapshot only describes the stage if it still
            # matches the live record; otherwise something changed the
            # record without going through history and no generation
            # can honestly be claimed.
            if newest.record_id != live.id or newest.record_version != live.version:
                return None
            stage_refs[upstream] = StageInputRef(
                generation=newest.generation,
                record_id=live.id,
                record_version=live.version,
            )

        return StageInputs(profile=profile_ref, stages=stage_refs)
