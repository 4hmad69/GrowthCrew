"""Decide which generated stages were built from inputs that have since moved on.

Read-only and derived: nothing is stored. A stage is stale when the
snapshot it was last generated as recorded an input - the business
profile at some version, or an upstream stage at some generation - that
is now newer. Staleness is information only. It never blocks generation
and never touches an approval.

Upstream stages are compared by *generation*, not by record version. A
record's version moves only when its content changes, so a forced
regeneration that happens to produce identical content leaves it
unchanged; the generation still advanced, and that is what proves the
stage was regenerated. The business profile has no such notion - a save
that changes nothing changes nothing - so its version is the right
comparator.

This module is kept apart from ``stage_history`` on purpose: the status
service needs it, and ``stage_history`` imports the status module for
the prerequisite map. Nothing here imports either.

Callers are responsible for checking that the workspace exists.
"""

from collections.abc import Mapping
from uuid import UUID

from sqlalchemy.orm import Session

from backend.app.db.repositories.business_profiles import BusinessProfileRepository
from backend.app.db.repositories.stage_snapshots import StageSnapshotRepository
from backend.app.schemas.stage_history import StageInputs
from backend.app.schemas.strategy_orchestration import (
    ProvenanceState,
    StageStaleness,
    StaleInput,
    StrategyPrerequisite,
    StrategyStage,
)


def assess_staleness(
    inputs: StageInputs | None,
    *,
    current_generations: Mapping[StrategyStage, int],
    current_profile_version: int | None,
) -> StageStaleness:
    """Judge one generation's recorded inputs against where they stand now.

    ``inputs`` is None when provenance is unknown, which can never be
    stale. An input that no longer exists (no history for the stage, no
    profile) cannot be said to have moved, so it is not reported. The
    stale inputs come back in canonical order: the profile first, then
    stages in chain order.
    """

    if inputs is None:
        return StageStaleness(provenance=ProvenanceState.UNKNOWN)

    stale: list[StaleInput] = []

    if (
        inputs.profile is not None
        and current_profile_version is not None
        and current_profile_version > inputs.profile.version
    ):
        stale.append(
            StaleInput(
                prerequisite=StrategyPrerequisite.BUSINESS_PROFILE,
                built_from=inputs.profile.version,
                current=current_profile_version,
            )
        )

    for upstream in StrategyStage:
        recorded = inputs.stages.get(upstream)
        current = current_generations.get(upstream)
        if recorded is not None and current is not None and current > recorded.generation:
            stale.append(
                StaleInput(
                    prerequisite=StrategyPrerequisite(upstream.value),
                    built_from=recorded.generation,
                    current=current,
                )
            )

    return StageStaleness(provenance=ProvenanceState.TRACKED, stale_inputs=stale)


class StageStalenessService:
    """Assess every stage of a workspace that has history."""

    def __init__(self, session: Session) -> None:
        self._snapshots = StageSnapshotRepository(session)
        self._profiles = BusinessProfileRepository(session)

    def staleness_by_stage(self, workspace_id: UUID) -> dict[StrategyStage, StageStaleness]:
        """Return the staleness of each stage that has been generated, in chain order.

        A stage's current generation is its newest snapshot, and it is
        judged by that snapshot's own recorded inputs. Stages with no
        history are absent. A stage whose record was later deleted still
        has history and is still reported; callers that only care about
        stages that currently exist should intersect with what exists.
        """

        latest = {
            StrategyStage(snapshot.stage): snapshot
            for snapshot in self._snapshots.list_latest_by_workspace(workspace_id)
        }
        current_generations = {stage: snapshot.generation for stage, snapshot in latest.items()}
        profile = self._profiles.get_by_workspace(workspace_id)
        current_profile_version = profile.version if profile is not None else None

        result: dict[StrategyStage, StageStaleness] = {}
        for stage in StrategyStage:
            snapshot = latest.get(stage)
            if snapshot is None:
                continue
            recorded = (
                StageInputs.model_validate(snapshot.inputs) if snapshot.inputs is not None else None
            )
            result[stage] = assess_staleness(
                recorded,
                current_generations=current_generations,
                current_profile_version=current_profile_version,
            )
        return result
