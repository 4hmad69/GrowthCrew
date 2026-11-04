"""Stage history REST API routes.

Every generated stage is overwritten in place when it is regenerated;
history is what remembers the past. These routes only read it: they list
the generations a stage has had and return any one of them in full,
including what it was built from. There is deliberately no way to change
or remove history here, and no way to restore a past generation - a
snapshot is a record of what was generated, read-only.

Like the approval routes they never call an LLM, retrieval, or web
search, so they depend on the database session alone and keep working
when the model provider is down.
"""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Path, status
from sqlalchemy.orm import Session

from backend.app.db.dependencies import get_db_session
from backend.app.schemas.stage_history import (
    StageHistoryResponse,
    StageSnapshotDetail,
    StageSnapshotSummary,
)
from backend.app.schemas.strategy_orchestration import StrategyStage
from backend.app.services.stage_history import StageHistoryService

router = APIRouter(
    prefix="/workspaces/{workspace_id}/history",
    tags=["stage history"],
)

DbSession = Annotated[Session, Depends(get_db_session)]

# The generation column is a 32-bit integer; a larger number can never exist,
# and sending one to the database would be an error rather than a not-found.
MAX_GENERATION = 2_147_483_647
Generation = Annotated[
    int,
    Path(
        ge=1,
        le=MAX_GENERATION,
        description="Which generation of the stage; the first one generated is 1.",
    ),
]


@router.get(
    "/{stage}",
    response_model=StageHistoryResponse,
    status_code=status.HTTP_200_OK,
    summary="List every generation of one stage",
    responses={404: {"description": "Workspace not found."}},
)
def list_stage_history(
    workspace_id: UUID,
    stage: StrategyStage,
    session: DbSession,
) -> StageHistoryResponse:
    """Return a stage's generations, newest first, without their content.

    Each entry says which generation it is, which stage record it left
    behind (record_id and record_version, the pair an approval is pinned
    to), the model and token usage, and whether it is known what the
    generation was built from. A stage that has never been generated
    has an empty history, not a 404.

    Fetch one generation with its content and inputs from the detail
    route.
    """

    snapshots = StageHistoryService(session).list_history(workspace_id, stage)

    return StageHistoryResponse(
        workspace_id=workspace_id,
        stage=stage,
        snapshots=[StageSnapshotSummary.model_validate(snapshot) for snapshot in snapshots],
    )


@router.get(
    "/{stage}/{generation}",
    response_model=StageSnapshotDetail,
    status_code=status.HTTP_200_OK,
    summary="Read one generation of a stage in full",
    responses={404: {"description": "Workspace not found, or that generation does not exist."}},
)
def get_stage_snapshot(
    workspace_id: UUID,
    stage: StrategyStage,
    generation: Generation,
    session: DbSession,
) -> StageSnapshotDetail:
    """Return one generation exactly as it was generated.

    content holds the stage's generated fields as that generation
    produced them, even if the stage has been regenerated since. inputs
    says which business profile version and which upstream generations
    it was built from; it is null, with provenance "unknown", for stages
    generated before history was recorded.
    """

    snapshot = StageHistoryService(session).get_snapshot(workspace_id, stage, generation)

    return StageSnapshotDetail.model_validate(snapshot)
