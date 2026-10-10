"""Stage approval REST API routes.

Approval is the human checkpoint between the strategy chain's stages: a
person reviews a generated stage and approves it, and only then can the
stages that build on it be generated. These routes only record and read
that decision. They never call an LLM, retrieval, or web search, so they
depend on the database session alone and keep working when the model
provider is down.

Every approval made here is a human approval. Automatic approval exists
only as an explicit option of the full-strategy run, and a client cannot
claim it: the request body accepts nothing but the version reviewed.
"""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy.orm import Session

from backend.app.db.dependencies import get_db_session
from backend.app.schemas.stage_approval import (
    StageApprovalRequest,
    StageApprovalResponse,
)
from backend.app.schemas.strategy_orchestration import StrategyStage
from backend.app.services.stage_approval import StageApprovalService

router = APIRouter(
    prefix="/workspaces/{workspace_id}/approvals",
    tags=["stage approvals"],
)

DbSession = Annotated[Session, Depends(get_db_session)]


@router.post(
    "/{stage}",
    response_model=StageApprovalResponse,
    status_code=status.HTTP_200_OK,
    summary="Approve one generated stage",
    responses={
        404: {"description": "Workspace not found, or the stage has not been generated yet."},
        409: {"description": "The stage changed since the reviewed version."},
    },
)
def approve_stage(
    workspace_id: UUID,
    stage: StrategyStage,
    payload: StageApprovalRequest,
    session: DbSession,
) -> StageApprovalResponse:
    """Record that the reviewer accepts this stage, at the version they reviewed.

    version must be the stage's current version, as reported by the
    strategy status or the stage's own endpoint. If the stage has been
    regenerated since, the request is refused with 409 so a person can
    never approve content they have not seen; fetch the latest version,
    review it, and approve again.

    Idempotent: approving a stage that is already approved at that
    version succeeds and changes nothing. Approving is not a change to
    the stage itself - its version and content are untouched.
    """

    approval = StageApprovalService(session).approve(workspace_id, stage, payload.version)

    return StageApprovalResponse.model_validate(approval)


@router.get(
    "/{stage}",
    response_model=StageApprovalResponse,
    status_code=status.HTTP_200_OK,
    summary="Read one stage's current approval",
    responses={404: {"description": "Workspace not found, or the stage is not approved."}},
)
def get_stage_approval(
    workspace_id: UUID,
    stage: StrategyStage,
    session: DbSession,
) -> StageApprovalResponse:
    """Return who approved the stage's current version, and which record it was.

    A stage that was approved but has since been regenerated is not
    approved any more, so this is a 404 for it exactly as for a stage
    that was never approved. The strategy status shows the same thing as
    a draft stage.
    """

    approval = StageApprovalService(session).get(workspace_id, stage)

    return StageApprovalResponse.model_validate(approval)


@router.delete(
    "/{stage}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Revoke one stage's approval",
    responses={404: {"description": "Workspace not found."}},
)
def revoke_stage_approval(
    workspace_id: UUID,
    stage: StrategyStage,
    session: DbSession,
) -> Response:
    """Pull an approval back, returning the stage to draft.

    Idempotent: revoking a stage that has no approval still succeeds.
    Stages that were generated from this one are not touched; they stay
    as they are, but can no longer be regenerated until this stage is
    approved again.
    """

    StageApprovalService(session).revoke(workspace_id, stage)

    return Response(status_code=status.HTTP_204_NO_CONTENT)
