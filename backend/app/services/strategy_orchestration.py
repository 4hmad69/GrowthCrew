"""Strategy orchestration: run the seven-agent chain as one workflow.

This service adds no persisted state and no agent logic of its own. It
walks the chain in order and calls each existing agent service's
generate(), which already owns its prerequisite checks, transaction
boundary, and optimistic concurrency. Each stage therefore commits
independently, which is what makes a run resumable: if stage N fails,
stages 1..N-1 are already saved, and calling again without
force_regenerate skips them and continues from N.

Behavior worth knowing:

- Without force_regenerate, stages that already exist are skipped
  without being called at all (no LLM, retrieval, or web-search cost).
  With it, every stage is regenerated, all or nothing.
- A run stops at the first failed stage; later stages are reported as
  not attempted. Only expected failures - domain, database, LLM, and
  embeddings errors - are turned into a failed stage. Anything else is
  a programming error and propagates, exactly as it would from the
  single-agent endpoints.
- A stage whose prerequisites are not yet approved stops the run too,
  but as ``awaiting_approval`` rather than ``failed``: nothing went
  wrong, a person has a review to do. The stage's own generate() is the
  authority on that (it raises ApprovalRequiredError), so this service
  never re-derives the rule. A person approves what the stage names and
  runs again; stages already saved are skipped as usual.
- With auto_approve, each stage this run generates is approved
  immediately after it is saved, recorded as an automatic approval.
  Stages the run skipped are never approved - a skipped stage is one a
  person may not have reviewed, so it can still hold the run at an
  approval checkpoint. Approving is a best-effort follow-up: if it
  fails, the stage stays correctly reported as generated and not
  auto-approved, and the next stage will stop the run at its checkpoint.
- A missing workspace is not a failed stage: it raises
  ResourceNotFoundError before anything runs.
- Failure messages returned to the client are the same safe, fixed
  messages the API's own error handlers use; raw exception text from
  the database, LLM, or embeddings layers never reaches the response.

Not solved here, by design (Step 18): regenerating an upstream stage
does not invalidate stages built from its old output, and two
simultaneous runs for one workspace are not serialized.
"""

import logging
from collections.abc import Mapping
from typing import Protocol
from uuid import UUID

from sqlalchemy.orm import Session

from backend.app.config import Settings
from backend.app.db.errors import DatabaseError
from backend.app.embeddings.errors import EmbeddingsError
from backend.app.embeddings.gateway import EmbeddingsGateway
from backend.app.exceptions import ApprovalRequiredError, DomainError
from backend.app.llm.errors import LLMGatewayError
from backend.app.llm.gateway import LLMGateway
from backend.app.schemas.stage_approval import ApprovalSource
from backend.app.schemas.strategy_orchestration import (
    StrategyGenerateResponse,
    StrategyStage,
    StrategyStageOutcome,
    StrategyStageResult,
    StrategyStatusResponse,
)
from backend.app.services.brand_strategy import BrandStrategyService
from backend.app.services.business_understanding import BusinessUnderstandingService
from backend.app.services.competitor_analysis import CompetitorAnalysisService
from backend.app.services.content_plan import ContentPlanService
from backend.app.services.customer_personas import CustomerPersonasService
from backend.app.services.market_research import MarketResearchService
from backend.app.services.marketing_strategy import MarketingStrategyService
from backend.app.websearch.gateway import WebSearchGateway

logger = logging.getLogger(__name__)

_MAX_ERROR_LENGTH = 500

_DATABASE_FAILURE = "A required database operation could not be completed."
_LLM_FAILURE = "A required LLM operation could not be completed."
_EMBEDDINGS_FAILURE = "A required embeddings operation could not be completed."


class GeneratedRecord(Protocol):
    """What the orchestrator needs from any stage's persisted record."""

    @property
    def version(self) -> int: ...


class StageGenerator(Protocol):
    """The generate() contract every agent service already satisfies."""

    def generate(
        self,
        workspace_id: UUID,
        *,
        force_regenerate: bool = False,
    ) -> GeneratedRecord: ...


class StatusReader(Protocol):
    """Read access to the chain's current status."""

    def get_status(self, workspace_id: UUID) -> StrategyStatusResponse: ...


class StageApprover(Protocol):
    """The approve() contract StageApprovalService already satisfies."""

    def approve(
        self,
        workspace_id: UUID,
        stage: StrategyStage,
        version: int,
        *,
        source: ApprovalSource = ApprovalSource.HUMAN,
    ) -> object: ...


def build_stage_generators(
    session: Session,
    settings: Settings,
    llm_gateway: LLMGateway,
    embeddings_gateway: EmbeddingsGateway,
    web_search_gateway: WebSearchGateway,
) -> dict[StrategyStage, StageGenerator]:
    """Construct the seven real agent services, sharing one session.

    Wired exactly as each agent's own API route wires it: the CRAG-backed
    agents get all three gateways, the direct-call agents only the LLM
    gateway.
    """

    return {
        StrategyStage.BUSINESS_UNDERSTANDING: BusinessUnderstandingService(
            session, llm_gateway, settings
        ),
        StrategyStage.MARKET_RESEARCH: MarketResearchService(
            session, llm_gateway, embeddings_gateway, web_search_gateway, settings
        ),
        StrategyStage.COMPETITOR_ANALYSIS: CompetitorAnalysisService(
            session, llm_gateway, embeddings_gateway, web_search_gateway, settings
        ),
        StrategyStage.CUSTOMER_PERSONAS: CustomerPersonasService(session, llm_gateway, settings),
        StrategyStage.BRAND_STRATEGY: BrandStrategyService(session, llm_gateway, settings),
        StrategyStage.MARKETING_STRATEGY: MarketingStrategyService(
            session, llm_gateway, embeddings_gateway, web_search_gateway, settings
        ),
        StrategyStage.CONTENT_PLAN: ContentPlanService(session, llm_gateway, settings),
    }


def _describe_failure(exc: Exception) -> str:
    """Return a client-safe message for an expected stage failure."""

    if isinstance(exc, DomainError):
        # Domain errors are written to be shown to clients; the API's own
        # handler returns str(exc) verbatim too.
        message = str(exc).strip() or "The stage could not be generated."
        return message[:_MAX_ERROR_LENGTH]
    if isinstance(exc, DatabaseError):
        return _DATABASE_FAILURE
    if isinstance(exc, LLMGatewayError):
        return _LLM_FAILURE
    return _EMBEDDINGS_FAILURE


class StrategyOrchestrationService:
    """Generate the full strategy chain for a workspace, in dependency order."""

    def __init__(
        self,
        status_reader: StatusReader,
        generators: Mapping[StrategyStage, StageGenerator],
        approver: StageApprover | None = None,
    ) -> None:
        missing = [stage.value for stage in StrategyStage if stage not in generators]
        if missing:
            raise ValueError(f"A generator is required for every stage; missing: {missing}.")

        self._status = status_reader
        self._generators = dict(generators)
        self._approver = approver

    def generate_full_strategy(
        self,
        workspace_id: UUID,
        *,
        force_regenerate: bool = False,
        auto_approve: bool = False,
    ) -> StrategyGenerateResponse:
        """Run the chain, skipping existing stages unless forced.

        Raises ResourceNotFoundError if the workspace does not exist.
        Expected failures of an individual stage, and stages stopped at
        an approval checkpoint, are reported in the response, not
        raised. auto_approve needs an approver to have been supplied;
        asking for it without one is a programming error.
        """

        if auto_approve and self._approver is None:
            raise ValueError("auto_approve requires an approver.")
        approver = self._approver if auto_approve else None

        status = self._status.get_status(workspace_id)
        existing_versions = {
            item.stage: item.version
            for item in status.stages
            if item.generated and item.version is not None
        }

        results: list[StrategyStageResult] = []
        stopped = False

        for stage in StrategyStage:
            if stopped:
                results.append(
                    StrategyStageResult(stage=stage, outcome=StrategyStageOutcome.NOT_ATTEMPTED)
                )
                continue

            if not force_regenerate and stage in existing_versions:
                results.append(
                    StrategyStageResult(
                        stage=stage,
                        outcome=StrategyStageOutcome.SKIPPED,
                        version=existing_versions[stage],
                    )
                )
                continue

            try:
                record = self._generators[stage].generate(
                    workspace_id,
                    force_regenerate=force_regenerate,
                )
            except ApprovalRequiredError as exc:
                # Not a failure: a person has stages to review first. Must
                # be caught before the generic domain-error branch below,
                # because it is a subclass of DomainError.
                stopped = True
                results.append(
                    StrategyStageResult(
                        stage=stage,
                        outcome=StrategyStageOutcome.AWAITING_APPROVAL,
                        unapproved_prerequisites=[StrategyStage(value) for value in exc.unapproved],
                    )
                )
                continue
            except (DomainError, DatabaseError, LLMGatewayError, EmbeddingsError) as exc:
                stopped = True
                logger.warning(
                    "Strategy stage %s failed for workspace %s (%s)",
                    stage.value,
                    workspace_id,
                    type(exc).__name__,
                )
                logger.debug(
                    "Strategy stage exception",
                    exc_info=(type(exc), exc, exc.__traceback__),
                )
                results.append(
                    StrategyStageResult(
                        stage=stage,
                        outcome=StrategyStageOutcome.FAILED,
                        error=_describe_failure(exc),
                    )
                )
                continue

            auto_approved = approver is not None and self._auto_approve(
                approver, workspace_id, stage, record.version
            )
            results.append(
                StrategyStageResult(
                    stage=stage,
                    outcome=StrategyStageOutcome.GENERATED,
                    version=record.version,
                    auto_approved=auto_approved,
                )
            )

        return StrategyGenerateResponse(
            workspace_id=workspace_id,
            force_regenerate=force_regenerate,
            auto_approve=auto_approve,
            stages=results,
        )

    @staticmethod
    def _auto_approve(
        approver: StageApprover,
        workspace_id: UUID,
        stage: StrategyStage,
        version: int,
    ) -> bool:
        """Approve a stage this run just generated; report whether it worked.

        Best-effort by design. The stage is already saved and correctly
        reported as generated, so a failure to approve it must not turn
        that into a failed stage. It also needs no special handling
        here: the next stage's own check will stop the run at the
        resulting approval checkpoint, which tells the caller exactly
        what is left to approve. Only expected failures are swallowed
        (and logged); anything else is a programming error and
        propagates.
        """

        try:
            approver.approve(workspace_id, stage, version, source=ApprovalSource.AUTO)
        except (DomainError, DatabaseError) as exc:
            logger.warning(
                "Auto-approval of strategy stage %s failed for workspace %s (%s)",
                stage.value,
                workspace_id,
                type(exc).__name__,
            )
            logger.debug(
                "Auto-approval exception",
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            return False

        return True
