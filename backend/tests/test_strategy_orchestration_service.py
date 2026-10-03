"""Unit tests for the strategy orchestration loop.

Stage generators and the status reader are injected stubs (not mocked
sessions), so these tests exercise ordering, skip/force behavior,
stop-at-first-failure, and error mapping in isolation. The same behavior
against real PostgreSQL and the real agent services is covered by the
orchestration integration tests.
"""

from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest
from sqlalchemy.orm import Session

from backend.app.config import Settings
from backend.app.db.errors import DatabaseOperationError
from backend.app.embeddings.errors import EmbeddingsResponseError
from backend.app.embeddings.gateway import EmbeddingsGateway
from backend.app.exceptions import ResourceNotFoundError, StaleResourceError
from backend.app.llm.errors import LLMProviderUnavailableError
from backend.app.llm.gateway import LLMGateway
from backend.app.schemas.strategy_orchestration import (
    StrategyStage,
    StrategyStageOutcome,
    StrategyStageStatus,
    StrategyStatusResponse,
)
from backend.app.services.brand_strategy import BrandStrategyService
from backend.app.services.business_understanding import BusinessUnderstandingService
from backend.app.services.competitor_analysis import CompetitorAnalysisService
from backend.app.services.content_plan import ContentPlanService
from backend.app.services.customer_personas import CustomerPersonasService
from backend.app.services.market_research import MarketResearchService
from backend.app.services.marketing_strategy import MarketingStrategyService
from backend.app.services.strategy_orchestration import (
    StrategyOrchestrationService,
    build_stage_generators,
)
from backend.app.websearch.gateway import WebSearchGateway

S = StrategyStage
CHAIN = list(StrategyStage)
GEN = StrategyStageOutcome.GENERATED
SKIP = StrategyStageOutcome.SKIPPED
FAIL = StrategyStageOutcome.FAILED
NOPE = StrategyStageOutcome.NOT_ATTEMPTED


class StubStatus:
    """Status reader reporting a fixed set of already-generated stages."""

    def __init__(self, existing: dict[StrategyStage, int] | None = None) -> None:
        self._existing = existing or {}
        self.calls: list[UUID] = []

    def get_status(self, workspace_id: UUID) -> StrategyStatusResponse:
        self.calls.append(workspace_id)
        return StrategyStatusResponse(
            workspace_id=workspace_id,
            stages=[
                StrategyStageStatus(
                    stage=stage,
                    generated=stage in self._existing,
                    version=self._existing.get(stage),
                    can_generate=True,
                    missing_prerequisites=[],
                )
                for stage in CHAIN
            ],
        )


class MissingWorkspaceStatus:
    """Status reader for a workspace that does not exist."""

    def get_status(self, workspace_id: UUID) -> StrategyStatusResponse:
        raise ResourceNotFoundError("Workspace not found.")


class StubGenerator:
    """Records calls to generate() and returns a record or raises."""

    def __init__(
        self,
        stage: StrategyStage,
        log: list[tuple[StrategyStage, UUID, bool]],
        *,
        version: int = 1,
        error: Exception | None = None,
    ) -> None:
        self._stage = stage
        self._log = log
        self._version = version
        self._error = error

    def generate(self, workspace_id: UUID, *, force_regenerate: bool = False) -> Any:
        self._log.append((self._stage, workspace_id, force_regenerate))
        if self._error is not None:
            raise self._error
        return SimpleNamespace(version=self._version)


def _service(
    *,
    existing: dict[StrategyStage, int] | None = None,
    errors: dict[StrategyStage, Exception] | None = None,
    versions: dict[StrategyStage, int] | None = None,
) -> tuple[StrategyOrchestrationService, list[tuple[StrategyStage, UUID, bool]]]:
    """Build a service over stub generators; return it and the shared call log."""

    log: list[tuple[StrategyStage, UUID, bool]] = []
    errors = errors or {}
    versions = versions or {}
    generators = {
        stage: StubGenerator(stage, log, version=versions.get(stage, 1), error=errors.get(stage))
        for stage in CHAIN
    }
    return StrategyOrchestrationService(StubStatus(existing), generators), log


def _outcomes(response: Any) -> list[StrategyStageOutcome]:
    return [item.outcome for item in response.stages]


def test_empty_workspace_generates_every_stage_in_chain_order() -> None:
    """A fresh workspace runs the whole chain, once each, in dependency order."""

    service, log = _service()
    workspace_id = uuid4()

    response = service.generate_full_strategy(workspace_id)

    assert [stage for stage, _, _ in log] == CHAIN
    assert all(called_id == workspace_id for _, called_id, _ in log)
    assert _outcomes(response) == [GEN] * 7
    assert response.complete is True
    assert response.workspace_id == workspace_id
    assert response.force_regenerate is False


def test_generated_stages_report_the_version_returned_by_the_agent() -> None:
    """Versions in the response come from the saved records, not a counter."""

    service, _ = _service(versions={S.MARKET_RESEARCH: 4, S.CONTENT_PLAN: 2})

    response = service.generate_full_strategy(uuid4())
    by_stage = {item.stage: item.version for item in response.stages}

    assert by_stage[S.MARKET_RESEARCH] == 4
    assert by_stage[S.CONTENT_PLAN] == 2
    assert by_stage[S.BRAND_STRATEGY] == 1


def test_existing_stages_are_skipped_without_being_called() -> None:
    """A complete workspace costs nothing: no agent is invoked at all."""

    existing = {stage: index + 2 for index, stage in enumerate(CHAIN)}
    service, log = _service(existing=existing)

    response = service.generate_full_strategy(uuid4())

    assert log == []
    assert _outcomes(response) == [SKIP] * 7
    assert [item.version for item in response.stages] == [2, 3, 4, 5, 6, 7, 8]
    assert response.complete is True


def test_partial_workspace_resumes_where_it_left_off() -> None:
    """Existing stages are skipped; only the missing ones are generated."""

    service, log = _service(existing={S.BUSINESS_UNDERSTANDING: 1, S.MARKET_RESEARCH: 3})

    response = service.generate_full_strategy(uuid4())

    assert [stage for stage, _, _ in log] == CHAIN[2:]
    assert _outcomes(response) == [SKIP, SKIP, GEN, GEN, GEN, GEN, GEN]
    assert all(force is False for _, _, force in log)


def test_skipping_is_by_stage_not_by_position() -> None:
    """A gap in the middle is filled even when later stages exist."""

    service, log = _service(existing={stage: 1 for stage in CHAIN if stage is not S.BRAND_STRATEGY})

    response = service.generate_full_strategy(uuid4())

    assert [stage for stage, _, _ in log] == [S.BRAND_STRATEGY]
    assert _outcomes(response) == [SKIP, SKIP, SKIP, SKIP, GEN, SKIP, SKIP]


def test_force_regenerate_calls_every_stage_with_force() -> None:
    """Force is all-or-nothing: existing stages are regenerated too."""

    service, log = _service(existing={stage: 1 for stage in CHAIN})

    response = service.generate_full_strategy(uuid4(), force_regenerate=True)

    assert [stage for stage, _, _ in log] == CHAIN
    assert all(force is True for _, _, force in log)
    assert _outcomes(response) == [GEN] * 7
    assert response.force_regenerate is True


def test_failure_stops_the_run_and_marks_later_stages_not_attempted() -> None:
    """Nothing after a failed stage is called, and earlier work is kept."""

    service, log = _service(
        existing={S.BUSINESS_UNDERSTANDING: 1},
        errors={
            S.CUSTOMER_PERSONAS: ResourceNotFoundError("Market research has not been generated.")
        },
    )

    response = service.generate_full_strategy(uuid4())

    assert [stage for stage, _, _ in log] == [
        S.MARKET_RESEARCH,
        S.COMPETITOR_ANALYSIS,
        S.CUSTOMER_PERSONAS,
    ]
    assert _outcomes(response) == [SKIP, GEN, GEN, FAIL, NOPE, NOPE, NOPE]
    assert response.failed_stage is S.CUSTOMER_PERSONAS
    assert response.complete is False
    assert response.stages[3].error == "Market research has not been generated."
    assert response.stages[3].version is None


def test_failure_on_the_first_stage() -> None:
    """The very first stage failing leaves the rest not attempted."""

    service, log = _service(
        errors={S.BUSINESS_UNDERSTANDING: ResourceNotFoundError("Business profile not found.")}
    )

    response = service.generate_full_strategy(uuid4())

    assert len(log) == 1
    assert _outcomes(response) == [FAIL, NOPE, NOPE, NOPE, NOPE, NOPE, NOPE]


def test_failure_on_the_last_stage() -> None:
    """The last stage failing still reports everything before it as done."""

    service, _ = _service(errors={S.CONTENT_PLAN: StaleResourceError("Content plan changed.")})

    response = service.generate_full_strategy(uuid4())

    assert _outcomes(response) == [GEN, GEN, GEN, GEN, GEN, GEN, FAIL]
    assert response.stages[6].error == "Content plan changed."


def test_failure_during_a_forced_run_is_reported_without_skips() -> None:
    """A forced run that fails midway is still a valid, skip-free result."""

    service, _ = _service(
        existing={stage: 1 for stage in CHAIN},
        errors={S.MARKETING_STRATEGY: LLMProviderUnavailableError("down")},
    )

    response = service.generate_full_strategy(uuid4(), force_regenerate=True)

    assert _outcomes(response) == [GEN, GEN, GEN, GEN, GEN, FAIL, NOPE]


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            DatabaseOperationError("connection to 10.0.0.5 refused: password=hunter2"),
            "A required database operation could not be completed.",
        ),
        (
            LLMProviderUnavailableError("401 from https://llm.example/v1 key=sk-secret"),
            "A required LLM operation could not be completed.",
        ),
        (
            EmbeddingsResponseError("bad vector for token sk-secret"),
            "A required embeddings operation could not be completed.",
        ),
    ],
)
def test_infrastructure_failures_return_fixed_safe_messages(
    error: Exception, expected: str
) -> None:
    """Raw provider or database text must never reach the client."""

    service, _ = _service(errors={S.MARKET_RESEARCH: error})

    response = service.generate_full_strategy(uuid4())

    assert response.stages[1].outcome is FAIL
    assert response.stages[1].error == expected
    assert "secret" not in (response.stages[1].error or "")
    assert "hunter2" not in (response.stages[1].error or "")


def test_domain_error_message_is_passed_through_and_capped() -> None:
    """Domain messages are client-safe, but the schema's length cap must hold."""

    service, _ = _service(errors={S.MARKET_RESEARCH: ResourceNotFoundError("x" * 900)})

    response = service.generate_full_strategy(uuid4())

    assert response.stages[1].error == "x" * 500


def test_blank_domain_error_message_gets_a_fallback() -> None:
    """An empty message must not produce an invalid failed stage."""

    service, _ = _service(errors={S.MARKET_RESEARCH: ResourceNotFoundError("   ")})

    response = service.generate_full_strategy(uuid4())

    assert response.stages[1].error == "The stage could not be generated."


def test_unexpected_errors_propagate_instead_of_being_swallowed() -> None:
    """Programming errors are not stage failures; they surface as 500s."""

    service, log = _service(errors={S.MARKET_RESEARCH: ValueError("bug")})

    with pytest.raises(ValueError, match="bug"):
        service.generate_full_strategy(uuid4())

    assert [stage for stage, _, _ in log] == [S.BUSINESS_UNDERSTANDING, S.MARKET_RESEARCH]


def test_missing_workspace_raises_before_any_stage_runs() -> None:
    """An unknown workspace is a 404, not a failed first stage."""

    log: list[tuple[StrategyStage, UUID, bool]] = []
    generators = {stage: StubGenerator(stage, log) for stage in CHAIN}
    service = StrategyOrchestrationService(MissingWorkspaceStatus(), generators)

    with pytest.raises(ResourceNotFoundError):
        service.generate_full_strategy(uuid4())

    assert log == []


def test_status_is_read_once_per_run() -> None:
    """One consistent snapshot decides skips for the whole run."""

    status = StubStatus()
    log: list[tuple[StrategyStage, UUID, bool]] = []
    service = StrategyOrchestrationService(
        status, {stage: StubGenerator(stage, log) for stage in CHAIN}
    )
    workspace_id = uuid4()

    service.generate_full_strategy(workspace_id)

    assert status.calls == [workspace_id]


def test_constructor_requires_a_generator_for_every_stage() -> None:
    """A wiring mistake should fail at construction, not mid-run."""

    log: list[tuple[StrategyStage, UUID, bool]] = []
    generators = {
        stage: StubGenerator(stage, log) for stage in CHAIN if stage is not S.CONTENT_PLAN
    }

    with pytest.raises(ValueError, match="content_plan"):
        StrategyOrchestrationService(StubStatus(), generators)


def test_build_stage_generators_wires_the_real_agent_services() -> None:
    """The factory must produce the right service class for every stage."""

    settings = Settings(
        environment="test",
        llm_provider="local",
        embeddings_provider="local",
        tavily_api_key=None,
    )

    generators = build_stage_generators(
        Session(),
        settings,
        LLMGateway(settings),
        EmbeddingsGateway(settings),
        WebSearchGateway(settings),
    )

    assert list(generators) == CHAIN
    assert {stage: type(service) for stage, service in generators.items()} == {
        S.BUSINESS_UNDERSTANDING: BusinessUnderstandingService,
        S.MARKET_RESEARCH: MarketResearchService,
        S.COMPETITOR_ANALYSIS: CompetitorAnalysisService,
        S.CUSTOMER_PERSONAS: CustomerPersonasService,
        S.BRAND_STRATEGY: BrandStrategyService,
        S.MARKETING_STRATEGY: MarketingStrategyService,
        S.CONTENT_PLAN: ContentPlanService,
    }
