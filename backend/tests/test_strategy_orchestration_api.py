"""Unit tests for the strategy orchestration HTTP contract.

The database session dependency and the two services behind the routes
are replaced with stubs, so these tests pin the HTTP behavior only:
paths, request validation, status codes, error sanitization, and how the
request is handed to the orchestrator. Behavior against real PostgreSQL
and the real agent services is covered by the integration tests.
"""

from collections.abc import Iterator
from typing import Any
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.api import strategy_orchestration as api_module
from backend.app.config import Settings
from backend.app.db.dependencies import get_db_session
from backend.app.db.errors import DatabaseUnavailableError
from backend.app.exceptions import ResourceNotFoundError
from backend.app.main import create_application
from backend.app.schemas.strategy_orchestration import (
    StageApprovalState,
    StrategyGenerateResponse,
    StrategyStage,
    StrategyStageOutcome,
    StrategyStageResult,
    StrategyStageStatus,
    StrategyStatusResponse,
)

CHAIN = list(StrategyStage)
GEN = StrategyStageOutcome.GENERATED
SKIP = StrategyStageOutcome.SKIPPED
FAIL = StrategyStageOutcome.FAILED
NOPE = StrategyStageOutcome.NOT_ATTEMPTED
AWAIT = StrategyStageOutcome.AWAITING_APPROVAL


def _status_response(workspace_id: UUID, generated: int) -> StrategyStatusResponse:
    return StrategyStatusResponse(
        workspace_id=workspace_id,
        stages=[
            StrategyStageStatus(
                stage=stage,
                generated=index < generated,
                version=1 if index < generated else None,
                approval=StageApprovalState.DRAFT if index < generated else None,
                can_generate=True,
                missing_prerequisites=[],
            )
            for index, stage in enumerate(CHAIN)
        ],
    )


def _generate_response(
    workspace_id: UUID,
    outcomes: list[StrategyStageOutcome],
    *,
    force: bool = False,
    auto: bool = False,
) -> StrategyGenerateResponse:
    stages: list[StrategyStageResult] = []
    for stage, outcome in zip(CHAIN, outcomes, strict=True):
        if outcome is GEN:
            stages.append(
                StrategyStageResult(stage=stage, outcome=outcome, version=1, auto_approved=auto)
            )
        elif outcome is SKIP:
            stages.append(StrategyStageResult(stage=stage, outcome=outcome, version=1))
        elif outcome is AWAIT:
            stages.append(
                StrategyStageResult(
                    stage=stage,
                    outcome=outcome,
                    unapproved_prerequisites=[CHAIN[0]],
                )
            )
        elif outcome is FAIL:
            stages.append(
                StrategyStageResult(
                    stage=stage, outcome=outcome, error="Business profile not found."
                )
            )
        else:
            stages.append(StrategyStageResult(stage=stage, outcome=outcome))
    return StrategyGenerateResponse(
        workspace_id=workspace_id, force_regenerate=force, auto_approve=auto, stages=stages
    )


class Recorder:
    """Shared record of how the stubbed services were used."""

    def __init__(self) -> None:
        self.status_calls: list[UUID] = []
        self.generate_calls: list[tuple[UUID, bool]] = []
        self.generators_built = 0
        self.auto_approve_calls: list[bool] = []
        self.approvers: list[object] = []
        self.approval_services_built = 0
        self.status_error: Exception | None = None
        self.generate_error: Exception | None = None
        self.generate_outcomes: list[StrategyStageOutcome] = [GEN] * 7
        self.generated_stages = 3


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    """Replace the route's services with stubs and return their call record."""

    record = Recorder()

    class StubStatusService:
        def __init__(self, session: object) -> None:
            pass

        def get_status(self, workspace_id: UUID) -> StrategyStatusResponse:
            record.status_calls.append(workspace_id)
            if record.status_error is not None:
                raise record.status_error
            return _status_response(workspace_id, record.generated_stages)

    class StubApprovalService:
        def __init__(self, session: object) -> None:
            record.approval_services_built += 1

    class StubOrchestrator:
        def __init__(
            self,
            status_service: object,
            generators: object,
            approver: object | None = None,
        ) -> None:
            record.approvers.append(approver)

        def generate_full_strategy(
            self,
            workspace_id: UUID,
            *,
            force_regenerate: bool = False,
            auto_approve: bool = False,
        ) -> StrategyGenerateResponse:
            record.generate_calls.append((workspace_id, force_regenerate))
            record.auto_approve_calls.append(auto_approve)
            if record.generate_error is not None:
                raise record.generate_error
            return _generate_response(
                workspace_id,
                record.generate_outcomes,
                force=force_regenerate,
                auto=auto_approve,
            )

    def stub_build_generators(*args: Any, **kwargs: Any) -> dict[str, Any]:
        record.generators_built += 1
        return {}

    monkeypatch.setattr(api_module, "StrategyStatusService", StubStatusService)
    monkeypatch.setattr(api_module, "StageApprovalService", StubApprovalService)
    monkeypatch.setattr(api_module, "StrategyOrchestrationService", StubOrchestrator)
    monkeypatch.setattr(api_module, "build_stage_generators", stub_build_generators)
    return record


@pytest.fixture
def application() -> FastAPI:
    """An app whose database session dependency yields a placeholder."""

    app = create_application(
        Settings(
            environment="test",
            llm_provider="local",
            embeddings_provider="local",
            tavily_api_key=None,
        )
    )

    def fake_session() -> Iterator[object]:
        yield object()

    app.dependency_overrides[get_db_session] = fake_session
    return app


@pytest.fixture
def client(application: FastAPI) -> Iterator[TestClient]:
    with TestClient(application, raise_server_exceptions=False) as test_client:
        yield test_client


def _status_url(workspace_id: UUID) -> str:
    return f"/api/v1/workspaces/{workspace_id}/strategy-status"


def _generate_url(workspace_id: UUID) -> str:
    return f"/api/v1/workspaces/{workspace_id}/generate-full-strategy"


def test_routes_are_registered_with_the_expected_contract(application: FastAPI) -> None:
    """Both routes are published in OpenAPI with typed responses."""

    spec = application.openapi()
    status_path = spec["paths"]["/api/v1/workspaces/{workspace_id}/strategy-status"]
    generate_path = spec["paths"]["/api/v1/workspaces/{workspace_id}/generate-full-strategy"]

    assert set(status_path) == {"get"}
    assert set(generate_path) == {"post"}
    assert status_path["get"]["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/StrategyStatusResponse"
    }
    assert generate_path["post"]["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/StrategyGenerateResponse"
    }
    assert "404" in status_path["get"]["responses"]
    assert "404" in generate_path["post"]["responses"]


def test_get_status_returns_every_stage_and_derived_fields(
    client: TestClient, recorder: Recorder
) -> None:
    """The status body lists all seven stages plus complete and next_stage."""

    workspace_id = uuid4()

    response = client.get(_status_url(workspace_id))
    body = response.json()

    assert response.status_code == 200
    assert recorder.status_calls == [workspace_id]
    assert body["workspace_id"] == str(workspace_id)
    assert [item["stage"] for item in body["stages"]] == [stage.value for stage in CHAIN]
    assert body["stages"][0] == {
        "stage": "business_understanding",
        "generated": True,
        "version": 1,
        "approval": "draft",
        "can_generate": True,
        "missing_prerequisites": [],
        "unapproved_prerequisites": [],
    }
    assert body["stages"][3]["approval"] is None
    assert body["complete"] is False
    assert body["next_stage"] == CHAIN[3].value
    assert body["approved"] is False
    assert body["next_to_approve"] == CHAIN[0].value


def test_get_status_for_unknown_workspace_is_404(client: TestClient, recorder: Recorder) -> None:
    """A missing workspace is a 404 with the domain message."""

    recorder.status_error = ResourceNotFoundError("Workspace not found.")

    response = client.get(_status_url(uuid4()))

    assert response.status_code == 404
    assert response.json() == {"detail": "Workspace not found."}


def test_get_status_database_failure_is_sanitized(client: TestClient, recorder: Recorder) -> None:
    """Infrastructure detail must not leak through the status route."""

    recorder.status_error = DatabaseUnavailableError("host=10.0.0.5 password=hunter2")

    response = client.get(_status_url(uuid4()))

    assert response.status_code == 503
    assert response.json() == {"detail": "A required database operation could not be completed."}
    assert "hunter2" not in response.text


def test_post_defaults_to_not_forcing_regeneration(client: TestClient, recorder: Recorder) -> None:
    """An empty body runs the resumable, skip-existing mode."""

    workspace_id = uuid4()

    response = client.post(_generate_url(workspace_id), json={})
    body = response.json()

    assert response.status_code == 200
    assert recorder.generate_calls == [(workspace_id, False)]
    assert recorder.generators_built == 1
    assert body["force_regenerate"] is False
    assert body["complete"] is True
    assert body["failed_stage"] is None
    assert [item["outcome"] for item in body["stages"]] == ["generated"] * 7


def test_post_passes_force_regenerate_through(client: TestClient, recorder: Recorder) -> None:
    """The caller's force flag reaches the orchestrator unchanged."""

    workspace_id = uuid4()

    response = client.post(_generate_url(workspace_id), json={"force_regenerate": True})

    assert response.status_code == 200
    assert recorder.generate_calls == [(workspace_id, True)]
    assert response.json()["force_regenerate"] is True


def test_post_reports_a_failed_stage_with_http_200(client: TestClient, recorder: Recorder) -> None:
    """A stage failure is data in the body, not an error status."""

    recorder.generate_outcomes = [FAIL, NOPE, NOPE, NOPE, NOPE, NOPE, NOPE]

    response = client.post(_generate_url(uuid4()), json={})
    body = response.json()

    assert response.status_code == 200
    assert body["complete"] is False
    assert body["failed_stage"] == "business_understanding"
    assert body["stages"][0] == {
        "stage": "business_understanding",
        "outcome": "failed",
        "version": None,
        "error": "Business profile not found.",
        "unapproved_prerequisites": [],
        "auto_approved": False,
    }
    assert [item["outcome"] for item in body["stages"][1:]] == ["not_attempted"] * 6


def test_post_reports_skipped_stages(client: TestClient, recorder: Recorder) -> None:
    """A resumed run shows what was skipped and what was generated."""

    recorder.generate_outcomes = [SKIP, SKIP, SKIP, GEN, GEN, GEN, GEN]

    body = client.post(_generate_url(uuid4()), json={}).json()

    assert [item["outcome"] for item in body["stages"]] == ["skipped"] * 3 + ["generated"] * 4
    assert body["complete"] is True


def test_post_for_unknown_workspace_is_404(client: TestClient, recorder: Recorder) -> None:
    """A missing workspace is the one case that is an error status."""

    recorder.generate_error = ResourceNotFoundError("Workspace not found.")

    response = client.post(_generate_url(uuid4()), json={})

    assert response.status_code == 404
    assert response.json() == {"detail": "Workspace not found."}


def test_post_database_failure_before_the_run_is_sanitized(
    client: TestClient, recorder: Recorder
) -> None:
    """An infrastructure failure outside any stage is a sanitized 503."""

    recorder.generate_error = DatabaseUnavailableError("host=10.0.0.5 password=hunter2")

    response = client.post(_generate_url(uuid4()), json={})

    assert response.status_code == 503
    assert response.json() == {"detail": "A required database operation could not be completed."}
    assert "hunter2" not in response.text


def test_post_unexpected_error_is_a_generic_500(client: TestClient, recorder: Recorder) -> None:
    """Programming errors never leak details to the client."""

    recorder.generate_error = RuntimeError("secret internals")

    response = client.post(_generate_url(uuid4()), json={})

    assert response.status_code == 500
    assert response.json() == {"detail": "An unexpected server error occurred."}
    assert "secret" not in response.text


@pytest.mark.parametrize(
    "payload",
    [
        {"force_regenerat": True},
        {"force_regenerate": "maybe"},
        {"force_regenerate": True, "extra": 1},
        [],
        "not an object",
    ],
)
def test_post_rejects_invalid_bodies_without_running_anything(
    client: TestClient, recorder: Recorder, payload: Any
) -> None:
    """Bad input is a 422 and never reaches the orchestrator."""

    response = client.post(_generate_url(uuid4()), json=payload)

    assert response.status_code == 422
    assert recorder.generate_calls == []
    assert recorder.generators_built == 0


def test_post_requires_a_body_like_the_other_agent_endpoints(
    client: TestClient, recorder: Recorder
) -> None:
    """Clients send {} for defaults, consistent with every other generate route."""

    response = client.post(_generate_url(uuid4()))

    assert response.status_code == 422
    assert recorder.generate_calls == []


@pytest.mark.parametrize("builder", [_status_url, _generate_url])
def test_invalid_workspace_ids_are_rejected(
    client: TestClient, recorder: Recorder, builder: Any
) -> None:
    """A non-UUID workspace id is a 422 on both routes."""

    url = builder("not-a-uuid")
    response = client.get(url) if builder is _status_url else client.post(url, json={})

    assert response.status_code == 422
    assert recorder.status_calls == []
    assert recorder.generate_calls == []


def test_wrong_methods_are_not_allowed(client: TestClient, recorder: Recorder) -> None:
    """Status is read-only; generation is POST-only."""

    workspace_id = uuid4()

    assert client.post(_status_url(workspace_id), json={}).status_code == 405
    assert client.get(_generate_url(workspace_id)).status_code == 405


def test_post_defaults_to_not_auto_approving(client: TestClient, recorder: Recorder) -> None:
    """Human review stays on unless the caller explicitly turns it off."""

    response = client.post(_generate_url(uuid4()), json={})

    assert recorder.auto_approve_calls == [False]
    assert response.json()["auto_approve"] is False
    assert all(item["auto_approved"] is False for item in response.json()["stages"])


def test_post_passes_auto_approve_through(client: TestClient, recorder: Recorder) -> None:
    """The caller's auto_approve flag reaches the orchestrator and is echoed back."""

    workspace_id = uuid4()

    response = client.post(
        _generate_url(workspace_id), json={"auto_approve": True, "force_regenerate": True}
    )
    body = response.json()

    assert response.status_code == 200
    assert recorder.generate_calls == [(workspace_id, True)]
    assert recorder.auto_approve_calls == [True]
    assert body["auto_approve"] is True
    assert all(item["auto_approved"] is True for item in body["stages"])


def test_post_gives_the_orchestrator_an_approver_built_from_the_request_session(
    client: TestClient, recorder: Recorder
) -> None:
    """Approvals made by a run use the same approval service the routes use."""

    client.post(_generate_url(uuid4()), json={})

    assert recorder.approval_services_built == 1
    assert len(recorder.approvers) == 1
    assert recorder.approvers[0] is not None


@pytest.mark.parametrize("value", ["yes", "true", 1, 0, None, []])
def test_post_rejects_a_non_boolean_auto_approve(
    client: TestClient, recorder: Recorder, value: Any
) -> None:
    """Switching off human review must never happen through sloppy coercion."""

    response = client.post(_generate_url(uuid4()), json={"auto_approve": value})

    assert response.status_code == 422
    assert recorder.generate_calls == []


def test_post_reports_an_approval_checkpoint_with_http_200(
    client: TestClient, recorder: Recorder
) -> None:
    """Stopping for approval is data in the body, not an error status."""

    recorder.generate_outcomes = [GEN, GEN, GEN, AWAIT, NOPE, NOPE, NOPE]

    response = client.post(_generate_url(uuid4()), json={})
    body = response.json()

    assert response.status_code == 200
    assert body["complete"] is False
    assert body["failed_stage"] is None
    assert body["awaiting_approval_stage"] == "customer_personas"
    assert body["stages"][3] == {
        "stage": "customer_personas",
        "outcome": "awaiting_approval",
        "version": None,
        "error": None,
        "unapproved_prerequisites": ["business_understanding"],
        "auto_approved": False,
    }
    assert [item["outcome"] for item in body["stages"][4:]] == ["not_attempted"] * 3


def test_request_schema_documents_auto_approve(application: FastAPI) -> None:
    """The option is visible in the published OpenAPI contract, defaulting to off."""

    schema = application.openapi()["components"]["schemas"]["StrategyGenerateRequest"]

    assert schema["properties"]["auto_approve"]["default"] is False
    assert schema["properties"]["auto_approve"]["type"] == "boolean"
