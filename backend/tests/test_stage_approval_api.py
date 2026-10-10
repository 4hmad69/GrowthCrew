"""Unit tests for the stage approval HTTP contract.

The database session dependency and the approval service behind the
routes are replaced with stubs, so these tests pin the HTTP behavior
only: paths, request validation, status codes, error sanitization, and
how the request is handed to the service. Behavior against real
PostgreSQL is covered by the integration tests.
"""

from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.api import stage_approvals as api_module
from backend.app.config import Settings
from backend.app.db.dependencies import get_db_session
from backend.app.db.errors import DatabaseUnavailableError
from backend.app.db.models.stage_approval import StageApproval
from backend.app.exceptions import (
    ResourceConflictError,
    ResourceNotFoundError,
    StaleResourceError,
)
from backend.app.llm.dependencies import get_llm_gateway
from backend.app.main import create_application
from backend.app.schemas.strategy_orchestration import StrategyStage

CHAIN = list(StrategyStage)
PATH = "/api/v1/workspaces/{workspace_id}/approvals/{stage}"
APPROVED_AT = datetime(2026, 10, 3, 12, 0, tzinfo=UTC)


def _row(workspace_id: UUID, stage: StrategyStage, version: int = 1) -> StageApproval:
    """Build an approval row the way the service would return it."""

    return StageApproval(
        id=uuid4(),
        workspace_id=workspace_id,
        stage=stage.value,
        record_id=uuid4(),
        approved_version=version,
        source="human",
        approved_at=APPROVED_AT,
    )


class Recorder:
    """Shared record of how the stubbed service was used."""

    def __init__(self) -> None:
        self.approve_calls: list[tuple[UUID, StrategyStage, int, dict[str, Any]]] = []
        self.get_calls: list[tuple[UUID, StrategyStage]] = []
        self.revoke_calls: list[tuple[UUID, StrategyStage]] = []
        self.error: Exception | None = None


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    """Replace the route's service with a stub and return its call record."""

    record = Recorder()

    class StubService:
        def __init__(self, session: object) -> None:
            pass

        def approve(
            self, workspace_id: UUID, stage: StrategyStage, version: int, **kwargs: Any
        ) -> StageApproval:
            record.approve_calls.append((workspace_id, stage, version, kwargs))
            if record.error is not None:
                raise record.error
            return _row(workspace_id, stage, version)

        def get(self, workspace_id: UUID, stage: StrategyStage) -> StageApproval:
            record.get_calls.append((workspace_id, stage))
            if record.error is not None:
                raise record.error
            return _row(workspace_id, stage)

        def revoke(self, workspace_id: UUID, stage: StrategyStage) -> None:
            record.revoke_calls.append((workspace_id, stage))
            if record.error is not None:
                raise record.error

    monkeypatch.setattr(api_module, "StageApprovalService", StubService)
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


def _url(workspace_id: UUID, stage: StrategyStage | str) -> str:
    value = stage.value if isinstance(stage, StrategyStage) else stage
    return f"/api/v1/workspaces/{workspace_id}/approvals/{value}"


# --- contract ------------------------------------------------------------------


def test_routes_are_registered_with_the_expected_contract(application: FastAPI) -> None:
    """One path with three methods, typed responses, and documented failures."""

    path = application.openapi()["paths"][PATH]
    ok = "#/components/schemas/StageApprovalResponse"

    assert set(path) == {"get", "post", "delete"}
    for method in ("get", "post"):
        schema = path[method]["responses"]["200"]["content"]["application/json"]["schema"]
        assert schema == {"$ref": ok}
    assert "content" not in path["delete"]["responses"]["204"]
    assert {"404", "409"} <= set(path["post"]["responses"])
    assert "404" in path["get"]["responses"]
    assert "404" in path["delete"]["responses"]


def test_stage_path_accepts_exactly_the_seven_generated_stages(application: FastAPI) -> None:
    """The business profile is not a generated stage and cannot be approved."""

    spec = application.openapi()
    stage_schema = spec["components"]["schemas"]["StrategyStage"]

    assert stage_schema["enum"] == [stage.value for stage in CHAIN]
    assert "business_profile" not in stage_schema["enum"]


def test_approval_routes_do_not_depend_on_the_llm_gateway(
    application: FastAPI, recorder: Recorder
) -> None:
    """Approving is a database-only action and must work with the model provider down."""

    def broken_gateway() -> None:
        raise RuntimeError("LLM gateway must not be constructed for approval routes")

    application.dependency_overrides[get_llm_gateway] = broken_gateway

    with TestClient(application, raise_server_exceptions=False) as client:
        approved = client.post(_url(uuid4(), CHAIN[1]), json={"version": 1})
        read = client.get(_url(uuid4(), CHAIN[1]))
        revoked = client.delete(_url(uuid4(), CHAIN[1]))

    assert (approved.status_code, read.status_code, revoked.status_code) == (200, 200, 204)


# --- POST: approve ---------------------------------------------------------------


def test_post_approves_the_stage_at_the_reviewed_version(
    client: TestClient, recorder: Recorder
) -> None:
    """The path's workspace and stage and the body's version reach the service unchanged."""

    workspace_id = uuid4()

    response = client.post(_url(workspace_id, StrategyStage.BRAND_STRATEGY), json={"version": 4})
    body = response.json()

    assert response.status_code == 200
    assert recorder.approve_calls == [(workspace_id, StrategyStage.BRAND_STRATEGY, 4, {})]
    assert set(body) == {
        "workspace_id",
        "stage",
        "record_id",
        "approved_version",
        "source",
        "approved_at",
    }
    assert body["workspace_id"] == str(workspace_id)
    assert body["stage"] == "brand_strategy"
    assert body["approved_version"] == 4


def test_post_always_records_a_human_approval(client: TestClient, recorder: Recorder) -> None:
    """The route never passes a source, so the service's human default applies."""

    response = client.post(_url(uuid4(), CHAIN[0]), json={"version": 1})

    assert response.json()["source"] == "human"
    assert recorder.approve_calls[0][3] == {}


@pytest.mark.parametrize("stage", CHAIN)
def test_post_accepts_every_generated_stage(
    client: TestClient, recorder: Recorder, stage: StrategyStage
) -> None:
    """Each of the seven stages is a valid approval target."""

    response = client.post(_url(uuid4(), stage), json={"version": 1})

    assert response.status_code == 200
    assert recorder.approve_calls[0][1] is stage


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"version": 0},
        {"version": -3},
        {"version": "3"},
        {"version": True},
        {"version": 2.5},
        {"version": None},
        {"version": 1, "source": "auto"},
        {"version": 1, "extra": "field"},
    ],
)
def test_post_rejects_invalid_bodies_without_calling_the_service(
    client: TestClient, recorder: Recorder, body: dict[str, Any]
) -> None:
    """Bad versions, and any attempt to smuggle in a source, never reach the service."""

    response = client.post(_url(uuid4(), CHAIN[0]), json=body)

    assert response.status_code == 422
    assert recorder.approve_calls == []


def test_post_rejects_a_missing_body(client: TestClient, recorder: Recorder) -> None:
    """An approval needs the reviewed version - there is no sensible default."""

    response = client.post(_url(uuid4(), CHAIN[0]))

    assert response.status_code == 422
    assert recorder.approve_calls == []


@pytest.mark.parametrize("stage", ["business_profile", "seo_audit", "BRAND_STRATEGY", "personas"])
def test_post_rejects_anything_that_is_not_a_stage(
    client: TestClient, recorder: Recorder, stage: str
) -> None:
    """Only the seven stage names are valid in the path."""

    response = client.post(_url(uuid4(), stage), json={"version": 1})

    assert response.status_code == 422
    assert recorder.approve_calls == []


def test_post_rejects_a_malformed_workspace_id(client: TestClient, recorder: Recorder) -> None:
    """The workspace id must be a UUID."""

    response = client.post(
        "/api/v1/workspaces/not-a-uuid/approvals/market_research", json={"version": 1}
    )

    assert response.status_code == 422
    assert recorder.approve_calls == []


@pytest.mark.parametrize(
    ("error", "status_code", "detail"),
    [
        (ResourceNotFoundError("Workspace not found."), 404, "Workspace not found."),
        (
            ResourceNotFoundError("The market research stage has not been generated yet."),
            404,
            "The market research stage has not been generated yet.",
        ),
        (
            StaleResourceError(
                "The market research stage is now at version 3, not version 2. "
                "Review the latest version before approving it."
            ),
            409,
            "The market research stage is now at version 3, not version 2. "
            "Review the latest version before approving it.",
        ),
        (
            ResourceConflictError("The market research stage's approval was changed. Try again."),
            409,
            "The market research stage's approval was changed. Try again.",
        ),
    ],
)
def test_post_translates_domain_errors(
    client: TestClient,
    recorder: Recorder,
    error: Exception,
    status_code: int,
    detail: str,
) -> None:
    """Domain failures become 404 and 409 with their own safe message."""

    recorder.error = error

    response = client.post(_url(uuid4(), CHAIN[1]), json={"version": 2})

    assert response.status_code == status_code
    assert response.json() == {"detail": detail}


def test_post_database_failure_is_sanitized(client: TestClient, recorder: Recorder) -> None:
    """Infrastructure detail must not leak through the approve route."""

    recorder.error = DatabaseUnavailableError("host=10.0.0.5 password=hunter2")

    response = client.post(_url(uuid4(), CHAIN[1]), json={"version": 1})

    assert response.status_code == 503
    assert response.json() == {"detail": "A required database operation could not be completed."}
    assert "hunter2" not in response.text


# --- GET: read ---------------------------------------------------------------------


def test_get_returns_the_current_approval(client: TestClient, recorder: Recorder) -> None:
    """A current approval is returned with the record it was given for."""

    workspace_id = uuid4()

    response = client.get(_url(workspace_id, CHAIN[2]))
    body = response.json()

    assert response.status_code == 200
    assert recorder.get_calls == [(workspace_id, CHAIN[2])]
    assert body["stage"] == "competitor_analysis"
    assert body["source"] == "human"
    assert body["approved_at"].startswith("2026-10-03T12:00:00")


def test_get_for_an_unapproved_stage_is_404(client: TestClient, recorder: Recorder) -> None:
    """A stage with no current approval is not found, not an empty approval."""

    recorder.error = ResourceNotFoundError("The competitor analysis stage is not approved.")

    response = client.get(_url(uuid4(), CHAIN[2]))

    assert response.status_code == 404
    assert response.json() == {"detail": "The competitor analysis stage is not approved."}


def test_get_rejects_something_that_is_not_a_stage(client: TestClient, recorder: Recorder) -> None:
    """An unknown stage name is a validation error, not a lookup."""

    response = client.get(_url(uuid4(), "business_profile"))

    assert response.status_code == 422
    assert recorder.get_calls == []


def test_get_database_failure_is_sanitized(client: TestClient, recorder: Recorder) -> None:
    """Infrastructure detail must not leak through the read route."""

    recorder.error = DatabaseUnavailableError("host=10.0.0.5 password=hunter2")

    response = client.get(_url(uuid4(), CHAIN[2]))

    assert response.status_code == 503
    assert "hunter2" not in response.text


# --- DELETE: revoke ------------------------------------------------------------------


def test_delete_revokes_with_an_empty_204(client: TestClient, recorder: Recorder) -> None:
    """Revoking succeeds with no body and hands the right stage to the service."""

    workspace_id = uuid4()

    response = client.delete(_url(workspace_id, CHAIN[4]))

    assert response.status_code == 204
    assert response.content == b""
    assert recorder.revoke_calls == [(workspace_id, CHAIN[4])]


def test_delete_needs_no_body_or_version(client: TestClient, recorder: Recorder) -> None:
    """Revoking is idempotent, so unlike approving it carries no concurrency token."""

    response = client.delete(_url(uuid4(), CHAIN[4]))

    assert response.status_code == 204


def test_delete_for_unknown_workspace_is_404(client: TestClient, recorder: Recorder) -> None:
    """A workspace that does not exist cannot have approvals revoked."""

    recorder.error = ResourceNotFoundError("Workspace not found.")

    response = client.delete(_url(uuid4(), CHAIN[4]))

    assert response.status_code == 404
    assert response.json() == {"detail": "Workspace not found."}


def test_delete_rejects_something_that_is_not_a_stage(
    client: TestClient, recorder: Recorder
) -> None:
    """An unknown stage name is a validation error."""

    response = client.delete(_url(uuid4(), "business_profile"))

    assert response.status_code == 422
    assert recorder.revoke_calls == []


def test_delete_database_failure_is_sanitized(client: TestClient, recorder: Recorder) -> None:
    """Infrastructure detail must not leak through the revoke route."""

    recorder.error = DatabaseUnavailableError("host=10.0.0.5 password=hunter2")

    response = client.delete(_url(uuid4(), CHAIN[4]))

    assert response.status_code == 503
    assert "hunter2" not in response.text
