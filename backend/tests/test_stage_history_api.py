"""Unit tests for the stage history HTTP contract.

The database session dependency and the history service behind the
routes are replaced with stubs, so these tests pin the HTTP behavior
only: paths, parameter validation, response shapes, status codes, and
error sanitization. Behavior against real PostgreSQL is covered by the
integration tests.
"""

from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.api import stage_history as api_module
from backend.app.config import Settings
from backend.app.db.dependencies import get_db_session
from backend.app.db.errors import DatabaseUnavailableError
from backend.app.db.models.stage_snapshot import StageSnapshot
from backend.app.exceptions import ResourceNotFoundError
from backend.app.llm.dependencies import get_llm_gateway
from backend.app.main import create_application
from backend.app.schemas.strategy_orchestration import StrategyStage

CHAIN = list(StrategyStage)
LIST_PATH = "/api/v1/workspaces/{workspace_id}/history/{stage}"
DETAIL_PATH = "/api/v1/workspaces/{workspace_id}/history/{stage}/{generation}"
CREATED_AT = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)

SUMMARY_FIELDS = {
    "stage",
    "generation",
    "record_id",
    "record_version",
    "model_used",
    "input_tokens",
    "output_tokens",
    "created_at",
    "provenance",
}


def _row(
    workspace_id: UUID,
    stage: StrategyStage,
    generation: int,
    *,
    inputs: dict[str, Any] | None = None,
) -> StageSnapshot:
    """Build a snapshot row the way the service would return it."""

    return StageSnapshot(
        id=uuid4(),
        workspace_id=workspace_id,
        stage=stage.value,
        generation=generation,
        record_id=uuid4(),
        record_version=generation,
        content={"overview": f"Generation {generation}."},
        inputs=inputs,
        model_used="test-model",
        input_tokens=10 * generation,
        output_tokens=20 * generation,
        created_at=CREATED_AT,
    )


def _tracked_inputs() -> dict[str, Any]:
    """Recorded inputs in the shape stored in the database."""

    return {
        "profile": {"id": str(uuid4()), "version": 2},
        "stages": {
            "business_understanding": {
                "generation": 3,
                "record_id": str(uuid4()),
                "record_version": 2,
            }
        },
    }


class Recorder:
    """Shared record of how the stubbed service was used."""

    def __init__(self) -> None:
        self.list_calls: list[tuple[UUID, StrategyStage]] = []
        self.get_calls: list[tuple[UUID, StrategyStage, int]] = []
        self.generations: list[int] = [3, 2, 1]
        self.inputs: dict[str, Any] | None = None
        self.error: Exception | None = None


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    """Replace the route's service with a stub and return its call record."""

    record = Recorder()

    class StubService:
        def __init__(self, session: object) -> None:
            pass

        def list_history(self, workspace_id: UUID, stage: StrategyStage) -> list[StageSnapshot]:
            record.list_calls.append((workspace_id, stage))
            if record.error is not None:
                raise record.error
            return [
                _row(workspace_id, stage, generation, inputs=record.inputs)
                for generation in record.generations
            ]

        def get_snapshot(
            self, workspace_id: UUID, stage: StrategyStage, generation: int
        ) -> StageSnapshot:
            record.get_calls.append((workspace_id, stage, generation))
            if record.error is not None:
                raise record.error
            return _row(workspace_id, stage, generation, inputs=record.inputs)

    monkeypatch.setattr(api_module, "StageHistoryService", StubService)
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


def _list_url(workspace_id: UUID | str, stage: StrategyStage | str) -> str:
    value = stage.value if isinstance(stage, StrategyStage) else stage
    return f"/api/v1/workspaces/{workspace_id}/history/{value}"


def _detail_url(workspace_id: UUID | str, stage: StrategyStage | str, generation: int | str) -> str:
    return f"{_list_url(workspace_id, stage)}/{generation}"


# --- contract ------------------------------------------------------------------


def test_routes_are_registered_with_the_expected_contract(application: FastAPI) -> None:
    """Two read-only paths, typed responses, and documented failures."""

    spec = application.openapi()
    listing = spec["paths"][LIST_PATH]
    detail = spec["paths"][DETAIL_PATH]

    assert set(listing) == {"get"}
    assert set(detail) == {"get"}
    list_schema = listing["get"]["responses"]["200"]["content"]["application/json"]["schema"]
    detail_schema = detail["get"]["responses"]["200"]["content"]["application/json"]["schema"]
    assert list_schema == {"$ref": "#/components/schemas/StageHistoryResponse"}
    assert detail_schema == {"$ref": "#/components/schemas/StageSnapshotDetail"}
    assert "404" in listing["get"]["responses"]
    assert "404" in detail["get"]["responses"]


def test_generation_parameter_is_documented_as_a_bounded_positive_integer(
    application: FastAPI,
) -> None:
    """The bounds clients see are the ones the route enforces."""

    parameters = application.openapi()["paths"][DETAIL_PATH]["get"]["parameters"]
    [generation] = [item for item in parameters if item["name"] == "generation"]

    assert generation["schema"]["type"] == "integer"
    assert generation["schema"]["minimum"] == 1
    assert generation["schema"]["maximum"] == 2_147_483_647


def test_summary_leaves_out_content_and_inputs_but_detail_has_them(application: FastAPI) -> None:
    """Listing stays light; the detail view carries the full generation."""

    schemas = application.openapi()["components"]["schemas"]

    assert set(schemas["StageSnapshotSummary"]["properties"]) == SUMMARY_FIELDS
    assert set(schemas["StageSnapshotDetail"]["properties"]) == SUMMARY_FIELDS | {
        "workspace_id",
        "content",
        "inputs",
    }


def test_stage_path_accepts_exactly_the_seven_generated_stages(application: FastAPI) -> None:
    """The business profile has no generations, so it is not a valid stage here."""

    stage_schema = application.openapi()["components"]["schemas"]["StrategyStage"]

    assert stage_schema["enum"] == [stage.value for stage in CHAIN]


def test_history_routes_do_not_depend_on_the_llm_gateway(
    application: FastAPI, recorder: Recorder
) -> None:
    """Reading history is database-only and must work with the model provider down."""

    def broken_gateway() -> None:
        raise RuntimeError("LLM gateway must not be constructed for history routes")

    application.dependency_overrides[get_llm_gateway] = broken_gateway

    with TestClient(application, raise_server_exceptions=False) as test_client:
        listed = test_client.get(_list_url(uuid4(), CHAIN[1]))
        read = test_client.get(_detail_url(uuid4(), CHAIN[1], 1))

    assert (listed.status_code, read.status_code) == (200, 200)


@pytest.mark.parametrize("method", ["post", "put", "patch", "delete"])
def test_history_cannot_be_changed_through_the_api(client: TestClient, method: str) -> None:
    """There is no write route: history is read-only, with no restore either."""

    for url in (_list_url(uuid4(), CHAIN[0]), _detail_url(uuid4(), CHAIN[0], 1)):
        assert getattr(client, method)(url).status_code == 405


# --- GET: list -----------------------------------------------------------------


def test_list_returns_generations_newest_first_without_content(
    client: TestClient, recorder: Recorder
) -> None:
    """The path's workspace and stage reach the service; summaries carry no content."""

    workspace_id = uuid4()

    response = client.get(_list_url(workspace_id, StrategyStage.BRAND_STRATEGY))
    body = response.json()

    assert response.status_code == 200
    assert recorder.list_calls == [(workspace_id, StrategyStage.BRAND_STRATEGY)]
    assert set(body) == {"workspace_id", "stage", "snapshots", "latest_generation"}
    assert body["workspace_id"] == str(workspace_id)
    assert body["stage"] == "brand_strategy"
    assert body["latest_generation"] == 3
    assert [item["generation"] for item in body["snapshots"]] == [3, 2, 1]
    for item in body["snapshots"]:
        assert set(item) == SUMMARY_FIELDS
    assert body["snapshots"][0]["input_tokens"] == 30
    assert body["snapshots"][0]["created_at"].startswith("2026-10-10T12:00:00")


def test_list_reports_provenance_without_listing_the_inputs(
    client: TestClient, recorder: Recorder
) -> None:
    """Whether inputs are known is shown; what they are is left to the detail view."""

    recorder.inputs = _tracked_inputs()

    tracked = client.get(_list_url(uuid4(), CHAIN[3])).json()["snapshots"]
    recorder.inputs = None
    unknown = client.get(_list_url(uuid4(), CHAIN[3])).json()["snapshots"]

    assert {item["provenance"] for item in tracked} == {"tracked"}
    assert {item["provenance"] for item in unknown} == {"unknown"}
    assert all("inputs" not in item for item in tracked)


def test_list_of_a_stage_never_generated_is_empty_not_missing(
    client: TestClient, recorder: Recorder
) -> None:
    """No history is an empty list with a null latest generation."""

    recorder.generations = []

    response = client.get(_list_url(uuid4(), CHAIN[6]))

    assert response.status_code == 200
    assert response.json()["snapshots"] == []
    assert response.json()["latest_generation"] is None


@pytest.mark.parametrize("stage", CHAIN)
def test_list_accepts_every_generated_stage(
    client: TestClient, recorder: Recorder, stage: StrategyStage
) -> None:
    """Each of the seven stages has a history."""

    response = client.get(_list_url(uuid4(), stage))

    assert response.status_code == 200
    assert recorder.list_calls[0][1] is stage


@pytest.mark.parametrize("stage", ["business_profile", "seo_audit", "BRAND_STRATEGY", "personas"])
def test_list_rejects_anything_that_is_not_a_stage(
    client: TestClient, recorder: Recorder, stage: str
) -> None:
    """Only the seven stage names are valid in the path."""

    response = client.get(_list_url(uuid4(), stage))

    assert response.status_code == 422
    assert recorder.list_calls == []


def test_list_rejects_a_malformed_workspace_id(client: TestClient, recorder: Recorder) -> None:
    """The workspace id must be a UUID."""

    response = client.get(_list_url("not-a-uuid", CHAIN[0]))

    assert response.status_code == 422
    assert recorder.list_calls == []


def test_list_for_an_unknown_workspace_is_404(client: TestClient, recorder: Recorder) -> None:
    """A missing workspace is reported with its own safe message."""

    recorder.error = ResourceNotFoundError("Workspace not found.")

    response = client.get(_list_url(uuid4(), CHAIN[0]))

    assert response.status_code == 404
    assert response.json() == {"detail": "Workspace not found."}


def test_list_failures_are_sanitized(client: TestClient, recorder: Recorder) -> None:
    """Infrastructure detail and unexpected errors never leak through the list route."""

    recorder.error = DatabaseUnavailableError("host=10.0.0.5 password=hunter2")
    database = client.get(_list_url(uuid4(), CHAIN[0]))
    recorder.error = RuntimeError("secret internal detail")
    unexpected = client.get(_list_url(uuid4(), CHAIN[0]))

    assert database.status_code == 503
    assert database.json() == {"detail": "A required database operation could not be completed."}
    assert unexpected.status_code == 500
    assert "secret" not in unexpected.text
    assert "hunter2" not in database.text


# --- GET: one generation ---------------------------------------------------------


def test_detail_returns_one_generation_in_full(client: TestClient, recorder: Recorder) -> None:
    """The path's workspace, stage, and generation reach the service as typed values."""

    workspace_id = uuid4()
    recorder.inputs = _tracked_inputs()

    response = client.get(_detail_url(workspace_id, StrategyStage.CUSTOMER_PERSONAS, 2))
    body = response.json()

    assert response.status_code == 200
    assert recorder.get_calls == [(workspace_id, StrategyStage.CUSTOMER_PERSONAS, 2)]
    assert set(body) == SUMMARY_FIELDS | {"workspace_id", "content", "inputs"}
    assert body["workspace_id"] == str(workspace_id)
    assert body["stage"] == "customer_personas"
    assert body["generation"] == 2
    assert body["content"] == {"overview": "Generation 2."}
    assert body["provenance"] == "tracked"
    assert body["inputs"]["profile"]["version"] == 2
    assert body["inputs"]["stages"]["business_understanding"]["generation"] == 3


def test_detail_of_a_stage_from_before_history_has_null_inputs(
    client: TestClient, recorder: Recorder
) -> None:
    """Backfilled generations say plainly that what they were built from is unknown."""

    recorder.inputs = None

    body = client.get(_detail_url(uuid4(), CHAIN[1], 1)).json()

    assert body["inputs"] is None
    assert body["provenance"] == "unknown"


def test_detail_accepts_the_largest_possible_generation(
    client: TestClient, recorder: Recorder
) -> None:
    """The upper bound is inclusive and reaches the service."""

    response = client.get(_detail_url(uuid4(), CHAIN[1], 2_147_483_647))

    assert response.status_code == 200
    assert recorder.get_calls[0][2] == 2_147_483_647


@pytest.mark.parametrize(
    "generation", ["0", "-1", "abc", "1.5", "2147483648", "99999999999999999999"]
)
def test_detail_rejects_invalid_generations_without_calling_the_service(
    client: TestClient, recorder: Recorder, generation: str
) -> None:
    """Non-positive, non-integer, and out-of-range numbers are 422, never a database error."""

    response = client.get(_detail_url(uuid4(), CHAIN[1], generation))

    assert response.status_code == 422
    assert recorder.get_calls == []


@pytest.mark.parametrize("stage", ["business_profile", "seo_audit", "BRAND_STRATEGY"])
def test_detail_rejects_anything_that_is_not_a_stage(
    client: TestClient, recorder: Recorder, stage: str
) -> None:
    """An unknown stage name is a validation error, not a lookup."""

    response = client.get(_detail_url(uuid4(), stage, 1))

    assert response.status_code == 422
    assert recorder.get_calls == []


@pytest.mark.parametrize(
    "message",
    ["Workspace not found.", "That generation of this stage does not exist."],
)
def test_detail_translates_not_found_with_its_own_safe_message(
    client: TestClient, recorder: Recorder, message: str
) -> None:
    """A missing workspace and a missing generation are both 404."""

    recorder.error = ResourceNotFoundError(message)

    response = client.get(_detail_url(uuid4(), CHAIN[1], 7))

    assert response.status_code == 404
    assert response.json() == {"detail": message}


def test_detail_failures_are_sanitized(client: TestClient, recorder: Recorder) -> None:
    """Infrastructure detail never leaks through the detail route."""

    recorder.error = DatabaseUnavailableError("host=10.0.0.5 password=hunter2")

    response = client.get(_detail_url(uuid4(), CHAIN[1], 1))

    assert response.status_code == 503
    assert "hunter2" not in response.text
