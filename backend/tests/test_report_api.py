"""Unit tests for the report HTTP contract.

The database session dependency and the services behind the routes are
replaced with stubs, so these tests pin the HTTP behavior only: paths,
parameter validation, response shapes, headers, status codes, and error
sanitization. Behavior against real PostgreSQL is covered by the
integration tests.
"""

from collections.abc import Iterator
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.api import reports as api_module
from backend.app.config import Settings
from backend.app.db.dependencies import get_db_session
from backend.app.db.errors import DatabaseUnavailableError
from backend.app.exceptions import ResourceConflictError, ResourceNotFoundError
from backend.app.llm.dependencies import get_llm_gateway
from backend.app.main import create_application
from backend.app.reports.delivery import MEDIA_TYPES
from backend.app.schemas.report import (
    STAGE_ORDER,
    OmissionReason,
    OmittedStage,
    ReportDocument,
    ReportFormat,
    ReportStageSection,
)
from backend.app.schemas.strategy_orchestration import (
    ProvenanceState,
    StageApprovalState,
    StageStaleness,
    StrategyStage,
)
from backend.app.services.report_export import ReportFile

JSON_PATH = "/api/v1/workspaces/{workspace_id}/report"
FILE_PATH = "/api/v1/workspaces/{workspace_id}/report/{report_format}"
NOW = datetime(2026, 11, 4, 12, 0, tzinfo=UTC)


def _report(workspace_id: UUID, *, drafts: bool = False) -> ReportDocument:
    sections = [
        ReportStageSection(
            stage=stage,
            generation=2,
            record_id=uuid4(),
            record_version=1,
            approval=StageApprovalState.DRAFT if drafts else StageApprovalState.APPROVED,
            staleness=StageStaleness(provenance=ProvenanceState.TRACKED),
            model_used="test-model",
            generated_at=NOW,
            content={"overview": f"Overview of {stage.value}."},
        )
        for stage in STAGE_ORDER[:2]
    ]
    return ReportDocument(
        workspace_id=workspace_id,
        business_name="Acme Coffee",
        generated_at=NOW,
        sections=sections,
        omitted=[
            OmittedStage(stage=stage, reason=OmissionReason.NOT_GENERATED)
            for stage in STAGE_ORDER[2:]
        ],
    )


class Recorder:
    """Shared record of how the stubbed services were used."""

    def __init__(self) -> None:
        self.report_calls: list[UUID] = []
        self.export_calls: list[tuple[UUID, ReportFormat]] = []
        self.error: Exception | None = None
        self.filename = "acme-coffee-strategy-report-2026-11-04.pdf"
        self.content = b"%PDF-stub-bytes"
        self.drafts = False


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    """Replace the route's services with stubs and return their call record."""

    record = Recorder()

    class StubContent:
        def __init__(self, session: object) -> None:
            pass

        def build(self, workspace_id: UUID) -> ReportDocument:
            record.report_calls.append(workspace_id)
            if record.error is not None:
                raise record.error
            return _report(workspace_id, drafts=record.drafts)

    class StubExport:
        def __init__(self, session: object) -> None:
            pass

        def export(self, workspace_id: UUID, report_format: ReportFormat) -> ReportFile:
            record.export_calls.append((workspace_id, report_format))
            if record.error is not None:
                raise record.error
            return ReportFile(
                filename=record.filename,
                media_type=MEDIA_TYPES[report_format],
                content=record.content,
            )

    monkeypatch.setattr(api_module, "ReportContentService", StubContent)
    monkeypatch.setattr(api_module, "ReportExportService", StubExport)
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


def _json_url(workspace_id: UUID | str) -> str:
    return f"/api/v1/workspaces/{workspace_id}/report"


def _file_url(workspace_id: UUID | str, report_format: ReportFormat | str) -> str:
    value = report_format.value if isinstance(report_format, ReportFormat) else report_format
    return f"{_json_url(workspace_id)}/{value}"


# --- contract ---------------------------------------------------------------------------


def test_routes_are_registered_with_the_expected_contract(application: FastAPI) -> None:
    """Two read-only paths, a typed JSON response, and documented failures."""

    spec = application.openapi()
    as_data = spec["paths"][JSON_PATH]
    as_file = spec["paths"][FILE_PATH]

    assert set(as_data) == {"get"}
    assert set(as_file) == {"get"}
    schema = as_data["get"]["responses"]["200"]["content"]["application/json"]["schema"]
    assert schema == {"$ref": "#/components/schemas/ReportDocument"}
    assert "404" in as_data["get"]["responses"]
    assert {"404", "409"} <= set(as_file["get"]["responses"])
    assert as_data["get"]["tags"] == as_file["get"]["tags"] == ["reports"]


def test_the_file_route_documents_each_media_type_as_binary(application: FastAPI) -> None:
    content = application.openapi()["paths"][FILE_PATH]["get"]["responses"]["200"]["content"]

    assert set(content) == set(MEDIA_TYPES.values())
    for media_type in MEDIA_TYPES.values():
        assert content[media_type]["schema"] == {"type": "string", "format": "binary"}


def test_the_format_parameter_accepts_exactly_the_three_formats(application: FastAPI) -> None:
    spec = application.openapi()
    parameters = spec["paths"][FILE_PATH]["get"]["parameters"]
    [report_format] = [item for item in parameters if item["name"] == "report_format"]

    assert report_format["schema"] == {"$ref": "#/components/schemas/ReportFormat"}
    assert spec["components"]["schemas"]["ReportFormat"]["enum"] == ["pdf", "docx", "csv"]


def test_the_report_schema_includes_its_computed_summary(application: FastAPI) -> None:
    schemas = application.openapi()["components"]["schemas"]

    assert {"is_complete", "is_clean", "flagged_stages"} <= set(
        schemas["ReportDocument"]["properties"]
    )
    assert {"flags", "is_clean"} <= set(schemas["ReportStageSection"]["properties"])


def test_report_routes_do_not_depend_on_the_llm_gateway(
    application: FastAPI, recorder: Recorder
) -> None:
    """Reading a report is database-only and must work with the model provider down."""

    def broken_gateway() -> None:
        raise RuntimeError("LLM gateway must not be constructed for report routes")

    application.dependency_overrides[get_llm_gateway] = broken_gateway

    with TestClient(application, raise_server_exceptions=False) as test_client:
        as_data = test_client.get(_json_url(uuid4()))
        as_file = test_client.get(_file_url(uuid4(), ReportFormat.PDF))

    assert (as_data.status_code, as_file.status_code) == (200, 200)


def test_report_routes_depend_only_on_the_database_session() -> None:
    routes = api_module.router.routes

    assert len(routes) == 2
    for route in routes:
        assert [item.call for item in route.dependant.dependencies] == [get_db_session]  # type: ignore[attr-defined]


@pytest.mark.parametrize("method", ["post", "put", "patch", "delete"])
def test_a_report_cannot_be_changed_through_the_api(client: TestClient, method: str) -> None:
    """There is no write route: a report is derived, never stored."""

    for url in (_json_url(uuid4()), _file_url(uuid4(), ReportFormat.PDF)):
        assert getattr(client, method)(url).status_code == 405


# --- GET: the report as data ------------------------------------------------------------


def test_the_report_is_returned_as_json_for_the_workspace(
    client: TestClient, recorder: Recorder
) -> None:
    workspace_id = uuid4()

    response = client.get(_json_url(workspace_id))

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    assert recorder.report_calls == [workspace_id]
    body = response.json()
    assert body["workspace_id"] == str(workspace_id)
    assert body["business_name"] == "Acme Coffee"
    assert [section["stage"] for section in body["sections"]] == [
        stage.value for stage in STAGE_ORDER[:2]
    ]
    assert [item["stage"] for item in body["omitted"]] == [stage.value for stage in STAGE_ORDER[2:]]
    assert body["omitted"][0]["reason"] == "not_generated"


def test_the_json_report_carries_the_computed_trust_summary(
    client: TestClient, recorder: Recorder
) -> None:
    recorder.drafts = True

    body = client.get(_json_url(uuid4())).json()

    assert body["is_complete"] is False
    assert body["is_clean"] is False
    assert body["flagged_stages"] == [stage.value for stage in STAGE_ORDER[:2]]
    assert body["sections"][0]["flags"] == ["draft"]
    assert body["sections"][0]["is_clean"] is False
    assert body["sections"][0]["approval"] == "draft"
    assert body["sections"][0]["generation"] == 2
    assert body["sections"][0]["content"] == {"overview": "Overview of business_understanding."}


def test_a_clean_section_has_no_flags_in_the_json(client: TestClient, recorder: Recorder) -> None:
    body = client.get(_json_url(uuid4())).json()

    assert body["sections"][0]["flags"] == []
    assert body["sections"][0]["is_clean"] is True
    assert body["flagged_stages"] == []


def test_the_json_report_is_never_cached(client: TestClient, recorder: Recorder) -> None:
    assert client.get(_json_url(uuid4())).headers["cache-control"] == "no-store"


def test_a_malformed_workspace_id_is_rejected_before_the_service(
    client: TestClient, recorder: Recorder
) -> None:
    for url in (_json_url("not-a-uuid"), _file_url("not-a-uuid", ReportFormat.PDF)):
        assert client.get(url).status_code == 422

    assert recorder.report_calls == []
    assert recorder.export_calls == []


def test_an_unknown_workspace_is_404_with_its_own_safe_message(
    client: TestClient, recorder: Recorder
) -> None:
    recorder.error = ResourceNotFoundError("Workspace not found.")

    as_data = client.get(_json_url(uuid4()))
    as_file = client.get(_file_url(uuid4(), ReportFormat.DOCX))

    assert (as_data.status_code, as_file.status_code) == (404, 404)
    assert as_data.json() == {"detail": "Workspace not found."}
    assert as_file.json() == {"detail": "Workspace not found."}


def test_json_failures_are_sanitized(client: TestClient, recorder: Recorder) -> None:
    recorder.error = DatabaseUnavailableError("host=10.0.0.5 password=hunter2")
    database = client.get(_json_url(uuid4()))
    recorder.error = RuntimeError("secret internal detail")
    unexpected = client.get(_json_url(uuid4()))

    assert database.status_code == 503
    assert database.json() == {"detail": "A required database operation could not be completed."}
    assert unexpected.status_code == 500
    assert "secret" not in unexpected.text
    assert "hunter2" not in database.text


# --- GET: the report as a file ----------------------------------------------------------


@pytest.mark.parametrize("report_format", list(ReportFormat))
def test_each_format_is_returned_as_an_attachment(
    client: TestClient, recorder: Recorder, report_format: ReportFormat
) -> None:
    workspace_id = uuid4()
    recorder.filename = f"acme-coffee-strategy-report-2026-11-04.{report_format.value}"

    response = client.get(_file_url(workspace_id, report_format))

    assert response.status_code == 200
    assert recorder.export_calls == [(workspace_id, report_format)]
    assert response.content == recorder.content
    assert response.headers["content-type"] == MEDIA_TYPES[report_format]
    assert response.headers["content-disposition"] == (
        f'attachment; filename="acme-coffee-strategy-report-2026-11-04.{report_format.value}"'
    )
    assert response.headers["content-length"] == str(len(recorder.content))


def test_a_file_is_never_cached_or_sniffed(client: TestClient, recorder: Recorder) -> None:
    response = client.get(_file_url(uuid4(), ReportFormat.CSV))

    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"


def test_the_filename_header_cannot_be_broken_out_of(
    client: TestClient, recorder: Recorder
) -> None:
    recorder.filename = 'evil"\r\nSet-Cookie: x=1.pdf'

    response = client.get(_file_url(uuid4(), ReportFormat.PDF))

    assert response.status_code == 200
    assert (
        response.headers["content-disposition"]
        == 'attachment; filename="evil___Set-Cookie__x_1.pdf"'
    )
    assert "set-cookie" not in response.headers


@pytest.mark.parametrize("report_format", ["xlsx", "PDF", "Pdf", "json", "txt", "pdf%20", "0"])
def test_any_other_format_is_rejected_before_the_service(
    client: TestClient, recorder: Recorder, report_format: str
) -> None:
    response = client.get(_file_url(uuid4(), report_format))

    assert response.status_code == 422
    assert recorder.export_calls == []


def test_a_workspace_with_nothing_to_export_is_a_409_with_its_own_message(
    client: TestClient, recorder: Recorder
) -> None:
    recorder.error = ResourceConflictError("No stage has been generated yet.")

    response = client.get(_file_url(uuid4(), ReportFormat.PDF))

    assert response.status_code == 409
    assert response.json() == {"detail": "No stage has been generated yet."}


def test_file_failures_are_sanitized(client: TestClient, recorder: Recorder) -> None:
    recorder.error = DatabaseUnavailableError("host=10.0.0.5 password=hunter2")
    database = client.get(_file_url(uuid4(), ReportFormat.PDF))
    recorder.error = RuntimeError("secret internal detail")
    unexpected = client.get(_file_url(uuid4(), ReportFormat.DOCX))

    assert database.status_code == 503
    assert database.json() == {"detail": "A required database operation could not be completed."}
    assert unexpected.status_code == 500
    assert "secret" not in unexpected.text
    assert "hunter2" not in database.text


def test_an_error_response_carries_no_attachment_header(
    client: TestClient, recorder: Recorder
) -> None:
    recorder.error = ResourceNotFoundError("Workspace not found.")

    response = client.get(_file_url(uuid4(), ReportFormat.PDF))

    assert "content-disposition" not in response.headers
    assert response.headers["content-type"] == "application/json"


def test_the_json_and_file_routes_do_not_shadow_each_other(
    client: TestClient, recorder: Recorder
) -> None:
    workspace_id = uuid4()

    as_data = client.get(_json_url(workspace_id))
    as_file = client.get(_file_url(workspace_id, ReportFormat.CSV))

    assert as_data.headers["content-type"] == "application/json"
    assert as_file.headers["content-type"] == MEDIA_TYPES[ReportFormat.CSV]
    assert recorder.report_calls == [workspace_id]
    assert recorder.export_calls == [(workspace_id, ReportFormat.CSV)]


def test_every_stage_is_still_accounted_for_in_the_stub_report() -> None:
    """Guards the helper: the stub must be a valid report for the tests above to mean anything."""

    report = _report(uuid4())

    assert {section.stage for section in report.sections} | {
        item.stage for item in report.omitted
    } == set(StrategyStage)
