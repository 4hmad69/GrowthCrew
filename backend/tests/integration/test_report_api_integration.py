"""Real PostgreSQL integration tests: the report routes.

Drives the real agents through the API with the deterministic "local"
providers and no Tavily key, then reads the report back over HTTP as data
and as PDF, Word and CSV files, and opens each file with an independent
reader. The headline requirements are that every format tells the same
story as the data (approval, staleness, omissions), that nothing is
written, that a report never needs the model provider, and that a stage
that has not earned trust is never presented as if it had.
"""

import csv
import io
import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

import pytest
from docx import Document
from fastapi.testclient import TestClient
from pypdf import PdfReader
from sqlalchemy import func, select

from backend.app.config import Settings
from backend.app.db.database import Database
from backend.app.db.models.market_research import MarketResearch
from backend.app.db.models.stage_approval import StageApproval
from backend.app.db.models.stage_snapshot import StageSnapshot
from backend.app.db.models.workspace import Workspace
from backend.app.llm.dependencies import get_llm_gateway
from backend.app.main import create_application
from backend.app.reports.delivery import MEDIA_TYPES
from backend.app.schemas.report import STAGE_ORDER, ReportFormat
from backend.app.schemas.strategy_orchestration import StrategyStage
from backend.app.services.stage_history import StageHistoryService

RUN_INTEGRATION_TESTS = os.getenv("GROWTHCREW_RUN_INTEGRATION_TESTS") == "1"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not RUN_INTEGRATION_TESTS,
        reason=("Set GROWTHCREW_RUN_INTEGRATION_TESTS=1 to run PostgreSQL integration tests."),
    ),
]

S = StrategyStage
BUSINESS_NAME = "Café Müller & Söhne <GmbH>"
STAGE_VALUES = [stage.value for stage in STAGE_ORDER]


@dataclass
class Context:
    """An API client and its settings."""

    client: TestClient
    settings: Settings

    def url(self, workspace_id: UUID, path: str) -> str:
        return f"/api/v1/workspaces/{workspace_id}/{path}"


@pytest.fixture
def context() -> Iterator[Context]:
    """An API client backed by real PostgreSQL and the local stubs."""

    settings = Settings(
        environment="test",
        llm_provider="local",
        embeddings_provider="local",
        tavily_api_key=None,
    )
    if settings.database_url is None:
        pytest.fail("GROWTHCREW_DATABASE_URL is required for integration tests.")

    with TestClient(create_application(settings)) as client:
        yield Context(client=client, settings=settings)


def _delete_workspace(ctx: Context, workspace_id: UUID) -> None:
    database = Database.from_settings(ctx.settings)
    try:
        with database.session() as session:
            row = session.get(Workspace, workspace_id)
            if row is not None:
                session.delete(row)
                session.commit()
    finally:
        database.dispose()


@pytest.fixture
def workspace(context: Context) -> Iterator[UUID]:
    """A workspace with a business profile and no generated stages."""

    response = context.client.post("/api/v1/workspaces", json={"name": "Report API Workspace"})
    assert response.status_code == 201
    workspace_id = UUID(response.json()["id"])
    profile = context.client.post(
        context.url(workspace_id, "business-profile"),
        json={"business_name": BUSINESS_NAME, "industry": "Coffee"},
    )
    assert profile.status_code == 201

    yield workspace_id

    _delete_workspace(context, workspace_id)


def _with_session(ctx: Context, action: Any) -> Any:
    database = Database.from_settings(ctx.settings)
    try:
        with database.session() as session:
            result = action(session)
            session.commit()
            return result
    finally:
        database.dispose()


def _build_chain(ctx: Context, workspace_id: UUID) -> None:
    response = ctx.client.post(
        ctx.url(workspace_id, "generate-full-strategy"),
        json={"force_regenerate": False, "auto_approve": True},
    )
    assert response.status_code == 200
    assert [item["outcome"] for item in response.json()["stages"]] == ["generated"] * 7


def _revoke(ctx: Context, workspace_id: UUID, stage: StrategyStage) -> None:
    response = ctx.client.delete(ctx.url(workspace_id, f"approvals/{stage.value}"))
    assert response.status_code == 204


def _force(ctx: Context, workspace_id: UUID, path: str) -> None:
    response = ctx.client.post(ctx.url(workspace_id, path), json={"force_regenerate": True})
    assert response.status_code in {200, 201}


def _change_market_research(ctx: Context, workspace_id: UUID, **changes: Any) -> None:
    """Overwrite market research with new content, the way an agent regenerating it would."""

    def action(session: Any) -> None:
        record = session.scalars(
            select(MarketResearch).where(MarketResearch.workspace_id == workspace_id)
        ).one()
        for name, value in changes.items():
            setattr(record, name, value)
        StageHistoryService(session).record(workspace_id, S.MARKET_RESEARCH, record)

    _with_session(ctx, action)


def _get(ctx: Context, workspace_id: UUID, report_format: ReportFormat) -> Any:
    return ctx.client.get(ctx.url(workspace_id, f"report/{report_format.value}"))


def _json(ctx: Context, workspace_id: UUID) -> Any:
    response = ctx.client.get(ctx.url(workspace_id, "report"))
    assert response.status_code == 200
    return response.json()


def _csv_rows(response: Any) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(response.content.decode("utf-8-sig"), newline="")))


def _docx(response: Any) -> Any:
    return Document(io.BytesIO(response.content))


def _pdf_text(response: Any) -> str:
    reader = PdfReader(io.BytesIO(response.content))
    return " ".join(" ".join(page.extract_text().split()) for page in reader.pages)


def _counts(ctx: Context) -> tuple[int, int, int]:
    def action(session: Any) -> tuple[int, int, int]:
        return (
            session.scalar(select(func.count()).select_from(StageSnapshot)),
            session.scalar(select(func.count()).select_from(StageApproval)),
            session.scalar(select(func.count()).select_from(MarketResearch)),
        )

    return _with_session(ctx, action)


# --- an empty workspace -----------------------------------------------------------------


def test_a_workspace_with_nothing_generated_has_a_report_but_nothing_to_export(
    context: Context, workspace: UUID
) -> None:
    body = _json(context, workspace)

    assert body["sections"] == []
    assert [item["stage"] for item in body["omitted"]] == STAGE_VALUES
    assert body["is_complete"] is False
    for report_format in ReportFormat:
        response = _get(context, workspace, report_format)
        assert response.status_code == 409, report_format
        assert response.json() == {
            "detail": "No stage has been generated yet, so there is nothing to export."
        }
        assert "content-disposition" not in response.headers


def test_an_unknown_workspace_is_404_for_the_data_and_every_file(context: Context) -> None:
    unknown = uuid4()

    assert context.client.get(context.url(unknown, "report")).status_code == 404
    for report_format in ReportFormat:
        response = _get(context, unknown, report_format)
        assert response.status_code == 404, report_format
        assert response.json() == {"detail": "Workspace not found."}


def test_bad_inputs_are_rejected_before_any_work(context: Context, workspace: UUID) -> None:
    assert context.client.get(context.url(workspace, "report/xlsx")).status_code == 422
    assert context.client.get(context.url(workspace, "report/PDF")).status_code == 422
    assert context.client.get("/api/v1/workspaces/not-a-uuid/report").status_code == 422
    assert context.client.get("/api/v1/workspaces/not-a-uuid/report/pdf").status_code == 422


# --- a freshly built, approved chain ----------------------------------------------------


def test_the_json_report_of_a_fresh_chain_is_complete_and_clean(
    context: Context, workspace: UUID
) -> None:
    _build_chain(context, workspace)

    response = context.client.get(context.url(workspace, "report"))
    body = response.json()

    assert response.headers["cache-control"] == "no-store"
    assert body["workspace_id"] == str(workspace)
    assert body["business_name"] == BUSINESS_NAME
    assert body["is_complete"] is True
    assert body["is_clean"] is True
    assert body["flagged_stages"] == []
    assert body["omitted"] == []
    assert [section["stage"] for section in body["sections"]] == STAGE_VALUES
    for section in body["sections"]:
        assert section["generation"] == 1, section["stage"]
        assert section["approval"] == "approved", section["stage"]
        assert section["flags"] == [], section["stage"]
        assert section["staleness"]["provenance"] == "tracked", section["stage"]


@pytest.mark.parametrize("report_format", list(ReportFormat))
def test_every_format_downloads_with_the_right_headers(
    context: Context, workspace: UUID, report_format: ReportFormat
) -> None:
    _build_chain(context, workspace)

    response = _get(context, workspace, report_format)

    assert response.status_code == 200
    assert response.headers["content-type"] == MEDIA_TYPES[report_format]
    assert response.headers["content-disposition"].startswith(
        'attachment; filename="cafe-muller-sohne-gmbh-strategy-report-'
    )
    assert response.headers["content-disposition"].endswith(f'.{report_format.value}"')
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["content-length"] == str(len(response.content))


def test_the_pdf_opens_and_names_the_business_and_every_stage(
    context: Context, workspace: UUID
) -> None:
    _build_chain(context, workspace)

    response = _get(context, workspace, ReportFormat.PDF)
    reader = PdfReader(io.BytesIO(response.content))
    text = _pdf_text(response)

    assert response.content.startswith(b"%PDF-")
    assert len(reader.pages) >= 2
    assert BUSINESS_NAME in text
    assert "Complete and current" in text
    assert [item.title for item in reader.outline] == [
        "Business Understanding",
        "Market Research",
        "Competitor Analysis",
        "Customer Personas",
        "Brand Strategy",
        "Marketing Strategy",
        "Content Plan",
    ]


def test_the_word_file_opens_and_names_the_business_and_every_stage(
    context: Context, workspace: UUID
) -> None:
    _build_chain(context, workspace)

    document = _docx(_get(context, workspace, ReportFormat.DOCX))
    headings = [p.text for p in document.paragraphs if p.style.name == "Heading 1"]

    assert document.paragraphs[0].text == BUSINESS_NAME
    assert "Complete and current" in document.paragraphs[3].text
    assert headings == [
        "Business Understanding",
        "Market Research",
        "Competitor Analysis",
        "Customer Personas",
        "Brand Strategy",
        "Marketing Strategy",
        "Content Plan",
    ]


def test_the_csv_has_the_stage_values_and_every_stage_is_clean(
    context: Context, workspace: UUID
) -> None:
    _build_chain(context, workspace)

    rows = _csv_rows(_get(context, workspace, ReportFormat.CSV))

    assert {row["stage"] for row in rows} == set(STAGE_VALUES)
    assert [row["stage"] for row in rows] == sorted(
        (row["stage"] for row in rows), key=STAGE_VALUES.index
    )
    assert {row["approval"] for row in rows} == {"approved"}
    assert {row["flags"] for row in rows} == {""}
    assert {row["generation"] for row in rows} == {"1"}


def test_the_files_carry_the_same_content_as_the_data(context: Context, workspace: UUID) -> None:
    """Whatever the JSON says a stage holds, the CSV rows for that stage say too."""

    _build_chain(context, workspace)
    body = _json(context, workspace)
    rows = _csv_rows(_get(context, workspace, ReportFormat.CSV))

    for section in body["sections"]:
        stage_rows = {row["path"]: row["value"] for row in rows if row["stage"] == section["stage"]}
        for field, value in section["content"].items():
            if isinstance(value, str) and value and not value.startswith(("=", "+", "-", "@")):
                assert stage_rows[field] == value, (section["stage"], field)


# --- the report is derived and read-only ------------------------------------------------


def test_exporting_writes_nothing_and_is_repeatable(context: Context, workspace: UUID) -> None:
    _build_chain(context, workspace)
    before = _counts(context)

    first_json = _json(context, workspace)
    first_csv = _get(context, workspace, ReportFormat.CSV)
    for report_format in ReportFormat:
        assert _get(context, workspace, report_format).status_code == 200
    second_json = _json(context, workspace)
    second_csv = _get(context, workspace, ReportFormat.CSV)

    assert _counts(context) == before
    assert {k: v for k, v in first_json.items() if k != "generated_at"} == {
        k: v for k, v in second_json.items() if k != "generated_at"
    }
    assert first_csv.content == second_csv.content


def test_a_report_never_needs_the_model_provider(context: Context, workspace: UUID) -> None:
    _build_chain(context, workspace)

    def broken_gateway() -> None:
        raise RuntimeError("LLM gateway must not be constructed for report routes")

    context.client.app.dependency_overrides[get_llm_gateway] = broken_gateway  # type: ignore[attr-defined]
    try:
        statuses = [context.client.get(context.url(workspace, "report")).status_code]
        statuses += [_get(context, workspace, fmt).status_code for fmt in ReportFormat]
    finally:
        context.client.app.dependency_overrides.pop(get_llm_gateway, None)  # type: ignore[attr-defined]

    assert statuses == [200, 200, 200, 200]


# --- every format says what the data says -----------------------------------------------


def _drafts_and_stale(context: Context, workspace: UUID) -> None:
    """Personas is unapproved and out of date; marketing strategy is out of date."""

    _build_chain(context, workspace)
    _force(context, workspace, "market-research")
    _revoke(context, workspace, S.CUSTOMER_PERSONAS)


def test_flags_appear_in_the_data_and_in_every_file(context: Context, workspace: UUID) -> None:
    _drafts_and_stale(context, workspace)

    body = _json(context, workspace)
    flagged = {
        section["stage"]: section["flags"] for section in body["sections"] if section["flags"]
    }
    csv_rows = _csv_rows(_get(context, workspace, ReportFormat.CSV))
    docx_text = [p.text for p in _docx(_get(context, workspace, ReportFormat.DOCX)).paragraphs]
    pdf_text = _pdf_text(_get(context, workspace, ReportFormat.PDF))

    assert body["is_clean"] is False
    assert flagged == {
        "customer_personas": ["draft", "stale"],
        "marketing_strategy": ["stale"],
    }
    assert body["flagged_stages"] == ["customer_personas", "marketing_strategy"]
    flags_by_stage = {row["stage"]: row["flags"] for row in csv_rows}
    assert flags_by_stage["customer_personas"] == "draft;stale"
    assert flags_by_stage["marketing_strategy"] == "stale"
    assert flags_by_stage["market_research"] == ""
    assert any(text.startswith("Draft: ") for text in docx_text)
    assert any(text.startswith("Out of date: ") for text in docx_text)
    assert "Draft: this version of the stage has not been approved." in pdf_text
    assert "Out of date: built from inputs that have since changed" in pdf_text
    assert "2 of 7 stages need attention (Customer Personas, Marketing Strategy)" in pdf_text


def test_a_content_change_is_reported_as_the_new_unapproved_generation_everywhere(
    context: Context, workspace: UUID
) -> None:
    _build_chain(context, workspace)
    _change_market_research(context, workspace, market_overview="A rewritten overview.")

    section = next(
        item for item in _json(context, workspace)["sections"] if item["stage"] == "market_research"
    )
    rows = [
        row
        for row in _csv_rows(_get(context, workspace, ReportFormat.CSV))
        if row["stage"] == "market_research"
    ]

    assert (section["generation"], section["approval"]) == (2, "draft")
    assert section["content"]["market_overview"] == "A rewritten overview."
    assert {row["generation"] for row in rows} == {"2"}
    assert {row["approval"] for row in rows} == {"draft"}
    assert {row["value"] for row in rows if row["path"] == "market_overview"} == {
        "A rewritten overview."
    }


def test_a_stage_the_report_cannot_tie_to_a_generation_is_left_out_everywhere(
    context: Context, workspace: UUID
) -> None:
    _build_chain(context, workspace)

    def action(session: Any) -> None:
        record = session.scalars(
            select(MarketResearch).where(MarketResearch.workspace_id == workspace)
        ).one()
        record.market_overview = "Edited without going through history."

    _with_session(context, action)

    body = _json(context, workspace)
    rows = _csv_rows(_get(context, workspace, ReportFormat.CSV))
    docx_text = [p.text for p in _docx(_get(context, workspace, ReportFormat.DOCX)).paragraphs]
    pdf_text = _pdf_text(_get(context, workspace, ReportFormat.PDF))

    assert body["omitted"] == [{"stage": "market_research", "reason": "history_out_of_sync"}]
    assert body["is_complete"] is False
    [omitted_row] = [row for row in rows if row["stage"] == "market_research"]
    assert omitted_row["flags"] == "omitted:history_out_of_sync"
    assert omitted_row["value"] == ""
    assert any("no longer matches its version history" in text for text in docx_text)
    assert "no longer matches its version history" in pdf_text
    assert "Edited without going through history." not in pdf_text


# --- text is not altered on its way out -------------------------------------------------


def test_text_is_exactly_as_stored_in_the_data_and_in_the_files(
    context: Context, workspace: UUID
) -> None:
    padded = "  padded start\n\nsecond paragraph  "
    _build_chain(context, workspace)
    _change_market_research(context, workspace, market_overview=padded)

    section = next(
        item for item in _json(context, workspace)["sections"] if item["stage"] == "market_research"
    )
    [csv_row] = [
        row
        for row in _csv_rows(_get(context, workspace, ReportFormat.CSV))
        if row["stage"] == "market_research" and row["path"] == "market_overview"
    ]
    docx_texts = [p.text for p in _docx(_get(context, workspace, ReportFormat.DOCX)).paragraphs]

    assert section["content"]["market_overview"] == padded
    assert csv_row["value"] == padded
    assert "  padded start" in docx_texts


def test_a_business_name_full_of_special_characters_is_safe_in_every_place(
    context: Context, workspace: UUID
) -> None:
    _build_chain(context, workspace)

    pdf = _get(context, workspace, ReportFormat.PDF)
    disposition = pdf.headers["content-disposition"]

    assert "<" not in disposition
    assert "&" not in disposition
    assert "\r" not in disposition
    assert disposition.count('"') == 2
    assert BUSINESS_NAME in _pdf_text(pdf)
    assert _docx(_get(context, workspace, ReportFormat.DOCX)).paragraphs[0].text == BUSINESS_NAME
