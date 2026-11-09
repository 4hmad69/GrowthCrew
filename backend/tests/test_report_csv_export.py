"""Unit tests for the CSV report exporter."""

import csv
import io
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest

from backend.app.reports.csv_export import CSV_COLUMNS, escape_formula, render_csv
from backend.app.schemas.report import (
    STAGE_ORDER,
    OmissionReason,
    OmittedStage,
    ReportDocument,
    ReportStageSection,
)
from backend.app.schemas.stage_history import STAGE_CONTENT_FIELDS
from backend.app.schemas.strategy_orchestration import (
    ProvenanceState,
    StageApprovalState,
    StageStaleness,
    StaleInput,
    StrategyPrerequisite,
    StrategyStage,
)

NOW = datetime(2026, 11, 4, 12, 0, tzinfo=UTC)
S = StrategyStage


def _section(
    stage: StrategyStage = S.BUSINESS_UNDERSTANDING,
    content: dict[str, Any] | None = None,
    *,
    approval: StageApprovalState = StageApprovalState.APPROVED,
    staleness: StageStaleness | None = None,
    **overrides: Any,
) -> ReportStageSection:
    values: dict[str, Any] = {
        "stage": stage,
        "generation": 3,
        "record_id": uuid4(),
        "record_version": 2,
        "approval": approval,
        "staleness": staleness or StageStaleness(provenance=ProvenanceState.TRACKED),
        "model_used": "gpt-oss:120b-cloud",
        "generated_at": NOW,
        "content": {"summary": "A summary."} if content is None else content,
    }
    values.update(overrides)
    return ReportStageSection(**values)


def _report(
    sections: list[ReportStageSection] | None = None,
    reasons: dict[StrategyStage, OmissionReason] | None = None,
) -> ReportDocument:
    """A report whose given stages are sections and whose other stages are omitted."""

    sections = sections or []
    present = {section.stage for section in sections}
    reasons = reasons or {}
    omitted = [
        OmittedStage(stage=stage, reason=reasons.get(stage, OmissionReason.NOT_GENERATED))
        for stage in STAGE_ORDER
        if stage not in present
    ]
    return ReportDocument(
        workspace_id=uuid4(),
        business_name="Acme Coffee",
        generated_at=NOW,
        sections=sections,
        omitted=omitted,
    )


def _rows(report: ReportDocument) -> list[dict[str, str]]:
    """Parse the exporter's bytes back the way a spreadsheet would read them."""

    text = render_csv(report).decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(text, newline="")))


def _values(rows: list[dict[str, str]], stage: StrategyStage) -> dict[str, str]:
    return {row["path"]: row["value"] for row in rows if row["stage"] == stage.value}


def _single(content: dict[str, Any], stage: StrategyStage = S.BUSINESS_UNDERSTANDING) -> Any:
    return _values(_rows(_report([_section(stage, content)])), stage)


# --- file shape -------------------------------------------------------------------------


def test_the_file_is_utf8_with_a_byte_order_mark() -> None:
    data = render_csv(_report([_section(content={"summary": "Café ☕ 日本語"})]))

    assert data.startswith(b"\xef\xbb\xbf")
    assert "Café ☕ 日本語" in data.decode("utf-8-sig")


def test_rows_end_with_crlf() -> None:
    data = render_csv(_report([_section()])).decode("utf-8-sig")

    assert data.endswith("\r\n")
    assert "\n" not in data.replace("\r\n", "")


def test_the_header_names_every_column_in_order() -> None:
    first_line = render_csv(_report()).decode("utf-8-sig").splitlines()[0]

    assert CSV_COLUMNS == (
        "stage",
        "generation",
        "record_version",
        "approval",
        "flags",
        "path",
        "value",
    )
    assert first_line == ",".join(CSV_COLUMNS)


def test_every_row_has_every_column() -> None:
    text = render_csv(_report([_section(content={"a": [1, {"b": None}]})])).decode("utf-8-sig")

    for row in csv.reader(io.StringIO(text, newline="")):
        assert len(row) == len(CSV_COLUMNS), row


def test_the_same_report_gives_the_same_bytes() -> None:
    report = _report([_section(content={"summary": "x", "other": [1, 2]})])

    assert render_csv(report) == render_csv(report)


# --- what each row says -----------------------------------------------------------------


def test_a_row_carries_its_stage_generation_version_and_approval() -> None:
    rows = _rows(_report([_section(S.MARKET_RESEARCH, {"market_overview": "Big."})]))
    [row] = [item for item in rows if item["stage"] == "market_research"]

    assert row["stage"] == "market_research"
    assert row["generation"] == "3"
    assert row["record_version"] == "2"
    assert row["approval"] == "approved"
    assert row["flags"] == ""
    assert row["path"] == "market_overview"
    assert row["value"] == "Big."


def test_flags_are_on_every_row_of_a_flagged_stage() -> None:
    stale = StageStaleness(
        provenance=ProvenanceState.TRACKED,
        stale_inputs=[
            StaleInput(
                prerequisite=StrategyPrerequisite.BUSINESS_PROFILE,
                built_from=1,
                current=2,
            )
        ],
    )
    section = _section(
        content={"a": "1", "b": "2"},
        approval=StageApprovalState.DRAFT,
        staleness=stale,
    )

    rows = _rows(_report([section]))
    stage_rows = [row for row in rows if row["stage"] == "business_understanding"]

    assert len(stage_rows) == 2
    assert {row["flags"] for row in stage_rows} == {"draft;stale"}
    assert {row["approval"] for row in stage_rows} == {"draft"}


def test_unknown_provenance_is_flagged_on_its_rows() -> None:
    section = _section(staleness=StageStaleness(provenance=ProvenanceState.UNKNOWN))

    [row] = _rows(_report([section]))[:1]

    assert row["flags"] == "unknown_provenance"


def test_stages_are_written_in_generation_order() -> None:
    sections = [_section(stage) for stage in STAGE_ORDER]

    stages = [row["stage"] for row in _rows(_report(sections))]

    assert stages == [stage.value for stage in STAGE_ORDER]


# --- omitted stages ---------------------------------------------------------------------


def test_an_empty_report_lists_every_stage_as_omitted() -> None:
    rows = _rows(_report())

    assert [row["stage"] for row in rows] == [stage.value for stage in STAGE_ORDER]
    assert {row["flags"] for row in rows} == {"omitted:not_generated"}
    for row in rows:
        assert row["generation"] == row["record_version"] == row["approval"] == ""
        assert row["path"] == row["value"] == ""


def test_omission_reasons_are_written_per_stage() -> None:
    sections = [_section(stage) for stage in STAGE_ORDER if stage is not S.MARKET_RESEARCH]
    reasons = {S.MARKET_RESEARCH: OmissionReason.HISTORY_OUT_OF_SYNC}

    rows = _rows(_report(sections, reasons))

    omitted = [row for row in rows if row["flags"].startswith("omitted")]
    assert [(row["stage"], row["flags"]) for row in omitted] == [
        ("market_research", "omitted:history_out_of_sync")
    ]


# --- flattening -------------------------------------------------------------------------


def test_top_level_text_is_one_row_per_field() -> None:
    assert _single({"summary": "One.", "confidence_notes": "Two."}) == {
        "summary": "One.",
        "confidence_notes": "Two.",
    }


def test_nested_values_get_dotted_and_indexed_paths() -> None:
    content = {
        "personas": [
            {"name": "Ada", "goals": ["grow", "save"]},
            {"name": "Bo", "goals": []},
        ]
    }

    assert _single(content, S.CUSTOMER_PERSONAS) == {
        "personas[0].name": "Ada",
        "personas[0].goals[0]": "grow",
        "personas[0].goals[1]": "save",
        "personas[1].name": "Bo",
        "personas[1].goals": "[]",
    }


def test_scalars_are_written_as_plain_text() -> None:
    values = _single(
        {
            "a_null": None,
            "a_true": True,
            "a_false": False,
            "an_int": 3,
            "a_float": 2.5,
            "a_negative": -5,
            "empty_text": "",
        }
    )

    assert values == {
        "a_null": "",
        "a_true": "true",
        "a_false": "false",
        "an_int": "3",
        "a_float": "2.5",
        "a_negative": "-5",
        "empty_text": "",
    }


def test_empty_containers_still_appear() -> None:
    assert _single({"a_list": [], "an_object": {}}) == {"a_list": "[]", "an_object": "{}"}


def test_a_stage_with_no_content_still_has_a_row() -> None:
    [row] = _rows(_report([_section(content={})]))[:1]

    assert row["stage"] == "business_understanding"
    assert row["approval"] == "approved"
    assert row["path"] == row["value"] == ""


def test_known_fields_come_back_in_their_documented_order() -> None:
    known = STAGE_CONTENT_FIELDS[S.MARKET_RESEARCH]
    content = {name: name for name in reversed(known)}

    rows = _rows(_report([_section(S.MARKET_RESEARCH, content)]))
    paths = [row["path"] for row in rows if row["stage"] == "market_research"]

    assert paths == list(known)


def test_unknown_fields_follow_the_known_ones_in_a_stable_order() -> None:
    content = {"zeta": "z", "market_overview": "m", "alpha": "a"}

    rows = _rows(_report([_section(S.MARKET_RESEARCH, content)]))
    paths = [row["path"] for row in rows if row["stage"] == "market_research"]

    assert paths == ["market_overview", "alpha", "zeta"]


def test_text_with_commas_quotes_and_newlines_survives_a_round_trip() -> None:
    text = 'He said, "go"\nthen left,\r\nthe line'

    assert _single({"summary": text}) == {"summary": text}


# --- formula injection ------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["=1+1", "+1", "-1", "@SUM(A1:A2)", "\tcmd", "\rcmd", '=HYPERLINK("http://evil","x")'],
)
def test_text_that_would_run_as_a_formula_is_quoted(text: str) -> None:
    assert escape_formula(text) == "'" + text
    assert _single({"summary": text}) == {"summary": "'" + text}


@pytest.mark.parametrize("text", ["", "plain", " =leading space", "a=b", "'quoted", "1+1", "x-1"])
def test_other_text_is_left_alone(text: str) -> None:
    assert escape_formula(text) == text
    assert _single({"summary": text}) == {"summary": text}


def test_a_formula_inside_nested_content_is_quoted() -> None:
    values = _single({"personas": [{"name": "=cmd|' /C calc'!A0", "goals": ["+x", "fine"]}]})

    assert values["personas[0].name"] == "'=cmd|' /C calc'!A0"
    assert values["personas[0].goals[0]"] == "'+x"
    assert values["personas[0].goals[1]"] == "fine"


def test_a_formula_in_a_field_name_is_quoted() -> None:
    paths = list(_single({"=cmd": "x"}))

    assert paths == ["'=cmd"]


def test_numbers_are_not_quoted() -> None:
    assert _single({"a_negative": -5, "text_negative": "-5"}) == {
        "a_negative": "-5",
        "text_negative": "'-5",
    }
