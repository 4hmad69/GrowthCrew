"""Unit tests for the Word report exporter.

Each test builds a report, writes it, and reads the file back with
python-docx, so what is checked is the document a person would open.
"""

import io
import zipfile
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest
from docx import Document
from docx.oxml.ns import qn

from backend.app.reports.docx_export import _write_stage, clean_text, render_docx
from backend.app.reports.labels import REPORT_TITLE, status_summary
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
EMPTY = "\u2014"


def _section(
    stage: StrategyStage = S.BUSINESS_UNDERSTANDING,
    content: dict[str, Any] | None = None,
    *,
    approval: StageApprovalState = StageApprovalState.APPROVED,
    staleness: StageStaleness | None = None,
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
    return ReportStageSection(**values)


def _report(
    sections: list[ReportStageSection] | None = None,
    reasons: dict[StrategyStage, OmissionReason] | None = None,
    business_name: str = "Acme Coffee",
) -> ReportDocument:
    sections = sections or []
    present = {section.stage for section in sections}
    reasons = reasons or {}
    return ReportDocument(
        workspace_id=uuid4(),
        business_name=business_name,
        generated_at=NOW,
        sections=sections,
        omitted=[
            OmittedStage(stage=stage, reason=reasons.get(stage, OmissionReason.NOT_GENERATED))
            for stage in STAGE_ORDER
            if stage not in present
        ],
    )


def _full(**contents: dict[str, Any]) -> ReportDocument:
    """Every stage present; stages named in contents get that content."""

    return _report([_section(stage, contents.get(stage.value)) for stage in STAGE_ORDER])


def _cm(length: Any) -> float:
    """A length in centimetres. Word stores twips, so EMU values come back slightly rounded."""

    return round(length.cm, 2)


def _open(report: ReportDocument) -> Any:
    return Document(io.BytesIO(render_docx(report)))


def _body(doc: Any, stage_title: str) -> list[tuple[str, str]]:
    """(style, text) of every paragraph between a stage's heading and the next one."""

    result: list[tuple[str, str]] = []
    inside = False
    for paragraph in doc.paragraphs:
        style = paragraph.style.name
        if style == "Heading 1":
            if inside:
                break
            inside = paragraph.text == stage_title
            continue
        if inside:
            result.append((style, paragraph.text))
    return result


def _texts(body: list[tuple[str, str]], style: str | None = None) -> list[str]:
    return [text for kind, text in body if style is None or kind == style]


# --- the file ---------------------------------------------------------------------------


def test_the_output_is_a_word_file_python_docx_can_open() -> None:
    data = render_docx(_full())

    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        assert "word/document.xml" in archive.namelist()
        assert archive.testzip() is None
    assert len(Document(io.BytesIO(data)).paragraphs) > 0


def test_the_page_is_a4_with_even_margins() -> None:
    section = _open(_full()).sections[0]

    assert _cm(section.page_width) == 21.0
    assert _cm(section.page_height) == 29.7
    for margin in (section.left_margin, section.right_margin, section.top_margin):
        assert _cm(margin) == 2.2


def test_the_zoom_setting_has_the_attribute_the_schema_requires() -> None:
    zoom = _open(_full()).settings.element.find(qn("w:zoom"))

    assert zoom is not None
    assert zoom.get(qn("w:percent")) == "100"


def test_core_properties_describe_the_report() -> None:
    properties = _open(_full()).core_properties

    assert properties.title == f"Acme Coffee - {REPORT_TITLE}"
    assert properties.author == "GrowthCrew"
    assert properties.created == NOW


def test_the_footer_has_live_page_number_fields() -> None:
    footer = _open(_full()).sections[0].footer.paragraphs[0]
    xml = footer._p.xml

    assert "Acme Coffee" in footer.text
    assert 'w:instr="PAGE"' in xml
    assert 'w:instr="NUMPAGES"' in xml


def test_the_same_report_gives_the_same_text() -> None:
    report = _full()

    first = [p.text for p in _open(report).paragraphs]
    second = [p.text for p in _open(report).paragraphs]

    assert first == second


# --- cover and overview -----------------------------------------------------------------


def test_the_cover_names_the_business_the_report_and_the_time() -> None:
    doc = _open(_full())

    assert (doc.paragraphs[0].style.name, doc.paragraphs[0].text) == ("Title", "Acme Coffee")
    assert (doc.paragraphs[1].style.name, doc.paragraphs[1].text) == ("Subtitle", REPORT_TITLE)
    assert doc.paragraphs[2].text == "Generated 04 November 2026, 12:00 UTC"


def test_the_status_line_is_the_summary_in_bold_lead() -> None:
    report = _report([_section(S.BUSINESS_UNDERSTANDING, approval=StageApprovalState.DRAFT)])

    paragraph = _open(report).paragraphs[3]

    assert paragraph.text == f"Status: {status_summary(report)}"
    assert paragraph.runs[0].text == "Status: "
    assert paragraph.runs[0].bold is True


def test_a_clean_report_says_so() -> None:
    assert "Complete and current" in _open(_full()).paragraphs[3].text


def test_the_overview_table_has_a_row_per_stage() -> None:
    table = _open(_full()).tables[0]

    assert [cell.text for cell in table.rows[0].cells] == [
        "Stage",
        "Generation",
        "Approval",
        "Status",
    ]
    assert [row.cells[0].text for row in table.rows[1:]] == [
        "Business Understanding",
        "Market Research",
        "Competitor Analysis",
        "Customer Personas",
        "Brand Strategy",
        "Marketing Strategy",
        "Content Plan",
    ]


def test_the_overview_shows_generation_approval_and_status() -> None:
    stale = StageStaleness(
        provenance=ProvenanceState.TRACKED,
        stale_inputs=[
            StaleInput(prerequisite=StrategyPrerequisite.BUSINESS_PROFILE, built_from=1, current=2)
        ],
    )
    sections = [
        _section(S.BUSINESS_UNDERSTANDING),
        _section(S.MARKET_RESEARCH, approval=StageApprovalState.DRAFT, staleness=stale),
        _section(
            S.COMPETITOR_ANALYSIS,
            staleness=StageStaleness(provenance=ProvenanceState.UNKNOWN),
        ),
    ]

    rows = {
        row.cells[0].text: [cell.text for cell in row.cells]
        for row in _open(_report(sections)).tables[0].rows
    }

    assert rows["Business Understanding"] == ["Business Understanding", "3", "Approved", "Current"]
    assert rows["Market Research"] == ["Market Research", "3", "Draft", "Draft, Out of date"]
    assert rows["Competitor Analysis"][3] == "Unknown provenance"


def test_omitted_stages_are_in_the_overview_with_their_reason() -> None:
    reasons = {S.MARKET_RESEARCH: OmissionReason.HISTORY_OUT_OF_SYNC}
    sections = [_section(S.BUSINESS_UNDERSTANDING)]

    rows = {
        row.cells[0].text: [cell.text for cell in row.cells]
        for row in _open(_report(sections, reasons)).tables[0].rows
    }

    assert rows["Market Research"] == [
        "Market Research",
        EMPTY,
        EMPTY,
        "Not included (history out of sync)",
    ]
    assert rows["Content Plan"][3] == "Not generated"


def test_the_header_row_repeats_and_rows_do_not_split() -> None:
    table = _open(_full()).tables[0]

    assert table.rows[0]._tr.xml.count("w:tblHeader") == 1
    assert all("w:cantSplit" in row._tr.xml for row in table.rows)
    assert "w:tblHeader" not in table.rows[1]._tr.xml


def test_the_table_columns_have_fixed_widths_on_the_grid() -> None:
    table = _open(_full()).tables[0]

    assert [_cm(column.width) for column in table.columns] == [5.0, 2.4, 2.6, 6.6]


def test_the_overview_is_followed_by_a_page_break_before_the_first_stage() -> None:
    doc = _open(_full())
    paragraphs = list(doc.paragraphs)
    first_heading = next(i for i, p in enumerate(paragraphs) if p.style.name == "Heading 1")
    breaks = [i for i, p in enumerate(paragraphs) if 'w:type="page"' in p._p.xml]

    assert breaks == [first_heading - 1]


def test_writing_a_stage_with_no_section_and_no_reason_is_refused() -> None:
    with pytest.raises(ValueError, match="neither a section nor an omission reason"):
        _write_stage(Document(), S.MARKET_RESEARCH, None, None)


# --- stages -----------------------------------------------------------------------------


def test_every_stage_has_a_heading_in_generation_order() -> None:
    doc = _open(_report([_section(S.BUSINESS_UNDERSTANDING)]))

    headings = [p.text for p in doc.paragraphs if p.style.name == "Heading 1"]

    assert headings == [
        "Business Understanding",
        "Market Research",
        "Competitor Analysis",
        "Customer Personas",
        "Brand Strategy",
        "Marketing Strategy",
        "Content Plan",
    ]


def test_a_stage_says_which_generation_and_record_it_is() -> None:
    body = _body(_open(_full()), "Business Understanding")

    assert body[0][1] == (
        "Generation 3  \u00b7  record version 2  \u00b7  Approved  "
        "\u00b7  model gpt-oss:120b-cloud  \u00b7  generated 04 November 2026, 12:00 UTC"
    )


def test_a_clean_stage_has_no_notice() -> None:
    body = _body(_open(_full()), "Business Understanding")

    assert not any(
        text.startswith(("Draft:", "Out of date:", "Unknown provenance:")) for text in _texts(body)
    )


def test_flagged_stages_carry_a_notice_directly_under_the_heading() -> None:
    stale = StageStaleness(
        provenance=ProvenanceState.TRACKED,
        stale_inputs=[
            StaleInput(prerequisite=StrategyPrerequisite.MARKET_RESEARCH, built_from=1, current=2)
        ],
    )
    report = _report(
        [
            _section(S.BUSINESS_UNDERSTANDING, approval=StageApprovalState.DRAFT, staleness=stale),
            _section(
                S.MARKET_RESEARCH,
                staleness=StageStaleness(provenance=ProvenanceState.UNKNOWN),
            ),
        ]
    )
    doc = _open(report)

    flagged = _texts(_body(doc, "Business Understanding"))
    unknown = _texts(_body(doc, "Market Research"))

    assert flagged[1].startswith("Draft: ")
    assert flagged[2].startswith("Out of date: ")
    assert "Market Research (built from generation 1, now generation 2)" in flagged[2]
    assert unknown[1].startswith("Unknown provenance: ")


def test_a_notice_leads_in_bold() -> None:
    doc = _open(_report([_section(approval=StageApprovalState.DRAFT)]))
    notice = next(p for p in doc.paragraphs if p.text.startswith("Draft: "))

    assert notice.runs[0].text == "Draft: "
    assert notice.runs[0].bold is True


def _fill(paragraph: Any) -> str:
    return paragraph._p.pPr.find(qn("w:shd")).get(qn("w:fill"))


def test_notices_are_coloured_by_what_they_say() -> None:
    clean = _open(_full()).paragraphs[3]
    flagged = _open(_report([_section(approval=StageApprovalState.DRAFT)]))
    warning = next(p for p in flagged.paragraphs if p.text.startswith("Draft: "))
    omitted = next(p for p in flagged.paragraphs if p.text.startswith("Not included: "))

    assert clean.text.startswith("Status: Complete")
    assert _fill(clean) == "E6F4EA"
    assert _fill(flagged.paragraphs[3]) == "FFF4D6"
    assert _fill(warning) == "FFF4D6"
    assert _fill(omitted) == "EDF2F7"


def test_the_overview_header_row_is_shaded() -> None:
    header = _open(_full()).tables[0].rows[0]

    assert all('w:fill="D9E2F3"' in cell._tc.xml for cell in header.cells)
    assert all("w:fill" not in cell._tc.xml for cell in _open(_full()).tables[0].rows[1].cells)


def test_notice_paragraph_properties_are_in_schema_order() -> None:
    """Border, then shading, then spacing and indent: Word rejects any other order."""

    doc = _open(_report([_section(approval=StageApprovalState.DRAFT)]))
    notice = next(p for p in doc.paragraphs if p.text.startswith("Draft: "))

    tags = [child.tag.split("}")[1] for child in notice._p.pPr]

    assert tags.index("pBdr") < tags.index("shd") < tags.index("spacing") < tags.index("ind")


def test_an_omitted_stage_keeps_its_heading_and_says_why() -> None:
    reasons = {S.MARKET_RESEARCH: OmissionReason.HISTORY_OUT_OF_SYNC}
    doc = _open(_report([_section(S.BUSINESS_UNDERSTANDING)], reasons))

    market = _body(doc, "Market Research")
    plan = _body(doc, "Content Plan")

    assert len(market) == 1
    assert "no longer matches its version history" in market[0][1]
    assert plan == [("Normal", "Not included: this stage has not been generated.")]


def test_stage_fields_are_headed_in_their_documented_order() -> None:
    known = STAGE_CONTENT_FIELDS[S.MARKET_RESEARCH]
    content = {name: f"text of {name}" for name in reversed(known)}
    doc = _open(_report([_section(S.MARKET_RESEARCH, content)]))

    headings = _texts(_body(doc, "Market Research"), "Heading 2")

    assert headings == [name.replace("_", " ").capitalize() for name in known]


def test_unknown_fields_follow_the_known_ones() -> None:
    content = {"zeta": "z", "market_overview": "m", "alpha": "a"}
    doc = _open(_report([_section(S.MARKET_RESEARCH, content)]))

    assert _texts(_body(doc, "Market Research"), "Heading 2") == [
        "Market overview",
        "Alpha",
        "Zeta",
    ]


def test_a_stage_with_no_content_shows_a_dash() -> None:
    body = _body(_open(_report([_section(content={})])), "Business Understanding")

    assert body[-1] == ("Normal", EMPTY)


# --- text -------------------------------------------------------------------------------


def test_text_keeps_its_leading_whitespace() -> None:
    body = _body(
        _open(_report([_section(content={"summary": "  indented\n\n"})])), "Business Understanding"
    )

    assert "  indented" in _texts(body)


def test_trailing_blank_lines_do_not_make_an_empty_paragraph() -> None:
    content = {"summary": "Only this.\n\n\n  \n"}
    body = _body(_open(_report([_section(content=content)])), "Business Understanding")

    assert _texts(body, "Normal")[-1] == "Only this."
    assert "" not in _texts(body)


@pytest.mark.parametrize("blank", ["", "   ", "\n\n", "\t \n"])
def test_text_with_nothing_in_it_is_a_dash(blank: str) -> None:
    body = _body(_open(_report([_section(content={"summary": blank})])), "Business Understanding")

    assert body[-1] == ("Normal", EMPTY)


def test_blank_lines_split_paragraphs_and_single_newlines_are_line_breaks() -> None:
    text = "First line\nsame paragraph\n\nSecond paragraph"
    body = _body(_open(_report([_section(content={"summary": text})])), "Business Understanding")

    assert _texts(body, "Normal")[-2:] == ["First line\nsame paragraph", "Second paragraph"]


def test_windows_line_endings_make_one_break_not_two() -> None:
    body = _body(
        _open(_report([_section(content={"summary": "a\r\nb\rc"})])), "Business Understanding"
    )

    assert _texts(body, "Normal")[-1] == "a\nb\nc"


def test_characters_xml_cannot_carry_are_dropped_not_fatal() -> None:
    text = "a\x00b\x0bc\x1fd\ufffee\ud800f ok"
    body = _body(_open(_report([_section(content={"summary": text})])), "Business Understanding")

    assert _texts(body, "Normal")[-1] == "abcdef ok"


def test_markup_looking_text_is_shown_literally() -> None:
    text = '<b>bold</b> & "quotes" <script>x</script> =1+1'
    body = _body(_open(_report([_section(content={"summary": text})])), "Business Understanding")

    assert _texts(body, "Normal")[-1] == text


def test_non_latin_text_survives() -> None:
    text = "Café ☕ 日本語 \U0001f600"
    body = _body(_open(_report([_section(content={"summary": text})])), "Business Understanding")

    assert _texts(body, "Normal")[-1] == text


@pytest.mark.parametrize(
    ("raw", "cleaned"),
    [
        ("plain", "plain"),
        ("a\r\nb", "a\nb"),
        ("a\rb", "a\nb"),
        ("tab\there", "tab\there"),
        ("a\x00\x08\x0b\x0c\x0e\x1fb", "ab"),
        ("\ud800\udfff", ""),
        ("\ufffe\uffff", ""),
        ("\ufffd stays", "\ufffd stays"),
    ],
)
def test_clean_text(raw: str, cleaned: str) -> None:
    assert clean_text(raw) == cleaned


def test_an_unusual_business_name_does_not_break_the_file() -> None:
    doc = _open(_report([_section()], business_name="Fish & Chips <Ltd>\x00"))

    assert doc.paragraphs[0].text == "Fish & Chips <Ltd>"
    assert "Fish & Chips <Ltd>" in doc.sections[0].footer.paragraphs[0].text


# --- structure --------------------------------------------------------------------------


def test_a_list_of_plain_values_is_a_bullet_list() -> None:
    content = {"key_differentiators": ["Fast", "Fresh", 3, True, None]}
    body = _body(_open(_report([_section(content=content)])), "Business Understanding")

    assert _texts(body, "List Bullet") == ["Fast", "Fresh", "3", "Yes", EMPTY]


def test_an_empty_list_or_object_shows_a_dash() -> None:
    content = {"a_list": [], "an_object": {}}
    body = _body(_open(_report([_section(content=content)])), "Business Understanding")

    assert body[-4:] == [
        ("Heading 2", "A list"),
        ("Normal", EMPTY),
        ("Heading 2", "An object"),
        ("Normal", EMPTY),
    ]


def test_a_list_of_objects_is_one_labelled_block_per_item() -> None:
    content = {
        "sources": [
            {"section": "overview", "source": "https://a.example", "snippet": "One."},
            {"section": "risks", "source": "https://b.example", "snippet": "Two."},
        ]
    }
    body = _body(_open(_report([_section(S.MARKET_RESEARCH, content)])), "Market Research")

    assert [item for item in body if item[0] == "Heading 3"] == [
        ("Heading 3", "Item 1"),
        ("Heading 3", "Item 2"),
    ]
    assert "Source: https://a.example" in _texts(body)
    assert "Snippet: Two." in _texts(body)


def test_an_item_is_headed_by_its_name_and_the_name_is_not_repeated() -> None:
    content = {
        "personas": [
            {"name": "Weekend Wen", "age_range": "28-35"},
            {"title": "Busy Bo", "age_range": "35-45"},
        ]
    }
    body = _body(_open(_report([_section(S.CUSTOMER_PERSONAS, content)])), "Customer Personas")
    texts = _texts(body)

    assert [text for kind, text in body if kind == "Heading 3"] == ["Weekend Wen", "Busy Bo"]
    assert not any(text.startswith(("Name:", "Title:")) for text in texts)
    assert "Age range: 28-35" in texts


def test_a_blank_name_is_not_used_as_a_heading() -> None:
    content = {"personas": [{"name": "   ", "age_range": "28-35"}]}
    body = _body(_open(_report([_section(S.CUSTOMER_PERSONAS, content)])), "Customer Personas")

    assert ("Heading 3", "Item 1") in body
    assert f"Name: {EMPTY}" in _texts(body)


def test_a_list_mixing_plain_values_and_objects_is_written_item_by_item() -> None:
    content = {"things": ["plain", {"name": "Wen", "age": 3}, 7]}
    body = _body(_open(_report([_section(content=content)])), "Business Understanding")

    after_heading = body[body.index(("Heading 2", "Things")) + 1 :]

    assert after_heading == [
        ("Heading 3", "Item 1"),
        ("Normal", "plain"),
        ("Heading 3", "Wen"),
        ("Normal", "Age: 3"),
        ("Heading 3", "Item 3"),
        ("Normal", "7"),
    ]


def test_labels_are_bold_and_values_are_not() -> None:
    doc = _open(_report([_section(S.CUSTOMER_PERSONAS, {"personas": [{"age_range": "28-35"}]})]))
    line = next(p for p in doc.paragraphs if p.text == "Age range: 28-35")

    assert [(run.text, run.bold) for run in line.runs] == [("Age range: ", True), ("28-35", None)]


def test_nested_objects_and_lists_are_indented_below_their_label() -> None:
    content = {
        "personas": [
            {"name": "Wen", "channels": {"primary": "Instagram", "secondary": ["YouTube"]}}
        ]
    }
    doc = _open(_report([_section(S.CUSTOMER_PERSONAS, content)]))

    by_text = {p.text: p for p in doc.paragraphs}
    indents = {
        text: _cm(by_text[text].paragraph_format.left_indent)
        for text in ("Channels:", "Primary: Instagram", "Secondary:", "YouTube")
    }

    assert indents == {
        "Channels:": 0.6,
        "Primary: Instagram": 1.2,
        "Secondary:": 1.2,
        "YouTube": 1.8,
    }


def test_a_multi_line_value_in_an_object_is_a_block_under_its_label() -> None:
    content = {"personas": [{"name": "Wen", "bio": "Line one.\n\nLine two."}]}
    body = _body(_open(_report([_section(S.CUSTOMER_PERSONAS, content)])), "Customer Personas")
    texts = _texts(body)

    start = texts.index("Bio:")
    assert texts[start : start + 3] == ["Bio:", "Line one.", "Line two."]


def test_numbers_and_booleans_in_objects_are_written_plainly() -> None:
    content = {"personas": [{"name": "Wen", "age": 31, "active": True, "gone": None}]}
    texts = _texts(
        _body(_open(_report([_section(S.CUSTOMER_PERSONAS, content)])), "Customer Personas")
    )

    assert {"Age: 31", "Active: Yes", f"Gone: {EMPTY}"} <= set(texts)


def test_keys_that_are_not_plain_names_are_shown_as_they_are() -> None:
    content = {"personas": [{"name": "Wen", "=odd key": "x"}]}
    texts = _texts(
        _body(_open(_report([_section(S.CUSTOMER_PERSONAS, content)])), "Customer Personas")
    )

    assert "=odd key: x" in texts


def test_a_large_report_is_written() -> None:
    items = [{"name": f"Item {number}", "detail": "word " * 200} for number in range(300)]
    doc = _open(_report([_section(S.CUSTOMER_PERSONAS, {"personas": items})]))

    headings = [p.text for p in doc.paragraphs if p.style.name == "Heading 3"]

    assert len(headings) == 300
