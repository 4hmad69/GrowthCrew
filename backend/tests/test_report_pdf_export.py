"""Unit tests for the PDF report exporter.

Each test builds a report, writes it, and reads the PDF back with pypdf, so
what is checked is the file a person would open: its text, outline,
metadata, embedded fonts and colours.
"""

import io
import re
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest
from pypdf import PdfReader

from backend.app.reports.labels import REPORT_TITLE, status_summary
from backend.app.reports.pdf_export import (
    REPLACEMENT,
    _covered_characters,
    _overview,
    _stage,
    _Styles,
    _Text,
    render_pdf,
)
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
    return _report([_section(stage, contents.get(stage.value)) for stage in STAGE_ORDER])


def _one(
    content: dict[str, Any], stage: StrategyStage = S.BUSINESS_UNDERSTANDING
) -> ReportDocument:
    return _report([_section(stage, content)])


def _read(report: ReportDocument) -> PdfReader:
    return PdfReader(io.BytesIO(render_pdf(report)))


def _text(report: ReportDocument) -> str:
    return "\n".join(page.extract_text() for page in _read(report).pages)


def _flat(text: str) -> str:
    """Whitespace collapsed to single spaces, so text the PDF wrapped can be compared."""

    return " ".join(text.split())


def _lines(report: ReportDocument) -> list[str]:
    return [line.strip() for line in _text(report).splitlines() if line.strip()]


def _positions(report: ReportDocument) -> dict[str, float]:
    """The x position of the first appearance of each piece of text, across all pages."""

    found: dict[str, float] = {}

    def visit(text: str, cm: list[float], tm: list[float], *_: Any) -> None:
        if text.strip():
            found.setdefault(text.strip(), round(tm[4] * cm[0] + cm[4], 1))

    for page in _read(report).pages:
        page.extract_text(visitor_text=visit)
    return found


def _rgb(fill: str) -> tuple[float, float, float]:
    return tuple(round(int(fill[i : i + 2], 16) / 255, 2) for i in (0, 2, 4))  # type: ignore[return-value]


def _fill_colours(report: ReportDocument) -> set[tuple[float, float, float]]:
    colours: set[tuple[float, float, float]] = set()
    for page in _read(report).pages:
        data = page.get_contents().get_data().decode("latin-1")
        for match in re.finditer(r"([\d.]+) ([\d.]+) ([\d.]+) rg", data):
            colours.add(tuple(round(float(value), 2) for value in match.groups()))  # type: ignore[arg-type]
    return colours


# --- the file ---------------------------------------------------------------------------


def test_the_output_is_a_pdf_pypdf_can_open() -> None:
    data = render_pdf(_full())

    assert data.startswith(b"%PDF-")
    assert data.rstrip().endswith(b"%%EOF")
    assert len(PdfReader(io.BytesIO(data)).pages) >= 2


def test_the_page_is_a4() -> None:
    box = _read(_full()).pages[0].mediabox

    assert round(float(box.width), 1) == 595.3
    assert round(float(box.height), 1) == 841.9


def test_metadata_describes_the_report() -> None:
    metadata = _read(_full()).metadata

    assert metadata is not None
    assert metadata.title == f"Acme Coffee - {REPORT_TITLE}"
    assert metadata.author == "GrowthCrew"
    assert metadata.creator == "GrowthCrew"
    assert metadata.subject == REPORT_TITLE


def test_the_pdf_is_dated_when_the_report_was_generated() -> None:
    metadata = _read(_full()).metadata

    assert metadata is not None
    assert metadata.creation_date == NOW
    assert metadata.modification_date == NOW


def test_the_same_report_gives_the_same_bytes() -> None:
    report = _full()

    assert render_pdf(report) == render_pdf(report)


def test_every_font_that_draws_text_is_embedded() -> None:
    reader = _read(_full())
    drawn: set[str] = set()
    for page in reader.pages:
        fonts = {key: value.get_object() for key, value in page["/Resources"]["/Font"].items()}
        data = page.get_contents().get_data().decode("latin-1")
        for name in re.findall(r"/(F\d+(?:\+\d+)?) [\d.]+ Tf", data):
            drawn.add(str(fonts[f"/{name}"]["/BaseFont"]))

    assert drawn
    assert all(re.fullmatch(r"/[A-Z]{6}\+BitstreamVeraSans-\w+", name) for name in drawn), drawn


def test_each_page_has_a_footer_with_its_number_and_the_total() -> None:
    reader = _read(_full())
    total = len(reader.pages)

    for number, page in enumerate(reader.pages, start=1):
        assert f"Acme Coffee \u00b7 Page {number} of {total}" in _flat(page.extract_text())


def test_the_outline_has_one_entry_per_stage_on_the_page_the_stage_starts() -> None:
    reader = _read(_full())
    entries = [(item.title, reader.get_destination_page_number(item)) for item in reader.outline]

    assert [title for title, _ in entries] == [
        "Business Understanding",
        "Market Research",
        "Competitor Analysis",
        "Customer Personas",
        "Brand Strategy",
        "Marketing Strategy",
        "Content Plan",
    ]
    assert all(page >= 1 for _, page in entries)
    assert [page for _, page in entries] == sorted(page for _, page in entries)
    for title, page in entries:
        assert title in reader.pages[page].extract_text().splitlines()


# --- cover and overview -----------------------------------------------------------------


def test_the_cover_names_the_business_the_report_and_the_time() -> None:
    lines = _lines(_full())

    assert lines[:3] == [
        "Acme Coffee",
        REPORT_TITLE,
        "Generated 04 November 2026, 12:00 UTC",
    ]


def test_the_status_line_is_the_summary() -> None:
    report = _report([_section(S.BUSINESS_UNDERSTANDING, approval=StageApprovalState.DRAFT)])

    assert f"Status: {status_summary(report)}" in _flat(_text(report))


def test_a_clean_report_says_so() -> None:
    assert "Status: Complete and current" in _flat(_text(_full()))


def test_the_overview_lists_every_stage_with_its_status() -> None:
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
            S.COMPETITOR_ANALYSIS, staleness=StageStaleness(provenance=ProvenanceState.UNKNOWN)
        ),
    ]
    reasons = {S.CUSTOMER_PERSONAS: OmissionReason.HISTORY_OUT_OF_SYNC}

    first_page = _flat(_read(_report(sections, reasons)).pages[0].extract_text())

    assert "Stages at a glance" in first_page
    assert "Business Understanding 3 Approved Current" in first_page
    assert "Market Research 3 Draft Draft, Out of date" in first_page
    assert "Competitor Analysis 3 Approved Unknown provenance" in first_page
    assert f"Customer Personas {EMPTY} {EMPTY} Not included (history out of sync)" in first_page
    assert f"Content Plan {EMPTY} {EMPTY} Not generated" in first_page


def test_the_overview_table_repeats_its_header_row() -> None:
    table = _overview(_Text(), _Styles(), _full())[1]

    assert table.repeatRows == 1


def test_stages_start_after_the_cover_page() -> None:
    reader = _read(_full())

    assert "Business Understanding" in reader.pages[0].extract_text()  # in the table
    assert "Generation 3" not in reader.pages[0].extract_text()
    assert "Generation 3" in reader.pages[1].extract_text()


def test_notices_and_the_table_are_coloured_by_what_they_say() -> None:
    clean = _fill_colours(_full())
    flagged = _fill_colours(
        _report(
            [_section(approval=StageApprovalState.DRAFT)],
            {S.MARKET_RESEARCH: OmissionReason.NOT_GENERATED},
        )
    )

    assert _rgb("E6F4EA") in clean  # all clear
    assert _rgb("FFF4D6") not in clean
    assert _rgb("FFF4D6") in flagged  # needs attention, and the draft notice
    assert _rgb("EDF2F7") in flagged  # a stage that is not included
    assert _rgb("D9E2F3") in clean  # the table's header row


# --- stages -----------------------------------------------------------------------------


def test_every_stage_has_a_heading_in_generation_order() -> None:
    text = _text(_report([_section(S.BUSINESS_UNDERSTANDING)]))
    # The first occurrence is the overview table; the headings are the lines that follow it.
    after_cover = text[text.index("Generation 3") :]

    positions = [
        after_cover.index(f"\n{stage.value.replace('_', ' ').title()}\n")
        for stage in STAGE_ORDER[1:]
    ]

    assert positions == sorted(positions)


def test_a_stage_says_which_generation_and_record_it_is() -> None:
    text = _flat(_text(_full()))

    assert (
        "Generation 3 \u00b7 record version 2 \u00b7 Approved \u00b7 model gpt-oss:120b-cloud "
        "\u00b7 generated 04 November 2026, 12:00 UTC"
    ) in text


def test_a_clean_stage_has_no_notice() -> None:
    text = _flat(_text(_full()))

    for lead in ("Draft:", "Out of date:", "Unknown provenance:", "Not included:"):
        assert lead not in text


def test_flagged_stages_carry_a_notice_under_the_heading() -> None:
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
                S.MARKET_RESEARCH, staleness=StageStaleness(provenance=ProvenanceState.UNKNOWN)
            ),
        ]
    )

    text = _flat(_text(report))

    assert "Draft: this version of the stage has not been approved." in text
    assert (
        "Out of date: built from inputs that have since changed: "
        "Market Research (built from generation 1, now generation 2)."
    ) in text
    assert "Unknown provenance: nothing records what this stage was built from" in text


def test_an_omitted_stage_keeps_its_heading_and_says_why() -> None:
    reasons = {S.MARKET_RESEARCH: OmissionReason.HISTORY_OUT_OF_SYNC}
    text = _flat(_text(_report([_section(S.BUSINESS_UNDERSTANDING)], reasons)))

    assert "Market Research Not included: this stage's stored record no longer matches" in text
    assert "Content Plan Not included: this stage has not been generated." in text


def test_stage_fields_are_headed_in_their_documented_order() -> None:
    known = STAGE_CONTENT_FIELDS[S.MARKET_RESEARCH]
    content = {name: f"text of {name}" for name in reversed(known)}
    lines = _lines(_one(content, S.MARKET_RESEARCH))

    headings = [name.replace("_", " ").capitalize() for name in known]
    positions = [lines.index(heading) for heading in headings]

    assert positions == sorted(positions)


def test_unknown_fields_follow_the_known_ones() -> None:
    lines = _lines(_one({"zeta": "z", "market_overview": "m", "alpha": "a"}, S.MARKET_RESEARCH))

    order = [lines.index(heading) for heading in ("Market overview", "Alpha", "Zeta")]

    assert order == sorted(order)


def test_a_stage_with_no_content_shows_a_dash() -> None:
    lines = _lines(_one({}))
    start = next(i for i, line in enumerate(lines) if line.startswith("Generation 3 \u00b7 record"))

    assert lines[start + 1] == EMPTY


# --- text -------------------------------------------------------------------------------


def test_paragraph_and_line_breaks_are_kept() -> None:
    lines = _lines(_one({"summary": "First line\nsame paragraph\n\nSecond paragraph"}))
    start = lines.index("Summary")

    assert lines[start + 1 : start + 4] == ["First line", "same paragraph", "Second paragraph"]


def test_markup_looking_text_is_shown_literally() -> None:
    text = '<b>bold</b> & "quotes" <font color=red>x</font> </para> &nbsp; &amp;'

    assert text in _text(_one({"summary": text}))


def test_runs_of_spaces_and_tabs_collapse_as_in_any_reflowed_document() -> None:
    assert "a b c" in _text(_one({"summary": "a    b\tc"}))


def test_characters_that_cannot_be_written_are_dropped() -> None:
    assert "abcdef ok" in _text(_one({"summary": "a\x00b\x0bc\x1fd\ufffee\ud800f ok"}))


def test_accented_latin_text_survives() -> None:
    text = "Café Müller señor łódz šíp"

    assert text in _text(_one({"summary": text}))


def test_text_is_composed_so_combining_marks_still_show() -> None:
    assert "caf\u00e9" in _text(_one({"summary": "cafe\u0301"}))


def test_invisible_joiners_are_dropped_without_a_placeholder() -> None:
    text = _text(_one({"summary": "Hello\u200dworld\u200b!"}))

    assert "Helloworld!" in text
    assert "Characters not shown" not in text


def test_a_long_unbreakable_string_is_wrapped_across_lines_not_run_off_the_page() -> None:
    url = "https://example.com/" + "x" * 600
    lines = _lines(_one({"summary": url}))
    pieces = [line for line in lines if line.count("x") > 40]

    assert len(pieces) >= 2
    assert "".join("".join(lines).split()).count("x") >= 600


def test_text_starts_at_the_left_margin_and_boxes_line_up_with_it() -> None:
    found = _positions(_report([_section(approval=StageApprovalState.DRAFT)]))
    margin = round(2.2 * 72 / 2.54, 1)

    assert found["Acme Coffee"] == margin
    assert found["Summary"] == margin
    # A notice's text sits inside its box: 9 pt of padding in from the margin.
    assert found["Status:"] == round(margin + 9, 1)
    assert found["Draft:"] == round(margin + 9, 1)


def test_writing_a_stage_with_no_section_and_no_reason_is_refused() -> None:
    with pytest.raises(ValueError, match="neither a section nor an omission reason"):
        _stage(_Text(), _Styles(), S.MARKET_RESEARCH, None, None)


# --- characters the font cannot show ----------------------------------------------------


def test_a_run_of_unsupported_characters_becomes_one_question_mark() -> None:
    assert "Hello ? world" in _text(_one({"summary": "Hello \u65e5\u672c\u8a9e world"}))


def test_emoji_and_other_scripts_are_replaced() -> None:
    text = _text(
        _one({"summary": "Great \U0001f44d\U0001f44d and \u041f\u0440\u0438\u0432\u0435\u0442"})
    )

    assert "Great ? and ?" in text


def test_a_notice_says_how_many_places_were_affected() -> None:
    text = _flat(_text(_one({"summary": "a \u65e5\u672c b \U0001f44d c", "other": "\u041f"})))

    assert "Characters not shown: this PDF's font cannot show some characters" in text
    assert "replaced with \u201c?\u201d in 3 places." in text
    assert "The Word and CSV exports keep the original text." in text


def test_one_replacement_is_one_place() -> None:
    assert "in 1 place." in _flat(_text(_one({"summary": "just \u65e5\u672c here"})))


def test_the_notice_is_in_the_stage_that_needed_it() -> None:
    sections = [
        _section(S.BUSINESS_UNDERSTANDING, {"summary": "plain"}),
        _section(S.MARKET_RESEARCH, {"market_overview": "\u65e5\u672c"}),
        _section(S.COMPETITOR_ANALYSIS, {"summary": "plain too"}),
    ]
    text = _text(_report(sections))

    assert text.count("Characters not shown") == 1
    market = text.index("Market Research\nGeneration")
    competitor = text.index("Competitor Analysis\nGeneration")
    notice = text.index("Characters not shown")
    assert market < notice < text.index("Market overview") < competitor


def test_text_the_font_can_show_gets_no_notice() -> None:
    assert "Characters not shown" not in _text(_full())


def test_a_literal_question_mark_is_not_counted() -> None:
    assert "Characters not shown" not in _text(_one({"summary": "Really? Yes?"}))


def test_one_report_does_not_leak_its_replacements_into_the_next() -> None:
    first = _text(_one({"summary": "\u65e5\u672c"}))
    second = _text(_one({"summary": "plain"}))

    assert "Characters not shown" in first
    assert "Characters not shown" not in second


def test_an_unsupported_business_name_is_replaced_and_noted_on_the_cover() -> None:
    report = _report([_section()], business_name="\u30b3\u30fc\u30d2\u30fc Coffee")
    reader = _read(report)

    assert reader.pages[0].extract_text().splitlines()[0] == "? Coffee"
    assert "Characters not shown" in _flat(reader.pages[0].extract_text())
    assert "? Coffee \u00b7 Page 1 of" in _flat(reader.pages[0].extract_text())
    assert reader.metadata is not None
    assert reader.metadata.title == f"? Coffee - {REPORT_TITLE}"


# --- the character rules directly -------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "shown", "replaced"),
    [
        ("plain", "plain", 0),
        ("", "", 0),
        (
            "Caf\u00e9 \u201cquoted\u201d \u2014 \u2022",
            "Caf\u00e9 \u201cquoted\u201d \u2014 \u2022",
            0,
        ),
        ("\u65e5\u672c\u8a9e", REPLACEMENT, 1),
        ("a\u65e5\u672c\u8a9eb \u65e5\u672c\u8a9e", "a?b ?", 2),
        ("\U0001f44d\U0001f44d \U0001f44d", "? ?", 2),
        ("\u65e5\u2003\u65e5", "? ?", 2),
        ("a\u65e5b\u65e5c", "a?b?c", 2),
        ("\u0144", REPLACEMENT, 1),
        ("\u0416", REPLACEMENT, 1),
        ("Hello\u200dworld", "Helloworld", 0),
        ("a\u00a0b", "a\u00a0b", 0),
        ("tab\there", "tab here", 0),
        ("one\ntwo", "one\ntwo", 0),
        ("a\r\nb", "a\nb", 0),
        ("\u2003em space", " em space", 0),
        ("cafe\u0301", "caf\u00e9", 0),
        ("a\x00b", "ab", 0),
        ("?", "?", 0),
    ],
)
def test_visible_text(raw: str, shown: str, replaced: int) -> None:
    text = _Text()

    assert text.visible(raw) == shown
    assert text.replaced == replaced


def test_markup_escapes_and_turns_newlines_into_breaks() -> None:
    assert _Text().markup('a < b & c > "d"\nnext') == 'a &lt; b &amp; c &gt; "d"<br/>next'


def test_the_font_covers_what_the_report_itself_writes() -> None:
    covered = _covered_characters()

    for character in "\u2014\u2022\u00b7\u201c\u201d\u2019 ?:()":
        assert ord(character) in covered, character


def test_the_font_does_not_claim_what_it_lacks() -> None:
    covered = _covered_characters()

    for character in "\u65e5\u0416\u03b1\u2192\u2713\U0001f44d":
        assert ord(character) not in covered, character


# --- structure --------------------------------------------------------------------------


def test_a_list_of_plain_values_is_a_bullet_list() -> None:
    lines = _lines(_one({"key_differentiators": ["Fast", "Fresh", 3, True, None]}))
    start = lines.index("Key differentiators")

    assert lines[start + 1 : start + 6] == [
        "\u2022 Fast",
        "\u2022 Fresh",
        "\u2022 3",
        "\u2022 Yes",
        f"\u2022 {EMPTY}",
    ]


def test_an_empty_list_or_object_shows_a_dash() -> None:
    lines = _lines(_one({"a_list": [], "an_object": {}}))

    assert lines[lines.index("A list") + 1] == EMPTY
    assert lines[lines.index("An object") + 1] == EMPTY


def test_a_list_of_objects_is_one_labelled_block_per_item() -> None:
    content = {
        "sources": [
            {"section": "overview", "source": "https://a.example", "snippet": "One."},
            {"section": "risks", "source": "https://b.example", "snippet": "Two."},
        ]
    }
    text = _flat(_text(_one(content, S.MARKET_RESEARCH)))

    assert (
        "Item 1 Section: overview Source: https://a.example Snippet: One. "
        "Item 2 Section: risks Source: https://b.example Snippet: Two."
    ) in text


def test_an_item_is_headed_by_its_name_and_the_name_is_not_repeated() -> None:
    content = {
        "personas": [
            {"name": "Weekend Wen", "age_range": "28-35"},
            {"title": "Busy Bo", "age_range": "35-45"},
        ]
    }
    text = _flat(_text(_one(content, S.CUSTOMER_PERSONAS)))

    assert "Weekend Wen Age range: 28-35 Busy Bo Age range: 35-45" in text
    assert "Name:" not in text
    assert "Title:" not in text


def test_a_blank_name_is_not_used_as_a_heading() -> None:
    text = _flat(
        _text(_one({"personas": [{"name": "   ", "age_range": "1"}]}, S.CUSTOMER_PERSONAS))
    )

    assert f"Item 1 Name: {EMPTY} Age range: 1" in text


def test_a_list_mixing_plain_values_and_objects_is_written_item_by_item() -> None:
    text = _flat(_text(_one({"things": ["plain", {"name": "Wen", "age": 3}, 7]})))

    assert "Things Item 1 plain Wen Age: 3 Item 3 7" in text


def test_labels_are_bold_and_values_are_not() -> None:
    reader = _read(_one({"personas": [{"age_range": "28-35"}]}, S.CUSTOMER_PERSONAS))
    fonts: dict[str, str] = {}

    def visit(text: str, cm: list[float], tm: list[float], font: Any, size: float) -> None:
        if text.strip() in {"Age range:", "28-35"}:
            fonts[text.strip()] = str(font["/BaseFont"])

    for page in reader.pages:
        page.extract_text(visitor_text=visit)

    assert fonts["Age range:"].endswith("Bold")
    assert fonts["28-35"].endswith("Roman")


def test_nested_objects_and_lists_are_indented_below_their_label() -> None:
    content = {
        "personas": [
            {"name": "Wen", "channels": {"primary": "Instagram", "secondary": ["YouTube"]}}
        ]
    }
    found = _positions(_one(content, S.CUSTOMER_PERSONAS))

    chain = [found["Personas"], found["Channels:"], found["Primary:"], found["YouTube"]]

    assert chain == sorted(chain)
    assert len(set(chain)) == 4
    assert found["Secondary:"] == found["Primary:"]
    assert found["Wen"] == found["Personas"]


def test_a_multi_line_value_in_an_object_is_a_block_under_its_label() -> None:
    content = {"personas": [{"name": "Wen", "bio": "Line one.\n\nLine two."}]}
    lines = _lines(_one(content, S.CUSTOMER_PERSONAS))
    start = lines.index("Bio:")

    assert lines[start : start + 3] == ["Bio:", "Line one.", "Line two."]


def test_numbers_and_booleans_in_objects_are_written_plainly() -> None:
    content = {"personas": [{"name": "Wen", "age": 31, "active": True, "gone": None}]}
    text = _flat(_text(_one(content, S.CUSTOMER_PERSONAS)))

    assert f"Age: 31 Active: Yes Gone: {EMPTY}" in text


def test_keys_that_are_not_plain_names_are_shown_as_they_are() -> None:
    content = {"personas": [{"name": "Wen", "=odd <key>": "x"}]}

    assert "=odd <key>: x" in _flat(_text(_one(content, S.CUSTOMER_PERSONAS)))


def test_headings_stay_with_what_follows_them() -> None:
    styles = _Styles()

    for heading in (styles.h1, styles.h2, styles.h3, styles.label):
        assert heading.keepWithNext == 1, heading.name


def test_a_large_report_is_written_with_a_full_outline() -> None:
    items = [{"name": f"Persona {number}", "detail": "word " * 200} for number in range(300)]
    reader = _read(_one({"personas": items}, S.CUSTOMER_PERSONAS))

    assert len(reader.pages) > 20
    assert len(reader.outline) == 7
    last = reader.pages[-1].extract_text()
    assert f"Page {len(reader.pages)} of {len(reader.pages)}" in _flat(last)
