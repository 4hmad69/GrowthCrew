"""Write a report as a Word document (.docx).

Layout: a cover block (business name, report title, when it was
generated, and a one-line status), a table giving every stage's
generation, approval and status, then one section per stage. A stage
that is flagged (draft, out of date, unknown provenance) says so in a
shaded notice directly under its heading, and a stage that is not in the
report still has its heading with the reason, so nothing is silently
missing.

Stage content is generic JSON, so it is written generically: text becomes
paragraphs, a list of plain values becomes bullets, a list of objects
becomes one labelled block per item, and an object becomes labelled
lines. A field's heading is its key made readable.

Text is written as it was stored, with two changes a Word file forces.
Line endings are normalised to line breaks, and characters XML cannot
carry (control characters and unpaired surrogates) are dropped, because
they would otherwise make the file unreadable or fail the export. A run
of blank lines becomes a paragraph break.
"""

import io
from datetime import UTC, datetime
from typing import Any

from docx import Document
from docx.document import Document as WordDocument
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from docx.styles.style import BaseStyle
from docx.text.paragraph import Paragraph
from pydantic import JsonValue

from backend.app.reports.content import ordered_content
from backend.app.reports.labels import (
    CLEAN_LABEL,
    FLAG_LABELS,
    REPORT_TITLE,
    approval_label,
    field_label,
    flag_notice,
    omission_label,
    omission_notice,
    stage_title,
    status_summary,
)
from backend.app.reports.text import (
    EMPTY,
    clean_text,
    is_scalar,
    item_label,
    paragraph_texts,
    scalar_text,
)
from backend.app.schemas.report import (
    STAGE_ORDER,
    OmissionReason,
    ReportDocument,
    ReportStageSection,
)
from backend.app.schemas.strategy_orchestration import StrategyStage

FONT = "Calibri"
INK = "1F2933"
MUTED = "616E7C"
ACCENT = "1F3A5F"

# Notice fills: a left bar and a soft background.
GOOD_BAR, GOOD_FILL = "2F855A", "E6F4EA"
WARN_BAR, WARN_FILL = "B7791F", "FFF4D6"
INFO_BAR, INFO_FILL = "718096", "EDF2F7"

# A4, with margins that leave 16.6 cm of text width.
PAGE_WIDTH_CM, PAGE_HEIGHT_CM, MARGIN_CM = 21.0, 29.7, 2.2
TABLE_WIDTHS_CM = (5.0, 2.4, 2.6, 6.6)
INDENT_CM = 0.6
FOOTER_SIZE_PT = 9


# --- low-level formatting ---------------------------------------------------------------


def _style_font(
    style: BaseStyle, *, size: float, bold: bool | None = None, color: str = INK
) -> None:
    """Set a style's font, including the theme font overrides built-in styles carry."""

    font = style.font  # type: ignore[attr-defined]
    font.name = FONT
    font.size = Pt(size)
    font.color.rgb = RGBColor.from_string(color)
    if bold is not None:
        font.bold = bold
    rfonts = style.element.get_or_add_rPr().get_or_add_rFonts()  # type: ignore[attr-defined]
    for attribute in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        if rfonts.get(qn(attribute)) is not None:
            del rfonts.attrib[qn(attribute)]
    rfonts.set(qn("w:eastAsia"), FONT)
    rfonts.set(qn("w:cs"), FONT)


def _style_spacing(
    style: BaseStyle, *, before: float, after: float, keep_with_next: bool = False
) -> None:
    paragraph_format = style.paragraph_format  # type: ignore[attr-defined]
    paragraph_format.space_before = Pt(before)
    paragraph_format.space_after = Pt(after)
    paragraph_format.keep_with_next = keep_with_next


def _configure_styles(document: WordDocument) -> None:
    styles = document.styles
    _style_font(styles["Normal"], size=10.5)
    styles["Normal"].paragraph_format.line_spacing = 1.15  # type: ignore[attr-defined]
    _style_spacing(styles["Normal"], before=0, after=6)

    _style_font(styles["Title"], size=26, bold=True, color=ACCENT)
    _style_spacing(styles["Title"], before=0, after=4, keep_with_next=True)
    _style_font(styles["Subtitle"], size=13, bold=False, color=MUTED)
    _style_spacing(styles["Subtitle"], before=0, after=4, keep_with_next=True)

    _style_font(styles["Heading 1"], size=17, bold=True, color=ACCENT)
    _style_spacing(styles["Heading 1"], before=20, after=6, keep_with_next=True)
    _style_font(styles["Heading 2"], size=12.5, bold=True, color=ACCENT)
    _style_spacing(styles["Heading 2"], before=12, after=3, keep_with_next=True)
    _style_font(styles["Heading 3"], size=11, bold=True, color=INK)
    _style_spacing(styles["Heading 3"], before=8, after=2, keep_with_next=True)

    _style_font(styles["List Bullet"], size=10.5)
    _style_spacing(styles["List Bullet"], before=0, after=3)


def _fix_default_template(document: WordDocument) -> None:
    """Give the template's zoom setting the percent attribute the schema requires."""

    zoom = document.settings.element.find(qn("w:zoom"))
    if zoom is not None and zoom.get(qn("w:percent")) is None:
        zoom.set(qn("w:percent"), "100")


def _configure_page(document: WordDocument) -> None:
    section = document.sections[0]
    section.page_width = Cm(PAGE_WIDTH_CM)
    section.page_height = Cm(PAGE_HEIGHT_CM)
    for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(section, side, Cm(MARGIN_CM))


# Children of a paragraph's properties that must come after a border or shading, in schema order.
_AFTER_SHADING = (
    "w:tabs",
    "w:suppressAutoHyphens",
    "w:kinsoku",
    "w:wordWrap",
    "w:overflowPunct",
    "w:topLinePunct",
    "w:autoSpaceDE",
    "w:autoSpaceDN",
    "w:bidi",
    "w:adjustRightInd",
    "w:snapToGrid",
    "w:spacing",
    "w:ind",
    "w:contextualSpacing",
    "w:mirrorIndents",
    "w:suppressOverlap",
    "w:jc",
    "w:textDirection",
    "w:textAlignment",
    "w:textboxTightWrap",
    "w:outlineLvl",
    "w:divId",
    "w:cnfStyle",
    "w:rPr",
    "w:sectPr",
    "w:pPrChange",
)


def _shading_element(fill: str) -> Any:
    shading = OxmlElement("w:shd")
    shading.set(qn("w:val"), "clear")
    shading.set(qn("w:color"), "auto")
    shading.set(qn("w:fill"), fill)
    return shading


def _shade_cell(cell: Any, fill: str) -> None:
    """Shade a table cell. python-docx places the element in schema order."""

    cell._tc.get_or_add_tcPr().append(_shading_element(fill))


def _add_field(paragraph: Paragraph, instruction: str) -> None:
    """A simple field such as the page number, which Word fills in itself.

    The field's placeholder result carries the footer's formatting, since
    that is what is shown until Word recalculates it.
    """

    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), instruction)
    run = OxmlElement("w:r")
    properties = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), MUTED)
    size = OxmlElement("w:sz")
    size.set(qn("w:val"), str(FOOTER_SIZE_PT * 2))
    properties.append(color)
    properties.append(size)
    run.append(properties)
    text = OxmlElement("w:t")
    text.text = "1"
    run.append(text)
    field.append(run)
    paragraph._p.append(field)


def _footer(document: WordDocument, business_name: str) -> None:
    paragraph = document.sections[0].footer.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.add_run(f"{clean_text(business_name)}  \u00b7  Page ")
    _add_field(paragraph, "PAGE")
    paragraph.add_run(" of ")
    _add_field(paragraph, "NUMPAGES")
    for run in paragraph.runs:
        run.font.size = Pt(FOOTER_SIZE_PT)
        run.font.color.rgb = RGBColor.from_string(MUTED)


def _notice(document: WordDocument, text: str, *, bar: str, fill: str) -> None:
    """A shaded paragraph with a coloured bar on its left."""

    paragraph = document.add_paragraph()
    lead, separator, rest = clean_text(text).partition(": ")
    if separator:
        paragraph.add_run(f"{lead}: ").bold = True
        paragraph.add_run(rest)
    else:
        paragraph.add_run(lead)
    paragraph.paragraph_format.space_after = Pt(6)
    paragraph.paragraph_format.left_indent = Cm(0.2)
    properties = paragraph._p.get_or_add_pPr()
    borders = OxmlElement("w:pBdr")
    left = OxmlElement("w:left")
    left.set(qn("w:val"), "single")
    left.set(qn("w:sz"), "24")
    left.set(qn("w:space"), "6")
    left.set(qn("w:color"), bar)
    borders.append(left)
    # The schema fixes the order of paragraph properties: border, then shading,
    # then spacing and indents. Word and validators reject any other order.
    properties.insert_element_before(borders, "w:shd", *_AFTER_SHADING)
    properties.insert_element_before(_shading_element(fill), *_AFTER_SHADING)


def _cell(cell: Any, text: str, *, bold: bool = False, fill: str | None = None) -> None:
    paragraph = cell.paragraphs[0]
    paragraph.paragraph_format.space_after = Pt(2)
    paragraph.paragraph_format.space_before = Pt(2)
    run = paragraph.add_run(clean_text(text))
    run.font.size = Pt(9.5)
    run.bold = bold
    if fill is not None:
        _shade_cell(cell, fill)


def _row_flags(row: Any, *, header: bool) -> None:
    """Keep a row on one page, and repeat the header row on every page."""

    properties = row._tr.get_or_add_trPr()
    properties.append(OxmlElement("w:cantSplit"))
    if header:
        properties.append(OxmlElement("w:tblHeader"))


# --- content ----------------------------------------------------------------------------


def _indent(paragraph: Paragraph, depth: int) -> None:
    if depth:
        paragraph.paragraph_format.left_indent = Cm(INDENT_CM * depth)


def _write_text(document: WordDocument, text: str, depth: int) -> None:
    pieces = paragraph_texts(text) or [EMPTY]
    for piece in pieces:
        _indent(document.add_paragraph(piece), depth)


def _write_label(document: WordDocument, label: str, depth: int) -> None:
    """A label above nested content: a heading at the top level, bold text below it."""

    if depth == 0:
        document.add_heading(clean_text(label), level=3)
        return
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.keep_with_next = True
    paragraph.paragraph_format.space_after = Pt(2)
    paragraph.add_run(clean_text(label)).bold = True
    _indent(paragraph, depth)


def _write_value(document: WordDocument, value: JsonValue, depth: int) -> None:
    if isinstance(value, dict):
        _write_object(document, value, depth)
    elif isinstance(value, list):
        _write_list(document, value, depth)
    elif isinstance(value, str):
        _write_text(document, value, depth)
    else:
        _indent(document.add_paragraph(scalar_text(value)), depth)


def _write_list(document: WordDocument, items: list[JsonValue], depth: int) -> None:
    if not items:
        _indent(document.add_paragraph(EMPTY), depth)
        return
    if all(is_scalar(item) for item in items):
        for item in items:
            paragraph = document.add_paragraph(scalar_text(item), style="List Bullet")
            _indent(paragraph, depth)
        return
    for number, item in enumerate(items, start=1):
        label, used_key = item_label(item, number)
        _write_label(document, label, depth)
        if isinstance(item, dict) and used_key is not None:
            item = {key: value for key, value in item.items() if key != used_key}
        _write_value(document, item, depth + 1)


def _write_object(document: WordDocument, fields: dict[str, JsonValue], depth: int) -> None:
    if not fields:
        _indent(document.add_paragraph(EMPTY), depth)
        return
    for key, value in fields.items():
        label = clean_text(field_label(key))
        inline = scalar_text(value) if is_scalar(value) else None
        paragraph = document.add_paragraph()
        if inline is not None and "\n" not in inline:
            paragraph.add_run(f"{label}: ").bold = True
            paragraph.add_run(inline)
            _indent(paragraph, depth)
            continue
        paragraph.paragraph_format.keep_with_next = True
        paragraph.paragraph_format.space_after = Pt(2)
        paragraph.add_run(f"{label}:").bold = True
        _indent(paragraph, depth)
        _write_value(document, value, depth + 1)


def _write_field(document: WordDocument, name: str, value: JsonValue) -> None:
    document.add_heading(clean_text(field_label(name)), level=2)
    _write_value(document, value, 0)


# --- sections ---------------------------------------------------------------------------


def _write_cover(document: WordDocument, report: ReportDocument) -> None:
    document.add_paragraph(clean_text(report.business_name), style="Title")
    document.add_paragraph(REPORT_TITLE, style="Subtitle")
    generated = report.generated_at.astimezone(UTC).strftime("%d %B %Y, %H:%M UTC")
    meta = document.add_paragraph(f"Generated {generated}")
    meta.runs[0].font.size = Pt(9.5)
    meta.runs[0].font.color.rgb = RGBColor.from_string(MUTED)

    good = report.is_clean
    _notice(
        document,
        f"Status: {status_summary(report)}",
        bar=GOOD_BAR if good else WARN_BAR,
        fill=GOOD_FILL if good else WARN_FILL,
    )


def _write_overview(document: WordDocument, report: ReportDocument) -> None:
    document.add_heading("Stages at a glance", level=2)
    sections = {section.stage: section for section in report.sections}
    omitted = {item.stage: item.reason for item in report.omitted}

    table = document.add_table(rows=1, cols=4)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    for cell, title in zip(
        table.rows[0].cells, ("Stage", "Generation", "Approval", "Status"), strict=True
    ):
        _cell(cell, title, bold=True, fill="D9E2F3")
    _row_flags(table.rows[0], header=True)

    for stage in STAGE_ORDER:
        section = sections.get(stage)
        row = table.add_row()
        if section is None:
            values = (stage_title(stage), EMPTY, EMPTY, omission_label(omitted[stage]))
        else:
            status = ", ".join(FLAG_LABELS[flag] for flag in section.flags) or CLEAN_LABEL
            values = (
                stage_title(stage),
                str(section.generation),
                approval_label(section.approval),
                status,
            )
        for cell, text in zip(row.cells, values, strict=True):
            _cell(cell, text)
        _row_flags(row, header=False)

    # Widths go on the grid columns as well as the cells: LibreOffice reads the grid.
    for column, width in zip(table.columns, TABLE_WIDTHS_CM, strict=True):
        column.width = Cm(width)
    for row in table.rows:
        for cell, width in zip(row.cells, TABLE_WIDTHS_CM, strict=True):
            cell.width = Cm(width)


def _write_stage(
    document: WordDocument,
    stage: StrategyStage,
    section: ReportStageSection | None,
    reason: OmissionReason | None,
) -> None:
    document.add_heading(stage_title(stage), level=1)

    if section is None:
        if reason is None:
            raise ValueError(f"{stage.value} has neither a section nor an omission reason.")
        _notice(document, omission_notice(reason), bar=INFO_BAR, fill=INFO_FILL)
        return

    generated = section.generated_at.astimezone(UTC).strftime("%d %B %Y, %H:%M UTC")
    meta = document.add_paragraph(
        f"Generation {section.generation}  \u00b7  record version {section.record_version}  "
        f"\u00b7  {approval_label(section.approval)}  "
        f"\u00b7  model {clean_text(section.model_used)}  \u00b7  generated {generated}"
    )
    meta.runs[0].font.size = Pt(9)
    meta.runs[0].font.color.rgb = RGBColor.from_string(MUTED)

    for flag in section.flags:
        _notice(document, flag_notice(section, flag), bar=WARN_BAR, fill=WARN_FILL)

    fields = ordered_content(section)
    if not fields:
        document.add_paragraph(EMPTY)
    for name, value in fields:
        _write_field(document, name, value)


def render_docx(report: ReportDocument) -> bytes:
    """The whole report as a .docx file's bytes."""

    document = Document()
    _fix_default_template(document)
    _configure_page(document)
    _configure_styles(document)
    _footer(document, report.business_name)

    properties = document.core_properties
    properties.title = clean_text(f"{report.business_name} - {REPORT_TITLE}")
    properties.author = "GrowthCrew"
    properties.last_modified_by = "GrowthCrew"
    moment: datetime = report.generated_at.astimezone(UTC).replace(tzinfo=None)
    properties.created = moment
    properties.modified = moment

    _write_cover(document, report)
    _write_overview(document, report)
    document.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    sections = {section.stage: section for section in report.sections}
    omitted = {item.stage: item.reason for item in report.omitted}
    for stage in STAGE_ORDER:
        _write_stage(document, stage, sections.get(stage), omitted.get(stage))

    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()
