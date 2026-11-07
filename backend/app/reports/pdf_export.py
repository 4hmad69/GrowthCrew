"""Write a report as a PDF.

Same layout and wording as the Word export: a cover (business name,
report title, when it was generated, a one-line status), a table of
every stage's generation, approval and status, then one section per
stage with a shaded notice under any flagged stage's heading, and a
heading with the reason for any stage that is not in the report. Every
stage heading is also a bookmark, so a reader can jump between stages.

Fonts. The PDF embeds the Bitstream Vera family that ships inside
reportlab, so it looks the same on every machine and needs nothing
installed. Vera covers English and most Western European text and the
usual punctuation (curly quotes, dashes, bullets, the euro sign), but
only part of Latin Extended (for example it has "\u0142" but not
"\u0144"), and it has no Greek, Cyrillic, Vietnamese, emoji, arrows,
check marks, rupee sign, or scripts such as Chinese, Arabic or Hindi. A
character the font cannot show is never left to print as a blank or a
black box: a run of such characters becomes a single "?", and the stage
(or the cover) says how many places were affected and that the Word and
CSV exports keep the original text. Invisible format characters
(zero-width joiners and the like) are dropped. Text is composed to
Unicode NFC first, so accented letters written as a base letter plus a
combining mark still show.

A PDF re-flows text to fit the page, so a run of spaces collapses to one
and leading spaces are not kept; the line breaks and paragraph breaks
that were stored are.
"""

import io
import unicodedata
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import reportlab
from pydantic import JsonValue
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.utils import TimeStamp
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

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

# Font names registered with reportlab; the files are the ones reportlab ships.
REGULAR, BOLD, ITALIC, BOLD_ITALIC = "GCSans", "GCSans-Bold", "GCSans-Italic", "GCSans-BoldItalic"
_FONT_FILES = {
    REGULAR: "Vera.ttf",
    BOLD: "VeraBd.ttf",
    ITALIC: "VeraIt.ttf",
    BOLD_ITALIC: "VeraBI.ttf",
}

# The same palette as the Word export.
INK = colors.HexColor("#1F2933")
MUTED = colors.HexColor("#616E7C")
ACCENT = colors.HexColor("#1F3A5F")
GOOD_BAR, GOOD_FILL = colors.HexColor("#2F855A"), colors.HexColor("#E6F4EA")
WARN_BAR, WARN_FILL = colors.HexColor("#B7791F"), colors.HexColor("#FFF4D6")
INFO_BAR, INFO_FILL = colors.HexColor("#718096"), colors.HexColor("#EDF2F7")
HEADER_FILL = colors.HexColor("#D9E2F3")
GRID = colors.HexColor("#A0AEC0")

MARGIN = 2.2 * cm
TEXT_WIDTH = A4[0] - 2 * MARGIN
TABLE_WIDTHS = tuple(width * cm for width in (5.0, 2.4, 2.6, 6.6))
INDENT = 0.6 * cm
REPLACEMENT = "?"


@lru_cache(maxsize=1)
def _covered_characters() -> frozenset[int]:
    """Register the fonts (once) and return the code points the regular face can show."""

    folder = Path(reportlab.__file__).parent / "fonts"
    for name, file in _FONT_FILES.items():
        if name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(name, str(folder / file)))
    pdfmetrics.registerFontFamily(
        REGULAR, normal=REGULAR, bold=BOLD, italic=ITALIC, boldItalic=BOLD_ITALIC
    )
    return frozenset(pdfmetrics.getFont(REGULAR).face.charToGlyph)


class _Text:
    """Turns raw text into something the PDF font can show, counting what it could not."""

    def __init__(self) -> None:
        self._covered = _covered_characters()
        self.replaced = 0

    def visible(self, text: str) -> str:
        """The text with every character the font cannot show replaced, or dropped if invisible."""

        pieces: list[str] = []
        in_gap = False
        for char in unicodedata.normalize("NFC", clean_text(text)):
            if char == "\n" or ord(char) in self._covered:
                pieces.append(char)
                in_gap = False
            elif unicodedata.category(char) == "Cf":
                continue
            elif char.isspace():
                pieces.append(" ")
                in_gap = False
            elif not in_gap:
                pieces.append(REPLACEMENT)
                self.replaced += 1
                in_gap = True
        return "".join(pieces)

    def markup(self, text: str) -> str:
        """The text as Paragraph markup: escaped, with line breaks."""

        return escape(self.visible(text)).replace("\n", "<br/>")


# --- styles -----------------------------------------------------------------------------


def _style(name: str, **values: Any) -> ParagraphStyle:
    defaults: dict[str, Any] = {
        "fontName": REGULAR,
        "fontSize": 9.5,
        "leading": 13.5,
        "textColor": INK,
        "spaceAfter": 6,
        "splitLongWords": 1,
        "bulletFontName": REGULAR,
    }
    defaults.update(values)
    return ParagraphStyle(name, **defaults)


class _Styles:
    def __init__(self) -> None:
        self.body = _style("body")
        self.title = _style(
            "title", fontName=BOLD, fontSize=24, leading=28, textColor=ACCENT, spaceAfter=4
        )
        self.subtitle = _style(
            "subtitle", fontName=ITALIC, fontSize=12.5, leading=16, textColor=MUTED, spaceAfter=4
        )
        self.meta = _style("meta", fontSize=8, leading=10.5, textColor=MUTED)
        self.h1 = _style(
            "h1",
            fontName=BOLD,
            fontSize=16,
            leading=20,
            textColor=ACCENT,
            spaceBefore=18,
            spaceAfter=6,
            keepWithNext=1,
        )
        self.h2 = _style(
            "h2",
            fontName=BOLD,
            fontSize=12,
            leading=15,
            textColor=ACCENT,
            spaceBefore=11,
            spaceAfter=3,
            keepWithNext=1,
        )
        self.h3 = _style(
            "h3",
            fontName=BOLD,
            fontSize=10.5,
            leading=13,
            spaceBefore=7,
            spaceAfter=2,
            keepWithNext=1,
        )
        self.label = _style("label", fontName=BOLD, spaceAfter=2, keepWithNext=1)
        self.notice = _style("notice", spaceAfter=0)
        self.cell = _style("cell", fontSize=8.8, leading=11, spaceAfter=0)
        self.cell_head = _style("cell_head", fontName=BOLD, fontSize=8.8, leading=11, spaceAfter=0)
        self._indented: dict[tuple[str, int], ParagraphStyle] = {}

    def indented(self, base: ParagraphStyle, depth: int) -> ParagraphStyle:
        """The style moved right by the nesting depth."""

        if depth == 0:
            return base
        key = (base.name, depth)
        if key not in self._indented:
            self._indented[key] = ParagraphStyle(
                f"{base.name}-{depth}", parent=base, leftIndent=base.leftIndent + INDENT * depth
            )
        return self._indented[key]

    def bullet(self, depth: int) -> ParagraphStyle:
        key = ("bullet", depth)
        if key not in self._indented:
            self._indented[key] = ParagraphStyle(
                f"bullet-{depth}",
                parent=self.body,
                leftIndent=INDENT * depth + 0.55 * cm,
                bulletIndent=INDENT * depth,
                spaceAfter=3,
            )
        return self._indented[key]


class _Heading(Paragraph):
    """A heading that is also a bookmark in the PDF's outline."""

    def __init__(self, text: str, style: ParagraphStyle, outline_title: str | None = None) -> None:
        super().__init__(text, style)
        self.outline_title = outline_title


class _ReportDocument(BaseDocTemplate):
    """A one-column A4 document with an outline entry for every stage heading.

    The text frame has no padding, so tables and shaded notices (which are as
    wide as the text) line up exactly with the paragraphs around them.
    """

    def __init__(self, target: io.BytesIO, **options: Any) -> None:
        super().__init__(target, **options)
        # (page number, title) of every stage heading, in order. The outline is written
        # when the pages are, because only then does a bookmark know which page it is on.
        self.headings: list[tuple[int, str]] = []
        frame = Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
            id="body",
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
        )
        self.addPageTemplates([PageTemplate(id="main", frames=[frame])])

    def afterFlowable(self, flowable: Flowable) -> None:
        title = getattr(flowable, "outline_title", None)
        if title is not None:
            self.headings.append((self.page, title))


def _creation_stamp(moment: datetime) -> TimeStamp:
    """The PDF's creation and modification time: the report's own, in UTC.

    reportlab's invariant mode pins the time to the year 2000 so that output
    is reproducible. The stamp is then set to when the report was generated,
    so the file is both reproducible and honest about its date.
    """

    utc = moment.astimezone(UTC)
    stamp = TimeStamp(invariant=1)
    stamp.t = utc.timestamp()
    stamp.lt = utc.timetuple()
    stamp.YMDhms = (utc.year, utc.month, utc.day, utc.hour, utc.minute, utc.second)
    stamp.tzname = "UTC"
    stamp.dhh = stamp.dmm = 0
    return stamp


def _canvas_class(
    footer_text: str, moment: datetime, headings: list[tuple[int, str]]
) -> type[Canvas]:
    """A canvas that writes the footer and the outline, which both need the finished pages.

    'Page N of M' needs the total, so pages are held back and written in a second pass.
    The same pass writes the outline, because a bookmark made earlier would not yet
    know which page it is on.
    """

    class NumberedCanvas(Canvas):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._saved_pages: list[dict[str, Any]] = []
            self._doc._timeStamp = _creation_stamp(moment)

        def showPage(self) -> None:
            self._saved_pages.append(dict(self.__dict__))
            self._startPage()

        def save(self) -> None:
            total = len(self._saved_pages)
            entry = 0
            for state in self._saved_pages:
                self.__dict__.update(state)
                self._footer(total)
                for page, title in headings:
                    if page == self._pageNumber:
                        entry += 1
                        self.bookmarkPage(f"stage-{entry}")
                        self.addOutlineEntry(title, f"stage-{entry}", 0, 0)
                super().showPage()
            super().save()

        def _footer(self, total: int) -> None:
            self.setFont(REGULAR, 8.5)
            self.setFillColor(MUTED)
            self.drawCentredString(
                A4[0] / 2, 1.1 * cm, f"{footer_text}  \u00b7  Page {self._pageNumber} of {total}"
            )

    return NumberedCanvas


# --- flowables --------------------------------------------------------------------------


def _notice(text: _Text, styles: _Styles, message: str, *, bar: Any, fill: Any) -> Table:
    """A shaded block with a coloured bar on its left."""

    lead, separator, rest = message.partition(": ")
    markup = f"<b>{text.markup(lead)}: </b>{text.markup(rest)}" if separator else text.markup(lead)
    table = Table([[Paragraph(markup, styles.notice)]], colWidths=[TEXT_WIDTH])
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), REGULAR),
                ("BACKGROUND", (0, 0), (-1, -1), fill),
                ("LINEBEFORE", (0, 0), (0, -1), 3, bar),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    table.spaceAfter = 6
    return table


def _replacement_notice(text: _Text, styles: _Styles, count: int) -> Table:
    places = "1 place" if count == 1 else f"{count} places"
    return _notice(
        text,
        styles,
        f"Characters not shown: this PDF's font cannot show some characters, so they were "
        f"replaced with \u201c{REPLACEMENT}\u201d in {places}. The Word and CSV exports keep "
        "the original text.",
        bar=INFO_BAR,
        fill=INFO_FILL,
    )


class _Writer:
    """Writes stage content into a list of flowables."""

    def __init__(self, text: _Text, styles: _Styles, story: list[Flowable]) -> None:
        self.text = text
        self.styles = styles
        self.story = story

    def paragraph(self, markup: str, style: ParagraphStyle, depth: int = 0) -> None:
        self.story.append(Paragraph(markup, self.styles.indented(style, depth)))

    def write_text(self, value: str, depth: int) -> None:
        for piece in paragraph_texts(value) or [EMPTY]:
            self.paragraph(self.text.markup(piece), self.styles.body, depth)

    def write_label(self, label: str, depth: int) -> None:
        if depth == 0:
            self.paragraph(self.text.markup(label), self.styles.h3)
        else:
            self.paragraph(self.text.markup(label), self.styles.label, depth)

    def write_value(self, value: JsonValue, depth: int) -> None:
        if isinstance(value, dict):
            self.write_object(value, depth)
        elif isinstance(value, list):
            self.write_list(value, depth)
        elif isinstance(value, str):
            self.write_text(value, depth)
        else:
            self.paragraph(self.text.markup(scalar_text(value)), self.styles.body, depth)

    def write_list(self, items: list[JsonValue], depth: int) -> None:
        if not items:
            self.paragraph(EMPTY, self.styles.body, depth)
            return
        if all(is_scalar(item) for item in items):
            for item in items:
                self.story.append(
                    Paragraph(
                        self.text.markup(scalar_text(item)),
                        self.styles.bullet(depth),
                        bulletText="\u2022",
                    )
                )
            return
        for number, item in enumerate(items, start=1):
            label, used_key = item_label(item, number)
            self.write_label(label, depth)
            if isinstance(item, dict) and used_key is not None:
                item = {key: value for key, value in item.items() if key != used_key}
            self.write_value(item, depth + 1)

    def write_object(self, fields: dict[str, JsonValue], depth: int) -> None:
        if not fields:
            self.paragraph(EMPTY, self.styles.body, depth)
            return
        for key, value in fields.items():
            label = self.text.markup(field_label(key))
            inline = scalar_text(value) if is_scalar(value) else None
            if inline is not None and "\n" not in inline:
                line = f"<b>{label}: </b>{self.text.markup(inline)}"
                self.paragraph(line, self.styles.body, depth)
                continue
            self.paragraph(f"{label}:", self.styles.label, depth)
            self.write_value(value, depth + 1)

    def write_field(self, name: str, value: JsonValue) -> None:
        self.paragraph(self.text.markup(field_label(name)), self.styles.h2)
        self.write_value(value, 0)


# --- sections ---------------------------------------------------------------------------


def _generated(report: ReportDocument) -> str:
    return report.generated_at.astimezone(UTC).strftime("%d %B %Y, %H:%M UTC")


def _cover(text: _Text, styles: _Styles, report: ReportDocument) -> list[Flowable]:
    story: list[Flowable] = [
        Paragraph(text.markup(report.business_name), styles.title),
        Paragraph(text.markup(REPORT_TITLE), styles.subtitle),
        Paragraph(f"Generated {_generated(report)}", styles.meta),
        Spacer(1, 4),
    ]
    good = report.is_clean
    story.append(
        _notice(
            text,
            styles,
            f"Status: {status_summary(report)}",
            bar=GOOD_BAR if good else WARN_BAR,
            fill=GOOD_FILL if good else WARN_FILL,
        )
    )
    return story


def _overview(text: _Text, styles: _Styles, report: ReportDocument) -> list[Flowable]:
    sections = {section.stage: section for section in report.sections}
    omitted = {item.stage: item.reason for item in report.omitted}

    rows: list[list[Paragraph]] = [
        [
            Paragraph(title, styles.cell_head)
            for title in ("Stage", "Generation", "Approval", "Status")
        ]
    ]
    for stage in STAGE_ORDER:
        section = sections.get(stage)
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
        rows.append([Paragraph(text.markup(value), styles.cell) for value in values])

    table = Table(rows, colWidths=list(TABLE_WIDTHS), repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), REGULAR),
                ("GRID", (0, 0), (-1, -1), 0.5, GRID),
                ("BACKGROUND", (0, 0), (-1, 0), HEADER_FILL),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return [Paragraph("Stages at a glance", styles.h2), table]


def _stage(
    text: _Text,
    styles: _Styles,
    stage: StrategyStage,
    section: ReportStageSection | None,
    reason: OmissionReason | None,
) -> list[Flowable]:
    title = stage_title(stage)
    story: list[Flowable] = [_Heading(text.markup(title), styles.h1, outline_title=title)]

    if section is None:
        if reason is None:
            raise ValueError(f"{stage.value} has neither a section nor an omission reason.")
        story.append(_notice(text, styles, omission_notice(reason), bar=INFO_BAR, fill=INFO_FILL))
        return story

    generated = section.generated_at.astimezone(UTC).strftime("%d %B %Y, %H:%M UTC")
    story.append(
        Paragraph(
            text.markup(
                f"Generation {section.generation}  \u00b7  record version "
                f"{section.record_version}  \u00b7  {approval_label(section.approval)}  "
                f"\u00b7  model {section.model_used}  \u00b7  generated {generated}"
            ),
            styles.meta,
        )
    )
    for flag in section.flags:
        story.append(
            _notice(text, styles, flag_notice(section, flag), bar=WARN_BAR, fill=WARN_FILL)
        )

    notice_position = len(story)
    replaced_before = text.replaced
    writer = _Writer(text, styles, story)
    fields = ordered_content(section)
    if not fields:
        writer.paragraph(EMPTY, styles.body)
    for name, value in fields:
        writer.write_field(name, value)
    if text.replaced > replaced_before:
        story.insert(
            notice_position, _replacement_notice(text, styles, text.replaced - replaced_before)
        )
    return story


def render_pdf(report: ReportDocument) -> bytes:
    """The whole report as a PDF file's bytes."""

    text = _Text()
    styles = _Styles()
    footer_text = _Text().visible(report.business_name).replace("\n", " ")

    replaced_before = text.replaced
    story: list[Flowable] = _cover(text, styles, report)
    if text.replaced > replaced_before:
        story.append(_replacement_notice(text, styles, text.replaced - replaced_before))
    story.extend(_overview(text, styles, report))
    story.append(PageBreak())

    sections = {section.stage: section for section in report.sections}
    omitted = {item.stage: item.reason for item in report.omitted}
    for stage in STAGE_ORDER:
        story.extend(_stage(text, styles, stage, sections.get(stage), omitted.get(stage)))

    buffer = io.BytesIO()
    document = _ReportDocument(
        buffer,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN + 0.4 * cm,
        title=_Text().visible(f"{report.business_name} - {REPORT_TITLE}"),
        author="GrowthCrew",
        subject=REPORT_TITLE,
        creator="GrowthCrew",
        invariant=1,
        # Without this reportlab starts every page in Helvetica, which is not embedded.
        initialFontName=REGULAR,
    )
    document.build(
        story, canvasmaker=_canvas_class(footer_text, report.generated_at, document.headings)
    )
    return buffer.getvalue()
