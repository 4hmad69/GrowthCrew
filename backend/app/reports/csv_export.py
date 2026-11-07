"""Write a report as CSV: one row per value, in a flat, sortable table.

A strategy report is nested text, so the CSV is a long table rather than
one wide row per stage. Every row says which stage and generation a
value belongs to and how far that stage can be trusted, so a row is never
separated from its flags when the file is filtered or pasted elsewhere:

    stage, generation, record_version, approval, flags, path, value

``path`` locates the value inside the stage's content: ``personas[0].name``
is the ``name`` of the first item of ``personas``. A stage that is not in
the report gets a single row with an empty path and a flag of
``omitted:<reason>``, so a missing stage is visible rather than silent.

Values are written as text. ``null`` becomes an empty cell, true and
false become ``true`` and ``false``, and an empty list or object becomes
``[]`` or ``{}`` so that a field with nothing in it still appears.

Generated text is untrusted: it comes from a model and from web pages. A
spreadsheet treats a cell that begins with ``=``, ``+``, ``-``, ``@``, a
tab or a carriage return as a formula, which can run commands or send data
elsewhere. Any text cell that begins with one of those has a single quote
put in front of it, which makes the spreadsheet read it as plain text.
Numbers produced by this module are not text and are left alone.

The file is UTF-8 with a byte-order mark, because Excel otherwise opens
UTF-8 as a legacy code page and garbles non-ASCII text.
"""

import csv
import io
import json

from pydantic import JsonValue

from backend.app.reports.content import ordered_content
from backend.app.schemas.report import STAGE_ORDER, ReportDocument, ReportStageSection

CSV_COLUMNS: tuple[str, ...] = (
    "stage",
    "generation",
    "record_version",
    "approval",
    "flags",
    "path",
    "value",
)

# Characters that make a spreadsheet read the start of a cell as a formula.
FORMULA_PREFIXES: tuple[str, ...] = ("=", "+", "-", "@", "\t", "\r")

FLAG_SEPARATOR = ";"


def escape_formula(text: str) -> str:
    """Return text that a spreadsheet will read as text, never as a formula."""

    if text.startswith(FORMULA_PREFIXES):
        return "'" + text
    return text


def _leaf(value: JsonValue) -> str:
    """A scalar as a cell, with text made safe for a spreadsheet."""

    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return escape_formula(value)
    return json.dumps(value)


def _flatten(path: str, value: JsonValue) -> list[tuple[str, str]]:
    """Every value under a path as (path, cell) pairs, in document order."""

    if isinstance(value, dict):
        if not value:
            return [(path, "{}")]
        rows: list[tuple[str, str]] = []
        for key, child in value.items():
            rows.extend(_flatten(f"{path}.{key}" if path else key, child))
        return rows
    if isinstance(value, list):
        if not value:
            return [(path, "[]")]
        rows = []
        for index, child in enumerate(value):
            rows.extend(_flatten(f"{path}[{index}]", child))
        return rows
    return [(path, _leaf(value))]


def _section_rows(section: ReportStageSection) -> list[list[str]]:
    prefix = [
        section.stage.value,
        str(section.generation),
        str(section.record_version),
        section.approval.value,
        FLAG_SEPARATOR.join(flag.value for flag in section.flags),
    ]
    values: list[tuple[str, str]] = []
    for name, value in ordered_content(section):
        values.extend(_flatten(name, value))
    if not values:
        # A stage with no content still shows up, so it cannot be mistaken for omitted.
        values = [("", "")]
    return [[*prefix, escape_formula(path), cell] for path, cell in values]


def render_csv(report: ReportDocument) -> bytes:
    """The whole report as CSV bytes (UTF-8 with a byte-order mark)."""

    sections = {section.stage: section for section in report.sections}
    omitted = {item.stage: item.reason for item in report.omitted}

    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\r\n")
    writer.writerow(CSV_COLUMNS)
    for stage in STAGE_ORDER:
        section = sections.get(stage)
        if section is not None:
            writer.writerows(_section_rows(section))
        else:
            writer.writerow([stage.value, "", "", "", f"omitted:{omitted[stage].value}", "", ""])
    return buffer.getvalue().encode("utf-8-sig")
