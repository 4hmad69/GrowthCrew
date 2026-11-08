"""How a report leaves the system as a file.

Which renderer produces each format, what the file is called, and how it
is labelled for a browser. Nothing here touches the database or the web:
it takes a report document and returns bytes and text.

A filename is built only from the report's own business name and date,
reduced to lowercase ASCII letters, digits and hyphens. A business name
is free text, and a filename goes into an HTTP header and onto someone's
disk, so nothing else is allowed through.
"""

import re
import unicodedata
from collections.abc import Callable
from datetime import UTC

from backend.app.reports.csv_export import render_csv
from backend.app.reports.docx_export import render_docx
from backend.app.reports.pdf_export import render_pdf
from backend.app.schemas.report import ReportDocument, ReportFormat

MEDIA_TYPES: dict[ReportFormat, str] = {
    ReportFormat.PDF: "application/pdf",
    ReportFormat.DOCX: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ReportFormat.CSV: "text/csv; charset=utf-8",
}

_RENDERERS: dict[ReportFormat, Callable[[ReportDocument], bytes]] = {
    ReportFormat.PDF: render_pdf,
    ReportFormat.DOCX: render_docx,
    ReportFormat.CSV: render_csv,
}

# Used when a business name has nothing in it a filename can keep (for example a name
# written entirely in a script with no ASCII equivalent).
FALLBACK_STEM = "growthcrew"
MAX_STEM_LENGTH = 60
_NOT_SAFE = re.compile(r"[^a-z0-9]+")
_NOT_HEADER_SAFE = re.compile(r"[^A-Za-z0-9._-]")


def render_report(report: ReportDocument, report_format: ReportFormat) -> bytes:
    """The report as a file in the given format."""

    return _RENDERERS[report_format](report)


def file_stem(business_name: str) -> str:
    """A business name as a safe, readable filename stem.

    Accents are folded to their plain letters (Caf\u00e9 becomes cafe), everything
    that is not a letter or digit becomes a single hyphen, and the result is
    cut to a sensible length without leaving a trailing hyphen.
    """

    folded = unicodedata.normalize("NFKD", business_name).encode("ascii", "ignore").decode("ascii")
    stem = _NOT_SAFE.sub("-", folded.lower()).strip("-")[:MAX_STEM_LENGTH].strip("-")
    return stem or FALLBACK_STEM


def report_filename(report: ReportDocument, report_format: ReportFormat) -> str:
    """For example ``acme-coffee-strategy-report-2026-11-04.pdf``."""

    day = report.generated_at.astimezone(UTC).strftime("%Y-%m-%d")
    return f"{file_stem(report.business_name)}-strategy-report-{day}.{report_format.value}"


def content_disposition(filename: str) -> str:
    """A header value that makes a browser save the response under this name.

    Anything outside letters, digits, dot, underscore and hyphen is replaced,
    so a filename can never break out of the quoted header value.
    """

    return f'attachment; filename="{_NOT_HEADER_SAFE.sub("_", filename)}"'
