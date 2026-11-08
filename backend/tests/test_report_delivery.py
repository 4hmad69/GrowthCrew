"""Unit tests for how a report is delivered as a file."""

import io
import zipfile
from datetime import UTC, datetime, timedelta, timezone
from uuid import uuid4

import pytest

from backend.app.reports.delivery import (
    FALLBACK_STEM,
    MAX_STEM_LENGTH,
    MEDIA_TYPES,
    content_disposition,
    file_stem,
    render_report,
    report_filename,
)
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

NOW = datetime(2026, 11, 4, 23, 30, tzinfo=UTC)


def _report(
    business_name: str = "Acme Coffee", generated_at: datetime = NOW, *, with_section: bool = True
) -> ReportDocument:
    sections = []
    if with_section:
        sections = [
            ReportStageSection(
                stage=StrategyStage.BUSINESS_UNDERSTANDING,
                generation=1,
                record_id=uuid4(),
                record_version=1,
                approval=StageApprovalState.APPROVED,
                staleness=StageStaleness(provenance=ProvenanceState.TRACKED),
                model_used="m",
                generated_at=generated_at,
                content={"summary": "A summary."},
            )
        ]
    present = {section.stage for section in sections}
    return ReportDocument(
        workspace_id=uuid4(),
        business_name=business_name,
        generated_at=generated_at,
        sections=sections,
        omitted=[
            OmittedStage(stage=stage, reason=OmissionReason.NOT_GENERATED)
            for stage in STAGE_ORDER
            if stage not in present
        ],
    )


# --- media types ------------------------------------------------------------------------


def test_every_format_has_its_own_media_type() -> None:
    assert MEDIA_TYPES == {
        ReportFormat.PDF: "application/pdf",
        ReportFormat.DOCX: (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ),
        ReportFormat.CSV: "text/csv; charset=utf-8",
    }
    assert set(MEDIA_TYPES) == set(ReportFormat)


# --- rendering --------------------------------------------------------------------------


def test_pdf_is_rendered_as_a_pdf() -> None:
    assert render_report(_report(), ReportFormat.PDF).startswith(b"%PDF-")


def test_docx_is_rendered_as_a_word_file() -> None:
    data = render_report(_report(), ReportFormat.DOCX)

    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        assert "word/document.xml" in archive.namelist()


def test_csv_is_rendered_as_utf8_csv_with_a_byte_order_mark() -> None:
    data = render_report(_report(), ReportFormat.CSV)

    assert data.startswith(b"\xef\xbb\xbf")
    assert data.decode("utf-8-sig").startswith("stage,generation,")


# --- filename stems ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "stem"),
    [
        ("Acme Coffee", "acme-coffee"),
        ("  Acme   Coffee  ", "acme-coffee"),
        ("ACME", "acme"),
        ("Fish & Chips <Ltd>", "fish-chips-ltd"),
        ("Café Müller", "cafe-muller"),
        ("Zoë's Crème Brûlée", "zoe-s-creme-brulee"),
        ("a/b\\c:d*e?f", "a-b-c-d-e-f"),
        ("../../etc/passwd", "etc-passwd"),
        ("\r\nSet-Cookie: x=1", "set-cookie-x-1"),
        ('quote"here', "quote-here"),
        ("100% Organic #1", "100-organic-1"),
        ("--dashes--", "dashes"),
        ("under_score", "under-score"),
        ("CON", "con"),
    ],
)
def test_file_stem_is_lowercase_ascii_letters_digits_and_hyphens(name: str, stem: str) -> None:
    assert file_stem(name) == stem


@pytest.mark.parametrize(
    "name", ["日本語", "コーヒー", "قهوة", "☕", "!!!", "   ", "---", "\x00\x01"]
)
def test_a_name_with_nothing_a_filename_can_keep_falls_back(name: str) -> None:
    assert file_stem(name) == FALLBACK_STEM


def test_a_mixed_script_name_keeps_what_it_can() -> None:
    assert file_stem("コーヒー Coffee") == "coffee"


def test_a_long_name_is_cut_to_sixty_characters() -> None:
    assert MAX_STEM_LENGTH == 60
    assert file_stem("x" * 500) == "x" * 60


def test_leading_punctuation_does_not_use_up_the_length_allowance() -> None:
    assert file_stem("!!!" + "a" * 100) == "a" * 60


def test_a_long_name_is_cut_without_a_trailing_hyphen() -> None:
    stem = file_stem("word " * 40)

    assert len(stem) <= MAX_STEM_LENGTH
    assert not stem.endswith("-")
    assert not stem.startswith("-")
    assert stem.startswith("word-word-")


def test_the_cut_never_ends_on_a_hyphen_even_when_it_lands_on_one() -> None:
    name = "a" * (MAX_STEM_LENGTH - 1) + " " + "b" * 10

    assert file_stem(name) == "a" * (MAX_STEM_LENGTH - 1)


@pytest.mark.parametrize(
    "name",
    ["Acme Coffee", "Café", "x" * 500, "\u0000\u202e evil", "A/B\\C", "<script>alert(1)</script>"],
)
def test_a_stem_is_never_anything_but_safe_characters(name: str) -> None:
    stem = file_stem(name)

    assert stem
    assert all(char in "abcdefghijklmnopqrstuvwxyz0123456789-" for char in stem)
    assert len(stem) <= MAX_STEM_LENGTH


# --- filenames --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("report_format", "extension"),
    [(ReportFormat.PDF, "pdf"), (ReportFormat.DOCX, "docx"), (ReportFormat.CSV, "csv")],
)
def test_the_filename_names_the_business_the_report_and_the_day(
    report_format: ReportFormat, extension: str
) -> None:
    assert (
        report_filename(_report(), report_format)
        == f"acme-coffee-strategy-report-2026-11-04.{extension}"
    )


def test_the_day_is_the_utc_day_whatever_the_zone_of_the_timestamp() -> None:
    late_in_karachi = datetime(2026, 11, 5, 3, 0, tzinfo=timezone(timedelta(hours=5)))

    report = _report(generated_at=late_in_karachi)

    assert report_filename(report, ReportFormat.PDF).endswith("2026-11-04.pdf")


def test_a_name_with_nothing_to_keep_still_gets_a_useful_filename() -> None:
    assert (
        report_filename(_report("日本語"), ReportFormat.CSV)
        == "growthcrew-strategy-report-2026-11-04.csv"
    )


# --- the header -------------------------------------------------------------------------


def test_the_header_asks_the_browser_to_save_the_file() -> None:
    assert content_disposition("a-report.pdf") == 'attachment; filename="a-report.pdf"'


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ('evil".pdf', 'attachment; filename="evil_.pdf"'),
        ("a\r\nSet-Cookie: x=1.pdf", 'attachment; filename="a__Set-Cookie__x_1.pdf"'),
        ("caf\u00e9.pdf", 'attachment; filename="caf_.pdf"'),
        ("a b;c.pdf", 'attachment; filename="a_b_c.pdf"'),
        ("../x.pdf", 'attachment; filename=".._x.pdf"'),
    ],
)
def test_the_header_cannot_be_broken_out_of(filename: str, expected: str) -> None:
    value = content_disposition(filename)

    assert value == expected
    assert "\r" not in value
    assert "\n" not in value
    assert value.count('"') == 2
