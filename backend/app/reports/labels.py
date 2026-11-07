"""The words a written report uses, shared by every document format.

The Word and PDF exporters describe a report in the same terms, so the
wording lives here once: stage titles, field labels, what each flag
means, and the one-line status at the top. Nothing here knows about a
file format.
"""

import re

from backend.app.schemas.report import (
    STAGE_ORDER,
    OmissionReason,
    ReportDocument,
    ReportStageFlag,
    ReportStageSection,
)
from backend.app.schemas.strategy_orchestration import (
    StageApprovalState,
    StrategyPrerequisite,
    StrategyStage,
)

REPORT_TITLE = "Marketing strategy report"

# A short name for a flag, for a table cell.
FLAG_LABELS: dict[ReportStageFlag, str] = {
    ReportStageFlag.DRAFT: "Draft",
    ReportStageFlag.STALE: "Out of date",
    ReportStageFlag.UNKNOWN_PROVENANCE: "Unknown provenance",
}

# A stage with no flags is approved, built from current inputs, and has
# recorded provenance.
CLEAN_LABEL = "Current"

_PLAIN_KEY = re.compile(r"^[a-z][a-z0-9_]*$")


def stage_title(stage: StrategyStage) -> str:
    """A stage's name for a heading: ``market_research`` -> ``Market Research``."""

    return stage.value.replace("_", " ").title()


def prerequisite_title(prerequisite: StrategyPrerequisite) -> str:
    """An input's name for a sentence: ``business_profile`` -> ``Business Profile``."""

    return prerequisite.value.replace("_", " ").title()


def field_label(key: str) -> str:
    """A content key for a heading: ``market_overview`` -> ``Market overview``.

    A key that is not a plain snake_case name is shown exactly as it is.
    """

    if _PLAIN_KEY.match(key):
        return key.replace("_", " ").capitalize()
    return key


def approval_label(approval: StageApprovalState) -> str:
    return "Approved" if approval is StageApprovalState.APPROVED else "Draft"


def stale_input_lines(section: ReportStageSection) -> list[str]:
    """What a stale stage was built from, and what it is now.

    The business profile is versioned; an upstream stage is counted in
    history generations, so the two are worded differently.
    """

    lines: list[str] = []
    for item in section.staleness.stale_inputs:
        unit = (
            "version"
            if item.prerequisite is StrategyPrerequisite.BUSINESS_PROFILE
            else "generation"
        )
        lines.append(
            f"{prerequisite_title(item.prerequisite)} "
            f"(built from {unit} {item.built_from}, now {unit} {item.current})"
        )
    return lines


def flag_notice(section: ReportStageSection, flag: ReportStageFlag) -> str:
    """A sentence saying what a flag means for this stage."""

    if flag is ReportStageFlag.DRAFT:
        return "Draft: this version of the stage has not been approved."
    if flag is ReportStageFlag.STALE:
        return (
            "Out of date: built from inputs that have since changed: "
            + "; ".join(stale_input_lines(section))
            + "."
        )
    return (
        "Unknown provenance: nothing records what this stage was built from, "
        "so it cannot be called current or out of date."
    )


def omission_notice(reason: OmissionReason) -> str:
    """A sentence saying why a stage is not in the report."""

    if reason is OmissionReason.NOT_GENERATED:
        return "Not included: this stage has not been generated."
    return (
        "Not included: this stage's stored record no longer matches its version "
        "history, so no generation can be named for it."
    )


def omission_label(reason: OmissionReason) -> str:
    """A short form of the omission reason, for a table cell."""

    if reason is OmissionReason.NOT_GENERATED:
        return "Not generated"
    return "Not included (history out of sync)"


def status_summary(report: ReportDocument) -> str:
    """The one-line verdict at the top of a report."""

    total = len(STAGE_ORDER)
    if not report.sections:
        return "Nothing to report yet: no stage can be included."
    if report.is_clean:
        return (
            f"Complete and current: all {total} stages are approved, built from "
            "current inputs, and have recorded provenance."
        )

    parts: list[str] = []
    if report.flagged_stages:
        names = ", ".join(stage_title(stage) for stage in report.flagged_stages)
        parts.append(f"{len(report.flagged_stages)} of {total} stages need attention ({names})")
    if report.omitted:
        names = ", ".join(stage_title(item.stage) for item in report.omitted)
        noun = "stage is" if len(report.omitted) == 1 else "stages are"
        parts.append(f"{len(report.omitted)} {noun} not included ({names})")
    return "Needs attention: " + "; ".join(parts) + "."
