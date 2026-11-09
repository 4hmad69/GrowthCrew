"""Unit tests for the wording shared by the report exporters."""

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest

from backend.app.reports.labels import (
    CLEAN_LABEL,
    FLAG_LABELS,
    REPORT_TITLE,
    approval_label,
    field_label,
    flag_notice,
    omission_label,
    omission_notice,
    prerequisite_title,
    stage_title,
    stale_input_lines,
    status_summary,
)
from backend.app.schemas.report import (
    STAGE_ORDER,
    OmissionReason,
    OmittedStage,
    ReportDocument,
    ReportStageFlag,
    ReportStageSection,
)
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


def _stale(*items: tuple[StrategyPrerequisite, int, int]) -> StageStaleness:
    return StageStaleness(
        provenance=ProvenanceState.TRACKED,
        stale_inputs=[
            StaleInput(prerequisite=prerequisite, built_from=built, current=current)
            for prerequisite, built, current in items
        ],
    )


def _section(
    stage: StrategyStage = S.BUSINESS_UNDERSTANDING,
    *,
    approval: StageApprovalState = StageApprovalState.APPROVED,
    staleness: StageStaleness | None = None,
) -> ReportStageSection:
    values: dict[str, Any] = {
        "stage": stage,
        "generation": 1,
        "record_id": uuid4(),
        "record_version": 1,
        "approval": approval,
        "staleness": staleness or StageStaleness(provenance=ProvenanceState.TRACKED),
        "model_used": "m",
        "generated_at": NOW,
        "content": {},
    }
    return ReportStageSection(**values)


def _report(
    sections: list[ReportStageSection], reasons: dict[StrategyStage, OmissionReason] | None = None
) -> ReportDocument:
    present = {section.stage for section in sections}
    reasons = reasons or {}
    return ReportDocument(
        workspace_id=uuid4(),
        business_name="Acme",
        generated_at=NOW,
        sections=sections,
        omitted=[
            OmittedStage(stage=stage, reason=reasons.get(stage, OmissionReason.NOT_GENERATED))
            for stage in STAGE_ORDER
            if stage not in present
        ],
    )


def test_the_report_title_is_the_wording_people_see() -> None:
    assert REPORT_TITLE == "Marketing strategy report"


# --- names ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("stage", "title"),
    [
        (S.BUSINESS_UNDERSTANDING, "Business Understanding"),
        (S.MARKET_RESEARCH, "Market Research"),
        (S.CONTENT_PLAN, "Content Plan"),
    ],
)
def test_stage_titles_are_readable(stage: StrategyStage, title: str) -> None:
    assert stage_title(stage) == title


def test_every_stage_has_a_distinct_title() -> None:
    titles = [stage_title(stage) for stage in STAGE_ORDER]

    assert len(set(titles)) == len(STAGE_ORDER)
    assert all(title[0].isupper() and "_" not in title for title in titles)


def test_prerequisite_titles_cover_the_business_profile() -> None:
    assert prerequisite_title(StrategyPrerequisite.BUSINESS_PROFILE) == "Business Profile"
    assert prerequisite_title(StrategyPrerequisite.MARKET_RESEARCH) == "Market Research"


@pytest.mark.parametrize(
    ("key", "label"),
    [
        ("market_overview", "Market overview"),
        ("summary", "Summary"),
        ("sources", "Sources"),
        ("ninety_day_roadmap", "Ninety day roadmap"),
        ("Already Nice", "Already Nice"),
        ("=cmd", "=cmd"),
        ("two words", "two words"),
        ("with-dash", "with-dash"),
        ("1leading", "1leading"),
        ("", ""),
    ],
)
def test_field_labels(key: str, label: str) -> None:
    assert field_label(key) == label


def test_approval_labels() -> None:
    assert approval_label(StageApprovalState.APPROVED) == "Approved"
    assert approval_label(StageApprovalState.DRAFT) == "Draft"


def test_every_flag_has_a_short_label() -> None:
    assert set(FLAG_LABELS) == set(ReportStageFlag)
    assert len(set(FLAG_LABELS.values())) == len(ReportStageFlag)
    assert CLEAN_LABEL not in FLAG_LABELS.values()


# --- notices ----------------------------------------------------------------------------


def test_the_profile_is_counted_in_versions_and_stages_in_generations() -> None:
    section = _section(
        staleness=_stale(
            (StrategyPrerequisite.BUSINESS_PROFILE, 1, 3),
            (StrategyPrerequisite.MARKET_RESEARCH, 2, 4),
        )
    )

    assert stale_input_lines(section) == [
        "Business Profile (built from version 1, now version 3)",
        "Market Research (built from generation 2, now generation 4)",
    ]


def test_a_stage_with_nothing_stale_lists_nothing() -> None:
    assert stale_input_lines(_section()) == []


def test_draft_notice() -> None:
    notice = flag_notice(_section(), ReportStageFlag.DRAFT)

    assert notice.startswith("Draft: ")
    assert "not been approved" in notice


def test_stale_notice_names_what_changed() -> None:
    section = _section(staleness=_stale((StrategyPrerequisite.MARKET_RESEARCH, 1, 2)))

    notice = flag_notice(section, ReportStageFlag.STALE)

    assert notice.startswith("Out of date: ")
    assert "Market Research (built from generation 1, now generation 2)" in notice
    assert notice.endswith(".")


def test_stale_notice_joins_several_inputs() -> None:
    section = _section(
        staleness=_stale(
            (StrategyPrerequisite.BUSINESS_PROFILE, 1, 2),
            (StrategyPrerequisite.MARKET_RESEARCH, 1, 2),
        )
    )

    notice = flag_notice(section, ReportStageFlag.STALE)

    assert "Business Profile (built from version 1, now version 2); Market Research" in notice


def test_unknown_provenance_notice_does_not_call_it_stale_or_current() -> None:
    section = _section(staleness=StageStaleness(provenance=ProvenanceState.UNKNOWN))

    notice = flag_notice(section, ReportStageFlag.UNKNOWN_PROVENANCE)

    assert notice.startswith("Unknown provenance: ")
    assert "cannot be called current or out of date" in notice


def test_omission_wording() -> None:
    assert omission_notice(OmissionReason.NOT_GENERATED).startswith("Not included: ")
    assert "has not been generated" in omission_notice(OmissionReason.NOT_GENERATED)
    assert "no longer matches its version history" in omission_notice(
        OmissionReason.HISTORY_OUT_OF_SYNC
    )
    assert omission_label(OmissionReason.NOT_GENERATED) == "Not generated"
    assert (
        omission_label(OmissionReason.HISTORY_OUT_OF_SYNC) == "Not included (history out of sync)"
    )


# --- the status line --------------------------------------------------------------------


def test_status_when_nothing_is_included() -> None:
    assert status_summary(_report([])) == "Nothing to report yet: no stage can be included."


def test_status_when_everything_is_current() -> None:
    summary = status_summary(_report([_section(stage) for stage in STAGE_ORDER]))

    assert summary == (
        "Complete and current: all 7 stages are approved, built from current inputs, "
        "and have recorded provenance."
    )


def test_status_names_flagged_stages() -> None:
    sections = [
        _section(stage, approval=StageApprovalState.DRAFT)
        if stage in {S.MARKET_RESEARCH, S.BRAND_STRATEGY}
        else _section(stage)
        for stage in STAGE_ORDER
    ]

    assert status_summary(_report(sections)) == (
        "Needs attention: 2 of 7 stages need attention (Market Research, Brand Strategy)."
    )


def test_status_names_omitted_stages_with_the_right_grammar() -> None:
    one = [_section(stage) for stage in STAGE_ORDER if stage is not S.CONTENT_PLAN]
    two = [_section(stage) for stage in STAGE_ORDER[:5]]

    assert status_summary(_report(one)) == (
        "Needs attention: 1 stage is not included (Content Plan)."
    )
    assert status_summary(_report(two)) == (
        "Needs attention: 2 stages are not included (Marketing Strategy, Content Plan)."
    )


def test_status_reports_flagged_and_omitted_together() -> None:
    sections = [_section(S.BUSINESS_UNDERSTANDING, approval=StageApprovalState.DRAFT)]

    assert status_summary(_report(sections)) == (
        "Needs attention: 1 of 7 stages need attention (Business Understanding); "
        "6 stages are not included (Market Research, Competitor Analysis, Customer Personas, "
        "Brand Strategy, Marketing Strategy, Content Plan)."
    )
