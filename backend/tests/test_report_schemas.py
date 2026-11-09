"""Unit tests for the report schemas."""

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest
from pydantic import ValidationError

from backend.app.schemas.report import (
    STAGE_ORDER,
    OmissionReason,
    OmittedStage,
    ReportDocument,
    ReportFormat,
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


def _tracked() -> StageStaleness:
    return StageStaleness(provenance=ProvenanceState.TRACKED)


def _stale() -> StageStaleness:
    return StageStaleness(
        provenance=ProvenanceState.TRACKED,
        stale_inputs=[
            StaleInput(
                prerequisite=StrategyPrerequisite.BUSINESS_PROFILE,
                built_from=1,
                current=2,
            )
        ],
    )


def _unknown() -> StageStaleness:
    return StageStaleness(provenance=ProvenanceState.UNKNOWN)


def _section(
    stage: StrategyStage = StrategyStage.BUSINESS_UNDERSTANDING,
    *,
    approval: StageApprovalState = StageApprovalState.APPROVED,
    staleness: StageStaleness | None = None,
    **overrides: Any,
) -> ReportStageSection:
    values: dict[str, Any] = {
        "stage": stage,
        "generation": 1,
        "record_id": uuid4(),
        "record_version": 1,
        "approval": approval,
        "staleness": staleness if staleness is not None else _tracked(),
        "model_used": "gpt-oss:120b-cloud",
        "generated_at": NOW,
        "content": {"summary": "A summary."},
    }
    values.update(overrides)
    return ReportStageSection(**values)


def _document(
    sections: list[ReportStageSection] | None = None,
    omitted: list[OmittedStage] | None = None,
    **overrides: Any,
) -> ReportDocument:
    values: dict[str, Any] = {
        "workspace_id": uuid4(),
        "business_name": "Acme Coffee",
        "generated_at": NOW,
        "sections": sections or [],
        "omitted": omitted or [],
    }
    values.update(overrides)
    return ReportDocument(**values)


def _all_sections() -> list[ReportStageSection]:
    return [_section(stage) for stage in STAGE_ORDER]


def _omit(*stages: StrategyStage) -> list[OmittedStage]:
    return [OmittedStage(stage=stage, reason=OmissionReason.NOT_GENERATED) for stage in stages]


# --- constants and enums ---------------------------------------------------


def test_stage_order_is_the_generation_order() -> None:
    assert STAGE_ORDER == (
        StrategyStage.BUSINESS_UNDERSTANDING,
        StrategyStage.MARKET_RESEARCH,
        StrategyStage.COMPETITOR_ANALYSIS,
        StrategyStage.CUSTOMER_PERSONAS,
        StrategyStage.BRAND_STRATEGY,
        StrategyStage.MARKETING_STRATEGY,
        StrategyStage.CONTENT_PLAN,
    )


def test_report_formats() -> None:
    assert [item.value for item in ReportFormat] == ["pdf", "docx", "csv"]


def test_flag_and_omission_values_are_the_wire_values() -> None:
    assert [item.value for item in ReportStageFlag] == [
        "draft",
        "stale",
        "unknown_provenance",
    ]
    assert [item.value for item in OmissionReason] == ["not_generated", "history_out_of_sync"]


# --- section flags ---------------------------------------------------------


def test_an_approved_tracked_current_stage_is_clean() -> None:
    section = _section()

    assert section.flags == []
    assert section.is_clean is True


def test_a_draft_stage_is_flagged_draft() -> None:
    section = _section(approval=StageApprovalState.DRAFT)

    assert section.flags == [ReportStageFlag.DRAFT]
    assert section.is_clean is False


def test_a_stale_stage_is_flagged_stale_even_when_approved() -> None:
    section = _section(staleness=_stale())

    assert section.flags == [ReportStageFlag.STALE]
    assert section.is_clean is False


def test_unknown_provenance_is_flagged_and_never_clean_even_when_approved() -> None:
    section = _section(staleness=_unknown())

    assert section.flags == [ReportStageFlag.UNKNOWN_PROVENANCE]
    assert section.is_clean is False


def test_unknown_provenance_is_not_called_stale() -> None:
    assert ReportStageFlag.STALE not in _section(staleness=_unknown()).flags


def test_flags_combine_in_a_fixed_order() -> None:
    draft_and_stale = _section(approval=StageApprovalState.DRAFT, staleness=_stale())
    draft_and_unknown = _section(approval=StageApprovalState.DRAFT, staleness=_unknown())

    assert draft_and_stale.flags == [ReportStageFlag.DRAFT, ReportStageFlag.STALE]
    assert draft_and_unknown.flags == [
        ReportStageFlag.DRAFT,
        ReportStageFlag.UNKNOWN_PROVENANCE,
    ]


def test_section_serializes_its_flags_and_cleanliness() -> None:
    dumped = _section(approval=StageApprovalState.DRAFT).model_dump(mode="json")

    assert dumped["flags"] == ["draft"]
    assert dumped["is_clean"] is False
    assert dumped["staleness"]["stale"] is False


# --- section validation ----------------------------------------------------


def test_section_rejects_unexpected_fields() -> None:
    with pytest.raises(ValidationError):
        _section(unexpected="x")


@pytest.mark.parametrize("field", ["generation", "record_version"])
@pytest.mark.parametrize("value", [0, -1])
def test_section_counters_must_be_positive(field: str, value: int) -> None:
    with pytest.raises(ValidationError):
        _section(**{field: value})


def test_section_requires_a_model_name() -> None:
    with pytest.raises(ValidationError):
        _section(model_used="")


def test_section_content_must_be_an_object() -> None:
    with pytest.raises(ValidationError):
        _section(content=["not", "an", "object"])


def test_section_content_keeps_whitespace_exactly_as_stored() -> None:
    content = {
        "  padded key\t": "  padded value\n\n",
        "items": ["\tfirst ", {"\nnested": " second\r\n"}],
    }

    section = _section(content=content)

    assert section.content == content
    assert list(section.content) == ["  padded key\t", "items"]


def test_section_content_is_not_held_to_a_field_list() -> None:
    section = _section(content={"a_field_the_stage_no_longer_has": [1, {"x": None}]})

    assert section.content == {"a_field_the_stage_no_longer_has": [1, {"x": None}]}


# --- document validation ---------------------------------------------------


def test_a_report_with_every_stage_is_complete() -> None:
    document = _document(sections=_all_sections())

    assert document.is_complete is True
    assert document.omitted == []


def test_a_report_with_nothing_generated_is_valid_but_incomplete() -> None:
    document = _document(omitted=_omit(*STAGE_ORDER))

    assert document.sections == []
    assert document.is_complete is False
    assert document.is_clean is False
    assert document.flagged_stages == []


def test_a_stage_may_be_a_section_or_omitted() -> None:
    sections = [_section(stage) for stage in STAGE_ORDER[:-1]]
    document = _document(sections=sections, omitted=_omit(StrategyStage.CONTENT_PLAN))

    assert document.is_complete is False
    assert [item.stage for item in document.omitted] == [StrategyStage.CONTENT_PLAN]


def test_every_stage_must_be_accounted_for() -> None:
    with pytest.raises(ValidationError, match="Every stage must appear"):
        _document(sections=[_section(stage) for stage in STAGE_ORDER[:-1]])


def test_a_stage_cannot_be_a_section_twice() -> None:
    sections = [*_all_sections(), _section(StrategyStage.CONTENT_PLAN)]

    with pytest.raises(ValidationError, match="only once"):
        _document(sections=sections)


def test_a_stage_cannot_be_both_a_section_and_omitted() -> None:
    with pytest.raises(ValidationError, match="only once"):
        _document(sections=_all_sections(), omitted=_omit(StrategyStage.CONTENT_PLAN))


def test_sections_must_be_in_generation_order() -> None:
    sections = list(reversed(_all_sections()))

    with pytest.raises(ValidationError, match="Sections must be listed in generation order"):
        _document(sections=sections)


def test_omitted_stages_must_be_in_generation_order() -> None:
    with pytest.raises(ValidationError, match="Omitted stages must be listed"):
        _document(omitted=_omit(*reversed(STAGE_ORDER)))


@pytest.mark.parametrize("name", ["", "x" * 161])
def test_business_name_is_bounded(name: str) -> None:
    with pytest.raises(ValidationError):
        _document(omitted=_omit(*STAGE_ORDER), business_name=name)


def test_document_rejects_unexpected_fields() -> None:
    with pytest.raises(ValidationError):
        _document(omitted=_omit(*STAGE_ORDER), unexpected="x")


# --- document summary ------------------------------------------------------


def test_a_complete_all_clean_report_is_clean() -> None:
    document = _document(sections=_all_sections())

    assert document.is_clean is True
    assert document.flagged_stages == []


def test_one_flagged_section_makes_the_report_unclean() -> None:
    sections = _all_sections()
    sections[3] = _section(StrategyStage.CUSTOMER_PERSONAS, approval=StageApprovalState.DRAFT)
    sections[5] = _section(StrategyStage.MARKETING_STRATEGY, staleness=_unknown())

    document = _document(sections=sections)

    assert document.is_clean is False
    assert document.flagged_stages == [
        StrategyStage.CUSTOMER_PERSONAS,
        StrategyStage.MARKETING_STRATEGY,
    ]


def test_an_incomplete_report_is_not_clean_even_if_its_sections_are() -> None:
    sections = [_section(stage) for stage in STAGE_ORDER[:-1]]
    document = _document(sections=sections, omitted=_omit(StrategyStage.CONTENT_PLAN))

    assert all(section.is_clean for section in document.sections)
    assert document.is_clean is False


def test_document_serializes_its_summary() -> None:
    dumped = _document(
        sections=[_section(stage) for stage in STAGE_ORDER[:-1]],
        omitted=[
            OmittedStage(
                stage=StrategyStage.CONTENT_PLAN,
                reason=OmissionReason.HISTORY_OUT_OF_SYNC,
            )
        ],
    ).model_dump(mode="json")

    assert dumped["is_complete"] is False
    assert dumped["is_clean"] is False
    assert dumped["flagged_stages"] == []
    assert dumped["omitted"] == [{"stage": "content_plan", "reason": "history_out_of_sync"}]
