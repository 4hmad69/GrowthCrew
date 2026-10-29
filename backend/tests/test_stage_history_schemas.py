"""Unit tests for stage history schemas, staleness types, and the content registry."""

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest
from pydantic import BaseModel, ValidationError

from backend.app.db.models.brand_strategy import BrandStrategy
from backend.app.db.models.business_understanding import BusinessUnderstanding
from backend.app.db.models.competitor_analysis import CompetitorAnalysis
from backend.app.db.models.content_plan import ContentPlan
from backend.app.db.models.customer_persona_set import CustomerPersonaSet
from backend.app.db.models.market_research import MarketResearch
from backend.app.db.models.marketing_strategy import MarketingStrategy
from backend.app.db.models.stage_snapshot import StageSnapshot
from backend.app.schemas.brand_strategy import BrandStrategyResponse
from backend.app.schemas.business_understanding import BusinessUnderstandingResponse
from backend.app.schemas.competitor_analysis import CompetitorAnalysisResponse
from backend.app.schemas.content_plan import ContentPlanResponse
from backend.app.schemas.customer_persona_set import CustomerPersonaSetResponse
from backend.app.schemas.market_research import MarketResearchResponse
from backend.app.schemas.marketing_strategy import MarketingStrategyResponse
from backend.app.schemas.stage_history import (
    STAGE_CONTENT_FIELDS,
    ProfileInputRef,
    StageHistoryResponse,
    StageInputRef,
    StageInputs,
    StageSnapshotDetail,
    StageSnapshotSummary,
)
from backend.app.schemas.strategy_orchestration import (
    ProvenanceState,
    StageStaleness,
    StaleInput,
    StrategyPrerequisite,
    StrategyStage,
)

STAGE_MODELS: dict[StrategyStage, type] = {
    StrategyStage.BUSINESS_UNDERSTANDING: BusinessUnderstanding,
    StrategyStage.MARKET_RESEARCH: MarketResearch,
    StrategyStage.COMPETITOR_ANALYSIS: CompetitorAnalysis,
    StrategyStage.CUSTOMER_PERSONAS: CustomerPersonaSet,
    StrategyStage.BRAND_STRATEGY: BrandStrategy,
    StrategyStage.MARKETING_STRATEGY: MarketingStrategy,
    StrategyStage.CONTENT_PLAN: ContentPlan,
}

STAGE_RESPONSES: dict[StrategyStage, type[BaseModel]] = {
    StrategyStage.BUSINESS_UNDERSTANDING: BusinessUnderstandingResponse,
    StrategyStage.MARKET_RESEARCH: MarketResearchResponse,
    StrategyStage.COMPETITOR_ANALYSIS: CompetitorAnalysisResponse,
    StrategyStage.CUSTOMER_PERSONAS: CustomerPersonaSetResponse,
    StrategyStage.BRAND_STRATEGY: BrandStrategyResponse,
    StrategyStage.MARKETING_STRATEGY: MarketingStrategyResponse,
    StrategyStage.CONTENT_PLAN: ContentPlanResponse,
}

# Columns every stage record has that are not generated content.
BOOKKEEPING_COLUMNS = {
    "id",
    "workspace_id",
    "profile_id",
    "model_used",
    "input_tokens",
    "output_tokens",
    "version",
    "created_at",
    "updated_at",
}

ALL_STAGES = list(StrategyStage)


def _inputs_payload() -> dict[str, Any]:
    """Build inputs the way they are stored: JSON-shaped, ids as strings."""

    return {
        "profile": {"id": str(uuid4()), "version": 2},
        "stages": {
            "business_understanding": {
                "generation": 3,
                "record_id": str(uuid4()),
                "record_version": 2,
            }
        },
    }


def _snapshot(**overrides: Any) -> StageSnapshot:
    """Build an unsaved snapshot row, with any field overridable."""

    values: dict[str, Any] = {
        "workspace_id": uuid4(),
        "stage": "market_research",
        "generation": 1,
        "record_id": uuid4(),
        "record_version": 1,
        "content": {"market_overview": "An overview."},
        "inputs": None,
        "model_used": "test-model",
        "input_tokens": 10,
        "output_tokens": 20,
        "created_at": datetime(2026, 10, 10, 12, 0, tzinfo=UTC),
    }
    values.update(overrides)
    return StageSnapshot(**values)


# --- content registry -------------------------------------------------------


def test_registry_lists_every_stage_once_in_chain_order() -> None:
    """A stage missing from the registry could never have a snapshot written."""

    assert tuple(STAGE_CONTENT_FIELDS) == tuple(StrategyStage)


@pytest.mark.parametrize("stage", ALL_STAGES)
def test_registry_fields_are_present_and_unique(stage: StrategyStage) -> None:
    """Each stage snapshots at least one field, and never the same one twice."""

    fields = STAGE_CONTENT_FIELDS[stage]

    assert fields
    assert len(set(fields)) == len(fields)


@pytest.mark.parametrize("stage", ALL_STAGES)
def test_registry_matches_the_stage_model_columns_exactly(stage: StrategyStage) -> None:
    """A column added to or dropped from a stage must be decided for history too.

    Every column that is not bookkeeping is generated content, and
    content that is not snapshotted is content that is silently lost
    when the stage is regenerated.
    """

    columns = {column.name for column in STAGE_MODELS[stage].__table__.columns}  # type: ignore[attr-defined]

    assert set(STAGE_CONTENT_FIELDS[stage]) == columns - BOOKKEEPING_COLUMNS


@pytest.mark.parametrize("stage", ALL_STAGES)
def test_registry_fields_exist_on_the_stage_response_schema(stage: StrategyStage) -> None:
    """Snapshot content must stay renderable as the stage's own response."""

    response_fields = set(STAGE_RESPONSES[stage].model_fields)

    assert set(STAGE_CONTENT_FIELDS[stage]) <= response_fields


# --- recorded inputs --------------------------------------------------------


def test_inputs_parse_from_their_stored_json_shape() -> None:
    """Ids and stage keys arrive as strings in JSONB and must coerce back."""

    inputs = StageInputs.model_validate(_inputs_payload())

    assert isinstance(inputs.profile, ProfileInputRef)
    assert inputs.profile.version == 2
    reference = inputs.stages[StrategyStage.BUSINESS_UNDERSTANDING]
    assert isinstance(reference, StageInputRef)
    assert reference.generation == 3
    assert reference.record_version == 2


def test_inputs_round_trip_through_json_unchanged() -> None:
    """What is written to the inputs column reads back as the same object."""

    inputs = StageInputs.model_validate(_inputs_payload())

    stored = inputs.model_dump(mode="json")

    assert StageInputs.model_validate(stored) == inputs
    assert "business_understanding" in stored["stages"]


def test_inputs_may_name_only_the_profile_or_only_stages() -> None:
    """The first three stages build from the profile alone; later ones from stages."""

    payload = _inputs_payload()

    assert StageInputs.model_validate({"profile": payload["profile"]}).stages == {}
    assert StageInputs.model_validate({"stages": payload["stages"]}).profile is None


def test_inputs_reject_recording_nothing() -> None:
    """Empty inputs would read as 'built from nothing'; unknown is None instead."""

    with pytest.raises(ValidationError, match="profile or at least one stage"):
        StageInputs.model_validate({})


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p["stages"].update({"not_a_stage": p["stages"]["business_understanding"]}),
        lambda p: p["stages"]["business_understanding"].update({"generation": 0}),
        lambda p: p["stages"]["business_understanding"].update({"record_version": 0}),
        lambda p: p["profile"].update({"version": 0}),
        lambda p: p["profile"].update({"id": "not-a-uuid"}),
        lambda p: p.update({"unexpected": 1}),
    ],
    ids=[
        "unknown-stage",
        "generation-zero",
        "record-version-zero",
        "profile-version-zero",
        "profile-id-not-uuid",
        "extra-field",
    ],
)
def test_inputs_reject_malformed_values(mutate: Any) -> None:
    """Provenance is only useful if it is exact, so malformed input fails loudly."""

    payload = _inputs_payload()
    mutate(payload)

    with pytest.raises(ValidationError):
        StageInputs.model_validate(payload)


# --- snapshot responses -----------------------------------------------------


def test_summary_validates_from_a_database_row() -> None:
    """The ORM row is converted directly, coercing the stage string to the enum."""

    summary = StageSnapshotSummary.model_validate(_snapshot(generation=4))

    assert summary.stage is StrategyStage.MARKET_RESEARCH
    assert summary.generation == 4
    assert summary.input_tokens == 10


def test_summary_without_inputs_reports_unknown_provenance() -> None:
    """NULL inputs - a pre-history stage - are reported as unknown, not as empty."""

    summary = StageSnapshotSummary.model_validate(_snapshot(inputs=None))

    assert summary.provenance is ProvenanceState.UNKNOWN
    assert summary.model_dump(mode="json")["provenance"] == "unknown"


def test_summary_with_inputs_reports_tracked_provenance_without_listing_them() -> None:
    """The summary says provenance is tracked but leaves the inputs to the detail view."""

    summary = StageSnapshotSummary.model_validate(_snapshot(inputs=_inputs_payload()))

    dumped = summary.model_dump(mode="json")
    assert summary.provenance is ProvenanceState.TRACKED
    assert dumped["provenance"] == "tracked"
    assert "inputs" not in dumped
    assert "content" not in dumped


def test_detail_returns_content_and_inputs() -> None:
    """The detail view is the full record of one generation."""

    payload = _inputs_payload()
    row = _snapshot(
        generation=2,
        content={"market_overview": "Overview.", "sources": [{"url": "https://example.com"}]},
        inputs=payload,
    )

    detail = StageSnapshotDetail.model_validate(row)

    dumped = detail.model_dump(mode="json")
    assert dumped["workspace_id"] == str(row.workspace_id)
    assert dumped["content"]["sources"] == [{"url": "https://example.com"}]
    assert dumped["inputs"]["profile"]["version"] == 2
    assert dumped["provenance"] == "tracked"


def test_detail_with_unknown_provenance_has_null_inputs() -> None:
    """A backfilled generation shows inputs as null alongside unknown provenance."""

    dumped = StageSnapshotDetail.model_validate(_snapshot(inputs=None)).model_dump(mode="json")

    assert dumped["inputs"] is None
    assert dumped["provenance"] == "unknown"


def test_detail_still_reads_content_from_before_a_stage_gained_a_field() -> None:
    """Old history must stay readable: content is not held to today's field list."""

    old_content = {"market_overview": "Written when the stage had fewer fields."}
    assert set(old_content) != set(STAGE_CONTENT_FIELDS[StrategyStage.MARKET_RESEARCH])

    detail = StageSnapshotDetail.model_validate(_snapshot(content=old_content))

    assert detail.content == old_content


def test_detail_rejects_content_that_is_not_json() -> None:
    """Content comes from a JSONB column, so anything else is a bug worth surfacing."""

    row = _snapshot(content={"market_overview": datetime(2026, 10, 10, tzinfo=UTC)})

    with pytest.raises(ValidationError):
        StageSnapshotDetail.model_validate(row)


def _summary(generation: int, stage: str = "market_research") -> StageSnapshotSummary:
    """Build a summary for one generation of a stage."""

    return StageSnapshotSummary.model_validate(_snapshot(generation=generation, stage=stage))


def test_history_lists_snapshots_newest_first() -> None:
    """A well-formed history reports its newest generation."""

    history = StageHistoryResponse(
        workspace_id=uuid4(),
        stage=StrategyStage.MARKET_RESEARCH,
        snapshots=[_summary(3), _summary(2), _summary(1)],
    )

    assert history.latest_generation == 3
    assert history.model_dump(mode="json")["latest_generation"] == 3


def test_empty_history_has_no_latest_generation() -> None:
    """A stage that was never generated has an empty history, not an error."""

    history = StageHistoryResponse(
        workspace_id=uuid4(), stage=StrategyStage.CONTENT_PLAN, snapshots=[]
    )

    assert history.latest_generation is None


@pytest.mark.parametrize(
    "generations",
    [[1, 2, 3], [3, 1, 2], [2, 2, 1]],
    ids=["oldest-first", "unsorted", "repeated-generation"],
)
def test_history_rejects_any_order_but_strictly_newest_first(generations: list[int]) -> None:
    """Clients rely on position, so a mis-ordered list is a service bug to catch."""

    with pytest.raises(ValidationError, match="newest first"):
        StageHistoryResponse(
            workspace_id=uuid4(),
            stage=StrategyStage.MARKET_RESEARCH,
            snapshots=[_summary(generation) for generation in generations],
        )


def test_history_rejects_a_snapshot_from_another_stage() -> None:
    """One stage's history can never carry another stage's snapshot."""

    with pytest.raises(ValidationError, match="belong to the stage"):
        StageHistoryResponse(
            workspace_id=uuid4(),
            stage=StrategyStage.MARKET_RESEARCH,
            snapshots=[_summary(2), _summary(1, stage="brand_strategy")],
        )


# --- staleness types --------------------------------------------------------


def test_provenance_states_are_tracked_and_unknown() -> None:
    """The two states clients branch on."""

    assert {state.value for state in ProvenanceState} == {"tracked", "unknown"}


def test_stale_input_accepts_the_profile_and_every_stage() -> None:
    """Any prerequisite - the profile or an upstream stage - can be the stale one."""

    for prerequisite in StrategyPrerequisite:
        item = StaleInput(prerequisite=prerequisite, built_from=1, current=2)
        assert item.prerequisite is prerequisite


@pytest.mark.parametrize(
    ("built_from", "current"),
    [(2, 2), (3, 2)],
    ids=["unchanged", "went-backwards"],
)
def test_stale_input_must_have_moved_forward(built_from: int, current: int) -> None:
    """An input that has not advanced is not stale, so it cannot be listed as such."""

    with pytest.raises(ValidationError, match="newer now"):
        StaleInput(
            prerequisite=StrategyPrerequisite.MARKET_RESEARCH,
            built_from=built_from,
            current=current,
        )


def test_stale_input_rejects_zero_generations() -> None:
    """Generations and versions start at one."""

    with pytest.raises(ValidationError):
        StaleInput(prerequisite=StrategyPrerequisite.BUSINESS_PROFILE, built_from=0, current=1)


def test_tracked_stage_with_no_moved_inputs_is_not_stale() -> None:
    """Fresh means tracked and nothing upstream has advanced."""

    staleness = StageStaleness(provenance=ProvenanceState.TRACKED)

    assert staleness.stale is False
    assert staleness.model_dump(mode="json") == {
        "provenance": "tracked",
        "stale_inputs": [],
        "stale": False,
    }


def test_stage_with_moved_inputs_is_stale_and_says_which() -> None:
    """Stale names every input that moved, and reports the flag in its output."""

    staleness = StageStaleness(
        provenance=ProvenanceState.TRACKED,
        stale_inputs=[
            StaleInput(prerequisite=StrategyPrerequisite.BUSINESS_PROFILE, built_from=1, current=3),
            StaleInput(
                prerequisite=StrategyPrerequisite.CUSTOMER_PERSONAS, built_from=2, current=3
            ),
        ],
    )

    dumped = staleness.model_dump(mode="json")
    assert staleness.stale is True
    assert dumped["stale"] is True
    assert [item["prerequisite"] for item in dumped["stale_inputs"]] == [
        "business_profile",
        "customer_personas",
    ]


def test_unknown_provenance_is_never_stale() -> None:
    """Nothing recorded the inputs, so nothing can be claimed about them moving."""

    assert StageStaleness(provenance=ProvenanceState.UNKNOWN).stale is False

    with pytest.raises(ValidationError, match="unknown provenance"):
        StageStaleness(
            provenance=ProvenanceState.UNKNOWN,
            stale_inputs=[
                StaleInput(
                    prerequisite=StrategyPrerequisite.MARKET_RESEARCH, built_from=1, current=2
                )
            ],
        )


def test_staleness_rejects_listing_the_same_input_twice() -> None:
    """Each input appears once, with the single generation it moved to."""

    item = StaleInput(prerequisite=StrategyPrerequisite.MARKET_RESEARCH, built_from=1, current=2)

    with pytest.raises(ValidationError, match="twice"):
        StageStaleness(provenance=ProvenanceState.TRACKED, stale_inputs=[item, item])
