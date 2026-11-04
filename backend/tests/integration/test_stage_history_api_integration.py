"""Real PostgreSQL integration tests for the stage history API.

Uses the deterministic "local" providers and no Tavily key. Where a test
needs generations whose content genuinely differs - the local providers
are deterministic, so regenerating through the agents repeats itself -
it writes them through the history service exactly as an agent does.
What it proves is the contract over real data: past generations stay
readable after their record is overwritten, provenance is reported as
recorded, bad input is a clean 4xx instead of a database error, and the
routes keep working with the model provider down.
"""

import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from backend.app.config import Settings
from backend.app.db.database import Database
from backend.app.db.models.market_research import MarketResearch
from backend.app.db.models.stage_snapshot import StageSnapshot
from backend.app.db.models.workspace import Workspace
from backend.app.main import create_application
from backend.app.schemas.strategy_orchestration import StrategyStage
from backend.app.services.stage_history import StageHistoryService

RUN_INTEGRATION_TESTS = os.getenv("GROWTHCREW_RUN_INTEGRATION_TESTS") == "1"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not RUN_INTEGRATION_TESTS,
        reason=("Set GROWTHCREW_RUN_INTEGRATION_TESTS=1 to run PostgreSQL integration tests."),
    ),
]

S = StrategyStage
STAGES = list(StrategyStage)

SUMMARY_FIELDS = {
    "stage",
    "generation",
    "record_id",
    "record_version",
    "model_used",
    "input_tokens",
    "output_tokens",
    "created_at",
    "provenance",
}


class ExplodingGateway:
    """An LLM gateway that fails the test if anything at all is asked of it."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"The history routes must not use the LLM gateway ({name}).")


@dataclass
class Context:
    """An API client and its settings."""

    client: TestClient
    settings: Settings

    def url(self, workspace_id: UUID | str, path: str) -> str:
        return f"/api/v1/workspaces/{workspace_id}/{path}"


def _settings() -> Settings:
    settings = Settings(
        environment="test",
        llm_provider="local",
        embeddings_provider="local",
        tavily_api_key=None,
    )
    if settings.database_url is None:
        pytest.fail("GROWTHCREW_DATABASE_URL is required for integration tests.")
    return settings


@pytest.fixture
def context() -> Iterator[Context]:
    """Create an API client backed by real PostgreSQL and the local stubs."""

    settings = _settings()
    with TestClient(create_application(settings)) as client:
        yield Context(client=client, settings=settings)


def _new_workspace(ctx: Context, name: str) -> UUID:
    response = ctx.client.post("/api/v1/workspaces", json={"name": name})
    assert response.status_code == 201
    workspace_id = UUID(response.json()["id"])
    profile = ctx.client.post(
        ctx.url(workspace_id, "business-profile"),
        json={"business_name": name, "industry": "Healthy Snacks"},
    )
    assert profile.status_code == 201
    return workspace_id


@pytest.fixture
def workspace(context: Context) -> Iterator[UUID]:
    """A workspace with a business profile and no generated stages."""

    workspace_id = _new_workspace(context, "History API Co")
    yield workspace_id
    _delete_workspace(context.settings, workspace_id)


@pytest.fixture
def other_workspace(context: Context) -> Iterator[UUID]:
    """A second, unrelated workspace."""

    workspace_id = _new_workspace(context, "Other History Co")
    yield workspace_id
    _delete_workspace(context.settings, workspace_id)


def _delete_workspace(settings: Settings, workspace_id: UUID) -> None:
    database = Database.from_settings(settings)
    try:
        with database.session() as session:
            row = session.get(Workspace, workspace_id)
            if row is not None:
                session.delete(row)
                session.commit()
    finally:
        database.dispose()


def _seed_generation(ctx: Context, workspace_id: UUID, **changes: Any) -> None:
    """Write one more market research generation the way an agent does."""

    database = Database.from_settings(ctx.settings)
    try:
        with database.session() as session:
            record = session.scalars(
                select(MarketResearch).where(MarketResearch.workspace_id == workspace_id)
            ).one_or_none()
            if record is None:
                record = MarketResearch(
                    workspace_id=workspace_id,
                    market_overview="Overview.",
                    target_customer_segments="Segments.",
                    competitive_landscape="Landscape.",
                    opportunities_and_risks="Risks.",
                    sources=[
                        {
                            "section": "market_overview",
                            "source": "https://example.com",
                            "snippet": "Café guide",
                        }
                    ],
                    model_used="test-model",
                    input_tokens=5,
                    output_tokens=6,
                )
                session.add(record)
            for name, value in changes.items():
                setattr(record, name, value)
            StageHistoryService(session).record(workspace_id, S.MARKET_RESEARCH, record)
            session.commit()
    finally:
        database.dispose()


def _history(ctx: Context, workspace_id: UUID | str, stage: str = "market_research") -> Any:
    return ctx.client.get(ctx.url(workspace_id, f"history/{stage}"))


def _snapshot(ctx: Context, workspace_id: UUID | str, generation: int | str) -> Any:
    return ctx.client.get(ctx.url(workspace_id, f"history/market_research/{generation}"))


# --- reading history -----------------------------------------------------------------


def test_a_stage_never_generated_has_an_empty_history(context: Context, workspace: UUID) -> None:
    """No generations is an empty list, for every stage, not a 404."""

    for stage in STAGES:
        response = _history(context, workspace, stage.value)

        assert response.status_code == 200, stage
        assert response.json() == {
            "workspace_id": str(workspace),
            "stage": stage.value,
            "snapshots": [],
            "latest_generation": None,
        }


def test_past_generations_stay_readable_after_the_stage_is_overwritten(
    context: Context, workspace: UUID
) -> None:
    """The point of history: what the stage was is still there once it has changed."""

    _seed_generation(context, workspace)
    _seed_generation(context, workspace, market_overview="A rewritten overview.")

    first = _snapshot(context, workspace, 1).json()
    second = _snapshot(context, workspace, 2).json()
    live = context.client.get(context.url(workspace, "market-research")).json()

    assert first["content"]["market_overview"] == "Overview."
    assert second["content"]["market_overview"] == "A rewritten overview."
    assert live["market_overview"] == "A rewritten overview."
    assert (first["record_version"], second["record_version"]) == (1, 2)
    assert first["record_id"] == second["record_id"]


def test_listing_is_newest_first_and_carries_no_content(context: Context, workspace: UUID) -> None:
    """The list is a light index of generations, newest at the top."""

    for overview in ("One.", "Two.", "Three."):
        _seed_generation(context, workspace, market_overview=overview)

    body = _history(context, workspace).json()

    assert body["latest_generation"] == 3
    assert [item["generation"] for item in body["snapshots"]] == [3, 2, 1]
    assert [item["record_version"] for item in body["snapshots"]] == [3, 2, 1]
    for item in body["snapshots"]:
        assert set(item) == SUMMARY_FIELDS
        assert item["model_used"] == "test-model"
        assert (item["input_tokens"], item["output_tokens"]) == (5, 6)


def test_unicode_and_nested_content_survive_the_round_trip(
    context: Context, workspace: UUID
) -> None:
    """Content is returned exactly as it was stored."""

    _seed_generation(context, workspace)

    body = _snapshot(context, workspace, 1).json()

    assert body["content"]["sources"] == [
        {"section": "market_overview", "source": "https://example.com", "snippet": "Café guide"}
    ]
    assert set(body["content"]) == {
        "market_overview",
        "target_customer_segments",
        "competitive_landscape",
        "opportunities_and_risks",
        "sources",
    }


def test_provenance_from_a_real_run_is_reported_in_detail_and_summary(
    context: Context, workspace: UUID
) -> None:
    """After a full run, personas report exactly what they were built from."""

    run = context.client.post(
        context.url(workspace, "generate-full-strategy"),
        json={"force_regenerate": False, "auto_approve": True},
    )
    assert run.status_code == 200

    summary = _history(context, workspace, "customer_personas").json()["snapshots"][0]
    detail = context.client.get(context.url(workspace, "history/customer_personas/1")).json()

    assert summary["provenance"] == "tracked"
    assert "inputs" not in summary
    assert detail["provenance"] == "tracked"
    assert detail["inputs"]["profile"]["version"] == 1
    assert set(detail["inputs"]["stages"]) == {"business_understanding", "market_research"}
    assert detail["inputs"]["stages"]["market_research"]["generation"] == 1


def test_a_generation_from_before_history_reports_unknown_provenance(
    context: Context, workspace: UUID
) -> None:
    """A backfilled generation has no recorded inputs and says so."""

    database = Database.from_settings(context.settings)
    try:
        with database.session() as session:
            session.add(
                StageSnapshot(
                    workspace_id=workspace,
                    stage=S.MARKET_RESEARCH.value,
                    generation=1,
                    record_id=uuid4(),
                    record_version=1,
                    content={"market_overview": "From before history."},
                    inputs=None,
                    model_used="legacy-model",
                )
            )
            session.commit()
    finally:
        database.dispose()

    summary = _history(context, workspace).json()["snapshots"][0]
    detail = _snapshot(context, workspace, 1).json()

    assert summary["provenance"] == "unknown"
    assert detail["provenance"] == "unknown"
    assert detail["inputs"] is None


def test_one_workspaces_history_is_invisible_to_another(
    context: Context, workspace: UUID, other_workspace: UUID
) -> None:
    """History is scoped to its workspace, in listing and in detail."""

    _seed_generation(context, workspace)

    assert _history(context, workspace).json()["latest_generation"] == 1
    assert _history(context, other_workspace).json()["snapshots"] == []
    assert _snapshot(context, other_workspace, 1).status_code == 404


# --- bad input ------------------------------------------------------------------------


def test_unknown_workspace_is_404_for_list_and_detail(context: Context) -> None:
    """A workspace that does not exist has no history, and says so."""

    nowhere = uuid4()

    listed = _history(context, nowhere)
    read = _snapshot(context, nowhere, 1)

    assert listed.status_code == read.status_code == 404
    assert listed.json() == {"detail": "Workspace not found."}
    assert read.json() == {"detail": "Workspace not found."}


def test_unknown_generation_is_404(context: Context, workspace: UUID) -> None:
    """A generation that was never recorded is not found."""

    _seed_generation(context, workspace)

    response = _snapshot(context, workspace, 2)

    assert response.status_code == 404
    assert response.json() == {"detail": "That generation of this stage does not exist."}


@pytest.mark.parametrize(
    "generation", ["0", "-1", "abc", "1.5", "2147483648", "99999999999999999999"]
)
def test_invalid_generations_are_a_clean_422_never_a_database_error(
    context: Context, workspace: UUID, generation: str
) -> None:
    """Out-of-range numbers must not reach the database, where they would be a 500."""

    response = _snapshot(context, workspace, generation)

    assert response.status_code == 422


def test_the_largest_possible_generation_is_a_plain_404(context: Context, workspace: UUID) -> None:
    """The upper bound is valid: it simply does not exist."""

    assert _snapshot(context, workspace, 2_147_483_647).status_code == 404


def test_history_cannot_be_written_through_the_api(context: Context, workspace: UUID) -> None:
    """There is no write route, so history cannot be altered or removed over HTTP."""

    _seed_generation(context, workspace)

    for method in ("post", "put", "patch", "delete"):
        response = getattr(context.client, method)(
            context.url(workspace, "history/market_research/1")
        )
        assert response.status_code == 405, method
    assert _snapshot(context, workspace, 1).status_code == 200


# --- independence from the model provider ---------------------------------------------


def test_history_is_readable_with_the_model_provider_down(workspace: UUID) -> None:
    """No LLM call is made to read history, so an outage cannot block it."""

    settings = _settings()
    with TestClient(create_application(settings, llm_gateway=ExplodingGateway())) as client:  # type: ignore[arg-type]
        ctx = Context(client=client, settings=settings)
        _seed_generation(ctx, workspace)

        listed = _history(ctx, workspace)
        read = _snapshot(ctx, workspace, 1)

    assert listed.status_code == 200
    assert read.status_code == 200
    assert read.json()["content"]["market_overview"] == "Overview."
