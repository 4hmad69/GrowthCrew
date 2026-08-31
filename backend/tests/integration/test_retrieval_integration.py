"""Real PostgreSQL integration tests for the retrieval service.

Uses the deterministic "local" embeddings provider - these tests prove
the mechanics (persistence, workspace isolation, limit, reindex), not
retrieval quality. The local provider's hash-based vectors have no real
semantic understanding, so which specific chunk ranks first for a given
query is not meaningful here. Real retrieval quality is only provable
against real Ollama Cloud embeddings, covered separately in
test_retrieval_llm_integration.py.
"""

import os
from collections.abc import Iterator
from uuid import UUID

import pytest

from backend.app.config import Settings
from backend.app.db.database import Database
from backend.app.db.models.workspace import Workspace
from backend.app.embeddings.gateway import EmbeddingsGateway
from backend.app.services.retrieval import RetrievalService

RUN_INTEGRATION_TESTS = os.getenv("GROWTHCREW_RUN_INTEGRATION_TESTS") == "1"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not RUN_INTEGRATION_TESTS,
        reason=("Set GROWTHCREW_RUN_INTEGRATION_TESTS=1 to run PostgreSQL integration tests."),
    ),
]


@pytest.fixture
def retrieval_context() -> Iterator[tuple[Database, RetrievalService, UUID]]:
    """Create a RetrievalService backed by real Postgres and a fresh workspace."""

    settings = Settings(environment="test", embeddings_provider="local")

    if settings.database_url is None:
        pytest.fail("GROWTHCREW_DATABASE_URL is required for integration tests.")

    database = Database.from_settings(settings)
    embeddings = EmbeddingsGateway(settings)

    with database.session() as session:
        workspace = Workspace(name="Retrieval Integration Co")
        session.add(workspace)
        session.commit()
        workspace_id = workspace.id

    with database.session() as session:
        service = RetrievalService(session, embeddings)
        yield database, service, workspace_id

    with database.session() as session:
        workspace = session.get(Workspace, workspace_id)
        if workspace is not None:
            session.delete(workspace)
            session.commit()

    database.dispose()


def test_add_chunk_persists_and_is_searchable(
    retrieval_context: tuple[Database, RetrievalService, UUID],
) -> None:
    """A single added chunk should be findable by search."""

    _database, service, workspace_id = retrieval_context

    service.add_chunk(workspace_id, "test:pricing", "Our pricing starts at $29/month.")

    results = service.search(workspace_id, "pricing question", limit=5)

    assert len(results) == 1
    assert results[0].content == "Our pricing starts at $29/month."
    assert results[0].source == "test:pricing"


def test_add_chunks_batch_persists_all(
    retrieval_context: tuple[Database, RetrievalService, UUID],
) -> None:
    """Batch-adding several chunks should persist every one of them."""

    _database, service, workspace_id = retrieval_context

    added = service.add_chunks(
        workspace_id,
        [
            ("test:a", "About pricing."),
            ("test:b", "About competitors."),
            ("test:c", "About customers."),
        ],
    )

    assert len(added) == 3

    results = service.search(workspace_id, "anything", limit=10)
    assert len(results) == 3


def test_search_respects_limit(
    retrieval_context: tuple[Database, RetrievalService, UUID],
) -> None:
    """Search should never return more than the requested limit."""

    _database, service, workspace_id = retrieval_context

    service.add_chunks(
        workspace_id,
        [(f"test:{i}", f"Chunk number {i}") for i in range(5)],
    )

    results = service.search(workspace_id, "anything", limit=2)

    assert len(results) == 2


def test_search_is_scoped_to_workspace(
    retrieval_context: tuple[Database, RetrievalService, UUID],
) -> None:
    """One workspace's chunks should never appear in another's search results."""

    database, service, workspace_id = retrieval_context

    with database.session() as session:
        other_workspace = Workspace(name="Other Retrieval Co")
        session.add(other_workspace)
        session.commit()
        other_workspace_id = other_workspace.id

    try:
        service.add_chunk(workspace_id, "test:mine", "My workspace's content.")

        with database.session() as session:
            other_embeddings = EmbeddingsGateway(
                Settings(environment="test", embeddings_provider="local")
            )
            RetrievalService(session, other_embeddings).add_chunk(
                other_workspace_id,
                "test:theirs",
                "The other workspace's content.",
            )

        results = service.search(workspace_id, "anything", limit=10)

        assert len(results) == 1
        assert results[0].source == "test:mine"
    finally:
        with database.session() as session:
            workspace = session.get(Workspace, other_workspace_id)
            if workspace is not None:
                session.delete(workspace)
                session.commit()


def test_reindex_replaces_existing_chunks(
    retrieval_context: tuple[Database, RetrievalService, UUID],
) -> None:
    """Reindexing should remove old chunks before adding the new ones."""

    _database, service, workspace_id = retrieval_context

    service.add_chunks(
        workspace_id,
        [("test:old-1", "Old content one."), ("test:old-2", "Old content two.")],
    )

    service.reindex(workspace_id, [("test:new", "Brand new content.")])

    results = service.search(workspace_id, "anything", limit=10)

    assert len(results) == 1
    assert results[0].source == "test:new"


def test_add_chunk_persists_metadata(
    retrieval_context: tuple[Database, RetrievalService, UUID],
) -> None:
    """Chunk metadata should round-trip through persistence unchanged."""

    _database, service, workspace_id = retrieval_context

    service.add_chunk(
        workspace_id,
        "test:with-metadata",
        "Some content.",
        metadata={"origin": "business_understanding", "confidence": "high"},
    )

    results = service.search(workspace_id, "anything", limit=1)

    assert results[0].chunk_metadata == {"origin": "business_understanding", "confidence": "high"}
