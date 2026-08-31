"""Persistence and similarity-search operations for knowledge chunks."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.db.models.knowledge_chunk import KnowledgeChunk


class KnowledgeChunkRepository:
    """Perform knowledge-chunk persistence and retrieval operations."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, chunk: KnowledgeChunk) -> None:
        """Stage a new knowledge chunk."""

        self._session.add(chunk)

    def add_many(self, chunks: list[KnowledgeChunk]) -> None:
        """Stage several new knowledge chunks at once."""

        self._session.add_all(chunks)

    def search(
        self,
        workspace_id: UUID,
        query_embedding: list[float],
        *,
        limit: int = 5,
    ) -> list[KnowledgeChunk]:
        """Return the chunks most similar to `query_embedding` for one workspace.

        Ordered by cosine distance (pgvector's <=> operator) - closest
        first. Filtering by workspace_id happens before the similarity
        ordering, so one workspace's data never leaks into another's
        search results.
        """

        statement = (
            select(KnowledgeChunk)
            .where(KnowledgeChunk.workspace_id == workspace_id)
            .order_by(KnowledgeChunk.embedding.cosine_distance(query_embedding))
            .limit(limit)
        )
        return list(self._session.scalars(statement))

    def delete_by_workspace(self, workspace_id: UUID) -> None:
        """Delete every chunk belonging to a workspace (e.g. before re-ingesting)."""

        statement = select(KnowledgeChunk).where(KnowledgeChunk.workspace_id == workspace_id)
        for chunk in self._session.scalars(statement):
            self._session.delete(chunk)
