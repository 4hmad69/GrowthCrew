"""Workspace-scoped retrieval: ingest text into the knowledge base, search it.

The service layer for the "business-specific memory" README requirement -
everything a future agent needs to add knowledge to a workspace and pull
back what's relevant, without touching embeddings or pgvector directly.

search() deliberately has no similarity-score threshold - it returns the
top-k nearest chunks and nothing more. Filtering by actual relevance is
the job of an LLM-based grading step (the "grade_documents" node the
CRAG graph will bring in a later step), matching how the source
agentic-rag repo works: a raw cosine-distance cutoff is a blunt
instrument next to asking a model whether a retrieved chunk actually
helps answer the question.
"""

from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

from backend.app.db.models.knowledge_chunk import KnowledgeChunk
from backend.app.db.repositories.knowledge_chunks import KnowledgeChunkRepository
from backend.app.embeddings.gateway import EmbeddingsGateway


class RetrievalService:
    """Add text to a workspace's knowledge base, and search it by similarity."""

    def __init__(self, session: Session, embeddings: EmbeddingsGateway) -> None:
        self._session = session
        self._embeddings = embeddings
        self._chunks = KnowledgeChunkRepository(session)

    def add_chunk(
        self,
        workspace_id: UUID,
        source: str,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> KnowledgeChunk:
        """Embed and persist one piece of text for a workspace."""

        [embedding] = self._embeddings.embed_documents([content])
        chunk = KnowledgeChunk(
            workspace_id=workspace_id,
            source=source,
            content=content,
            embedding=embedding,
            chunk_metadata=metadata or {},
        )
        self._chunks.add(chunk)
        self._session.commit()
        return chunk

    def add_chunks(
        self,
        workspace_id: UUID,
        items: list[tuple[str, str]],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> list[KnowledgeChunk]:
        """Embed and persist several (source, content) pairs in one batch call."""

        if not items:
            return []

        contents = [content for _source, content in items]
        embeddings = self._embeddings.embed_documents(contents)

        chunks = [
            KnowledgeChunk(
                workspace_id=workspace_id,
                source=source,
                content=content,
                embedding=embedding,
                chunk_metadata=metadata or {},
            )
            for (source, content), embedding in zip(items, embeddings, strict=True)
        ]
        self._chunks.add_many(chunks)
        self._session.commit()
        return chunks

    def search(
        self,
        workspace_id: UUID,
        query: str,
        *,
        limit: int = 5,
    ) -> list[KnowledgeChunk]:
        """Return the chunks most relevant to `query` for one workspace."""

        query_embedding = self._embeddings.embed_query(query)
        return self._chunks.search(workspace_id, query_embedding, limit=limit)

    def reindex(
        self,
        workspace_id: UUID,
        items: list[tuple[str, str]],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> list[KnowledgeChunk]:
        """Replace a workspace's entire knowledge base with new content."""

        self._chunks.delete_by_workspace(workspace_id)
        self._session.flush()
        return self.add_chunks(workspace_id, items, metadata=metadata)
