"""Workspace-scoped knowledge chunk ORM model.

The persistent vectorstore backing retrieval - replaces the local FAISS
files the agentic-rag source repo used with a real pgvector table, scoped
per workspace, matching the "business-specific memory" requirement in
GrowthCrew's README.

Unlike BusinessProfile/BusinessUnderstanding, this table has no
version_id_col optimistic concurrency - chunks are write-once/read-many;
a content change is a delete-and-reinsert, not an in-place edit, so
optimistic concurrency isn't a meaningful concept here.

The embedding column's width (768) is a literal, not derived from
Settings.embeddings_dimension - a pgvector column's width is fixed by
the migration that created it, so deriving it "dynamically" from a
Settings default would only be misleading: changing that setting alone
would never actually change the deployed column width. 768 matches
nomic-embed-text, the model both this project and the proven agentic-rag
repo use.
"""

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Index, String, Text, Uuid, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.db.base import Base

EMBEDDING_DIMENSION = 768


class KnowledgeChunk(Base):
    """One embedded piece of text belonging to a single workspace's knowledge base."""

    __tablename__ = "knowledge_chunks"
    __table_args__ = (
        Index(
            "ix_knowledge_chunks_workspace_id",
            "workspace_id",
        ),
        Index(
            "ix_knowledge_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    workspace_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey(
            "workspaces.id",
            ondelete="CASCADE",
        ),
        nullable=False,
    )

    source: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    embedding: Mapped[list[float]] = mapped_column(
        Vector(EMBEDDING_DIMENSION),
        nullable=False,
    )
    chunk_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
