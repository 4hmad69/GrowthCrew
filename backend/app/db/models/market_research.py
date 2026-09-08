"""Market Research agent output ORM model."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, ClassVar
from uuid import UUID, uuid4

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.base import Base

if TYPE_CHECKING:
    from backend.app.db.models.workspace import Workspace


class MarketResearch(Base):
    """AI-generated market research report for a workspace.

    Produced by the Market Research agent: four narrative sections, each
    the output of its own independent CRAG graph run over a research
    question built from the workspace's business profile - genuinely
    retrieval-grounded (internal knowledge base and/or live web), not a
    single unsourced LLM guess. `sources` is built directly from each
    run's own relevant_documents, not asked of the model, so citations
    can never be fabricated independently of what was actually retrieved.

    Keyed by workspace_id rather than profile_id (unlike
    BusinessUnderstanding) because retrieval itself is workspace-scoped -
    a market research report is fundamentally "what does this workspace's
    world look like", not a property of the profile record.
    """

    __tablename__ = "market_researches"
    __table_args__ = (
        UniqueConstraint(
            "workspace_id",
            name="uq_market_researches_workspace_id",
        ),
        CheckConstraint(
            "jsonb_typeof(sources) = 'array'",
            name="ck_market_researches_sources_array",
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

    market_overview: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    target_customer_segments: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    competitive_landscape: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    opportunities_and_risks: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    sources: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        server_default=text("'[]'::jsonb"),
    )

    model_used: Mapped[str] = mapped_column(
        String(120),
        nullable=False,
    )
    input_tokens: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
    )
    output_tokens: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
    )

    version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
        server_default=text("1"),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    workspace: Mapped[Workspace] = relationship(
        back_populates="market_research",
    )

    __mapper_args__: ClassVar[dict[str, Any]] = {
        "version_id_col": version,
    }
