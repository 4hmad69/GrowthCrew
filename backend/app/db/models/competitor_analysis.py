"""Competitor Analysis agent output ORM model."""

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


class CompetitorAnalysis(Base):
    """AI-generated competitor analysis report for a workspace.

    The CRAG graph's second consumer per Step 8's doc: four narrative
    sections, each the output of its own independent CRAG graph run over
    a research question built from the workspace's business profile,
    centered on BusinessProfile.known_competitors rather than the
    broader market questions Market Research already covers. `sources`
    is built directly from each run's own relevant_documents, not asked
    of the model, so citations can never be fabricated independently of
    what was actually retrieved - same discipline as MarketResearch.

    Keyed by workspace_id rather than profile_id, for the same reason as
    MarketResearch: a competitor analysis report is a self-contained
    document about the workspace's competitive world, not a live view
    over the profile it was built from, and should stay readable even if
    that profile is later edited or removed.
    """

    __tablename__ = "competitor_analyses"
    __table_args__ = (
        UniqueConstraint(
            "workspace_id",
            name="uq_competitor_analyses_workspace_id",
        ),
        CheckConstraint(
            "jsonb_typeof(sources) = 'array'",
            name="ck_competitor_analyses_sources_array",
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

    competitor_overview: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    strengths_and_weaknesses: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    pricing_and_positioning: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    differentiation_opportunities: Mapped[str] = mapped_column(
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
        back_populates="competitor_analysis",
    )

    __mapper_args__: ClassVar[dict[str, Any]] = {
        "version_id_col": version,
    }
