"""Marketing Strategy agent output ORM model."""

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


class MarketingStrategy(Base):
    """AI-generated marketing strategy for a workspace.

    The first agent that does not build its research questions from the
    business profile alone: each of its four sections is a CRAG graph run
    seeded with the *outputs* of Business Understanding, Market Research,
    and Competitor Analysis, layered on top of the profile's own planning
    fields (main_marketing_goal, existing_channels, monthly marketing
    budget, current_challenges). generate() hard-requires all three prior
    reports to already exist for the workspace - see
    MarketingStrategyService for why a strategy built on a partially
    missing research base was judged worse than asking the user to finish
    the earlier steps first.

    Keyed by workspace_id, same reasoning as MarketResearch and
    CompetitorAnalysis: a generated strategy is a self-contained document,
    not a live view over the reports it originally synthesized, and
    should stay readable even if those reports are later regenerated.
    """

    __tablename__ = "marketing_strategies"
    __table_args__ = (
        UniqueConstraint(
            "workspace_id",
            name="uq_marketing_strategies_workspace_id",
        ),
        CheckConstraint(
            "jsonb_typeof(sources) = 'array'",
            name="ck_marketing_strategies_sources_array",
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

    recommended_channels_and_tactics: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    content_and_messaging_pillars: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    ninety_day_roadmap: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    budget_allocation_and_kpis: Mapped[str] = mapped_column(
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
        back_populates="marketing_strategy",
    )

    __mapper_args__: ClassVar[dict[str, Any]] = {
        "version_id_col": version,
    }
