"""Workspace ORM model."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, ClassVar
from uuid import UUID, uuid4

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Index,
    Integer,
    String,
    Uuid,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.base import Base

if TYPE_CHECKING:
    from backend.app.db.models.business_profile import BusinessProfile
    from backend.app.db.models.competitor_analysis import CompetitorAnalysis
    from backend.app.db.models.content_plan import ContentPlan
    from backend.app.db.models.market_research import MarketResearch
    from backend.app.db.models.marketing_strategy import MarketingStrategy


class Workspace(Base):
    """Top-level business workspace owned by a future GrowthCrew user."""

    __tablename__ = "workspaces"
    __table_args__ = (
        CheckConstraint(
            "status IN ('active', 'archived')",
            name="ck_workspaces_status",
        ),
        Index(
            "ix_workspaces_status_created_at",
            "status",
            "created_at",
        ),
    )

    id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    name: Mapped[str] = mapped_column(
        String(160),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="active",
        server_default=text("'active'"),
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
    business_profile: Mapped[BusinessProfile | None] = relationship(
        back_populates="workspace",
        cascade="all, delete-orphan",
        passive_deletes=True,
        uselist=False,
    )
    market_research: Mapped[MarketResearch | None] = relationship(
        back_populates="workspace",
        cascade="all, delete-orphan",
        passive_deletes=True,
        uselist=False,
    )
    competitor_analysis: Mapped[CompetitorAnalysis | None] = relationship(
        back_populates="workspace",
        cascade="all, delete-orphan",
        passive_deletes=True,
        uselist=False,
    )
    marketing_strategy: Mapped[MarketingStrategy | None] = relationship(
        back_populates="workspace",
        cascade="all, delete-orphan",
        passive_deletes=True,
        uselist=False,
    )
    content_plan: Mapped[ContentPlan | None] = relationship(
        back_populates="workspace",
        cascade="all, delete-orphan",
        passive_deletes=True,
        uselist=False,
    )
    __mapper_args__: ClassVar[dict[str, Any]] = {
        "version_id_col": version,
    }
