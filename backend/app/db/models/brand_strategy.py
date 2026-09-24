"""Brand Strategy agent output ORM model."""

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


class BrandStrategy(Base):
    """AI-generated brand positioning, voice, and messaging strategy.

    Like Customer Personas and Content Planning, this agent runs no CRAG
    graph: it needs no external grounding, only synthesis over reports
    that already exist (Business Understanding's summary and
    differentiators, Competitor Analysis's four sections, and Customer
    Personas' overview and persona set), plus the business profile's own
    brand_tone field. One direct structured LLM call via
    LLMGateway.structured_with_usage(). See the Brand Strategy service
    for the generation flow.

    Keyed by workspace_id, same reasoning as every prior synthesizing
    agent: no foreign key to any of the reports it was generated from,
    so a generated strategy stays readable even if those reports are
    later regenerated or removed.

    brand_pillars and tagline_options are JSONB string arrays rather
    than a child table: both are short, always read and regenerated
    together with the rest of the strategy, and downstream agents (Step
    15's extended Marketing Strategy) consume the whole record as one
    unit, the same reasoning ContentPlan.entries and
    CustomerPersonaSet.personas already established.
    """

    __tablename__ = "brand_strategies"
    __table_args__ = (
        UniqueConstraint(
            "workspace_id",
            name="uq_brand_strategies_workspace_id",
        ),
        CheckConstraint(
            "jsonb_typeof(brand_pillars) = 'array'",
            name="ck_brand_strategies_brand_pillars_array",
        ),
        CheckConstraint(
            "jsonb_typeof(tagline_options) = 'array'",
            name="ck_brand_strategies_tagline_options_array",
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

    overview: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    positioning_statement: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    value_proposition: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    brand_voice_and_tone: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    brand_pillars: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        server_default=text("'[]'::jsonb"),
    )
    tagline_options: Mapped[list[str]] = mapped_column(
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
        back_populates="brand_strategy",
    )

    __mapper_args__: ClassVar[dict[str, Any]] = {
        "version_id_col": version,
    }
