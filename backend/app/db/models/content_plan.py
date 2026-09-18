"""Content Planning agent output ORM model."""

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


class ContentPlan(Base):
    """AI-generated 30-day content calendar for a workspace.

    Unlike Market Research, Competitor Analysis, and Marketing Strategy,
    this agent does not run the CRAG graph: it doesn't need external
    grounding, only synthesis over decisions Marketing Strategy already
    made (recommended channels, content and messaging pillars) plus the
    business profile's planning fields. Same shape as
    BusinessUnderstandingService - one direct structured LLM call via
    LLMGateway.structured_with_usage(), no retrieval or web search
    involved. See ContentPlanService for the generation flow.

    Keyed by workspace_id, same reasoning as MarketingStrategy: no
    foreign key to MarketingStrategy itself, so a generated calendar
    stays readable even if the strategy it was built from is later
    regenerated or removed.
    """

    __tablename__ = "content_plans"
    __table_args__ = (
        UniqueConstraint(
            "workspace_id",
            name="uq_content_plans_workspace_id",
        ),
        CheckConstraint(
            "jsonb_typeof(entries) = 'array'",
            name="ck_content_plans_entries_array",
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
    entries: Mapped[list[dict[str, Any]]] = mapped_column(
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
        back_populates="content_plan",
    )

    __mapper_args__: ClassVar[dict[str, Any]] = {
        "version_id_col": version,
    }
