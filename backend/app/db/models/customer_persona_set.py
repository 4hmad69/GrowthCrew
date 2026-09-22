"""Customer Personas agent output ORM model."""

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


class CustomerPersonaSet(Base):
    """AI-generated set of structured customer personas for a workspace.

    Like Content Planning, this agent runs no CRAG graph: it needs no
    external grounding, only synthesis over reports that already exist
    (Business Understanding's inferred pain points and Market Research's
    target customer segments, plus the business profile's own target
    fields). One direct structured LLM call via
    LLMGateway.structured_with_usage(). See the Customer Personas
    service for the generation flow.

    Keyed by workspace_id, same reasoning as MarketingStrategy and
    ContentPlan: no foreign key to any of the reports it was generated
    from, so a generated persona set stays readable even if those
    reports are later regenerated or removed.

    personas is a JSONB array of structured persona objects rather than
    a child table: personas are always read and regenerated together as
    one unit, and downstream agents consume the whole set.
    """

    __tablename__ = "customer_persona_sets"
    __table_args__ = (
        UniqueConstraint(
            "workspace_id",
            name="uq_customer_persona_sets_workspace_id",
        ),
        CheckConstraint(
            "jsonb_typeof(personas) = 'array'",
            name="ck_customer_persona_sets_personas_array",
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
    personas: Mapped[list[dict[str, Any]]] = mapped_column(
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
        back_populates="customer_persona_set",
    )

    __mapper_args__: ClassVar[dict[str, Any]] = {
        "version_id_col": version,
    }
