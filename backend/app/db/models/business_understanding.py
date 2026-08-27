"""Business Understanding agent output ORM model."""

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
    from backend.app.db.models.business_profile import BusinessProfile


class BusinessUnderstanding(Base):
    """AI-generated synthesis of a business profile.

    Produced by the Business Understanding agent: a narrative summary plus
    inferred attributes (stage, competitive category, differentiators,
    likely customer pain points) that were never asked for directly on the
    onboarding form - genuine synthesis, not a restatement of form fields.
    """

    __tablename__ = "business_understandings"
    __table_args__ = (
        UniqueConstraint(
            "profile_id",
            name="uq_business_understandings_profile_id",
        ),
        CheckConstraint(
            "jsonb_typeof(key_differentiators) = 'array'",
            name="ck_business_understandings_key_differentiators_array",
        ),
        CheckConstraint(
            "jsonb_typeof(likely_customer_pain_points) = 'array'",
            name="ck_business_understandings_pain_points_array",
        ),
    )

    id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    profile_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey(
            "business_profiles.id",
            ondelete="CASCADE",
        ),
        nullable=False,
    )

    summary: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    inferred_business_stage: Mapped[str] = mapped_column(
        String(160),
        nullable=False,
    )
    competitive_category: Mapped[str] = mapped_column(
        String(160),
        nullable=False,
    )
    key_differentiators: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        server_default=text("'[]'::jsonb"),
    )
    likely_customer_pain_points: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        server_default=text("'[]'::jsonb"),
    )
    confidence_notes: Mapped[str | None] = mapped_column(
        Text,
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

    business_profile: Mapped[BusinessProfile] = relationship(
        back_populates="business_understanding",
    )

    __mapper_args__: ClassVar[dict[str, Any]] = {
        "version_id_col": version,
    }
