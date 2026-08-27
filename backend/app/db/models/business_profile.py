"""Business onboarding profile ORM model."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, ClassVar
from uuid import UUID, uuid4

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
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
    from backend.app.db.models.business_understanding import BusinessUnderstanding
    from backend.app.db.models.workspace import Workspace


class BusinessProfile(Base):
    """User-provided business onboarding information for one workspace."""

    __tablename__ = "business_profiles"
    __table_args__ = (
        UniqueConstraint(
            "workspace_id",
            name="uq_business_profiles_workspace_id",
        ),
        CheckConstraint(
            (
                "(monthly_marketing_budget IS NULL "
                "AND marketing_budget_currency IS NULL) "
                "OR "
                "(monthly_marketing_budget IS NOT NULL "
                "AND marketing_budget_currency IS NOT NULL)"
            ),
            name="ck_business_profiles_budget_currency_pair",
        ),
        CheckConstraint(
            ("monthly_marketing_budget IS NULL OR monthly_marketing_budget >= 0"),
            name="ck_business_profiles_budget_non_negative",
        ),
        CheckConstraint(
            ("marketing_budget_currency IS NULL OR marketing_budget_currency ~ '^[A-Z]{3}$'"),
            name="ck_business_profiles_currency_format",
        ),
        CheckConstraint(
            "jsonb_typeof(existing_channels) = 'array'",
            name="ck_business_profiles_existing_channels_array",
        ),
        CheckConstraint(
            "jsonb_typeof(known_competitors) = 'array'",
            name="ck_business_profiles_known_competitors_array",
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

    business_name: Mapped[str] = mapped_column(
        String(160),
        nullable=False,
    )
    website: Mapped[str | None] = mapped_column(
        String(2048),
    )
    product_or_service: Mapped[str | None] = mapped_column(
        Text,
    )
    industry: Mapped[str | None] = mapped_column(
        String(120),
    )
    country: Mapped[str | None] = mapped_column(
        String(100),
    )
    target_market: Mapped[str | None] = mapped_column(
        Text,
    )
    target_customer: Mapped[str | None] = mapped_column(
        Text,
    )
    price_range: Mapped[str | None] = mapped_column(
        String(200),
    )
    monthly_marketing_budget: Mapped[Decimal | None] = mapped_column(
        Numeric(14, 2),
    )
    marketing_budget_currency: Mapped[str | None] = mapped_column(
        String(3),
    )
    main_marketing_goal: Mapped[str | None] = mapped_column(
        Text,
    )
    existing_channels: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        server_default=text("'[]'::jsonb"),
    )
    brand_tone: Mapped[str | None] = mapped_column(
        String(500),
    )
    known_competitors: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        server_default=text("'[]'::jsonb"),
    )
    current_challenges: Mapped[str | None] = mapped_column(
        Text,
    )
    additional_instructions: Mapped[str | None] = mapped_column(
        Text,
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
        back_populates="business_profile",
    )
    business_understanding: Mapped[BusinessUnderstanding | None] = relationship(
        back_populates="business_profile",
        cascade="all, delete-orphan",
        passive_deletes=True,
        uselist=False,
    )

    __mapper_args__: ClassVar[dict[str, Any]] = {
        "version_id_col": version,
    }
