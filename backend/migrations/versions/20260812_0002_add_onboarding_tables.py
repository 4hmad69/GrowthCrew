"""Add workspace and business profile tables.

Revision ID: 20260812_0002
Revises: 20260801_0001
Create Date: 2026-08-12
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260812_0002"
down_revision: str | None = "20260801_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the first GrowthCrew business-domain tables."""

    op.create_table(
        "workspaces",
        sa.Column(
            "id",
            sa.Uuid(),
            nullable=False,
        ),
        sa.Column(
            "name",
            sa.String(length=160),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.String(length=20),
            server_default=sa.text("'active'"),
            nullable=False,
        ),
        sa.Column(
            "version",
            sa.Integer(),
            server_default=sa.text("1"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status IN ('active', 'archived')",
            name="ck_workspaces_status",
        ),
        sa.PrimaryKeyConstraint(
            "id",
            name="pk_workspaces",
        ),
    )

    op.create_index(
        "ix_workspaces_status_created_at",
        "workspaces",
        ["status", "created_at"],
        unique=False,
    )

    op.create_table(
        "business_profiles",
        sa.Column(
            "id",
            sa.Uuid(),
            nullable=False,
        ),
        sa.Column(
            "workspace_id",
            sa.Uuid(),
            nullable=False,
        ),
        sa.Column(
            "business_name",
            sa.String(length=160),
            nullable=False,
        ),
        sa.Column(
            "website",
            sa.String(length=2048),
            nullable=True,
        ),
        sa.Column(
            "product_or_service",
            sa.Text(),
            nullable=True,
        ),
        sa.Column(
            "industry",
            sa.String(length=120),
            nullable=True,
        ),
        sa.Column(
            "country",
            sa.String(length=100),
            nullable=True,
        ),
        sa.Column(
            "target_market",
            sa.Text(),
            nullable=True,
        ),
        sa.Column(
            "target_customer",
            sa.Text(),
            nullable=True,
        ),
        sa.Column(
            "price_range",
            sa.String(length=200),
            nullable=True,
        ),
        sa.Column(
            "monthly_marketing_budget",
            sa.Numeric(precision=14, scale=2),
            nullable=True,
        ),
        sa.Column(
            "marketing_budget_currency",
            sa.String(length=3),
            nullable=True,
        ),
        sa.Column(
            "main_marketing_goal",
            sa.Text(),
            nullable=True,
        ),
        sa.Column(
            "existing_channels",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "brand_tone",
            sa.String(length=500),
            nullable=True,
        ),
        sa.Column(
            "known_competitors",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "current_challenges",
            sa.Text(),
            nullable=True,
        ),
        sa.Column(
            "additional_instructions",
            sa.Text(),
            nullable=True,
        ),
        sa.Column(
            "version",
            sa.Integer(),
            server_default=sa.text("1"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            (
                "(monthly_marketing_budget IS NULL "
                "AND marketing_budget_currency IS NULL) "
                "OR "
                "(monthly_marketing_budget IS NOT NULL "
                "AND marketing_budget_currency IS NOT NULL)"
            ),
            name="ck_business_profiles_budget_currency_pair",
        ),
        sa.CheckConstraint(
            ("monthly_marketing_budget IS NULL OR monthly_marketing_budget >= 0"),
            name="ck_business_profiles_budget_non_negative",
        ),
        sa.CheckConstraint(
            ("marketing_budget_currency IS NULL OR marketing_budget_currency ~ '^[A-Z]{3}$'"),
            name="ck_business_profiles_currency_format",
        ),
        sa.CheckConstraint(
            "jsonb_typeof(existing_channels) = 'array'",
            name="ck_business_profiles_existing_channels_array",
        ),
        sa.CheckConstraint(
            "jsonb_typeof(known_competitors) = 'array'",
            name="ck_business_profiles_known_competitors_array",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id"],
            ["workspaces.id"],
            name="fk_business_profiles_workspace_id_workspaces",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "id",
            name="pk_business_profiles",
        ),
        sa.UniqueConstraint(
            "workspace_id",
            name="uq_business_profiles_workspace_id",
        ),
    )


def downgrade() -> None:
    """Remove onboarding-domain tables."""

    op.drop_table("business_profiles")
    op.drop_index(
        "ix_workspaces_status_created_at",
        table_name="workspaces",
    )
    op.drop_table("workspaces")
