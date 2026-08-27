"""Add business understanding table.

Revision ID: 20260826_0003
Revises: 20260812_0002
Create Date: 2026-08-26
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260826_0003"
down_revision: str | None = "20260812_0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the business_understandings table."""

    op.create_table(
        "business_understandings",
        sa.Column(
            "id",
            sa.Uuid(),
            nullable=False,
        ),
        sa.Column(
            "profile_id",
            sa.Uuid(),
            nullable=False,
        ),
        sa.Column(
            "summary",
            sa.Text(),
            nullable=False,
        ),
        sa.Column(
            "inferred_business_stage",
            sa.String(length=160),
            nullable=False,
        ),
        sa.Column(
            "competitive_category",
            sa.String(length=160),
            nullable=False,
        ),
        sa.Column(
            "key_differentiators",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "likely_customer_pain_points",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "confidence_notes",
            sa.Text(),
            nullable=True,
        ),
        sa.Column(
            "model_used",
            sa.String(length=120),
            nullable=False,
        ),
        sa.Column(
            "input_tokens",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column(
            "output_tokens",
            sa.Integer(),
            server_default=sa.text("0"),
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
            "jsonb_typeof(key_differentiators) = 'array'",
            name="ck_business_understandings_key_differentiators_array",
        ),
        sa.CheckConstraint(
            "jsonb_typeof(likely_customer_pain_points) = 'array'",
            name="ck_business_understandings_pain_points_array",
        ),
        sa.ForeignKeyConstraint(
            ["profile_id"],
            ["business_profiles.id"],
            name="fk_business_understandings_profile_id_business_profiles",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "id",
            name="pk_business_understandings",
        ),
        sa.UniqueConstraint(
            "profile_id",
            name="uq_business_understandings_profile_id",
        ),
    )


def downgrade() -> None:
    """Remove the business_understandings table."""

    op.drop_table("business_understandings")
