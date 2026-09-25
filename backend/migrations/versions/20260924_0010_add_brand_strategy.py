"""Add brand strategies table.

Revision ID: 20260924_0010
Revises: 20260919_0009
Create Date: 2026-09-24
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260924_0010"
down_revision: str | None = "20260919_0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the brand_strategies table."""

    op.create_table(
        "brand_strategies",
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
            "overview",
            sa.Text(),
            nullable=False,
        ),
        sa.Column(
            "positioning_statement",
            sa.Text(),
            nullable=False,
        ),
        sa.Column(
            "value_proposition",
            sa.Text(),
            nullable=False,
        ),
        sa.Column(
            "brand_voice_and_tone",
            sa.Text(),
            nullable=False,
        ),
        sa.Column(
            "brand_pillars",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "tagline_options",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
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
            "jsonb_typeof(brand_pillars) = 'array'",
            name="ck_brand_strategies_brand_pillars_array",
        ),
        sa.CheckConstraint(
            "jsonb_typeof(tagline_options) = 'array'",
            name="ck_brand_strategies_tagline_options_array",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id"],
            ["workspaces.id"],
            name="fk_brand_strategies_workspace_id_workspaces",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "id",
            name="pk_brand_strategies",
        ),
        sa.UniqueConstraint(
            "workspace_id",
            name="uq_brand_strategies_workspace_id",
        ),
    )


def downgrade() -> None:
    """Remove the brand_strategies table."""

    op.drop_table("brand_strategies")
