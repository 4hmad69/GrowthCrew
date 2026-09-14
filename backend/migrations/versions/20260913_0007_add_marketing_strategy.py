"""Add marketing strategy table.

Revision ID: 20260913_0007
Revises: 20260911_0006
Create Date: 2026-09-13
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260913_0007"
down_revision: str | None = "20260911_0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the marketing_strategies table."""

    op.create_table(
        "marketing_strategies",
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
            "recommended_channels_and_tactics",
            sa.Text(),
            nullable=False,
        ),
        sa.Column(
            "content_and_messaging_pillars",
            sa.Text(),
            nullable=False,
        ),
        sa.Column(
            "ninety_day_roadmap",
            sa.Text(),
            nullable=False,
        ),
        sa.Column(
            "budget_allocation_and_kpis",
            sa.Text(),
            nullable=False,
        ),
        sa.Column(
            "sources",
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
            "jsonb_typeof(sources) = 'array'",
            name="ck_marketing_strategies_sources_array",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id"],
            ["workspaces.id"],
            name="fk_marketing_strategies_workspace_id_workspaces",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "id",
            name="pk_marketing_strategies",
        ),
        sa.UniqueConstraint(
            "workspace_id",
            name="uq_marketing_strategies_workspace_id",
        ),
    )


def downgrade() -> None:
    """Remove the marketing_strategies table."""

    op.drop_table("marketing_strategies")
