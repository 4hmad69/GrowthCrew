"""Add market research table.

Revision ID: 20260907_0005
Revises: 20260830_0004
Create Date: 2026-09-07
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260907_0005"
down_revision: str | None = "20260830_0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the market_researches table."""

    op.create_table(
        "market_researches",
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
            "market_overview",
            sa.Text(),
            nullable=False,
        ),
        sa.Column(
            "target_customer_segments",
            sa.Text(),
            nullable=False,
        ),
        sa.Column(
            "competitive_landscape",
            sa.Text(),
            nullable=False,
        ),
        sa.Column(
            "opportunities_and_risks",
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
            name="ck_market_researches_sources_array",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id"],
            ["workspaces.id"],
            name="fk_market_researches_workspace_id_workspaces",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "id",
            name="pk_market_researches",
        ),
        sa.UniqueConstraint(
            "workspace_id",
            name="uq_market_researches_workspace_id",
        ),
    )


def downgrade() -> None:
    """Remove the market_researches table."""

    op.drop_table("market_researches")
