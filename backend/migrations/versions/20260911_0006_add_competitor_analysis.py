"""Add competitor analysis table.

Revision ID: 20260911_0006
Revises: 20260907_0005
Create Date: 2026-09-11
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260911_0006"
down_revision: str | None = "20260907_0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the competitor_analyses table."""

    op.create_table(
        "competitor_analyses",
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
            "competitor_overview",
            sa.Text(),
            nullable=False,
        ),
        sa.Column(
            "strengths_and_weaknesses",
            sa.Text(),
            nullable=False,
        ),
        sa.Column(
            "pricing_and_positioning",
            sa.Text(),
            nullable=False,
        ),
        sa.Column(
            "differentiation_opportunities",
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
            name="ck_competitor_analyses_sources_array",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id"],
            ["workspaces.id"],
            name="fk_competitor_analyses_workspace_id_workspaces",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "id",
            name="pk_competitor_analyses",
        ),
        sa.UniqueConstraint(
            "workspace_id",
            name="uq_competitor_analyses_workspace_id",
        ),
    )


def downgrade() -> None:
    """Remove the competitor_analyses table."""

    op.drop_table("competitor_analyses")
