"""Add customer persona sets table.

Revision ID: 20260919_0009
Revises: 20260916_0008
Create Date: 2026-09-19
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260919_0009"
down_revision: str | None = "20260916_0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the customer_persona_sets table."""

    op.create_table(
        "customer_persona_sets",
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
            "personas",
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
            "jsonb_typeof(personas) = 'array'",
            name="ck_customer_persona_sets_personas_array",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id"],
            ["workspaces.id"],
            name="fk_customer_persona_sets_workspace_id_workspaces",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "id",
            name="pk_customer_persona_sets",
        ),
        sa.UniqueConstraint(
            "workspace_id",
            name="uq_customer_persona_sets_workspace_id",
        ),
    )


def downgrade() -> None:
    """Remove the customer_persona_sets table."""

    op.drop_table("customer_persona_sets")
