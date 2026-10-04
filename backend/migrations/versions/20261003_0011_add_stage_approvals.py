"""Add stage approvals table.

Revision ID: 20261003_0011
Revises: 20260924_0010
Create Date: 2026-10-03
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20261003_0011"
down_revision: str | None = "20260924_0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the stage_approvals table."""

    op.create_table(
        "stage_approvals",
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
            "stage",
            sa.String(length=40),
            nullable=False,
        ),
        sa.Column(
            "record_id",
            sa.Uuid(),
            nullable=False,
        ),
        sa.Column(
            "approved_version",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "source",
            sa.String(length=10),
            server_default=sa.text("'human'"),
            nullable=False,
        ),
        sa.Column(
            "approved_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "stage IN ('business_understanding', 'market_research', "
            "'competitor_analysis', 'customer_personas', 'brand_strategy', "
            "'marketing_strategy', 'content_plan')",
            name="ck_stage_approvals_stage",
        ),
        sa.CheckConstraint(
            "source IN ('human', 'auto')",
            name="ck_stage_approvals_source",
        ),
        sa.CheckConstraint(
            "approved_version >= 1",
            name="ck_stage_approvals_approved_version_positive",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id"],
            ["workspaces.id"],
            name="fk_stage_approvals_workspace_id_workspaces",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "id",
            name="pk_stage_approvals",
        ),
        sa.UniqueConstraint(
            "workspace_id",
            "stage",
            name="uq_stage_approvals_workspace_id_stage",
        ),
    )


def downgrade() -> None:
    """Remove the stage_approvals table."""

    op.drop_table("stage_approvals")
