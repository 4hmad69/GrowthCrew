"""Enable the pgvector PostgreSQL extension.

Revision ID: 20260801_0001
Revises: None
Create Date: 2026-08-01 17:12:00
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260801_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Enable pgvector in the GrowthCrew database."""

    op.execute("CREATE EXTENSION IF NOT EXISTS vector")


def downgrade() -> None:
    """Remove pgvector when reverting the complete schema to base."""

    op.execute("DROP EXTENSION IF EXISTS vector")