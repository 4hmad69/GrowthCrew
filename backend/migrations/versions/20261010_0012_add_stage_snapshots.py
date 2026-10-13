"""Add stage snapshots table.

Revision ID: 20261010_0012
Revises: 20261003_0011
Create Date: 2026-10-10
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20261010_0012"
down_revision: str | None = "20261003_0011"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# (stage, source table, workspace id expression, join clause, content fields).
# Self-contained on purpose: a migration must keep working even after the
# application's models change, so nothing here is imported from the app.
_BACKFILL: tuple[tuple[str, str, str, str, tuple[str, ...]], ...] = (
    (
        "business_understanding",
        "business_understandings",
        "bp.workspace_id",
        "JOIN business_profiles bp ON bp.id = t.profile_id",
        (
            "summary",
            "inferred_business_stage",
            "competitive_category",
            "key_differentiators",
            "likely_customer_pain_points",
            "confidence_notes",
        ),
    ),
    (
        "market_research",
        "market_researches",
        "t.workspace_id",
        "",
        (
            "market_overview",
            "target_customer_segments",
            "competitive_landscape",
            "opportunities_and_risks",
            "sources",
        ),
    ),
    (
        "competitor_analysis",
        "competitor_analyses",
        "t.workspace_id",
        "",
        (
            "competitor_overview",
            "strengths_and_weaknesses",
            "pricing_and_positioning",
            "differentiation_opportunities",
            "sources",
        ),
    ),
    (
        "customer_personas",
        "customer_persona_sets",
        "t.workspace_id",
        "",
        ("overview", "personas"),
    ),
    (
        "brand_strategy",
        "brand_strategies",
        "t.workspace_id",
        "",
        (
            "overview",
            "positioning_statement",
            "value_proposition",
            "brand_voice_and_tone",
            "brand_pillars",
            "tagline_options",
        ),
    ),
    (
        "marketing_strategy",
        "marketing_strategies",
        "t.workspace_id",
        "",
        (
            "recommended_channels_and_tactics",
            "content_and_messaging_pillars",
            "ninety_day_roadmap",
            "budget_allocation_and_kpis",
            "sources",
        ),
    ),
    (
        "content_plan",
        "content_plans",
        "t.workspace_id",
        "",
        ("overview", "entries"),
    ),
)


def _backfill_statement(
    stage: str,
    table: str,
    workspace_expression: str,
    join: str,
    fields: tuple[str, ...],
) -> sa.TextClause:
    """Build the INSERT ... SELECT that gives one existing stage generation 1.

    Every interpolated value is a literal from _BACKFILL above, never input.
    """

    content = ", ".join(f"'{field}', t.{field}" for field in fields)
    return sa.text(
        f"""
        INSERT INTO stage_snapshots (
            id, workspace_id, stage, generation, record_id, record_version,
            content, inputs, model_used, input_tokens, output_tokens, created_at
        )
        SELECT
            gen_random_uuid(), {workspace_expression}, '{stage}', 1, t.id, t.version,
            jsonb_build_object({content}), NULL,
            t.model_used, t.input_tokens, t.output_tokens, t.updated_at
        FROM {table} t
        {join}
        """
    )


def upgrade() -> None:
    """Create stage_snapshots, make it append-only, and backfill generation 1."""

    op.create_table(
        "stage_snapshots",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("stage", sa.String(length=40), nullable=False),
        sa.Column("generation", sa.Integer(), nullable=False),
        sa.Column("record_id", sa.Uuid(), nullable=False),
        sa.Column("record_version", sa.Integer(), nullable=False),
        sa.Column("content", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("inputs", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("model_used", sa.String(length=120), nullable=False),
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
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "stage IN ('business_understanding', 'market_research', "
            "'competitor_analysis', 'customer_personas', 'brand_strategy', "
            "'marketing_strategy', 'content_plan')",
            name="ck_stage_snapshots_stage",
        ),
        sa.CheckConstraint(
            "generation >= 1",
            name="ck_stage_snapshots_generation_positive",
        ),
        sa.CheckConstraint(
            "record_version >= 1",
            name="ck_stage_snapshots_record_version_positive",
        ),
        sa.CheckConstraint(
            "jsonb_typeof(content) = 'object'",
            name="ck_stage_snapshots_content_object",
        ),
        sa.CheckConstraint(
            "inputs IS NULL OR jsonb_typeof(inputs) = 'object'",
            name="ck_stage_snapshots_inputs_object",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id"],
            ["workspaces.id"],
            name="fk_stage_snapshots_workspace_id_workspaces",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_stage_snapshots"),
        sa.UniqueConstraint(
            "workspace_id",
            "stage",
            "generation",
            name="uq_stage_snapshots_workspace_id_stage_generation",
        ),
    )

    # History must not be rewritable. Deleting stays allowed so rows can
    # still cascade away with their workspace.
    op.execute(
        """
        CREATE FUNCTION stage_snapshots_reject_update() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
            RAISE EXCEPTION 'stage_snapshots rows are append-only'
                USING ERRCODE = 'restrict_violation';
        END;
        $$
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_stage_snapshots_reject_update
        BEFORE UPDATE ON stage_snapshots
        FOR EACH ROW EXECUTE FUNCTION stage_snapshots_reject_update()
        """
    )

    # Stages generated before history existed get a generation-1 snapshot
    # of their current content. Their inputs are unknown (NULL): nothing
    # recorded which upstream versions they were built from.
    for stage, table, workspace_expression, join, fields in _BACKFILL:
        op.execute(_backfill_statement(stage, table, workspace_expression, join, fields))


def downgrade() -> None:
    """Remove stage history (the stage records themselves are untouched)."""

    op.execute("DROP TRIGGER trg_stage_snapshots_reject_update ON stage_snapshots")
    op.execute("DROP FUNCTION stage_snapshots_reject_update()")
    op.drop_table("stage_snapshots")
