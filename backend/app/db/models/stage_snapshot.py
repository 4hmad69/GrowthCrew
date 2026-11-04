"""Stage snapshot ORM model: the append-only history of generated stages."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    Uuid,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.base import Base
from backend.app.db.models.stage_approval import STAGE_APPROVAL_STAGES

if TYPE_CHECKING:
    from backend.app.db.models.workspace import Workspace

# The seven generated stages of the strategy chain, in chain order: the same
# literals the stage_approvals CHECK uses. A unit test pins them to
# StrategyStage so the db layer and the schemas layer cannot drift apart.
STAGE_SNAPSHOT_STAGES: tuple[str, ...] = STAGE_APPROVAL_STAGES


def _stage_check_sql() -> str:
    """Build the ``stage IN (...)`` CHECK expression from the stage literals."""

    quoted = ", ".join(f"'{stage}'" for stage in STAGE_SNAPSHOT_STAGES)
    return f"stage IN ({quoted})"


class StageSnapshot(Base):
    """One generation of one stage of the strategy chain, kept forever.

    Every stage record is overwritten in place when it is regenerated, so
    on its own it remembers nothing. A snapshot is the memory: a full
    copy of what a generation produced, written in the same transaction
    as the generation itself and never changed afterwards. Rows are
    append-only; a database trigger rejects any UPDATE, so history cannot
    be rewritten by a bug or by hand.

    generation is this table's own counter, per (workspace, stage), and
    is deliberately independent of the stage record's version. The
    record's version only moves when its content changes, so a forced
    regeneration that produces identical content leaves it alone - the
    snapshot sequence is what proves a generation happened.

    record_id and record_version identify the stage record as it stood
    right after the generation, which is exactly what a stage approval
    pins. Two snapshots with the same pair therefore hold identical
    content, because any content change bumps the record's version.
    record_id is not a foreign key for the same reason it is not one in
    stage_approvals: it points into one of seven tables, and history must
    outlive a record that was deleted and recreated.

    content is the stage's generated fields as a JSON object (the field
    list per stage lives in the schemas layer). inputs records which
    upstream generations and which business profile version the stage
    was built from; NULL means provenance is unknown, which is what the
    migration records for stages that existed before history did.
    """

    __tablename__ = "stage_snapshots"
    __table_args__ = (
        UniqueConstraint(
            "workspace_id",
            "stage",
            "generation",
            name="uq_stage_snapshots_workspace_id_stage_generation",
        ),
        CheckConstraint(
            _stage_check_sql(),
            name="ck_stage_snapshots_stage",
        ),
        CheckConstraint(
            "generation >= 1",
            name="ck_stage_snapshots_generation_positive",
        ),
        CheckConstraint(
            "record_version >= 1",
            name="ck_stage_snapshots_record_version_positive",
        ),
        CheckConstraint(
            "jsonb_typeof(content) = 'object'",
            name="ck_stage_snapshots_content_object",
        ),
        CheckConstraint(
            "inputs IS NULL OR jsonb_typeof(inputs) = 'object'",
            name="ck_stage_snapshots_inputs_object",
        ),
    )

    id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    workspace_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey(
            "workspaces.id",
            ondelete="CASCADE",
        ),
        nullable=False,
    )

    stage: Mapped[str] = mapped_column(
        String(40),
        nullable=False,
    )
    generation: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    record_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        nullable=False,
    )
    record_version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    content: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
    )
    # none_as_null: without it SQLAlchemy stores Python None as the JSON
    # value null, not SQL NULL - which is not "provenance unknown" and
    # would fail the object-or-NULL CHECK.
    inputs: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB(none_as_null=True),
        nullable=True,
    )

    model_used: Mapped[str] = mapped_column(
        String(120),
        nullable=False,
    )
    input_tokens: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
    )
    output_tokens: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    workspace: Mapped[Workspace] = relationship(
        back_populates="stage_snapshots",
    )
