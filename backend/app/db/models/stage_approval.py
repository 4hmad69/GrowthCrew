"""Stage approval ORM model."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
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
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.base import Base

if TYPE_CHECKING:
    from backend.app.db.models.workspace import Workspace

# The seven generated stages of the strategy chain, in chain order. These
# are the literal values the stage CHECK constraint allows; a unit test
# pins them to StrategyStage so the two cannot drift apart silently (the
# db layer deliberately does not import from the schemas layer).
STAGE_APPROVAL_STAGES: tuple[str, ...] = (
    "business_understanding",
    "market_research",
    "competitor_analysis",
    "customer_personas",
    "brand_strategy",
    "marketing_strategy",
    "content_plan",
)

# Who recorded an approval: a person reviewing the artifact, or a
# full-strategy run that was explicitly asked to approve what it generated.
STAGE_APPROVAL_SOURCES: tuple[str, ...] = ("human", "auto")


def _in_list(column: str, values: tuple[str, ...]) -> str:
    """Build a ``column IN ('a', 'b')`` CHECK expression from literals."""

    quoted = ", ".join(f"'{value}'" for value in values)
    return f"{column} IN ({quoted})"


class StageApproval(Base):
    """A recorded approval of one generated stage of the strategy chain.

    An approval is pinned to the exact stage record that was reviewed:
    record_id is that record's id and approved_version is its version at
    approval time. A stage counts as approved only while the current
    record still has both values, so regenerating (or deleting and
    recreating) a stage silently retires its approval - no agent service
    needs reset logic, and approving never touches the artifact's own
    version or updated_at.

    record_id is deliberately not a foreign key: it points into one of
    seven different tables depending on stage, and an approval must be
    able to outlive (and be recognized as stale against) a record that
    was deleted and recreated. Rows are keyed by (workspace_id, stage) so
    there is at most one approval per stage; approving again replaces it.

    No approver identity is stored: the application has no users yet.
    """

    __tablename__ = "stage_approvals"
    __table_args__ = (
        UniqueConstraint(
            "workspace_id",
            "stage",
            name="uq_stage_approvals_workspace_id_stage",
        ),
        CheckConstraint(
            _in_list("stage", STAGE_APPROVAL_STAGES),
            name="ck_stage_approvals_stage",
        ),
        CheckConstraint(
            _in_list("source", STAGE_APPROVAL_SOURCES),
            name="ck_stage_approvals_source",
        ),
        CheckConstraint(
            "approved_version >= 1",
            name="ck_stage_approvals_approved_version_positive",
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
    record_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        nullable=False,
    )
    approved_version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    source: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
        default="human",
        server_default=text("'human'"),
    )
    approved_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    workspace: Mapped[Workspace] = relationship(
        back_populates="stage_approvals",
    )
