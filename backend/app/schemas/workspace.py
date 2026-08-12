"""Workspace request and response schemas."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import Field, model_validator

from backend.app.schemas.common import OrmSchema, StrictSchema


class WorkspaceCreate(StrictSchema):
    """Create a new business workspace."""

    name: str = Field(
        min_length=1,
        max_length=160,
    )


class WorkspaceUpdate(StrictSchema):
    """Update mutable workspace properties."""

    version: int = Field(ge=1)
    name: str | None = Field(
        default=None,
        min_length=1,
        max_length=160,
    )

    @model_validator(mode="after")
    def require_change(self) -> "WorkspaceUpdate":
        """Reject PATCH requests that contain no mutation."""

        if self.model_fields_set <= {"version"}:
            raise ValueError("At least one workspace field must be updated.")
        return self


class WorkspaceResponse(OrmSchema):
    """Public representation of a workspace."""

    id: UUID
    name: str
    status: Literal["active", "archived"]
    version: int
    created_at: datetime
    updated_at: datetime


class WorkspaceListResponse(StrictSchema):
    """Paginated workspace collection."""

    items: list[WorkspaceResponse]
    total: int
    limit: int
    offset: int
