"""Shared configuration for GrowthCrew API schemas."""

from pydantic import BaseModel, ConfigDict


class StrictSchema(BaseModel):
    """Base schema that rejects unexpected fields."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


class OrmSchema(StrictSchema):
    """Base response schema that can validate SQLAlchemy model attributes."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        from_attributes=True,
    )
