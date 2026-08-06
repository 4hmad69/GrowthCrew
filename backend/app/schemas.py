"""Pydantic response schemas for the GrowthCrew API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class HealthResponse(BaseModel):
    """Response returned by the process health endpoint."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"]
    service: str
    version: str
    environment: str


class DatabaseHealthResponse(BaseModel):
    """Response returned when PostgreSQL and pgvector are ready."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"]
    database: Literal["reachable"]
    pgvector: Literal["available"]