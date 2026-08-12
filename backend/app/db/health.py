"""Database readiness checks."""

from dataclasses import dataclass
from typing import Literal, Protocol

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from backend.app.db.database import Database
from backend.app.db.errors import (
    DatabaseNotConfiguredError,
    DatabaseNotReadyError,
    DatabaseUnavailableError,
)


@dataclass(frozen=True, slots=True)
class DatabaseHealthSnapshot:
    """Internal result of a successful database readiness check."""

    database: Literal["reachable"] = "reachable"
    pgvector: Literal["available"] = "available"


class DatabaseHealthChecker(Protocol):
    """Contract used by the API and test doubles."""

    def check(self) -> DatabaseHealthSnapshot:
        """Return readiness details or raise a controlled database error."""

        ...


class DatabaseHealthService:
    """Check PostgreSQL connectivity and the required pgvector extension."""

    def __init__(self, database: Database) -> None:
        self._database = database

    def check(self) -> DatabaseHealthSnapshot:
        """Confirm PostgreSQL responds and pgvector is enabled."""

        statement = text(
            """
            SELECT EXISTS (
                SELECT 1
                FROM pg_extension
                WHERE extname = 'vector'
            )
            """
        )

        try:
            with self._database.engine.connect() as connection:
                pgvector_enabled = connection.scalar(statement)
        except SQLAlchemyError as exc:
            raise DatabaseUnavailableError("PostgreSQL is unavailable.") from exc

        if pgvector_enabled is not True:
            raise DatabaseNotReadyError("The pgvector extension is not enabled.")

        return DatabaseHealthSnapshot()


class UnconfiguredDatabaseHealthService:
    """Represent a backend started without database configuration."""

    def check(self) -> DatabaseHealthSnapshot:
        """Fail with a controlled readiness error."""

        raise DatabaseNotConfiguredError("Database configuration is missing.")
