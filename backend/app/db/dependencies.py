"""FastAPI dependencies for database-backed routes."""

from collections.abc import Iterator
from typing import cast

from fastapi import Request
from sqlalchemy.orm import Session

from backend.app.db.database import Database
from backend.app.db.errors import DatabaseNotConfiguredError


def get_database(request: Request) -> Database:
    """Return the configured application database or fail safely."""

    database = cast(Database | None, request.app.state.database)
    if database is None:
        raise DatabaseNotConfiguredError("Database configuration is missing.")
    return database


def get_db_session(request: Request) -> Iterator[Session]:
    """Provide one short-lived SQLAlchemy session for a request."""

    database = get_database(request)
    with database.session() as session:
        yield session