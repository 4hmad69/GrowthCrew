"""FastAPI application factory."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.app.api.router import api_router
from backend.app.config import Settings, get_settings
from backend.app.db.database import Database
from backend.app.db.errors import DatabaseError
from backend.app.db.health import (
    DatabaseHealthChecker,
    DatabaseHealthService,
    UnconfiguredDatabaseHealthService,
)
from backend.app.errors import handle_database_error, handle_unexpected_error


def configure_logging(log_level: str) -> None:
    """Configure readable foundation logging for local development."""

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def create_application(
    settings: Settings | None = None,
    database: Database | None = None,
    database_health_checker: DatabaseHealthChecker | None = None,
) -> FastAPI:
    """Create and configure a GrowthCrew FastAPI application."""

    resolved_settings = settings or get_settings()
    configure_logging(resolved_settings.log_level)

    resolved_database = database
    owns_database = False
    if resolved_database is None and resolved_settings.database_url_value is not None:
        resolved_database = Database.from_settings(resolved_settings)
        owns_database = True

    resolved_health_checker = database_health_checker
    if resolved_health_checker is None:
        if resolved_database is None:
            resolved_health_checker = UnconfiguredDatabaseHealthService()
        else:
            resolved_health_checker = DatabaseHealthService(resolved_database)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        if owns_database and resolved_database is not None:
            resolved_database.dispose()

    application = FastAPI(
        title=resolved_settings.app_name,
        version=resolved_settings.app_version,
        docs_url=f"{resolved_settings.api_v1_prefix}/docs",
        redoc_url=f"{resolved_settings.api_v1_prefix}/redoc",
        openapi_url=f"{resolved_settings.api_v1_prefix}/openapi.json",
        lifespan=lifespan,
    )
    application.state.settings = resolved_settings
    application.state.database = resolved_database
    application.state.database_health_checker = resolved_health_checker
    application.add_exception_handler(DatabaseError, handle_database_error)
    application.add_exception_handler(Exception, handle_unexpected_error)
    application.include_router(
        api_router,
        prefix=resolved_settings.api_v1_prefix,
    )
    return application