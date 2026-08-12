"""Centralized handlers for controlled and unexpected API failures."""

import logging

from fastapi import Request
from fastapi.responses import JSONResponse

from backend.app.db.errors import DatabaseError

logger = logging.getLogger(__name__)


async def handle_database_error(request: Request, exc: DatabaseError) -> JSONResponse:
    """Return a safe readiness response for controlled database failures."""

    logger.warning(
        "Database readiness failure for path %s (%s)",
        request.url.path,
        type(exc).__name__,
    )
    logger.debug(
        "Database readiness exception",
        exc_info=(type(exc), exc, exc.__traceback__),
    )
    return JSONResponse(
        status_code=503,
        content={"detail": "The database is not ready."},
    )


async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    """Log an unexpected exception and return a safe generic response."""

    logger.exception("Unhandled API exception for path %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected server error occurred."},
    )
