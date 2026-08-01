"""Centralized handlers for unexpected API failures."""

import logging

from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    """Log an unexpected exception and return a safe generic response."""

    logger.exception("Unhandled API exception for path %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected server error occurred."},
    )
