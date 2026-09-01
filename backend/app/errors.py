"""Centralized handlers for controlled and unexpected API failures."""

import logging

from fastapi import Request
from fastapi.responses import JSONResponse

from backend.app.db.errors import DatabaseError
from backend.app.embeddings.errors import EmbeddingsError
from backend.app.exceptions import (
    DomainError,
    ResourceConflictError,
    ResourceNotFoundError,
)
from backend.app.llm.errors import LLMGatewayError

logger = logging.getLogger(__name__)


async def handle_domain_error(
    request: Request,
    exc: DomainError,
) -> JSONResponse:
    """Translate safe application-domain failures into HTTP responses."""

    if isinstance(exc, ResourceNotFoundError):
        status_code = 404
    elif isinstance(exc, ResourceConflictError):
        status_code = 409
    else:
        status_code = 400

    logger.info(
        "Domain failure for path %s (%s)",
        request.url.path,
        type(exc).__name__,
    )

    return JSONResponse(
        status_code=status_code,
        content={"detail": str(exc)},
    )


async def handle_database_error(
    request: Request,
    exc: DatabaseError,
) -> JSONResponse:
    """Return a safe response for controlled database failures."""

    logger.warning(
        "Database failure for path %s (%s)",
        request.url.path,
        type(exc).__name__,
    )
    logger.debug(
        "Database exception",
        exc_info=(type(exc), exc, exc.__traceback__),
    )

    return JSONResponse(
        status_code=503,
        content={"detail": "A required database operation could not be completed."},
    )


async def handle_llm_error(
    request: Request,
    exc: LLMGatewayError,
) -> JSONResponse:
    """Return a safe response for controlled LLM gateway failures."""

    logger.warning(
        "LLM gateway failure for path %s (%s)",
        request.url.path,
        type(exc).__name__,
    )
    logger.debug(
        "LLM gateway exception",
        exc_info=(type(exc), exc, exc.__traceback__),
    )

    return JSONResponse(
        status_code=503,
        content={"detail": "A required LLM operation could not be completed."},
    )


async def handle_embeddings_error(
    request: Request,
    exc: EmbeddingsError,
) -> JSONResponse:
    """Return a safe response for controlled embeddings failures."""

    logger.warning(
        "Embeddings failure for path %s (%s)",
        request.url.path,
        type(exc).__name__,
    )
    logger.debug(
        "Embeddings exception",
        exc_info=(type(exc), exc, exc.__traceback__),
    )

    return JSONResponse(
        status_code=503,
        content={"detail": "A required embeddings operation could not be completed."},
    )


async def handle_unexpected_error(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Log an unexpected exception and return a safe generic response."""

    logger.exception(
        "Unhandled API exception for path %s",
        request.url.path,
    )

    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected server error occurred."},
    )
