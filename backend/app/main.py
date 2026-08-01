"""FastAPI application factory and process-level app instance."""

import logging

from fastapi import FastAPI

from backend.app.api.router import api_router
from backend.app.config import Settings, get_settings
from backend.app.errors import handle_unexpected_error


def configure_logging(log_level: str) -> None:
    """Configure readable foundation logging for local development."""

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def create_application(settings: Settings | None = None) -> FastAPI:
    """Create and configure a GrowthCrew FastAPI application."""

    resolved_settings = settings or get_settings()
    configure_logging(resolved_settings.log_level)

    application = FastAPI(
        title=resolved_settings.app_name,
        version=resolved_settings.app_version,
        docs_url=f"{resolved_settings.api_v1_prefix}/docs",
        redoc_url=f"{resolved_settings.api_v1_prefix}/redoc",
        openapi_url=f"{resolved_settings.api_v1_prefix}/openapi.json",
    )
    application.state.settings = resolved_settings
    application.add_exception_handler(Exception, handle_unexpected_error)
    application.include_router(api_router, prefix=resolved_settings.api_v1_prefix)
    return application


app = create_application()
