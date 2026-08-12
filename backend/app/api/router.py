"""Top-level API router."""

from fastapi import APIRouter

from backend.app.api.health import router as health_router
from backend.app.api.workspaces import router as workspace_router

api_router = APIRouter()

api_router.include_router(health_router)
api_router.include_router(workspace_router)
