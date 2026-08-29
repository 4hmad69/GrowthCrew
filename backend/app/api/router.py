"""Top-level API router."""

from fastapi import APIRouter

from backend.app.api.business_profiles import (
    router as business_profile_router,
)
from backend.app.api.business_understanding import (
    router as business_understanding_router,
)
from backend.app.api.health import router as health_router
from backend.app.api.workspaces import router as workspace_router

api_router = APIRouter()

api_router.include_router(health_router)
api_router.include_router(workspace_router)
api_router.include_router(business_profile_router)
api_router.include_router(business_understanding_router)
