"""ORM model registry used by application code and Alembic."""

from backend.app.db.models.business_profile import BusinessProfile
from backend.app.db.models.workspace import Workspace

__all__ = [
    "BusinessProfile",
    "Workspace",
]
