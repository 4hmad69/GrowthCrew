"""Persistence operations for Customer Personas agent output."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.db.models.customer_persona_set import CustomerPersonaSet


class CustomerPersonaSetRepository:
    """Perform customer-persona-set persistence operations."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, persona_set: CustomerPersonaSet) -> None:
        """Stage a new customer persona set record."""

        self._session.add(persona_set)

    def get_by_workspace(self, workspace_id: UUID) -> CustomerPersonaSet | None:
        """Return the current customer persona set for a workspace, if any."""

        statement = select(CustomerPersonaSet).where(
            CustomerPersonaSet.workspace_id == workspace_id
        )
        return self._session.scalar(statement)

    def delete(self, persona_set: CustomerPersonaSet) -> None:
        """Stage a customer persona set record for deletion."""

        self._session.delete(persona_set)
