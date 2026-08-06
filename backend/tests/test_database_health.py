"""Unit tests for the database readiness API contract."""

from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.db.errors import DatabaseUnavailableError
from backend.app.db.health import DatabaseHealthSnapshot
from backend.app.main import create_application


class HealthyDatabaseChecker:
    """Return a deterministic healthy database result."""

    def check(self) -> DatabaseHealthSnapshot:
        return DatabaseHealthSnapshot()


class UnavailableDatabaseChecker:
    """Simulate a database connectivity failure."""

    def check(self) -> DatabaseHealthSnapshot:
        raise DatabaseUnavailableError("internal connection detail")


def test_database_health_endpoint_returns_expected_contract() -> None:
    """A healthy database should return a minimal readiness contract."""

    application = create_application(
        Settings(environment="test"),
        database_health_checker=HealthyDatabaseChecker(),
    )

    with TestClient(application) as client:
        response = client.get("/api/v1/health/database")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "database": "reachable",
        "pgvector": "available",
    }


def test_database_health_endpoint_sanitizes_failures() -> None:
    """Database failures should return 503 without internal details."""

    application = create_application(
        Settings(environment="test"),
        database_health_checker=UnavailableDatabaseChecker(),
    )

    with TestClient(
        application,
        raise_server_exceptions=False,
    ) as client:
        response = client.get("/api/v1/health/database")

    assert response.status_code == 503
    assert response.json() == {
        "detail": "The database is not ready."
    }
    assert "internal connection detail" not in response.text