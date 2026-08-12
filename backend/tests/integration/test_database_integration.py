"""Integration test for the local PostgreSQL and pgvector service."""

import os

import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_application

RUN_INTEGRATION_TESTS = os.getenv("GROWTHCREW_RUN_INTEGRATION_TESTS") == "1"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not RUN_INTEGRATION_TESTS,
        reason=("Set GROWTHCREW_RUN_INTEGRATION_TESTS=1 to run database integration tests."),
    ),
]


def test_database_readiness_against_local_postgres() -> None:
    """The real database should be reachable with pgvector enabled."""

    settings = Settings(environment="test")
    if settings.database_url is None:
        pytest.fail("GROWTHCREW_DATABASE_URL is required for integration tests.")

    application = create_application(settings)

    with TestClient(application) as client:
        response = client.get("/api/v1/health/database")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "database": "reachable",
        "pgvector": "available",
    }
