"""The approval gate is part of the published API contract of the gated routes.

Customer Personas, Brand Strategy, Marketing Strategy, and Content Planning
refuse to generate until the stages they build on are approved, answering
409. That has to be visible in the OpenAPI document so a client can plan for
it, and the three stages that need only the business profile must not
advertise it, since they are never gated.
"""

import pytest
from fastapi import FastAPI

from backend.app.config import Settings
from backend.app.main import create_application

GATED_PATHS = [
    "/api/v1/workspaces/{workspace_id}/personas",
    "/api/v1/workspaces/{workspace_id}/brand-strategy",
    "/api/v1/workspaces/{workspace_id}/marketing-strategy",
    "/api/v1/workspaces/{workspace_id}/content-plan",
]
UNGATED_PATHS = [
    "/api/v1/workspaces/{workspace_id}/business-profile/understanding",
    "/api/v1/workspaces/{workspace_id}/market-research",
    "/api/v1/workspaces/{workspace_id}/competitor-analysis",
]


@pytest.fixture(scope="module")
def application() -> FastAPI:
    return create_application(
        Settings(
            environment="test",
            llm_provider="local",
            embeddings_provider="local",
            tavily_api_key=None,
        )
    )


@pytest.mark.parametrize("path", GATED_PATHS)
def test_gated_routes_document_the_409(application: FastAPI, path: str) -> None:
    """A client can see from the spec that approval may block generation."""

    responses = application.openapi()["paths"][path]["post"]["responses"]

    assert "409" in responses
    assert "approved" in responses["409"]["description"]


@pytest.mark.parametrize("path", UNGATED_PATHS)
def test_ungated_routes_do_not_advertise_the_approval_409(application: FastAPI, path: str) -> None:
    """Stages that need only the profile are never blocked on an approval."""

    responses = application.openapi()["paths"][path]["post"]["responses"]

    assert "409" not in responses or "approved" not in responses["409"]["description"]
