"""Tests for the onboarding API client."""

from uuid import uuid4

import httpx
import pytest

from frontend.services.backend_http import BackendUnavailableError
from frontend.services.onboarding_api import (
    BusinessProfileCreateRequest,
    OnboardingApiClient,
    OnboardingConflictError,
    OnboardingValidationError,
    WorkspaceCreateRequest,
)


def _client(handler) -> OnboardingApiClient:
    return OnboardingApiClient(
        base_url="http://testserver",
        timeout_seconds=1.0,
        transport=httpx.MockTransport(handler),
    )


def test_create_workspace_returns_validated_model() -> None:
    """A valid workspace-creation response should become a typed model."""

    workspace_id = uuid4()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/workspaces"
        assert request.method == "POST"
        return httpx.Response(
            201,
            json={
                "id": str(workspace_id),
                "name": "FitMeal",
                "status": "active",
                "version": 1,
                "created_at": "2026-08-15T10:00:00Z",
                "updated_at": "2026-08-15T10:00:00Z",
            },
        )

    workspace = _client(handler).create_workspace(WorkspaceCreateRequest(name="FitMeal"))

    assert workspace.id == workspace_id
    assert workspace.name == "FitMeal"
    assert workspace.version == 1


def test_create_workspace_reports_connection_failure() -> None:
    """Connection errors should become a user-safe frontend exception."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused", request=request)

    with pytest.raises(BackendUnavailableError):
        _client(handler).create_workspace(WorkspaceCreateRequest(name="FitMeal"))


def test_create_business_profile_returns_validated_model() -> None:
    """A valid business-profile response should become a typed model."""

    workspace_id = uuid4()
    profile_id = uuid4()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == f"/api/v1/workspaces/{workspace_id}/business-profile"
        return httpx.Response(
            201,
            json={
                "id": str(profile_id),
                "workspace_id": str(workspace_id),
                "business_name": "FitMeal",
                "website": None,
                "product_or_service": None,
                "industry": "Healthy Food",
                "country": None,
                "target_market": None,
                "target_customer": "Busy professionals",
                "price_range": None,
                "monthly_marketing_budget": "500.00",
                "marketing_budget_currency": "USD",
                "main_marketing_goal": "Increase monthly subscriptions",
                "existing_channels": [],
                "brand_tone": "Friendly and energetic",
                "known_competitors": [],
                "current_challenges": None,
                "additional_instructions": None,
                "version": 1,
                "created_at": "2026-08-15T10:00:00Z",
                "updated_at": "2026-08-15T10:00:00Z",
            },
        )

    request = BusinessProfileCreateRequest(
        business_name="FitMeal",
        industry="Healthy Food",
        target_customer="Busy professionals",
        monthly_marketing_budget=500,
        marketing_budget_currency="usd",
        main_marketing_goal="Increase monthly subscriptions",
        brand_tone="Friendly and energetic",
    )

    profile = _client(handler).create_business_profile(workspace_id, request)

    assert profile.business_name == "FitMeal"
    assert profile.marketing_budget_currency == "USD"


def test_create_business_profile_raises_conflict_when_one_exists() -> None:
    """A 409 from the backend should surface as OnboardingConflictError."""

    workspace_id = uuid4()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            409,
            json={"detail": "A business profile already exists for this workspace."},
        )

    with pytest.raises(OnboardingConflictError, match="already exists"):
        _client(handler).create_business_profile(
            workspace_id,
            BusinessProfileCreateRequest(business_name="FitMeal"),
        )


def test_create_business_profile_raises_validation_error_with_detail() -> None:
    """A 422 from the backend should surface as OnboardingValidationError with detail."""

    workspace_id = uuid4()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            422,
            json={
                "detail": [
                    {
                        "loc": ["body", "business_name"],
                        "msg": "Field required",
                        "type": "missing",
                    }
                ]
            },
        )

    with pytest.raises(OnboardingValidationError) as excinfo:
        _client(handler).create_business_profile(
            workspace_id,
            BusinessProfileCreateRequest(business_name="FitMeal"),
        )

    assert excinfo.value.detail[0]["loc"] == ["body", "business_name"]
