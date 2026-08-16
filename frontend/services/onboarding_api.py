"""Typed client for the GrowthCrew onboarding endpoints.

Wraps workspace creation and business-profile creation - the two calls the
Step 4 onboarding UI needs - behind request/response models that mirror the
backend's Pydantic contracts (``backend/app/schemas``). Kept separate from
:mod:`frontend.services.api_client` so the health-check client and the
onboarding client can evolve independently while sharing the same HTTP
plumbing.
"""

from datetime import datetime
from decimal import Decimal
from typing import Annotated, Any, Literal, TypeVar
from uuid import UUID

import httpx
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from frontend.services.backend_http import (
    BackendConflictError,
    BackendHttpClient,
    BackendResponseError,
    BackendValidationError,
)

__all__ = [
    "BusinessProfileCreateRequest",
    "BusinessProfileRead",
    "OnboardingApiClient",
    "OnboardingConflictError",
    "OnboardingValidationError",
    "WorkspaceCreateRequest",
    "WorkspaceRead",
]

ListItem = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=160),
]

ModelT = TypeVar("ModelT", bound=BaseModel)


class WorkspaceCreateRequest(BaseModel):
    """Client-side copy of the workspace creation contract."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=160)


class WorkspaceRead(BaseModel):
    """Frontend copy of the workspace response contract."""

    model_config = ConfigDict(extra="forbid")

    id: UUID
    name: str
    status: Literal["active", "archived"]
    version: int
    created_at: datetime
    updated_at: datetime


class BusinessProfileCreateRequest(BaseModel):
    """Client-side copy of the business-profile creation contract."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    business_name: str = Field(min_length=1, max_length=160)
    website: HttpUrl | None = None
    product_or_service: str | None = Field(default=None, max_length=4000)
    industry: str | None = Field(default=None, max_length=120)
    country: str | None = Field(default=None, max_length=100)
    target_market: str | None = Field(default=None, max_length=2000)
    target_customer: str | None = Field(default=None, max_length=2000)
    price_range: str | None = Field(default=None, max_length=200)
    monthly_marketing_budget: Decimal | None = Field(
        default=None,
        ge=0,
        max_digits=14,
        decimal_places=2,
    )
    marketing_budget_currency: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z]{3}$",
    )
    main_marketing_goal: str | None = Field(default=None, max_length=2000)
    existing_channels: list[ListItem] = Field(default_factory=list, max_length=20)
    brand_tone: str | None = Field(default=None, max_length=500)
    known_competitors: list[ListItem] = Field(default_factory=list, max_length=20)
    current_challenges: str | None = Field(default=None, max_length=4000)
    additional_instructions: str | None = Field(default=None, max_length=8000)

    @field_validator("marketing_budget_currency")
    @classmethod
    def normalize_currency(cls, value: str | None) -> str | None:
        """Normalize ISO-style currency codes to uppercase, matching the backend."""

        if value is None:
            return None
        return value.upper()

    @model_validator(mode="after")
    def validate_budget_pair(self) -> "BusinessProfileCreateRequest":
        """Require budget amount and currency together, matching the backend."""

        budget_missing = self.monthly_marketing_budget is None
        currency_missing = self.marketing_budget_currency is None

        if budget_missing != currency_missing:
            raise ValueError("Marketing budget and currency must be supplied together.")

        return self


class BusinessProfileRead(BaseModel):
    """Frontend copy of the business-profile response contract."""

    model_config = ConfigDict(extra="forbid")

    id: UUID
    workspace_id: UUID
    business_name: str
    website: HttpUrl | None
    product_or_service: str | None
    industry: str | None
    country: str | None
    target_market: str | None
    target_customer: str | None
    price_range: str | None
    monthly_marketing_budget: Decimal | None
    marketing_budget_currency: str | None
    main_marketing_goal: str | None
    existing_channels: list[str]
    brand_tone: str | None
    known_competitors: list[str]
    current_challenges: str | None
    additional_instructions: str | None
    version: int
    created_at: datetime
    updated_at: datetime


class OnboardingConflictError(BackendConflictError):
    """Raised when a workspace already has a business profile."""


class OnboardingValidationError(BackendValidationError):
    """Raised when onboarding data fails backend validation."""


class OnboardingApiClient:
    """Typed client for creating workspaces and business profiles."""

    def __init__(
        self,
        base_url: str,
        timeout_seconds: float,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._http = BackendHttpClient(base_url, timeout_seconds, transport)

    def create_workspace(self, request: WorkspaceCreateRequest) -> WorkspaceRead:
        """Create a new workspace and return its stored representation."""

        payload = self._http.post(
            "/api/v1/workspaces",
            request.model_dump(mode="json"),
        )
        return self._parse(WorkspaceRead, payload, "workspace")

    def create_business_profile(
        self,
        workspace_id: UUID,
        request: BusinessProfileCreateRequest,
    ) -> BusinessProfileRead:
        """Create the onboarding business profile for a workspace."""

        try:
            payload = self._http.post(
                f"/api/v1/workspaces/{workspace_id}/business-profile",
                request.model_dump(mode="json"),
            )
        except BackendConflictError as exc:
            raise OnboardingConflictError(str(exc)) from exc
        except BackendValidationError as exc:
            raise OnboardingValidationError(str(exc), detail=exc.detail) from exc

        return self._parse(BusinessProfileRead, payload, "business profile")

    @staticmethod
    def _parse(
        model: type[ModelT],
        payload: dict[str, Any],
        label: str,
    ) -> ModelT:
        """Validate a raw JSON payload against a frontend response model."""

        try:
            return model.model_validate(payload)
        except ValidationError as exc:
            raise BackendResponseError(
                f"The backend returned an invalid {label} response."
            ) from exc
