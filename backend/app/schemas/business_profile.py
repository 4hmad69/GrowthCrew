"""Business onboarding request and response schemas."""

from datetime import datetime
from decimal import Decimal
from typing import Annotated
from uuid import UUID

from pydantic import (
    Field,
    HttpUrl,
    StringConstraints,
    field_validator,
    model_validator,
)

from backend.app.schemas.common import OrmSchema, StrictSchema

ListItem = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=160,
    ),
]


class BusinessProfileCreate(StrictSchema):
    """Create the user-provided onboarding profile."""

    business_name: str = Field(
        min_length=1,
        max_length=160,
    )
    website: HttpUrl | None = None
    product_or_service: str | None = Field(
        default=None,
        max_length=4000,
    )
    industry: str | None = Field(
        default=None,
        max_length=120,
    )
    country: str | None = Field(
        default=None,
        max_length=100,
    )
    target_market: str | None = Field(
        default=None,
        max_length=2000,
    )
    target_customer: str | None = Field(
        default=None,
        max_length=2000,
    )
    price_range: str | None = Field(
        default=None,
        max_length=200,
    )
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
    main_marketing_goal: str | None = Field(
        default=None,
        max_length=2000,
    )
    existing_channels: list[ListItem] = Field(
        default_factory=list,
        max_length=20,
    )
    brand_tone: str | None = Field(
        default=None,
        max_length=500,
    )
    known_competitors: list[ListItem] = Field(
        default_factory=list,
        max_length=20,
    )
    current_challenges: str | None = Field(
        default=None,
        max_length=4000,
    )
    additional_instructions: str | None = Field(
        default=None,
        max_length=8000,
    )

    @field_validator("marketing_budget_currency")
    @classmethod
    def normalize_currency(
        cls,
        value: str | None,
    ) -> str | None:
        """Normalize ISO-style currency codes to uppercase."""

        if value is None:
            return None
        return value.upper()

    @model_validator(mode="after")
    def validate_budget_pair(self) -> "BusinessProfileCreate":
        """Require budget amount and currency together."""

        budget_missing = self.monthly_marketing_budget is None
        currency_missing = self.marketing_budget_currency is None

        if budget_missing != currency_missing:
            raise ValueError("Marketing budget and currency must be supplied together.")

        return self


class BusinessProfileUpdate(StrictSchema):
    """Patch editable onboarding information."""

    version: int = Field(ge=1)

    business_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=160,
    )
    website: HttpUrl | None = None
    product_or_service: str | None = Field(
        default=None,
        max_length=4000,
    )
    industry: str | None = Field(
        default=None,
        max_length=120,
    )
    country: str | None = Field(
        default=None,
        max_length=100,
    )
    target_market: str | None = Field(
        default=None,
        max_length=2000,
    )
    target_customer: str | None = Field(
        default=None,
        max_length=2000,
    )
    price_range: str | None = Field(
        default=None,
        max_length=200,
    )
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
    main_marketing_goal: str | None = Field(
        default=None,
        max_length=2000,
    )
    existing_channels: list[ListItem] | None = Field(
        default=None,
        max_length=20,
    )
    brand_tone: str | None = Field(
        default=None,
        max_length=500,
    )
    known_competitors: list[ListItem] | None = Field(
        default=None,
        max_length=20,
    )
    current_challenges: str | None = Field(
        default=None,
        max_length=4000,
    )
    additional_instructions: str | None = Field(
        default=None,
        max_length=8000,
    )

    @field_validator("marketing_budget_currency")
    @classmethod
    def normalize_currency(
        cls,
        value: str | None,
    ) -> str | None:
        """Normalize supplied currency codes."""

        if value is None:
            return None
        return value.upper()

    @model_validator(mode="after")
    def validate_patch(self) -> "BusinessProfileUpdate":
        """Validate partial-update semantics."""

        changed_fields = self.model_fields_set - {"version"}

        if not changed_fields:
            raise ValueError("At least one business profile field must be updated.")

        if "business_name" in self.model_fields_set and self.business_name is None:
            raise ValueError("Business name cannot be cleared.")

        for list_field in (
            "existing_channels",
            "known_competitors",
        ):
            if list_field in self.model_fields_set and getattr(self, list_field) is None:
                raise ValueError(f"{list_field} must use an empty list to clear values.")

        budget_included = "monthly_marketing_budget" in self.model_fields_set
        currency_included = "marketing_budget_currency" in self.model_fields_set

        if budget_included != currency_included:
            raise ValueError("Budget and currency must be updated together.")

        if budget_included:
            budget_missing = self.monthly_marketing_budget is None
            currency_missing = self.marketing_budget_currency is None

            if budget_missing != currency_missing:
                raise ValueError("Budget and currency must both contain values or both be null.")

        return self


class BusinessProfileResponse(OrmSchema):
    """Public representation of the onboarding profile."""

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
