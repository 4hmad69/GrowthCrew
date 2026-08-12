"""Unit tests for onboarding API validation."""

from decimal import Decimal

import pytest
from pydantic import ValidationError

from backend.app.schemas.business_profile import (
    BusinessProfileCreate,
    BusinessProfileUpdate,
)


def test_create_profile_normalizes_currency() -> None:
    """Currency codes should be normalized before persistence."""

    profile = BusinessProfileCreate(
        business_name="Acme",
        monthly_marketing_budget=Decimal("500.00"),
        marketing_budget_currency="usd",
    )

    assert profile.marketing_budget_currency == "USD"


def test_create_profile_requires_budget_currency_pair() -> None:
    """A numeric budget without currency should be rejected."""

    with pytest.raises(ValidationError):
        BusinessProfileCreate(
            business_name="Acme",
            monthly_marketing_budget=Decimal("500.00"),
        )


def test_update_requires_budget_fields_together() -> None:
    """Partial budget-pair updates should be rejected."""

    with pytest.raises(ValidationError):
        BusinessProfileUpdate(
            version=1,
            monthly_marketing_budget=Decimal("750.00"),
        )


def test_update_rejects_cleared_business_name() -> None:
    """The required business name cannot be changed to null."""

    with pytest.raises(ValidationError):
        BusinessProfileUpdate(
            version=1,
            business_name=None,
        )


def test_update_requires_real_change() -> None:
    """A PATCH containing only a version should be rejected."""

    with pytest.raises(ValidationError):
        BusinessProfileUpdate(version=1)
