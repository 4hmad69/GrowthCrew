"""Tests for the business-profile form's pure logic (no Streamlit runtime needed)."""

from decimal import InvalidOperation

import pytest
from pydantic import ValidationError

from frontend.services.onboarding_api import OnboardingValidationError
from frontend.views.business_profile_form import (
    _build_request,
    _first_backend_message,
    _first_local_message,
    _parse_list,
)

EMPTY_DRAFT: dict[str, str] = {
    "business_name": "",
    "website": "",
    "product_or_service": "",
    "industry": "",
    "country": "",
    "target_customer": "",
    "price_range": "",
    "monthly_marketing_budget": "",
    "marketing_budget_currency": "",
    "main_marketing_goal": "",
    "existing_channels_raw": "",
    "brand_tone": "",
    "known_competitors_raw": "",
    "current_challenges": "",
    "additional_instructions": "",
}


def test_parse_list_splits_and_trims() -> None:
    """Comma-separated input should become a clean list with blanks dropped."""

    assert _parse_list("Instagram,  email newsletter ,,TikTok") == [
        "Instagram",
        "email newsletter",
        "TikTok",
    ]


def test_parse_list_handles_empty_input() -> None:
    """An empty field should produce an empty list, not [''].."""

    assert _parse_list("") == []
    assert _parse_list("   ") == []


def test_build_request_from_minimal_draft() -> None:
    """A draft with only the required field should build a valid request."""

    draft = {**EMPTY_DRAFT, "business_name": "FitMeal"}

    request = _build_request(draft)

    assert request.business_name == "FitMeal"
    assert request.monthly_marketing_budget is None
    assert request.marketing_budget_currency is None
    assert request.existing_channels == []


def test_build_request_pairs_budget_and_currency() -> None:
    """A budget with no currency typed should still validate: currency defaults to USD."""

    draft = {
        **EMPTY_DRAFT,
        "business_name": "FitMeal",
        "monthly_marketing_budget": "500",
        "marketing_budget_currency": "usd",
    }

    request = _build_request(draft)

    assert request.monthly_marketing_budget == 500
    assert request.marketing_budget_currency == "USD"


def test_build_request_drops_currency_when_budget_is_blank() -> None:
    """A leftover currency value with no budget amount should not cause a validation error."""

    draft = {
        **EMPTY_DRAFT,
        "business_name": "FitMeal",
        "monthly_marketing_budget": "",
        "marketing_budget_currency": "USD",
    }

    request = _build_request(draft)

    assert request.monthly_marketing_budget is None
    assert request.marketing_budget_currency is None


def test_build_request_rejects_invalid_budget_text() -> None:
    """Non-numeric budget input should raise InvalidOperation, not a silent 0."""

    draft = {
        **EMPTY_DRAFT,
        "business_name": "FitMeal",
        "monthly_marketing_budget": "not-a-number",
        "marketing_budget_currency": "USD",
    }

    with pytest.raises(InvalidOperation):
        _build_request(draft)


def test_build_request_rejects_missing_business_name() -> None:
    """An empty business name should fail Pydantic validation, not reach the API."""

    with pytest.raises(ValidationError):
        _build_request(EMPTY_DRAFT)


def test_first_local_message_for_invalid_operation() -> None:
    """InvalidOperation should map to a budget-specific message."""

    assert "budget" in _first_local_message(InvalidOperation()).lower()


def test_first_local_message_for_validation_error() -> None:
    """A Pydantic ValidationError should surface its first field message."""

    try:
        _build_request(EMPTY_DRAFT)
    except ValidationError as exc:
        message = _first_local_message(exc)
        assert message
        return
    pytest.fail("Expected ValidationError was not raised")


def test_first_backend_message_includes_field_path() -> None:
    """A backend 422 detail should be rendered as 'field: message'."""

    exc = OnboardingValidationError(
        "The backend rejected the request payload.",
        detail=[{"loc": ["body", "business_name"], "msg": "Field required"}],
    )

    assert _first_backend_message(exc) == "business_name: Field required"


def test_first_backend_message_falls_back_without_detail() -> None:
    """A 422 with no structured detail should fall back to the exception message."""

    exc = OnboardingValidationError("The backend rejected the request payload.", detail=None)

    assert _first_backend_message(exc) == "The backend rejected the request payload."
