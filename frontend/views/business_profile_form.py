"""Step 2 of onboarding: the business profile form."""

from decimal import Decimal, InvalidOperation

import streamlit as st
from pydantic import ValidationError

from frontend.config import FrontendSettings
from frontend.services.backend_http import BackendResponseError, BackendUnavailableError
from frontend.services.onboarding_api import (
    BusinessProfileCreateRequest,
    OnboardingApiClient,
    OnboardingConflictError,
    OnboardingValidationError,
)
from frontend.state.onboarding_state import advance_to, get_onboarding_state, set_error


def render_business_profile_form(settings: FrontendSettings) -> None:
    """Render the second onboarding screen: the business profile form."""

    state = get_onboarding_state()

    if state.workspace is None:
        st.warning("Create a workspace first.")
        advance_to("new_workspace")
        st.rerun()
        return

    st.subheader("Step 2 of 3 - Tell us about the business")
    st.caption(f"Workspace: {state.workspace.name}")

    if state.error_message:
        st.error(state.error_message)

    draft = state.business_profile_draft

    with st.form("business_profile_form"):
        business_name = st.text_input(
            "Business name *",
            value=draft.get("business_name", state.workspace.name),
            max_chars=160,
        )
        website = st.text_input(
            "Website",
            value=draft.get("website", ""),
            placeholder="https://example.com",
        )
        product_or_service = st.text_area(
            "Product or service",
            value=draft.get("product_or_service", ""),
            max_chars=4000,
        )
        industry = st.text_input(
            "Industry",
            value=draft.get("industry", ""),
            max_chars=120,
        )
        country = st.text_input(
            "Country",
            value=draft.get("country", ""),
            max_chars=100,
        )
        target_customer = st.text_area(
            "Target customer",
            value=draft.get("target_customer", ""),
            max_chars=2000,
        )
        price_range = st.text_input(
            "Price range",
            value=draft.get("price_range", ""),
            max_chars=200,
        )

        budget_col, currency_col = st.columns([3, 1])
        budget_raw = budget_col.text_input(
            "Monthly marketing budget",
            value=draft.get("monthly_marketing_budget", ""),
            placeholder="e.g. 500",
        )
        currency = currency_col.text_input(
            "Currency",
            value=draft.get("marketing_budget_currency", "USD"),
            max_chars=3,
        )

        main_marketing_goal = st.text_area(
            "Main marketing goal",
            value=draft.get("main_marketing_goal", ""),
            max_chars=2000,
        )
        existing_channels_raw = st.text_input(
            "Existing marketing channels (comma-separated)",
            value=draft.get("existing_channels_raw", ""),
            placeholder="Instagram, email newsletter",
        )
        brand_tone = st.text_input(
            "Brand tone",
            value=draft.get("brand_tone", ""),
            max_chars=500,
        )
        known_competitors_raw = st.text_input(
            "Known competitors (comma-separated)",
            value=draft.get("known_competitors_raw", ""),
        )
        current_challenges = st.text_area(
            "Current challenges",
            value=draft.get("current_challenges", ""),
            max_chars=4000,
        )
        additional_instructions = st.text_area(
            "Anything else GrowthCrew should know?",
            value=draft.get("additional_instructions", ""),
            max_chars=8000,
        )

        submitted = st.form_submit_button("Continue to review", type="primary")

    if not submitted:
        return

    state.business_profile_draft = {
        "business_name": business_name,
        "website": website,
        "product_or_service": product_or_service,
        "industry": industry,
        "country": country,
        "target_customer": target_customer,
        "price_range": price_range,
        "monthly_marketing_budget": budget_raw,
        "marketing_budget_currency": currency,
        "main_marketing_goal": main_marketing_goal,
        "existing_channels_raw": existing_channels_raw,
        "brand_tone": brand_tone,
        "known_competitors_raw": known_competitors_raw,
        "current_challenges": current_challenges,
        "additional_instructions": additional_instructions,
    }

    try:
        request = _build_request(state.business_profile_draft)
    except (ValidationError, InvalidOperation) as exc:
        set_error(_first_local_message(exc))
        st.rerun()
        return

    client = OnboardingApiClient(
        base_url=settings.normalized_backend_url,
        timeout_seconds=settings.http_timeout_seconds,
    )

    with st.spinner("Saving the business profile..."):
        try:
            profile = client.create_business_profile(state.workspace.id, request)
        except OnboardingConflictError as exc:
            set_error(str(exc))
            st.rerun()
            return
        except OnboardingValidationError as exc:
            set_error(_first_backend_message(exc))
            st.rerun()
            return
        except (BackendUnavailableError, BackendResponseError) as exc:
            set_error(str(exc))
            st.rerun()
            return

    state.business_profile = profile
    set_error(None)
    advance_to("review")
    st.rerun()


def _build_request(draft: dict[str, str]) -> BusinessProfileCreateRequest:
    """Convert raw form strings into a validated onboarding request."""

    budget_raw = draft["monthly_marketing_budget"].strip()
    budget = Decimal(budget_raw) if budget_raw else None

    return BusinessProfileCreateRequest(
        business_name=draft["business_name"],
        website=draft["website"] or None,
        product_or_service=draft["product_or_service"] or None,
        industry=draft["industry"] or None,
        country=draft["country"] or None,
        target_customer=draft["target_customer"] or None,
        price_range=draft["price_range"] or None,
        monthly_marketing_budget=budget,
        marketing_budget_currency=(draft["marketing_budget_currency"] or None) if budget else None,
        main_marketing_goal=draft["main_marketing_goal"] or None,
        existing_channels=_parse_list(draft["existing_channels_raw"]),
        brand_tone=draft["brand_tone"] or None,
        known_competitors=_parse_list(draft["known_competitors_raw"]),
        current_challenges=draft["current_challenges"] or None,
        additional_instructions=draft["additional_instructions"] or None,
    )


def _parse_list(raw: str) -> list[str]:
    """Split a comma-separated form field into a cleaned list of values."""

    return [item.strip() for item in raw.split(",") if item.strip()]


def _first_local_message(exc: ValidationError | InvalidOperation) -> str:
    """Return a user-facing message for a local (pre-submit) validation failure."""

    if isinstance(exc, InvalidOperation):
        return "Monthly marketing budget must be a valid number."

    errors = exc.errors()
    if not errors:
        return "One of the fields you entered is invalid."
    return str(errors[0]["msg"])


def _first_backend_message(exc: OnboardingValidationError) -> str:
    """Return a user-facing message for a backend validation failure."""

    if not exc.detail:
        return str(exc)

    first_error = exc.detail[0]
    field = ".".join(str(part) for part in first_error.get("loc", [])[1:])
    message = first_error.get("msg", str(exc))
    return f"{field}: {message}" if field else str(message)
