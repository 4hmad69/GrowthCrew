"""Streamlit entry point for the GrowthCrew frontend."""

import streamlit as st

from frontend.config import FrontendSettings, get_frontend_settings
from frontend.state.onboarding_state import get_onboarding_state
from frontend.views.business_profile_form import render_business_profile_form
from frontend.views.home import render_home
from frontend.views.new_workspace import render_new_workspace
from frontend.views.review_confirm import render_review_confirm

_PAGES = ("Home", "New business")


def main() -> None:
    """Configure the page and route to the selected page."""

    st.set_page_config(
        page_title="GrowthCrew",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    settings = get_frontend_settings()
    page = st.sidebar.radio("Navigate", _PAGES)

    if page == "Home":
        render_home(settings)
    else:
        render_onboarding(settings)


def render_onboarding(settings: FrontendSettings) -> None:
    """Route to the current step of the onboarding flow."""

    state = get_onboarding_state()

    if state.step == "new_workspace":
        render_new_workspace(settings)
    elif state.step == "business_profile":
        render_business_profile_form(settings)
    else:
        render_review_confirm()


if __name__ == "__main__":
    main()
