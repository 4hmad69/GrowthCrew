"""Step 1 of onboarding: create a workspace."""

import streamlit as st
from pydantic import ValidationError

from frontend.config import FrontendSettings
from frontend.services.backend_http import BackendResponseError, BackendUnavailableError
from frontend.services.onboarding_api import OnboardingApiClient, WorkspaceCreateRequest
from frontend.state.onboarding_state import advance_to, get_onboarding_state, set_error


def render_new_workspace(settings: FrontendSettings) -> None:
    """Render the first onboarding screen: naming a new workspace."""

    state = get_onboarding_state()

    st.subheader("Step 1 of 3 - Name your workspace")
    st.caption("A workspace holds one business's profile, research, and campaigns.")

    if state.error_message:
        st.error(state.error_message)

    with st.form("new_workspace_form"):
        name = st.text_input(
            "Workspace name",
            max_chars=160,
            placeholder="e.g. FitMeal",
        )
        submitted = st.form_submit_button("Create workspace", type="primary")

    if not submitted:
        return

    try:
        request = WorkspaceCreateRequest(name=name)
    except ValidationError as exc:
        set_error(_first_message(exc))
        st.rerun()
        return

    client = OnboardingApiClient(
        base_url=settings.normalized_backend_url,
        timeout_seconds=settings.http_timeout_seconds,
    )

    with st.spinner("Creating your workspace..."):
        try:
            workspace = client.create_workspace(request)
        except (BackendUnavailableError, BackendResponseError) as exc:
            set_error(str(exc))
            st.rerun()
            return

    state.workspace = workspace
    set_error(None)
    advance_to("business_profile")
    st.rerun()


def _first_message(exc: ValidationError) -> str:
    """Return the first Pydantic validation message, dropping field-path noise."""

    errors = exc.errors()
    if not errors:
        return "The workspace name is invalid."
    return str(errors[0]["msg"])
