"""Typed session-state helpers for the Step 4 onboarding flow.

Views should go through the functions here instead of touching
``st.session_state`` directly, so the shape of onboarding progress lives in
one typed place.
"""

from dataclasses import dataclass, field
from typing import Any, Literal

import streamlit as st

from frontend.services.onboarding_api import BusinessProfileRead, WorkspaceRead

_SESSION_KEY = "growthcrew_onboarding"

OnboardingStep = Literal["new_workspace", "business_profile", "review", "complete"]


@dataclass
class OnboardingState:
    """In-progress onboarding data for one workspace-creation attempt."""

    step: OnboardingStep = "new_workspace"
    workspace: WorkspaceRead | None = None
    business_profile_draft: dict[str, Any] = field(default_factory=dict)
    business_profile: BusinessProfileRead | None = None
    error_message: str | None = None


def get_onboarding_state() -> OnboardingState:
    """Return the current onboarding state, creating it on first access."""

    if _SESSION_KEY not in st.session_state:
        st.session_state[_SESSION_KEY] = OnboardingState()
    return st.session_state[_SESSION_KEY]


def reset_onboarding_state() -> None:
    """Clear onboarding progress entirely and start over from the first screen."""

    st.session_state[_SESSION_KEY] = OnboardingState()


def advance_to(step: OnboardingStep) -> None:
    """Move the onboarding flow to a specific step."""

    get_onboarding_state().step = step


def set_error(message: str | None) -> None:
    """Record (or clear) the error banner shown on the current step."""

    get_onboarding_state().error_message = message
