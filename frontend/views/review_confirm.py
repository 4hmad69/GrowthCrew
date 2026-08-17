"""Step 3 of onboarding: review, confirm, and finish."""

import streamlit as st

from frontend.state.onboarding_state import get_onboarding_state, reset_onboarding_state


def render_review_confirm() -> None:
    """Render the final onboarding screen: a read-only summary."""

    state = get_onboarding_state()

    if state.workspace is None or state.business_profile is None:
        st.warning("Onboarding data is incomplete. Start over.")
        reset_onboarding_state()
        st.rerun()
        return

    workspace = state.workspace
    profile = state.business_profile

    st.subheader("Step 3 of 3 - Review")
    st.success(f"'{workspace.name}' is set up and ready.")

    st.write(f"**Business name:** {profile.business_name}")
    if profile.website:
        st.write(f"**Website:** {profile.website}")
    if profile.industry:
        st.write(f"**Industry:** {profile.industry}")
    if profile.target_customer:
        st.write(f"**Target customer:** {profile.target_customer}")
    if profile.monthly_marketing_budget is not None:
        st.write(
            f"**Monthly marketing budget:** "
            f"{profile.monthly_marketing_budget} {profile.marketing_budget_currency}"
        )
    if profile.main_marketing_goal:
        st.write(f"**Main marketing goal:** {profile.main_marketing_goal}")
    if profile.existing_channels:
        st.write(f"**Existing channels:** {', '.join(profile.existing_channels)}")
    if profile.brand_tone:
        st.write(f"**Brand tone:** {profile.brand_tone}")
    if profile.known_competitors:
        st.write(f"**Known competitors:** {', '.join(profile.known_competitors)}")

    st.divider()
    st.info(
        "Research, brand strategy, and campaign generation come in later "
        "steps once the agent workflow (Step 5 onward) is built."
    )

    if st.button("Start another workspace"):
        reset_onboarding_state()
        st.rerun()
