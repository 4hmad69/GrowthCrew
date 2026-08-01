"""Foundation home page for GrowthCrew."""

import streamlit as st

from frontend.config import FrontendSettings
from frontend.services.api_client import (
    BackendResponseError,
    BackendUnavailableError,
    GrowthCrewApiClient,
)


def render_home(settings: FrontendSettings) -> None:
    """Render the Step 1 foundation screen and real backend status."""

    st.title("GrowthCrew")
    st.caption("Your AI marketing team for planning, creating, launching, and improving campaigns.")

    st.subheader("Foundation status")
    client = GrowthCrewApiClient(
        base_url=settings.normalized_backend_url,
        timeout_seconds=settings.http_timeout_seconds,
    )

    with st.spinner("Checking the GrowthCrew backend..."):
        try:
            health = client.get_health()
        except BackendUnavailableError as exc:
            st.error(str(exc))
            st.code(
                "python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000",
                language="text",
            )
        except BackendResponseError as exc:
            st.error(str(exc))
        else:
            st.success("Backend connected")
            service_column, version_column, environment_column = st.columns(3)
            service_column.metric("Service", health.service)
            version_column.metric("Version", health.version)
            environment_column.metric("Environment", health.environment)

    st.divider()
    st.subheader("Current milestone")
    st.write(
        "Step 1 establishes the FastAPI backend, Streamlit frontend, typed health "
        "contract, environment configuration, tests, and code-quality tooling."
    )
    st.info(
        "Ollama Cloud is the approved first LLM provider. Its provider integration "
        "begins in Step 5, after the local application and business-profile contracts "
        "are stable."
    )
