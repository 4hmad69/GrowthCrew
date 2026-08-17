"""Foundation home page for GrowthCrew."""

import streamlit as st

from frontend.config import FrontendSettings
from frontend.services.api_client import (
    BackendResponseError,
    BackendServiceUnavailableError,
    BackendUnavailableError,
    GrowthCrewApiClient,
)


def render_home(settings: FrontendSettings) -> None:
    """Render application and database readiness from real backend calls."""

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
                "python -m uvicorn backend.app.main:create_application "
                "--factory --reload --host 127.0.0.1 --port 8000",
                language="text",
            )
            return
        except BackendResponseError as exc:
            st.error(str(exc))
            return

    st.success("Backend connected")
    service_column, version_column, environment_column = st.columns(3)
    service_column.metric("Service", health.service)
    version_column.metric("Version", health.version)
    environment_column.metric("Environment", health.environment)

    with st.spinner("Checking PostgreSQL and pgvector..."):
        try:
            database_health = client.get_database_health()
        except BackendServiceUnavailableError as exc:
            st.warning(str(exc))
            st.code(
                "docker compose up -d postgres\n"
                "python -m alembic -c backend\\alembic.ini upgrade head",
                language="text",
            )
        except (BackendUnavailableError, BackendResponseError) as exc:
            st.error(str(exc))
        else:
            st.success("Database connected")
            database_column, pgvector_column = st.columns(2)
            database_column.metric(
                "PostgreSQL",
                database_health.database,
            )
            pgvector_column.metric(
                "pgvector",
                database_health.pgvector,
            )

    st.divider()
    st.subheader("Current milestone")
    st.write(
        "Step 4 adds the onboarding UI: create a workspace and business "
        "profile through the 'New business' page in the sidebar, backed "
        "by the Step 3 API."
    )
    st.info(
        "Ollama Cloud remains the approved first LLM provider. Its "
        "integration begins in Step 5 after business-profile contracts "
        "are stable."
    )
