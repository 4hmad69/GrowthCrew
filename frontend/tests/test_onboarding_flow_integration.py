"""Real end-to-end onboarding test: Streamlit UI -> real FastAPI app -> PostgreSQL.

Follows the same GROWTHCREW_RUN_INTEGRATION_TESTS gate and cleanup pattern
as backend/tests/integration/test_onboarding_api_integration.py.

The onboarding API client is synchronous (it matches Streamlit's own
synchronous execution model), and httpx.ASGITransport only implements the
*async* transport interface - it cannot be used with a sync httpx.Client.
So instead of an in-process ASGI transport, the real FastAPI app is run on
a background thread bound to a free local port, and the client talks to it
over a real (loopback) HTTP connection - the same way it talks to the
backend in normal use.
"""

import os
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from uuid import UUID

import pytest
import uvicorn
from streamlit.testing.v1 import AppTest

from backend.app.config import Settings
from backend.app.db.database import Database
from backend.app.db.models.workspace import Workspace
from backend.app.main import create_application
from frontend.services.onboarding_api import OnboardingApiClient

RUN_INTEGRATION_TESTS = os.getenv("GROWTHCREW_RUN_INTEGRATION_TESTS") == "1"

# AppTest.from_file() resolves relative paths against the file that calls it
# (this test module), not the pytest rootdir - so an absolute path is used
# instead of the fragile "frontend/app.py" relative form.
APP_PATH = Path(__file__).resolve().parents[1] / "app.py"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not RUN_INTEGRATION_TESTS,
        reason=("Set GROWTHCREW_RUN_INTEGRATION_TESTS=1 to run PostgreSQL integration tests."),
    ),
]


def _free_port() -> int:
    """Ask the OS for an unused local port to run the test server on."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def real_onboarding_client() -> Iterator[tuple[OnboardingApiClient, Settings]]:
    """Run the real FastAPI app on a background thread and return a client for it."""

    settings = Settings(environment="test")

    if settings.database_url is None:
        pytest.fail("GROWTHCREW_DATABASE_URL is required for integration tests.")

    application = create_application(settings)
    port = _free_port()
    config = uvicorn.Config(application, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    if not server.started:
        pytest.fail("The backend test server did not start within 10 seconds.")

    client = OnboardingApiClient(
        base_url=f"http://127.0.0.1:{port}",
        timeout_seconds=5.0,
    )

    try:
        yield client, settings
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def _cleanup(settings: Settings, workspace_id: UUID) -> None:
    """Delete the workspace this test created (cascades to its business profile)."""

    database = Database.from_settings(settings)
    try:
        with database.session() as session:
            workspace = session.get(Workspace, workspace_id)
            if workspace is not None:
                session.delete(workspace)
                session.commit()
    finally:
        database.dispose()


def test_onboarding_flow_end_to_end(real_onboarding_client, monkeypatch) -> None:
    """Drive New Workspace -> Business Profile -> Review through the real UI and API."""

    client, settings = real_onboarding_client

    def _use_real_client(*_args, **_kwargs) -> OnboardingApiClient:
        return client

    monkeypatch.setattr("frontend.views.new_workspace.OnboardingApiClient", _use_real_client)
    monkeypatch.setattr(
        "frontend.views.business_profile_form.OnboardingApiClient",
        _use_real_client,
    )

    at = AppTest.from_file(str(APP_PATH))
    at.run()
    assert not at.exception

    at.sidebar.radio[0].set_value("New business").run()
    assert not at.exception
    assert at.subheader[0].value == "Step 1 of 3 - Name your workspace"

    at.text_input[0].set_value("Integration Test Business").run()
    at.button[0].click().run()
    assert not at.exception
    assert at.subheader[0].value == "Step 2 of 3 - Tell us about the business"

    at.text_input[0].set_value("Integration Test Business").run()
    at.text_area[1].set_value("Busy professionals").run()
    at.button[-1].click().run()
    assert not at.exception
    assert at.subheader[0].value == "Step 3 of 3 - Review"

    workspace_id = UUID(str(at.session_state["growthcrew_onboarding"].workspace.id))

    try:
        assert any("Integration Test Business" in element.value for element in at.markdown)
    finally:
        _cleanup(settings, workspace_id)
