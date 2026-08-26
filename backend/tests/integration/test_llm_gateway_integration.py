"""Real integration tests against actual Ollama Cloud.

Gated separately from the PostgreSQL integration tests
(GROWTHCREW_RUN_INTEGRATION_TESTS) since this needs `ollama signin` plus a
local Ollama daemon reachable at Settings.llm_base_url, not Postgres - the
two external dependencies are independent, so requiring both just to run
either would be unnecessarily restrictive.

This is also where the one gap flagged when the gateway was built gets
closed: LangChain's own fake chat models don't implement
with_structured_output(), so the native structured-output path's usage
tracking could only be verified against a real provider. The structured()
test below asserts non-zero token counts specifically to confirm that.
"""

import logging
import os

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from backend.app.config import Settings
from backend.app.llm.gateway import LLMGateway
from backend.app.main import create_application

RUN_LLM_INTEGRATION_TESTS = os.getenv("GROWTHCREW_RUN_LLM_INTEGRATION_TESTS") == "1"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not RUN_LLM_INTEGRATION_TESTS,
        reason=(
            "Set GROWTHCREW_RUN_LLM_INTEGRATION_TESTS=1 to run tests against "
            "real Ollama Cloud (requires `ollama signin`)."
        ),
    ),
]


class _SimpleAnswer(BaseModel):
    is_greeting: bool
    reply: str


@pytest.fixture
def real_settings() -> Settings:
    """Settings pointed at the real Ollama Cloud provider."""

    return Settings(environment="test", llm_provider="ollama")


def test_llm_health_endpoint_reports_real_ollama_cloud(real_settings: Settings) -> None:
    """The health endpoint should confirm real Ollama Cloud connectivity end to end."""

    application = create_application(real_settings)

    with TestClient(application) as client:
        response = client.get("/api/v1/health/llm")

    assert response.status_code == 200
    body = response.json()
    assert body["provider"] == "ollama"
    assert body["reachable"] is True


def test_chat_against_real_ollama_cloud_tracks_real_usage(
    real_settings: Settings,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A real chat call should return text and log non-zero token usage."""

    gateway = LLMGateway(real_settings)

    with caplog.at_level(logging.INFO, logger="backend.app.llm.gateway"):
        result = gateway.chat("Reply with only the word: hello")

    assert isinstance(result, str)
    assert result.strip() != ""

    record = next(r for r in caplog.records if r.message == "llm_call_completed")
    assert record.llm_input_tokens > 0
    assert record.llm_output_tokens > 0


def test_structured_against_real_ollama_cloud_tracks_real_usage(
    real_settings: Settings,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A real structured call should validate and log non-zero token usage."""

    gateway = LLMGateway(real_settings)

    with caplog.at_level(logging.INFO, logger="backend.app.llm.gateway"):
        result = gateway.structured(
            "Say hello and confirm this is a greeting.",
            _SimpleAnswer,
        )

    assert isinstance(result, _SimpleAnswer)
    assert result.is_greeting is True

    record = next(r for r in caplog.records if r.message == "llm_call_completed")
    assert record.llm_call_kind == "structured"
    assert record.llm_input_tokens > 0
    assert record.llm_output_tokens > 0
