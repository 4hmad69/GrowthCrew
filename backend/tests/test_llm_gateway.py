"""Tests for the LLM gateway.

The "ollama" provider path is tested by monkeypatching LLMGateway's
internal chat model with a small fake that fires LangChain's real
callback machinery (using real LLMResult/ChatGeneration/AIMessage
objects, not hand-rolled stand-ins) - so usage-tracking is exercised
through the same code path a real ChatOllama response would trigger,
without needing network access to real Ollama Cloud.
"""

import logging
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from pydantic import BaseModel

from backend.app.config import Settings
from backend.app.llm.gateway import LLMGateway


class _Answer(BaseModel):
    ok: bool
    reason: str


def _fake_result(content: str, input_tokens: int, output_tokens: int) -> LLMResult:
    """Build a real LLMResult carrying usage metadata, as ChatOllama would return."""

    message = AIMessage(
        content=content,
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    )
    return LLMResult(generations=[[ChatGeneration(message=message)]])


class _FakeStructuredRunnable:
    """Stand-in for chat_model.with_structured_output(schema)'s return value."""

    def __init__(self, *, result: Any = None, error: Exception | None = None) -> None:
        self._result = result
        self._error = error
        self.calls = 0

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        self.calls += 1
        if self._error is not None:
            raise self._error
        return self._result


class _FakeChatModel:
    """A chat model double that fires real LangChain callbacks, like a real model."""

    def __init__(
        self,
        *,
        content: str = "fallback response",
        input_tokens: int = 5,
        output_tokens: int = 3,
        structured_result: Any = None,
        structured_error: Exception | None = None,
    ) -> None:
        self._content = content
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self._structured_runnable = _FakeStructuredRunnable(
            result=structured_result,
            error=structured_error,
        )
        self.invoke_calls = 0

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> AIMessage:
        self.invoke_calls += 1
        result = _fake_result(self._content, self._input_tokens, self._output_tokens)
        for callback in (config or {}).get("callbacks", []):
            callback.on_llm_end(result)
        return result.generations[0][0].message

    def with_structured_output(self, schema: type) -> _FakeStructuredRunnable:
        return self._structured_runnable


def _gateway_with_fake_model(
    fake_model: _FakeChatModel,
    **settings_overrides: object,
) -> LLMGateway:
    """Build a real LLMGateway, then swap in a fake chat model (no network)."""

    settings = Settings(
        environment="test",
        llm_provider="ollama",
        llm_retry_initial_delay_seconds=0.1,
        llm_retry_attempts=0,
        **settings_overrides,
    )
    gateway = LLMGateway.__new__(LLMGateway)
    gateway._settings = settings
    gateway._chat_model = fake_model
    return gateway


# --- local provider -----------------------------------------------------------


def test_chat_local_provider_returns_clearly_labeled_stub() -> None:
    gateway = LLMGateway(Settings(environment="test", llm_provider="local"))

    result = gateway.chat("hello")

    assert "local provider stub" in result


def test_structured_local_provider_fills_schema_without_a_model() -> None:
    gateway = LLMGateway(Settings(environment="test", llm_provider="local"))

    result = gateway.structured("should I retrieve?", _Answer)

    assert isinstance(result, _Answer)
    assert result.ok is True


# --- ollama provider: chat() ---------------------------------------------------


def test_chat_ollama_provider_returns_content_and_tracks_usage(caplog) -> None:
    fake_model = _FakeChatModel(content="hi there", input_tokens=12, output_tokens=4)
    gateway = _gateway_with_fake_model(fake_model)

    with caplog.at_level(logging.INFO, logger="backend.app.llm.gateway"):
        result = gateway.chat("say hi")

    assert result == "hi there"
    assert fake_model.invoke_calls == 1

    record = next(r for r in caplog.records if r.message == "llm_call_completed")
    assert record.llm_input_tokens == 12
    assert record.llm_output_tokens == 4
    assert record.llm_call_kind == "chat"


def test_chat_ollama_provider_computes_estimated_cost(caplog) -> None:
    fake_model = _FakeChatModel(input_tokens=1000, output_tokens=1000)
    gateway = _gateway_with_fake_model(
        fake_model,
        llm_cost_per_1k_input_tokens=0.5,
        llm_cost_per_1k_output_tokens=1.5,
    )

    with caplog.at_level(logging.INFO, logger="backend.app.llm.gateway"):
        gateway.chat("say hi")

    record = next(r for r in caplog.records if r.message == "llm_call_completed")
    assert record.llm_estimated_cost_usd == pytest.approx(2.0)


# --- ollama provider: structured() ----------------------------------------------


def test_structured_ollama_provider_uses_native_result_when_it_succeeds(caplog) -> None:
    fake_model = _FakeChatModel(structured_result=_Answer(ok=True, reason="native worked"))
    gateway = _gateway_with_fake_model(fake_model)

    with caplog.at_level(logging.INFO, logger="backend.app.llm.gateway"):
        result = gateway.structured("is this ok?", _Answer)

    assert result.reason == "native worked"
    assert fake_model.invoke_calls == 0  # fallback path never triggered

    record = next(r for r in caplog.records if r.message == "llm_call_completed")
    assert record.llm_call_kind == "structured"


def test_structured_ollama_provider_falls_back_and_still_tracks_usage() -> None:
    fake_model = _FakeChatModel(
        content='{"ok": true, "reason": "fallback worked"}',
        input_tokens=20,
        output_tokens=8,
        structured_error=ValueError("native structured output not supported"),
    )
    gateway = _gateway_with_fake_model(fake_model)

    result = gateway.structured("is this ok?", _Answer)

    assert result.reason == "fallback worked"
    assert fake_model.invoke_calls == 1  # the JSON-only fallback call
