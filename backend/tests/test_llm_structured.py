"""Tests for LLM structured-output robustness."""

import pytest
from pydantic import BaseModel

from backend.app.config import Settings
from backend.app.llm.errors import LLMStructuredOutputError
from backend.app.llm.structured import (
    LocalStructuredRunnable,
    RobustStructuredRunnable,
    coerce_structured_output,
    json_only_prompt,
    parse_json_object,
    stringify_messages,
)


class _SampleDecision(BaseModel):
    should_retrieve: bool
    tags: list[str]
    count: int
    score: float
    reason: str


class _FakeMessage:
    """Stand-in for a LangChain AIMessage - just needs a .content attribute."""

    def __init__(self, content: str) -> None:
        self.content = content


class _FakeRunnable:
    """A .invoke()-able test double that either returns a fixed result or raises."""

    def __init__(self, *, result: object = None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.calls = 0

    def invoke(self, input: object, config: object = None, **kwargs: object) -> object:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.result


def _fast_settings(**overrides: object) -> Settings:
    """Settings using the minimum allowed backoff delay so tests run fast."""

    defaults: dict[str, object] = {
        "environment": "test",
        "llm_retry_initial_delay_seconds": 0.1,
        "llm_retry_attempts": 0,
    }
    defaults.update(overrides)
    return Settings(**defaults)


# --- stringify_messages -----------------------------------------------------


def test_stringify_messages_passes_through_plain_strings() -> None:
    assert stringify_messages("hello") == "hello"


def test_stringify_messages_formats_dicts_as_lines() -> None:
    assert stringify_messages({"a": 1, "b": 2}) == "a: 1\nb: 2"


def test_stringify_messages_joins_message_like_objects() -> None:
    messages = [_FakeMessage("first"), _FakeMessage("second")]
    assert stringify_messages(messages) == "first\nsecond"


# --- parse_json_object -------------------------------------------------------


def test_parse_json_object_passes_through_dicts() -> None:
    assert parse_json_object({"a": 1}) == {"a": 1}


def test_parse_json_object_extracts_json_from_surrounding_text() -> None:
    text = 'Sure, here you go:\n```json\n{"a": 1, "b": [1, 2]}\n```\nHope that helps!'
    assert parse_json_object(text) == {"a": 1, "b": [1, 2]}


def test_parse_json_object_raises_when_no_json_present() -> None:
    with pytest.raises(ValueError, match="No JSON object found"):
        parse_json_object("I cannot help with that.")


# --- coerce_structured_output -------------------------------------------------


def test_coerce_structured_output_passes_through_correct_type() -> None:
    instance = _SampleDecision(
        should_retrieve=True,
        tags=["a"],
        count=1,
        score=0.5,
        reason="because",
    )
    assert coerce_structured_output(instance, _SampleDecision) is instance


def test_coerce_structured_output_validates_from_dict() -> None:
    data = {
        "should_retrieve": False,
        "tags": [],
        "count": 0,
        "score": 0.0,
        "reason": "n/a",
    }
    result = coerce_structured_output(data, _SampleDecision)
    assert isinstance(result, _SampleDecision)
    assert result.should_retrieve is False


# --- json_only_prompt ---------------------------------------------------------


def test_json_only_prompt_includes_schema_and_task() -> None:
    prompt = json_only_prompt("What's the weather?", _SampleDecision)
    assert "should_retrieve" in prompt
    assert "What's the weather?" in prompt
    assert "only one valid JSON object" in prompt


# --- RobustStructuredRunnable --------------------------------------------------


def test_robust_structured_runnable_uses_native_result_when_it_succeeds() -> None:
    native = _FakeRunnable(
        result=_SampleDecision(
            should_retrieve=True,
            tags=["x"],
            count=2,
            score=1.0,
            reason="ok",
        )
    )
    chat_model = _FakeRunnable()

    runnable = RobustStructuredRunnable(native, chat_model, _SampleDecision, _fast_settings())
    result = runnable.invoke("some question")

    assert result.should_retrieve is True
    assert native.calls == 1
    assert chat_model.calls == 0  # fallback should never be reached


def test_robust_structured_runnable_falls_back_to_json_prompt() -> None:
    native = _FakeRunnable(error=ValueError("native structured output not supported"))
    chat_model = _FakeRunnable(
        result=_FakeMessage(
            '{"should_retrieve": true, "tags": [], "count": 3, '
            '"score": 0.9, "reason": "fallback worked"}'
        )
    )

    runnable = RobustStructuredRunnable(native, chat_model, _SampleDecision, _fast_settings())
    result = runnable.invoke("some question")

    assert result.reason == "fallback worked"
    assert native.calls == 1
    assert chat_model.calls == 1


def test_robust_structured_runnable_raises_typed_error_when_both_fail() -> None:
    native = _FakeRunnable(error=ValueError("native failed"))
    chat_model = _FakeRunnable(error=RuntimeError("fallback failed too"))

    runnable = RobustStructuredRunnable(native, chat_model, _SampleDecision, _fast_settings())

    with pytest.raises(LLMStructuredOutputError, match="_SampleDecision"):
        runnable.invoke("some question")


# --- LocalStructuredRunnable ---------------------------------------------------


def test_local_structured_runnable_fills_every_field_by_type() -> None:
    runnable = LocalStructuredRunnable(_SampleDecision)
    result = runnable.invoke("Should I retrieve documents about pricing?")

    assert result.should_retrieve is True
    assert result.tags == []
    assert result.count == 0
    assert result.score == 0.0
    assert isinstance(result.reason, str) and result.reason
