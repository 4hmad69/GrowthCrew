"""Tests for the LLM retry/backoff wrapper."""

import pytest

from backend.app.config import Settings
from backend.app.llm.errors import LLMProviderUnavailableError
from backend.app.llm.retry import invoke_with_retries, is_transient_error


def _fast_settings(**overrides: object) -> Settings:
    """Settings using the minimum allowed backoff delay so tests run fast."""

    defaults: dict[str, object] = {
        "environment": "test",
        "llm_retry_initial_delay_seconds": 0.1,
    }
    defaults.update(overrides)
    return Settings(**defaults)


class _TransientFailure(Exception):
    """Stand-in for a provider error that looks transient."""


def test_is_transient_error_matches_known_markers() -> None:
    """Errors mentioning rate limits, timeouts, or 5xx codes are transient."""

    assert is_transient_error(Exception("HTTP 503 Service Unavailable"))
    assert is_transient_error(Exception("Connection timed out"))
    assert is_transient_error(Exception("rate limit exceeded"))


def test_is_transient_error_rejects_unrelated_errors() -> None:
    """A validation/auth-style error should not be treated as retryable."""

    assert not is_transient_error(ValueError("invalid schema field 'foo'"))


def test_invoke_with_retries_returns_on_first_success() -> None:
    """A successful call should not retry at all."""

    calls = {"count": 0}

    def operation() -> str:
        calls["count"] += 1
        return "ok"

    result = invoke_with_retries(operation, settings=_fast_settings())

    assert result == "ok"
    assert calls["count"] == 1


def test_invoke_with_retries_retries_transient_failures_then_succeeds() -> None:
    """A transient failure should be retried and can still succeed."""

    calls = {"count": 0}

    def operation() -> str:
        calls["count"] += 1
        if calls["count"] < 3:
            raise _TransientFailure("connection timed out")
        return "ok"

    result = invoke_with_retries(
        operation,
        settings=_fast_settings(llm_retry_attempts=5),
    )

    assert result == "ok"
    assert calls["count"] == 3


def test_invoke_with_retries_raises_immediately_on_non_transient_error() -> None:
    """A non-transient error should not be retried at all."""

    calls = {"count": 0}

    def operation() -> str:
        calls["count"] += 1
        raise ValueError("invalid request payload")

    with pytest.raises(ValueError, match="invalid request payload"):
        invoke_with_retries(operation, settings=_fast_settings(llm_retry_attempts=5))

    assert calls["count"] == 1


def test_invoke_with_retries_wraps_exhausted_transient_failures() -> None:
    """Exhausting every retry on a transient error raises a typed gateway error."""

    calls = {"count": 0}

    def operation() -> str:
        calls["count"] += 1
        raise _TransientFailure("503 Service Unavailable")

    with pytest.raises(LLMProviderUnavailableError):
        invoke_with_retries(operation, settings=_fast_settings(llm_retry_attempts=2))

    assert calls["count"] == 3  # the initial attempt plus 2 retries
