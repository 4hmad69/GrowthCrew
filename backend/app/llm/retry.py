"""Retry-with-backoff wrapper for LLM provider calls.

Ported from https://github.com/4hmad69/agentic-rag (agentic_rag/backends.py),
generalized to use GrowthCrew's Settings and typed exceptions instead of a
standalone BackendSettings dataclass.

One deliberate change from the source repo: backoff uses "full jitter"
(sleep for a random duration between 0 and the current exponential delay)
instead of a fixed exponential delay plus a small fixed jitter addition.
Full jitter is the strategy AWS's own backoff guidance recommends - it
spreads retries out more evenly and avoids many clients retrying in
lockstep, and it happens to make tests fast too, since a tiny configured
delay keeps the whole retry ceiling tiny.
"""

import logging
import random
import time
from collections.abc import Callable

from backend.app.config import Settings
from backend.app.llm.errors import LLMProviderUnavailableError

logger = logging.getLogger(__name__)

_TRANSIENT_ERROR_MARKERS = (
    "429",
    "502",
    "503",
    "504",
    "rate limit",
    "temporarily unavailable",
    "timeout",
    "timed out",
    "connection",
    "dns",
    "name resolution",
)


def is_transient_error(exc: Exception) -> bool:
    """Return True if an exception looks like a retryable, transient failure.

    Deliberately conservative: anything not matching a known transient
    marker (a bad prompt, an auth failure, a schema mismatch) is treated
    as permanent, since retrying those would just waste time and money on
    a call that will never succeed.
    """

    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in _TRANSIENT_ERROR_MARKERS)


def invoke_with_retries[ResultT](
    operation: Callable[[], ResultT],
    settings: Settings,
) -> ResultT:
    """Call operation(), retrying transient failures with exponential backoff.

    A non-transient failure is re-raised immediately, unchanged - the
    caller sees exactly what the provider raised. A transient failure that
    exhausts every retry attempt is raised as LLMProviderUnavailableError,
    since at that point "retry harder" isn't the right diagnosis anymore -
    the provider is genuinely unreachable.
    """

    last_error: Exception | None = None

    for attempt in range(settings.llm_retry_attempts + 1):
        try:
            return operation()
        except Exception as exc:  # provider clients raise several transport-specific types
            last_error = exc

            if not is_transient_error(exc):
                raise

            if attempt >= settings.llm_retry_attempts:
                raise LLMProviderUnavailableError(
                    "The LLM provider could not be reached after "
                    f"{settings.llm_retry_attempts + 1} attempts."
                ) from exc

            sleep_for = random.uniform(
                0,
                settings.llm_retry_initial_delay_seconds * (2**attempt),
            )
            logger.warning(
                "Transient LLM provider error on attempt %s/%s, retrying in %.2fs: %s",
                attempt + 1,
                settings.llm_retry_attempts + 1,
                sleep_for,
                exc,
            )
            time.sleep(sleep_for)

    # Unreachable: llm_retry_attempts is constrained to >= 0 by Settings, so
    # the loop above always returns or raises before exhausting its range.
    raise LLMProviderUnavailableError("Retry loop exited unexpectedly.") from last_error
