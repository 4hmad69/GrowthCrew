"""The LLM gateway: the one thing agents import instead of a provider SDK.

Wraps model construction, retries, structured-output robustness, and
token/cost/latency logging behind two methods - chat() and structured().
Provider selection (Ollama Cloud today, "local" for tests/offline dev) is
entirely driven by Settings.llm_provider; callers never construct a
provider client or touch retry logic themselves.

Token usage is captured via LangChain's callback system (on_llm_end),
not by inspecting the final return value - the structured() path can
return a validated schema instance with no trace of the raw model
response, so usage has to be captured at the point of the actual model
call, not after. This works identically whether structured() succeeds
through native structured output or falls back to the JSON-only prompt,
since both paths ultimately invoke a LangChain chat model and callbacks
fire from that invocation itself, propagated automatically through
Runnable.invoke()'s config parameter - a core LangChain guarantee, not
something specific to this wrapper.

Temperature is fixed at 0 for now (deterministic, matching the CRAG-style
grading/decision work agents will do first, and matching the proven
agentic-rag repo's own default). Nothing yet needs per-call temperature
override, so it isn't exposed - add it when an agent actually needs it.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from backend.app.config import Settings
from backend.app.llm.retry import invoke_with_retries
from backend.app.llm.structured import LocalStructuredRunnable, RobustStructuredRunnable

logger = logging.getLogger(__name__)


class _UsageCapture(BaseCallbackHandler):
    """Captures token usage from a single LLM call via LangChain's callbacks.

    A fresh instance is used per gateway call (never shared/reused across
    calls) so counts never leak between unrelated requests. When a call
    involves more than one underlying model invocation (e.g. structured()
    falling back from native structured output to the JSON-only prompt),
    passing the same capture instance into both invocations correctly
    accumulates usage across both real calls - both actually cost tokens,
    even though only the second one produced the final result.
    """

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        for generation_list in response.generations:
            for generation in generation_list:
                message = getattr(generation, "message", None)
                usage = getattr(message, "usage_metadata", None) if message else None
                if isinstance(usage, dict):
                    self.input_tokens += int(usage.get("input_tokens", 0) or 0)
                    self.output_tokens += int(usage.get("output_tokens", 0) or 0)


@dataclass(frozen=True)
class LLMUsage:
    """Token usage and estimated cost for a single gateway call."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost_usd: float


class _LocalChatModel:
    """Deterministic stand-in chat model for the "local" test/offline provider.

    Never calls a real LLM - returns a clearly-labeled stub response, so a
    "local" provider response can never be mistaken for a genuine model
    answer if it ends up in a log or a UI during local development.
    """

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> "_LocalMessage":
        return _LocalMessage("[local provider stub response - no real model was called]")


class _LocalMessage:
    """Minimal message stand-in with the attribute callers read: .content."""

    def __init__(self, content: str) -> None:
        self.content = content


class LLMGateway:
    """The single entry point every agent uses to call an LLM."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._chat_model = self._build_chat_model()

    def chat(self, prompt: str) -> str:
        """Send a plain-text prompt and return the model's text response."""

        result, _usage = self._chat_with_usage(prompt)
        return result

    def chat_with_usage(self, prompt: str) -> tuple[str, LLMUsage]:
        """Same as chat(), but also returns the call's token usage/cost.

        For callers that need to persist usage alongside the result (e.g.
        an agent service storing cost per generated record), not just
        have it appear in the logs.
        """

        return self._chat_with_usage(prompt)

    def structured[SchemaT](self, prompt: str, schema: type[SchemaT]) -> SchemaT:
        """Send a prompt and return a validated instance of `schema`."""

        result, _usage = self._structured_with_usage(prompt, schema)
        return result

    def structured_with_usage[SchemaT](
        self,
        prompt: str,
        schema: type[SchemaT],
    ) -> tuple[SchemaT, LLMUsage]:
        """Same as structured(), but also returns the call's token usage/cost."""

        return self._structured_with_usage(prompt, schema)

    def _chat_with_usage(self, prompt: str) -> tuple[str, LLMUsage]:
        started_at = time.monotonic()

        if self._settings.llm_provider == "local":
            message = self._chat_model.invoke(prompt)
            usage = self._log_call("chat", _UsageCapture(), time.monotonic() - started_at)
            return str(message.content), usage

        capture = _UsageCapture()
        message = invoke_with_retries(
            lambda: self._chat_model.invoke(prompt, config={"callbacks": [capture]}),
            settings=self._settings,
        )
        usage = self._log_call("chat", capture, time.monotonic() - started_at)
        return str(getattr(message, "content", message)), usage

    def _structured_with_usage[SchemaT](
        self,
        prompt: str,
        schema: type[SchemaT],
    ) -> tuple[SchemaT, LLMUsage]:
        started_at = time.monotonic()

        if self._settings.llm_provider == "local":
            result = LocalStructuredRunnable(schema).invoke(prompt)
            usage = self._log_call(
                "structured",
                _UsageCapture(),
                time.monotonic() - started_at,
            )
            return result, usage

        capture = _UsageCapture()
        structured_runnable = self._chat_model.with_structured_output(schema)
        runnable = RobustStructuredRunnable(
            structured_runnable,
            self._chat_model,
            schema,
            self._settings,
        )
        result = runnable.invoke(prompt, config={"callbacks": [capture]})
        usage = self._log_call("structured", capture, time.monotonic() - started_at)
        return result, usage

    def _build_chat_model(self) -> Any:
        """Construct the underlying provider client based on llm_provider."""

        if self._settings.llm_provider == "local":
            return _LocalChatModel()

        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=self._settings.llm_model,
            base_url=self._settings.llm_base_url,
            temperature=0,
            num_predict=self._settings.llm_num_predict,
            client_kwargs={"timeout": self._settings.llm_request_timeout_seconds},
        )

    def _log_call(self, kind: str, usage: _UsageCapture, duration_seconds: float) -> LLMUsage:
        """Log and return token usage, estimated cost, and latency for one call."""

        total_tokens = usage.input_tokens + usage.output_tokens
        estimated_cost = (
            usage.input_tokens / 1000 * self._settings.llm_cost_per_1k_input_tokens
            + usage.output_tokens / 1000 * self._settings.llm_cost_per_1k_output_tokens
        )
        rounded_cost = round(estimated_cost, 6)

        logger.info(
            "llm_call_completed",
            extra={
                "llm_provider": self._settings.llm_provider,
                "llm_model": self._settings.llm_model,
                "llm_call_kind": kind,
                "llm_input_tokens": usage.input_tokens,
                "llm_output_tokens": usage.output_tokens,
                "llm_total_tokens": total_tokens,
                "llm_estimated_cost_usd": rounded_cost,
                "llm_duration_seconds": round(duration_seconds, 3),
            },
        )

        return LLMUsage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=rounded_cost,
        )
