"""Structured-output robustness for LLM calls.

Ported and adapted from https://github.com/4hmad69/agentic-rag
(agentic_rag/backends.py: RobustStructuredRunnable, LocalStructuredRunnable,
json_only_prompt, parse_json_object, coerce_structured_output).

One deliberate departure from the source repo: there, when both structured-
output attempts failed, the runnable silently fell back to a rule-based
guess at the answer (LocalStructuredRunnable), keyed off the schema's
class name. That's a reasonable choice for a small demo with a handful of
known schemas, but it doesn't generalize - GrowthCrew will have many
agent-specific schemas this module has no way to know about in advance,
and returning a plausible-looking guess instead of a real answer is worse
than failing loudly. So here, exhausting both structured-output attempts
raises LLMStructuredOutputError instead. The schema-agnostic local
fallback still exists (LocalStructuredRunnable below), but only as the
deliberate "local" test/offline provider path (see the llm_provider
setting in backend/app/config.py) - never as a silent in-flight
substitute for a real answer.

This module has no hard dependency on langchain: every function accepts
anything with a LangChain-style .invoke(input, config=None, **kwargs)
method, so it can be tested without the real provider libraries installed.
"""

import json
from collections.abc import Iterable
from typing import Any, Protocol

from backend.app.config import Settings
from backend.app.llm.errors import LLMStructuredOutputError
from backend.app.llm.retry import invoke_with_retries


class SupportsInvoke(Protocol):
    """Anything with a LangChain-style .invoke(input, config=None, **kwargs)."""

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any: ...


class RobustStructuredRunnable:
    """Wraps a structured-output runnable with a JSON-only-prompt fallback.

    Tries the provider's native structured-output mode first (with retry
    on transient failures). If that fails to parse or validate, falls back
    to asking the raw chat model for JSON matching the schema and parses
    that instead. If both attempts fail, raises LLMStructuredOutputError -
    the caller decides what to do next, rather than silently receiving a
    plausible-looking but fabricated result.
    """

    def __init__(
        self,
        structured_runnable: SupportsInvoke,
        chat_model: SupportsInvoke,
        schema: type,
        settings: Settings,
    ) -> None:
        self._structured_runnable = structured_runnable
        self._chat_model = chat_model
        self._schema = schema
        self._settings = settings

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        """Return a validated instance of `schema`, or raise LLMStructuredOutputError."""

        native_error: Exception
        try:
            result = invoke_with_retries(
                lambda: self._structured_runnable.invoke(input, config=config, **kwargs),
                settings=self._settings,
            )
            return coerce_structured_output(result, self._schema)
        except Exception as exc:  # native structured output fails in provider-specific ways
            native_error = exc

        try:
            prompt = json_only_prompt(input, self._schema)
            raw = invoke_with_retries(
                lambda: self._chat_model.invoke(
                    prompt,
                    config=config,
                    options={"num_predict": self._settings.llm_num_predict},
                ),
                settings=self._settings,
            )
            data = parse_json_object(getattr(raw, "content", raw))
            return coerce_structured_output(data, self._schema)
        except Exception as fallback_error:
            raise LLMStructuredOutputError(
                f"Could not produce valid structured output for {self._schema.__name__}: "
                f"native structured-output call failed ({type(native_error).__name__}: "
                f"{native_error}), and the JSON-only fallback also failed."
            ) from fallback_error


class LocalStructuredRunnable:
    """Schema-agnostic structured output for the "local" test/offline provider.

    Fills every field of `schema` with a type-appropriate placeholder value
    instead of calling any LLM. Used only when Settings.llm_provider is
    "local" - for tests and offline development, never as a silent
    production fallback.
    """

    def __init__(self, schema: type) -> None:
        self._schema = schema

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        """Build a placeholder instance of `schema` without calling any model."""

        text = stringify_messages(input)
        fields = getattr(self._schema, "model_fields", {})
        values: dict[str, Any] = {}

        for name, field in fields.items():
            annotation = str(field.annotation).lower()
            if "bool" in annotation:
                values[name] = True
            elif "list" in annotation:
                values[name] = []
            elif "int" in annotation:
                values[name] = 0
            elif "float" in annotation:
                values[name] = 0.0
            else:
                values[name] = _compact_text(text)

        return self._schema(**values)


def json_only_prompt(input: Any, schema: type) -> str:
    """Build a prompt asking the model to return raw JSON matching `schema`."""

    schema_spec = schema.model_json_schema() if hasattr(schema, "model_json_schema") else schema

    return (
        "Return only one valid JSON object that matches the schema. "
        "Do not include markdown, code fences, explanations, or extra text.\n\n"
        f"Schema:\n{json.dumps(schema_spec, ensure_ascii=True)}\n\n"
        f"Task:\n{stringify_messages(input)}"
    )


def parse_json_object(value: Any) -> dict[str, Any]:
    """Extract the first valid JSON object embedded anywhere in `value`."""

    if isinstance(value, dict):
        return value

    text = str(value).strip()
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "{[":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    raise ValueError(f"No JSON object found in model output: {text[:200]}")


def coerce_structured_output(result: Any, schema: type) -> Any:
    """Validate/convert a raw result into an instance of `schema`."""

    if isinstance(result, schema):
        return result
    if hasattr(schema, "model_validate"):
        return schema.model_validate(result)
    return result


def stringify_messages(input: Any) -> str:
    """Render a LangChain-style prompt input (str/dict/message list) as plain text."""

    if isinstance(input, str):
        return input
    if isinstance(input, dict):
        return "\n".join(f"{key}: {value}" for key, value in input.items())
    if isinstance(input, Iterable):
        parts = []
        for item in input:
            content = getattr(item, "content", None)
            parts.append(content if content is not None else str(item))
        return "\n".join(parts)
    return str(input)


def _compact_text(text: str, max_words: int = 16) -> str:
    """Return a short, cleaned-up snippet of `text` for placeholder string fields."""

    words = [
        word.strip(".,?!:;()[]{}\"'")
        for word in text.replace("\n", " ").split()
        if len(word.strip(".,?!:;()[]{}\"'")) > 2
    ]
    return " ".join(words[:max_words])
