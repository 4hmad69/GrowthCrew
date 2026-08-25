"""LLM-gateway-specific application exceptions.

Mirrors the naming and layering style of backend/app/db/errors.py: one
base class per subsystem, with specific subclasses for the failure modes
that callers actually need to react to differently.
"""


class LLMGatewayError(RuntimeError):
    """Base class for controlled LLM gateway failures."""


class LLMProviderUnavailableError(LLMGatewayError):
    """Raised when the configured LLM provider cannot be reached."""


class LLMResponseError(LLMGatewayError):
    """Raised when the LLM provider returns an invalid or unusable response."""


class LLMStructuredOutputError(LLMResponseError):
    """Raised when a response cannot be parsed into the requested schema,
    even after fallback attempts."""
