"""Embeddings-specific application exceptions.

Mirrors backend/app/llm/errors.py's layering style: one base class per
subsystem, specific subclasses for failure modes callers need to react
to differently.
"""


class EmbeddingsError(RuntimeError):
    """Base class for controlled embeddings failures."""


class EmbeddingsProviderUnavailableError(EmbeddingsError):
    """Raised when the configured embeddings provider cannot be reached."""


class EmbeddingsResponseError(EmbeddingsError):
    """Raised when the embeddings provider returns an invalid or unusable response."""
