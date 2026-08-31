"""The embeddings gateway: the one thing agents import for embedding text.

Mirrors backend/app/llm/gateway.py's shape - provider selection, retry via
the same invoke_with_retries used for chat calls (same underlying Ollama
daemon, same transient-failure profile, so no separate retry settings),
and a deterministic "local" stub for tests/offline development.

Shares llm_base_url with the chat gateway rather than introducing a
separate embeddings base URL - both chat and embeddings go through the
same local Ollama daemon, matching the proven agentic-rag config where
OLLAMA_BASE_URL was never split between the two.
"""

import hashlib
import math
import random
from typing import Any

from backend.app.config import Settings
from backend.app.llm.retry import invoke_with_retries


class EmbeddingsGateway:
    """The single entry point for turning text into vectors."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = self._build_client()

    def embed_query(self, text: str) -> list[float]:
        """Embed a single piece of query text.

        For the "ollama" provider, prepends "search_query: " -
        nomic-embed-text's own model card documents this as required, not
        optional: embeddings without the task-instruction prefix use the
        wrong training objective and retrieval quality degrades
        significantly. The "local" stub provider has no such requirement.
        """

        if self._settings.embeddings_provider == "local":
            return _hash_embedding(text, self._settings.embeddings_dimension)

        return invoke_with_retries(
            lambda: self._client.embed_query(f"search_query: {text}"),
            settings=self._settings,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed several pieces of text in one batch call.

        For the "ollama" provider, prepends "search_document: " to each
        text - the indexing-side counterpart to embed_query()'s
        "search_query: " prefix, per nomic-embed-text's documented usage.
        """

        if self._settings.embeddings_provider == "local":
            return [_hash_embedding(text, self._settings.embeddings_dimension) for text in texts]

        prefixed = [f"search_document: {text}" for text in texts]
        return invoke_with_retries(
            lambda: self._client.embed_documents(prefixed),
            settings=self._settings,
        )

    def _build_client(self) -> Any:
        """Construct the underlying provider client based on embeddings_provider."""

        if self._settings.embeddings_provider == "local":
            return None

        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=self._settings.embeddings_model,
            base_url=self._settings.llm_base_url,
            client_kwargs={"timeout": self._settings.embeddings_request_timeout_seconds},
        )


def _hash_embedding(text: str, dimension: int) -> list[float]:
    """Deterministic, dependency-free embedding for the "local" test/offline provider.

    The same text always produces the same vector; different text produces
    a meaningfully different one (verified: near-zero cosine similarity
    for unrelated strings) - enough to exercise real similarity-search
    logic in tests without calling any real embedding model. Seeds a PRNG
    from a SHA-256 digest rather than reinterpreting hash bytes directly
    as floats, which can produce NaN/Inf for certain bit patterns.
    """

    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest(), "big")
    rng = random.Random(seed)
    values = [rng.uniform(-1.0, 1.0) for _ in range(dimension)]
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]
