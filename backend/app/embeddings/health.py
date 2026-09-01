"""Embeddings gateway readiness checks."""

from dataclasses import dataclass
from typing import Literal, Protocol

from backend.app.config import Settings
from backend.app.embeddings.errors import EmbeddingsResponseError
from backend.app.embeddings.gateway import EmbeddingsGateway


@dataclass(frozen=True, slots=True)
class EmbeddingsHealthSnapshot:
    """Internal result of a successful embeddings gateway readiness check."""

    provider: str
    model: str
    dimension: int
    reachable: Literal[True] = True


class EmbeddingsHealthChecker(Protocol):
    """Contract used by the API and test doubles."""

    def check(self) -> EmbeddingsHealthSnapshot:
        """Return readiness details or raise a controlled embeddings error."""

        ...


class EmbeddingsHealthService:
    """Confirm the configured embeddings provider actually responds correctly."""

    def __init__(self, gateway: EmbeddingsGateway, settings: Settings) -> None:
        self._gateway = gateway
        self._settings = settings

    def check(self) -> EmbeddingsHealthSnapshot:
        """Embed a minimal string and confirm the returned dimension is correct.

        Skipped for the "local" provider - there's no real backend to
        reach. The dimension check matters specifically for embeddings
        (unlike chat, which has no fixed-width response): the pgvector
        column is a fixed width set at migration time, so a provider or
        model change that silently returns a different dimension would
        otherwise only surface as a confusing insert failure later.
        """

        if self._settings.embeddings_provider != "local":
            vector = self._gateway.embed_query("readiness check")
            if len(vector) != self._settings.embeddings_dimension:
                raise EmbeddingsResponseError(
                    f"Expected {self._settings.embeddings_dimension}-dimensional "
                    f"embeddings, got {len(vector)}."
                )

        return EmbeddingsHealthSnapshot(
            provider=self._settings.embeddings_provider,
            model=self._settings.embeddings_model,
            dimension=self._settings.embeddings_dimension,
        )
