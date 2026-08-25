"""LLM gateway readiness checks."""

from dataclasses import dataclass
from typing import Literal, Protocol

from backend.app.config import Settings
from backend.app.llm.gateway import LLMGateway


@dataclass(frozen=True, slots=True)
class LLMHealthSnapshot:
    """Internal result of a successful LLM gateway readiness check."""

    provider: str
    model: str
    reachable: Literal[True] = True


class LLMHealthChecker(Protocol):
    """Contract used by the API and test doubles."""

    def check(self) -> LLMHealthSnapshot:
        """Return readiness details or raise a controlled LLM gateway error."""

        ...


class LLMHealthService:
    """Confirm the configured LLM provider actually responds."""

    def __init__(self, gateway: LLMGateway, settings: Settings) -> None:
        self._gateway = gateway
        self._settings = settings

    def check(self) -> LLMHealthSnapshot:
        """Send a minimal prompt and confirm the provider responds.

        Skipped for the "local" provider - there's no real backend to
        reach, so the check would be checking nothing.
        """

        if self._settings.llm_provider != "local":
            self._gateway.chat("Reply with only the single word: ready")

        return LLMHealthSnapshot(
            provider=self._settings.llm_provider,
            model=self._settings.llm_model,
        )
