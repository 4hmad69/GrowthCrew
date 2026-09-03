"""The web search gateway: the one thing agents import for live web search.

Wraps Tavily (via langchain-tavily) behind the same provider/retry shape
as LLMGateway and EmbeddingsGateway, with one deliberate difference:
there is no fabricated "local" stub that returns fake search results. A
fake LLM/embeddings response is obviously inert placeholder data; a fake
*web search result* looks exactly like real, current information but
isn't - actively misleading in a way the other two gateways' local stubs
never are. Instead, when no API key is configured, search() returns an
empty result list and is_available() reports False - matching exactly
how the proven agentic-rag repo already treats a missing
TAVILY_API_KEY: route around it, never fabricate results.

The API key is passed explicitly to TavilySearchAPIWrapper rather than
relying on it being present in os.environ - GrowthCrew's own Settings
already resolves .env correctly regardless of working directory (see
config.py's _PROJECT_ROOT_ENV_FILE), so there's no reason to depend on a
second, less reliable mechanism for the same value.
"""

import logging
from dataclasses import dataclass
from typing import Any

from backend.app.config import Settings
from backend.app.llm.retry import invoke_with_retries

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class WebSearchResult:
    """One web search result."""

    title: str
    url: str
    content: str


class WebSearchGateway:
    """The single entry point for live web search."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._tool = self._build_tool()

    def is_available(self) -> bool:
        """Return True if a real Tavily API key is configured."""

        return self._tool is not None

    def search(self, query: str) -> list[WebSearchResult]:
        """Search the web, or return an empty list if no API key is configured."""

        if self._tool is None:
            logger.warning("Web search requested but TAVILY_API_KEY is not configured.")
            return []

        raw_results = invoke_with_retries(
            lambda: self._tool.invoke({"query": query}),
            settings=self._settings,
        )
        return _parse_results(raw_results)

    def _build_tool(self) -> Any:
        """Construct the Tavily tool, or None if no API key is configured."""

        if self._settings.tavily_api_key is None:
            return None

        from langchain_tavily import TavilySearch
        from langchain_tavily.tavily_search import TavilySearchAPIWrapper

        wrapper = TavilySearchAPIWrapper(
            tavily_api_key=self._settings.tavily_api_key.get_secret_value()
        )
        return TavilySearch(
            max_results=self._settings.web_search_max_results,
            api_wrapper=wrapper,
        )


def _parse_results(raw_results: Any) -> list[WebSearchResult]:
    """Parse TavilySearch's response into typed results.

    TavilySearch.invoke() returns a dict with a "results" key (a list of
    dicts with title/url/content/score) - verified directly against the
    installed langchain-tavily package's source, not assumed.
    """

    if isinstance(raw_results, dict):
        entries = raw_results.get("results", [])
    elif isinstance(raw_results, list):
        entries = raw_results
    else:
        entries = []

    results = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        content = entry.get("content") or entry.get("snippet") or ""
        if not content:
            continue
        results.append(
            WebSearchResult(
                title=entry.get("title", ""),
                url=entry.get("url", ""),
                content=content,
            )
        )
    return results
