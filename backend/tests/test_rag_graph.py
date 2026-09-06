"""Tests for the assembled CRAG graph."""

from typing import Any
from uuid import uuid4

from backend.app.agents.rag.graph import build_rag_graph
from backend.app.agents.rag.schemas import (
    ContextNeedDecision,
    QueryRewrite,
    RelevanceGrade,
    SourceSelection,
    SupportGrade,
    UsefulnessGrade,
)
from backend.app.config import Settings
from backend.app.llm.gateway import LLMGateway
from backend.app.websearch.gateway import WebSearchGateway


class _EmptyRetrieval:
    def search(self, workspace_id, query, *, limit=5):
        return []


class _ScriptedGateway:
    """A fake LLMGateway returning scripted structured()/chat() results in order."""

    def __init__(self, structured_queue: dict[type, list[Any]], chat_responses: list[str]) -> None:
        self._queue = {schema: list(values) for schema, values in structured_queue.items()}
        self._chat_responses = list(chat_responses)

    def structured(self, prompt: str, schema: type) -> Any:
        return self._queue[schema].pop(0)

    def chat(self, prompt: str) -> str:
        if len(self._chat_responses) > 1:
            return self._chat_responses.pop(0)
        return self._chat_responses[0]


def _initial_state(query: str = "How much does this cost?") -> dict:
    return {
        "workspace_id": uuid4(),
        "original_query": query,
        "rewritten_query": "",
        "context_route": None,
        "retrieved_documents": [],
        "relevant_documents": [],
        "generation": "",
        "is_grounded": False,
        "is_useful": False,
        "retrieval_attempts": 0,
        "revision_attempts": 0,
        "web_attempts": 0,
        "trace": [],
    }


def test_graph_takes_direct_path_when_no_retrieval_needed() -> None:
    gateway = _ScriptedGateway(
        structured_queue={
            QueryRewrite: [QueryRewrite(rewritten_query="pricing")],
            ContextNeedDecision: [
                ContextNeedDecision(needs_retrieval=False, reasoning="general knowledge")
            ],
        },
        chat_responses=["a direct answer"],
    )

    settings = Settings(environment="test")
    graph = build_rag_graph(gateway, _EmptyRetrieval(), WebSearchGateway(settings))
    result = graph.invoke(_initial_state())

    assert result["generation"] == "a direct answer"
    assert "generate_direct" in result["trace"]
    assert "generate_grounded" not in result["trace"]


def test_graph_generates_grounded_answer_when_documents_are_relevant() -> None:
    gateway = _ScriptedGateway(
        structured_queue={
            QueryRewrite: [QueryRewrite(rewritten_query="pricing")],
            ContextNeedDecision: [
                ContextNeedDecision(needs_retrieval=True, reasoning="needs data")
            ],
            SourceSelection: [SourceSelection(source="vectorstore", reasoning="internal data")],
            RelevanceGrade: [RelevanceGrade(is_relevant=True, reasoning="matches")],
            SupportGrade: [SupportGrade(is_grounded=True, reasoning="fully grounded")],
            UsefulnessGrade: [UsefulnessGrade(is_useful=True, reasoning="answers it")],
        },
        chat_responses=["grounded answer [Source 1]"],
    )

    class _OneChunkRetrieval:
        def search(self, workspace_id, query, *, limit=5):
            return [type("Chunk", (), {"content": "Prices start at $29.", "source": "test:1"})()]

    graph = build_rag_graph(
        gateway, _OneChunkRetrieval(), WebSearchGateway(Settings(environment="test"))
    )
    result = graph.invoke(_initial_state())

    assert result["generation"] == "grounded answer [Source 1]"
    assert result["is_grounded"] is True
    assert result["is_useful"] is True
    assert result["revision_attempts"] == 0


def test_graph_falls_back_to_direct_answer_when_no_relevant_documents_found() -> None:
    """Empty vectorstore and no Tavily key -> retry loop exhausts -> falls back to direct."""

    gateway = LLMGateway(Settings(environment="test", llm_provider="local"))
    web_search = WebSearchGateway(Settings(environment="test"))  # no key -> always empty

    graph = build_rag_graph(gateway, _EmptyRetrieval(), web_search)
    result = graph.invoke(_initial_state())

    assert result["generation"]
    assert "generate_direct" in result["trace"]
    assert result["retrieval_attempts"] >= 1
