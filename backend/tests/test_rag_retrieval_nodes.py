"""Tests for the CRAG graph's retrieval and document-grading nodes."""

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from backend.app.agents.rag.retrieval_nodes import (
    build_grade_documents_node,
    build_retrieve_vectorstore_node,
    build_retrieve_web_node,
)
from backend.app.agents.rag.state import RagState, RetrievedDocument
from backend.app.config import Settings
from backend.app.llm.gateway import LLMGateway
from backend.app.websearch.gateway import WebSearchResult


def _base_state(**overrides: Any) -> RagState:
    state: RagState = {
        "workspace_id": uuid4(),
        "original_query": "How much does this typically cost?",
        "rewritten_query": "typical pricing benchmarks",
        "context_route": "vectorstore",
        "retrieved_documents": [],
        "relevant_documents": [],
        "generation": "",
        "is_grounded": False,
        "is_useful": False,
        "retrieval_attempts": 0,
        "revision_attempts": 0,
        "web_attempts": 0,
        "trace": [],
        "total_input_tokens": 0,
        "total_output_tokens": 0,
    }
    state.update(overrides)
    return state


@dataclass
class _FakeChunk:
    content: str
    source: str


class _FakeRetrievalService:
    def __init__(self, chunks: list[_FakeChunk]) -> None:
        self._chunks = chunks
        self.search_calls: list[tuple] = []

    def search(self, workspace_id, query, *, limit=5):
        self.search_calls.append((workspace_id, query, limit))
        return self._chunks[:limit]


class _FakeWebSearchGateway:
    def __init__(self, results: list[WebSearchResult]) -> None:
        self._results = results
        self.search_calls: list[str] = []

    def search(self, query: str) -> list[WebSearchResult]:
        self.search_calls.append(query)
        return self._results


# --- retrieve_vectorstore ------------------------------------------------------


def test_retrieve_vectorstore_maps_chunks_to_documents() -> None:
    fake_service = _FakeRetrievalService([_FakeChunk(content="c1", source="s1")])
    node = build_retrieve_vectorstore_node(fake_service)

    result = node(_base_state())

    assert result["retrieved_documents"] == [RetrievedDocument(content="c1", source="s1")]
    assert result["retrieval_attempts"] == 1
    assert fake_service.search_calls[0][2] == 5  # limit passed through


def test_retrieve_vectorstore_increments_existing_attempt_count() -> None:
    fake_service = _FakeRetrievalService([])
    node = build_retrieve_vectorstore_node(fake_service)

    result = node(_base_state(retrieval_attempts=2))

    assert result["retrieval_attempts"] == 3


# --- retrieve_web ----------------------------------------------------------------


def test_retrieve_web_maps_results_to_documents() -> None:
    fake_gateway = _FakeWebSearchGateway(
        [WebSearchResult(title="T", url="https://x.com", content="c1")]
    )
    node = build_retrieve_web_node(fake_gateway)

    result = node(_base_state())

    assert result["retrieved_documents"] == [
        RetrievedDocument(content="c1", source="https://x.com")
    ]
    assert result["web_attempts"] == 1


def test_retrieve_web_handles_empty_results() -> None:
    node = build_retrieve_web_node(_FakeWebSearchGateway([]))

    result = node(_base_state())

    assert result["retrieved_documents"] == []


# --- grade_documents -------------------------------------------------------------


def test_grade_documents_grades_each_document_individually() -> None:
    gateway = LLMGateway(Settings(environment="test", llm_provider="local"))
    node = build_grade_documents_node(gateway)
    documents = [
        RetrievedDocument(content="c1", source="s1"),
        RetrievedDocument(content="c2", source="s2"),
    ]

    result = node(_base_state(retrieved_documents=documents))

    # local provider fills bool fields as True, so both should pass
    assert len(result["relevant_documents"]) == 2
    assert "2/2" in result["trace"][0]
    assert isinstance(result["total_input_tokens"], int)
    assert isinstance(result["total_output_tokens"], int)


def test_grade_documents_handles_no_retrieved_documents() -> None:
    gateway = LLMGateway(Settings(environment="test", llm_provider="local"))
    node = build_grade_documents_node(gateway)

    result = node(_base_state(retrieved_documents=[]))

    assert result["relevant_documents"] == []
    assert "0/0" in result["trace"][0]
    assert result["total_input_tokens"] == 0
    assert result["total_output_tokens"] == 0
