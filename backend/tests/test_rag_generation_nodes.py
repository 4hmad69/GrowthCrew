"""Tests for the CRAG graph's generation, final grading, and revision nodes."""

from typing import Any
from uuid import uuid4

from backend.app.agents.rag.generation_nodes import (
    build_generate_direct_node,
    build_generate_grounded_node,
    build_grade_support_node,
    build_grade_usefulness_node,
    build_revise_answer_node,
)
from backend.app.agents.rag.state import RagState, RetrievedDocument
from backend.app.config import Settings
from backend.app.llm.gateway import LLMGateway


def _base_state(**overrides: Any) -> RagState:
    state: RagState = {
        "workspace_id": uuid4(),
        "original_query": "How much does this typically cost?",
        "rewritten_query": "typical pricing benchmarks",
        "context_route": "vectorstore",
        "retrieved_documents": [],
        "relevant_documents": [
            RetrievedDocument(content="Prices start at $29/month.", source="test:1"),
        ],
        "generation": "",
        "is_grounded": False,
        "is_useful": False,
        "retrieval_attempts": 0,
        "revision_attempts": 0,
        "web_attempts": 0,
        "trace": [],
    }
    state.update(overrides)
    return state


def _local_gateway() -> LLMGateway:
    return LLMGateway(Settings(environment="test", llm_provider="local"))


def test_generate_grounded_returns_generation_and_trace() -> None:
    node = build_generate_grounded_node(_local_gateway())

    result = node(_base_state())

    assert isinstance(result["generation"], str) and result["generation"]
    assert result["trace"] == ["generate_grounded"]


def test_generate_grounded_handles_no_relevant_documents() -> None:
    node = build_generate_grounded_node(_local_gateway())

    result = node(_base_state(relevant_documents=[]))

    assert isinstance(result["generation"], str)


def test_generate_direct_returns_generation_and_trace() -> None:
    node = build_generate_direct_node(_local_gateway())

    result = node(_base_state())

    assert isinstance(result["generation"], str) and result["generation"]
    assert result["trace"] == ["generate_direct"]


def test_grade_support_returns_boolean_and_trace() -> None:
    node = build_grade_support_node(_local_gateway())

    result = node(_base_state(generation="Prices start at $29/month [Source 1]."))

    assert isinstance(result["is_grounded"], bool)
    assert result["trace"][0].startswith("grade_support:")


def test_grade_usefulness_returns_boolean_and_trace() -> None:
    node = build_grade_usefulness_node(_local_gateway())

    result = node(_base_state(generation="Prices start at $29/month."))

    assert isinstance(result["is_useful"], bool)
    assert result["trace"][0].startswith("grade_usefulness:")


def test_revise_answer_increments_revision_attempts() -> None:
    node = build_revise_answer_node(_local_gateway())

    result = node(_base_state(generation="an earlier draft", revision_attempts=1))

    assert result["revision_attempts"] == 2
    assert isinstance(result["generation"], str) and result["generation"]
    assert result["trace"] == ["revise_answer"]
