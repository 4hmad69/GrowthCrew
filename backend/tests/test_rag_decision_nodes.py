"""Tests for the CRAG graph's decision nodes."""

from typing import Any
from uuid import uuid4

from backend.app.agents.rag.decision_nodes import (
    build_decide_context_need_node,
    build_rewrite_query_node,
    build_select_source_node,
)
from backend.app.agents.rag.state import RagState
from backend.app.config import Settings
from backend.app.llm.gateway import LLMGateway


def _base_state(**overrides: Any) -> RagState:
    state: RagState = {
        "workspace_id": uuid4(),
        "original_query": "What do people typically pay for this?",
        "rewritten_query": "typical pricing benchmarks",
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
        "total_input_tokens": 0,
        "total_output_tokens": 0,
    }
    state.update(overrides)
    return state


def _local_gateway() -> LLMGateway:
    return LLMGateway(Settings(environment="test", llm_provider="local"))


def test_rewrite_query_returns_rewritten_query_and_trace() -> None:
    node = build_rewrite_query_node(_local_gateway())

    result = node(_base_state())

    assert "rewritten_query" in result
    assert isinstance(result["rewritten_query"], str) and result["rewritten_query"]
    assert result["trace"] == ["rewrite_query"]
    assert isinstance(result["total_input_tokens"], int)
    assert isinstance(result["total_output_tokens"], int)


def test_decide_context_need_does_not_set_context_route_when_retrieval_needed() -> None:
    """LocalStructuredRunnable fills bool fields as True, so needs_retrieval=True here."""

    node = build_decide_context_need_node(_local_gateway())

    result = node(_base_state())

    assert "context_route" not in result
    assert result["trace"][0].startswith("decide_context_need:needs_retrieval")
    assert "total_input_tokens" in result
    assert "total_output_tokens" in result


def test_select_source_sets_context_route_to_a_valid_source() -> None:
    node = build_select_source_node(_local_gateway())

    result = node(_base_state())

    assert result["context_route"] in ("vectorstore", "web")
    assert result["trace"][0].startswith("select_source:")
    assert "total_input_tokens" in result
    assert "total_output_tokens" in result


def test_nodes_only_take_state_as_their_argument() -> None:
    """LangGraph requires node functions to accept only `state` - confirm the factories
    produce exactly that shape, with dependencies already closed over."""

    gateway = _local_gateway()
    for factory in (
        build_rewrite_query_node,
        build_decide_context_need_node,
        build_select_source_node,
    ):
        node = factory(gateway)
        result = node(_base_state())
        assert isinstance(result, dict)
