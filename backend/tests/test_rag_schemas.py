"""Tests for the CRAG graph's typed state and decision schemas."""

from uuid import uuid4

import pytest
from pydantic import ValidationError

from backend.app.agents.rag.schemas import (
    ContextNeedDecision,
    QueryRewrite,
    RelevanceGrade,
    SourceSelection,
    SupportGrade,
    UsefulnessGrade,
)
from backend.app.agents.rag.state import RagState, RetrievedDocument


def test_rag_state_constructs_with_all_fields() -> None:
    """RagState should accept a fully populated state dict."""

    state: RagState = {
        "workspace_id": uuid4(),
        "original_query": "What are typical prices in this market?",
        "rewritten_query": "typical market pricing benchmarks",
        "context_route": "vectorstore",
        "retrieved_documents": [RetrievedDocument(content="text", source="test:1")],
        "relevant_documents": [],
        "generation": "",
        "is_grounded": False,
        "is_useful": False,
        "retrieval_attempts": 0,
        "revision_attempts": 0,
        "web_attempts": 0,
        "trace": ["rewrite_query"],
        "total_input_tokens": 0,
        "total_output_tokens": 0,
    }

    assert state["context_route"] == "vectorstore"
    assert state["trace"] == ["rewrite_query"]


def test_query_rewrite_rejects_empty_string() -> None:
    with pytest.raises(ValidationError):
        QueryRewrite(rewritten_query="")


def test_query_rewrite_accepts_valid_text() -> None:
    result = QueryRewrite(rewritten_query="market pricing benchmarks")
    assert result.rewritten_query == "market pricing benchmarks"


def test_context_need_decision_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ContextNeedDecision(needs_retrieval=True, reasoning="because", extra_field="nope")


def test_source_selection_rejects_invalid_source() -> None:
    with pytest.raises(ValidationError):
        SourceSelection(source="database", reasoning="because")  # not a valid Literal


def test_source_selection_accepts_valid_sources() -> None:
    assert SourceSelection(source="vectorstore", reasoning="internal data exists").source == (
        "vectorstore"
    )
    assert SourceSelection(source="web", reasoning="needs current data").source == "web"


def test_relevance_grade_requires_reasoning() -> None:
    with pytest.raises(ValidationError):
        RelevanceGrade(is_relevant=True, reasoning="")


def test_support_grade_accepts_valid_input() -> None:
    grade = SupportGrade(is_grounded=False, reasoning="claim not found in retrieved text")
    assert grade.is_grounded is False


def test_usefulness_grade_accepts_valid_input() -> None:
    grade = UsefulnessGrade(is_useful=True, reasoning="directly answers the question")
    assert grade.is_useful is True
