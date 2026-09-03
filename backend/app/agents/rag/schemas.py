"""LLM-facing decision and grading schemas for the CRAG graph.

Every schema here is the output contract for exactly one graph node's
LLMGateway.structured() call - never exposed via the API, never persisted
directly. Each carries a `reasoning` field: not for display, but because
asking a model to justify a yes/no decision measurably improves the
decision itself (a form of forced chain-of-thought), and it makes a wrong
decision debuggable after the fact.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class QueryRewrite(BaseModel):
    """Output of the query-rewriting node."""

    model_config = ConfigDict(extra="forbid")

    rewritten_query: str = Field(min_length=1, max_length=500)


class ContextNeedDecision(BaseModel):
    """Whether a query can be answered directly or needs retrieval."""

    model_config = ConfigDict(extra="forbid")

    needs_retrieval: bool
    reasoning: str = Field(min_length=1, max_length=500)


class SourceSelection(BaseModel):
    """Which retrieval source to use, when retrieval is needed."""

    model_config = ConfigDict(extra="forbid")

    source: Literal["vectorstore", "web"]
    reasoning: str = Field(min_length=1, max_length=500)


class RelevanceGrade(BaseModel):
    """Whether a single retrieved document is actually relevant to the query."""

    model_config = ConfigDict(extra="forbid")

    is_relevant: bool
    reasoning: str = Field(min_length=1, max_length=500)


class SupportGrade(BaseModel):
    """Whether a generated answer is actually grounded in its retrieved context."""

    model_config = ConfigDict(extra="forbid")

    is_grounded: bool
    reasoning: str = Field(min_length=1, max_length=500)


class UsefulnessGrade(BaseModel):
    """Whether a generated answer actually addresses the original query."""

    model_config = ConfigDict(extra="forbid")

    is_useful: bool
    reasoning: str = Field(min_length=1, max_length=500)
