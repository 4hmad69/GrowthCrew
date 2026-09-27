"""LLM-facing decision and grading schemas for the CRAG graph.

Every schema here is the output contract for exactly one graph node's
LLMGateway.structured() call - never exposed via the API, never persisted
directly. Each carries a `reasoning` field: not for display, but because
asking a model to justify a yes/no decision measurably improves the
decision itself (a form of forced chain-of-thought), and it makes a wrong
decision debuggable after the fact.

`reasoning` is capped at 2000 characters, not 500. A real Ollama Cloud
run against gpt-oss:120b-cloud hit exactly this: `decide_context_need`'s
native structured-output call returned prose instead of JSON at all
(an OutputParserException), and the JSON-only fallback then produced
valid JSON whose `reasoning` exceeded a 500-char cap - both attempts
failing on the same call, in two different ways, raised
LLMStructuredOutputError all the way up to a 503. None of the five
prompts that ask for one of these schemas puts any limit on how long the
model's reasoning should be, and a large model given no length guidance
will sometimes write several sentences of justification. 2000 characters
is a generous margin above what was actually observed, not a precisely
measured bound - the corresponding prompts also now ask for reasoning
"in one or two sentences" to make hitting even this generous cap rare,
but the schema itself is the actual safety net, since a prompt asking
for brevity is a request, not a guarantee.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

_REASONING_MAX_LENGTH = 2000


class QueryRewrite(BaseModel):
    """Output of the query-rewriting node."""

    model_config = ConfigDict(extra="forbid")

    rewritten_query: str = Field(min_length=1, max_length=500)


class ContextNeedDecision(BaseModel):
    """Whether a query can be answered directly or needs retrieval."""

    model_config = ConfigDict(extra="forbid")

    needs_retrieval: bool
    reasoning: str = Field(min_length=1, max_length=_REASONING_MAX_LENGTH)


class SourceSelection(BaseModel):
    """Which retrieval source to use, when retrieval is needed."""

    model_config = ConfigDict(extra="forbid")

    source: Literal["vectorstore", "web"]
    reasoning: str = Field(min_length=1, max_length=_REASONING_MAX_LENGTH)


class RelevanceGrade(BaseModel):
    """Whether a single retrieved document is actually relevant to the query."""

    model_config = ConfigDict(extra="forbid")

    is_relevant: bool
    reasoning: str = Field(min_length=1, max_length=_REASONING_MAX_LENGTH)


class SupportGrade(BaseModel):
    """Whether a generated answer is actually grounded in its retrieved context."""

    model_config = ConfigDict(extra="forbid")

    is_grounded: bool
    reasoning: str = Field(min_length=1, max_length=_REASONING_MAX_LENGTH)


class UsefulnessGrade(BaseModel):
    """Whether a generated answer actually addresses the original query."""

    model_config = ConfigDict(extra="forbid")

    is_useful: bool
    reasoning: str = Field(min_length=1, max_length=_REASONING_MAX_LENGTH)
