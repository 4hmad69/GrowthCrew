"""Typed state for the CRAG-style retrieval graph.

Reusable across any agent that needs iterative, self-correcting retrieval
(Market Research and Competitor Analysis first) - not specific to any one
agent's domain. Ported and adapted from the agentic-rag source repo's
graph state, generalized to carry a workspace_id (retrieval is always
scoped to one business's knowledge base) and trimmed to the fields this
graph's nodes actually read or write.
"""

import operator
from typing import Annotated, Literal, TypedDict
from uuid import UUID


class RetrievedDocument(TypedDict):
    """One retrieved piece of context, carried through the graph."""

    content: str
    source: str


class RagState(TypedDict):
    """The full state threaded through every node in the CRAG graph."""

    workspace_id: UUID
    original_query: str
    rewritten_query: str
    context_route: Literal["direct", "vectorstore", "web"] | None
    retrieved_documents: list[RetrievedDocument]
    relevant_documents: list[RetrievedDocument]
    generation: str
    is_grounded: bool
    is_useful: bool
    retrieval_attempts: int
    revision_attempts: int
    web_attempts: int
    # operator.add as the reducer: each node appends to the trace rather
    # than overwriting it, so the full path through the graph survives
    # even when LangGraph merges partial state updates from a node.
    trace: Annotated[list[str], operator.add]
