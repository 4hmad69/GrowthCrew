"""Decision nodes for the CRAG graph: rewrite, route, select source.

Each node is built via a small factory function that closes over the
LLMGateway it needs - LangGraph node functions take only `state` as
their argument, so dependencies get injected at graph-assembly time
(Commit 6) rather than read from a global.

context_route is only ever set here to "direct" - when retrieval isn't
needed at all. It's deliberately left unset otherwise, so the graph's
conditional edge (built in Commit 6) can tell "no retrieval needed" apart
from "retrieval needed, source not chosen yet" and route to
select_source next.
"""

from collections.abc import Callable

from backend.app.agents.rag.schemas import ContextNeedDecision, QueryRewrite, SourceSelection
from backend.app.agents.rag.state import RagState
from backend.app.llm.gateway import LLMGateway


def build_rewrite_query_node(gateway: LLMGateway) -> Callable[[RagState], dict]:
    """Build the node that rewrites the original query into a clearer form."""

    def rewrite_query(state: RagState) -> dict:
        prompt = (
            "Rewrite the following question into a clear, self-contained "
            "search query. Keep it concise.\n\n"
            f"Question: {state['original_query']}"
        )
        result = gateway.structured(prompt, QueryRewrite)
        return {
            "rewritten_query": result.rewritten_query,
            "trace": ["rewrite_query"],
        }

    return rewrite_query


def build_decide_context_need_node(gateway: LLMGateway) -> Callable[[RagState], dict]:
    """Build the node that decides whether retrieval is needed at all."""

    def decide_context_need(state: RagState) -> dict:
        prompt = (
            "Decide whether answering this query requires retrieving "
            "additional information, or whether it can be answered directly "
            "from general knowledge.\n\n"
            f"Query: {state['rewritten_query']}"
        )
        result = gateway.structured(prompt, ContextNeedDecision)

        if not result.needs_retrieval:
            return {
                "context_route": "direct",
                "trace": [f"decide_context_need:direct ({result.reasoning})"],
            }

        return {"trace": [f"decide_context_need:needs_retrieval ({result.reasoning})"]}

    return decide_context_need


def build_select_source_node(gateway: LLMGateway) -> Callable[[RagState], dict]:
    """Build the node that picks vectorstore vs web, once retrieval is needed."""

    def select_source(state: RagState) -> dict:
        prompt = (
            "Decide whether this query is best answered from the business's "
            "own internal knowledge base (vectorstore) or from a live web "
            "search - use web search only for current events, live pricing, "
            "or anything unlikely to already be captured internally.\n\n"
            f"Query: {state['rewritten_query']}"
        )
        result = gateway.structured(prompt, SourceSelection)
        return {
            "context_route": result.source,
            "trace": [f"select_source:{result.source} ({result.reasoning})"],
        }

    return select_source
