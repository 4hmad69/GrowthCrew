"""Assembles the CRAG graph: nodes + conditional edges + loop limits."""

from langgraph.graph import END, StateGraph

from backend.app.agents.rag.decision_nodes import (
    build_decide_context_need_node,
    build_rewrite_query_node,
    build_select_source_node,
)
from backend.app.agents.rag.generation_nodes import (
    build_generate_direct_node,
    build_generate_grounded_node,
    build_grade_support_node,
    build_grade_usefulness_node,
    build_revise_answer_node,
)
from backend.app.agents.rag.retrieval_nodes import (
    build_grade_documents_node,
    build_retrieve_vectorstore_node,
    build_retrieve_web_node,
)
from backend.app.agents.rag.state import RagState
from backend.app.llm.gateway import LLMGateway
from backend.app.services.retrieval import RetrievalService
from backend.app.websearch.gateway import WebSearchGateway

MAX_RETRIEVAL_ATTEMPTS = 2
MAX_REVISION_ATTEMPTS = 2
MAX_WEB_ATTEMPTS = 1


def build_rag_graph(
    llm_gateway: LLMGateway,
    retrieval: RetrievalService,
    web_search: WebSearchGateway,
):
    """Build and compile the CRAG graph for one request's dependencies.

    Built fresh per invocation rather than once at app startup - `retrieval`
    is bound to a request-scoped database session, so the graph can't be a
    long-lived singleton the way the gateways themselves are.
    """

    graph = StateGraph(RagState)

    graph.add_node("rewrite_query", build_rewrite_query_node(llm_gateway))
    graph.add_node("decide_context_need", build_decide_context_need_node(llm_gateway))
    graph.add_node("select_source", build_select_source_node(llm_gateway))
    graph.add_node("retrieve_vectorstore", build_retrieve_vectorstore_node(retrieval))
    graph.add_node("retrieve_web", build_retrieve_web_node(web_search))
    graph.add_node("grade_documents", build_grade_documents_node(llm_gateway))
    graph.add_node("generate_grounded", build_generate_grounded_node(llm_gateway))
    graph.add_node("generate_direct", build_generate_direct_node(llm_gateway))
    graph.add_node("grade_support", build_grade_support_node(llm_gateway))
    graph.add_node("grade_usefulness", build_grade_usefulness_node(llm_gateway))
    graph.add_node("revise_answer", build_revise_answer_node(llm_gateway))

    graph.set_entry_point("rewrite_query")
    graph.add_edge("rewrite_query", "decide_context_need")

    graph.add_conditional_edges(
        "decide_context_need",
        lambda state: "direct" if state.get("context_route") == "direct" else "select_source",
        {"direct": "generate_direct", "select_source": "select_source"},
    )

    graph.add_conditional_edges(
        "select_source",
        lambda state: state["context_route"],
        {"vectorstore": "retrieve_vectorstore", "web": "retrieve_web"},
    )

    graph.add_edge("retrieve_vectorstore", "grade_documents")
    graph.add_edge("retrieve_web", "grade_documents")

    def _route_after_grading(state: RagState) -> str:
        if state["relevant_documents"]:
            return "generate"
        if state["context_route"] == "vectorstore" and state["web_attempts"] < MAX_WEB_ATTEMPTS:
            return "try_web"
        if state["retrieval_attempts"] < MAX_RETRIEVAL_ATTEMPTS:
            return "retry"
        return "give_up"

    graph.add_conditional_edges(
        "grade_documents",
        _route_after_grading,
        {
            "generate": "generate_grounded",
            "try_web": "retrieve_web",
            "retry": "retrieve_vectorstore",
            "give_up": "generate_direct",
        },
    )

    graph.add_edge("generate_grounded", "grade_support")
    graph.add_edge("generate_direct", END)

    def _route_after_support(state: RagState) -> str:
        if not state["is_grounded"] and state["revision_attempts"] < MAX_REVISION_ATTEMPTS:
            return "revise"
        return "check_usefulness"

    graph.add_conditional_edges(
        "grade_support",
        _route_after_support,
        {"revise": "revise_answer", "check_usefulness": "grade_usefulness"},
    )

    def _route_after_usefulness(state: RagState) -> str:
        if not state["is_useful"] and state["revision_attempts"] < MAX_REVISION_ATTEMPTS:
            return "revise"
        return "finish"

    graph.add_conditional_edges(
        "grade_usefulness",
        _route_after_usefulness,
        {"revise": "revise_answer", "finish": END},
    )

    graph.add_edge("revise_answer", "grade_support")

    return graph.compile()
