"""Retrieval and document-grading nodes for the CRAG graph."""

from collections.abc import Callable

from backend.app.agents.rag.schemas import RelevanceGrade
from backend.app.agents.rag.state import RagState, RetrievedDocument
from backend.app.llm.gateway import LLMGateway
from backend.app.services.retrieval import RetrievalService
from backend.app.websearch.gateway import WebSearchGateway

_VECTORSTORE_RESULTS_LIMIT = 5


def build_retrieve_vectorstore_node(retrieval: RetrievalService) -> Callable[[RagState], dict]:
    """Build the node that searches the workspace's pgvector knowledge base."""

    def retrieve_vectorstore(state: RagState) -> dict:
        chunks = retrieval.search(
            state["workspace_id"],
            state["rewritten_query"],
            limit=_VECTORSTORE_RESULTS_LIMIT,
        )
        documents = [
            RetrievedDocument(content=chunk.content, source=chunk.source) for chunk in chunks
        ]
        return {
            "retrieved_documents": documents,
            "retrieval_attempts": state["retrieval_attempts"] + 1,
            "trace": [f"retrieve_vectorstore:{len(documents)} chunks"],
        }

    return retrieve_vectorstore


def build_retrieve_web_node(web_search: WebSearchGateway) -> Callable[[RagState], dict]:
    """Build the node that searches the live web."""

    def retrieve_web(state: RagState) -> dict:
        results = web_search.search(state["rewritten_query"])
        documents = [
            RetrievedDocument(content=result.content, source=result.url) for result in results
        ]
        return {
            "retrieved_documents": documents,
            "web_attempts": state["web_attempts"] + 1,
            "trace": [f"retrieve_web:{len(documents)} results"],
        }

    return retrieve_web


def build_grade_documents_node(gateway: LLMGateway) -> Callable[[RagState], dict]:
    """Build the node that grades each retrieved document's relevance.

    Grades one document at a time rather than the whole batch together -
    a single combined prompt makes it easy for a model to rubber-stamp
    everything as relevant just because *something* in the batch is.
    """

    def grade_documents(state: RagState) -> dict:
        relevant: list[RetrievedDocument] = []
        total_input_tokens = 0
        total_output_tokens = 0
        for document in state["retrieved_documents"]:
            prompt = (
                "Does the following retrieved text help answer the query? "
                "Answer strictly based on whether it's actually relevant.\n\n"
                f"Query: {state['rewritten_query']}\n\n"
                f"Retrieved text: {document['content']}"
            )
            grade, usage = gateway.structured_with_usage(prompt, RelevanceGrade)
            total_input_tokens += usage.input_tokens
            total_output_tokens += usage.output_tokens
            if grade.is_relevant:
                relevant.append(document)

        total = len(state["retrieved_documents"])
        return {
            "relevant_documents": relevant,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "trace": [f"grade_documents:{len(relevant)}/{total} relevant"],
        }

    return grade_documents
