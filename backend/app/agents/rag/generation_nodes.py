"""Generation, final grading, and revision nodes for the CRAG graph."""

from collections.abc import Callable

from backend.app.agents.rag.schemas import SupportGrade, UsefulnessGrade
from backend.app.agents.rag.state import RagState
from backend.app.llm.gateway import LLMGateway


def _format_sources(state: RagState) -> str:
    """Render relevant_documents as numbered, citable source blocks."""

    return "\n\n".join(
        f"[Source {i + 1}] ({doc['source']}): {doc['content']}"
        for i, doc in enumerate(state["relevant_documents"])
    )


def build_generate_grounded_node(gateway: LLMGateway) -> Callable[[RagState], dict]:
    """Build the node that generates an answer grounded in relevant_documents."""

    def generate_grounded(state: RagState) -> dict:
        prompt = (
            "Answer the query using only the sources below. Cite sources "
            "inline using [Source N] tags. If the sources don't fully "
            "answer the query, say so rather than guessing.\n\n"
            f"Query: {state['rewritten_query']}\n\n"
            f"Sources:\n{_format_sources(state)}"
        )
        return {
            "generation": gateway.chat(prompt),
            "trace": ["generate_grounded"],
        }

    return generate_grounded


def build_generate_direct_node(gateway: LLMGateway) -> Callable[[RagState], dict]:
    """Build the node that generates an answer without any retrieved context."""

    def generate_direct(state: RagState) -> dict:
        prompt = (
            "Answer the following query directly, using general knowledge.\n\n"
            f"Query: {state['rewritten_query']}"
        )
        return {
            "generation": gateway.chat(prompt),
            "trace": ["generate_direct"],
        }

    return generate_direct


def build_grade_support_node(gateway: LLMGateway) -> Callable[[RagState], dict]:
    """Build the node that checks whether the generation is grounded in its sources."""

    def grade_support(state: RagState) -> dict:
        sources_text = "\n\n".join(doc["content"] for doc in state["relevant_documents"])
        prompt = (
            "Does the following answer make any claims that are NOT "
            "supported by the sources? Answer is_grounded=True only if "
            "every claim is backed by the sources.\n\n"
            f"Sources:\n{sources_text}\n\n"
            f"Answer: {state['generation']}"
        )
        grade = gateway.structured(prompt, SupportGrade)
        return {
            "is_grounded": grade.is_grounded,
            "trace": [f"grade_support:{grade.is_grounded} ({grade.reasoning})"],
        }

    return grade_support


def build_grade_usefulness_node(gateway: LLMGateway) -> Callable[[RagState], dict]:
    """Build the node that checks whether the generation actually answers the query."""

    def grade_usefulness(state: RagState) -> dict:
        prompt = (
            "Does the following answer actually address the query, "
            "completely and directly?\n\n"
            f"Query: {state['original_query']}\n\n"
            f"Answer: {state['generation']}"
        )
        grade = gateway.structured(prompt, UsefulnessGrade)
        return {
            "is_useful": grade.is_useful,
            "trace": [f"grade_usefulness:{grade.is_useful} ({grade.reasoning})"],
        }

    return grade_usefulness


def build_revise_answer_node(gateway: LLMGateway) -> Callable[[RagState], dict]:
    """Build the node that revises the generation based on grading feedback."""

    def revise_answer(state: RagState) -> dict:
        prompt = (
            "The following answer was graded as either not grounded in "
            "its sources or not useful. Revise it - stay strictly within "
            "the sources given, and make sure it directly answers the "
            "query.\n\n"
            f"Query: {state['rewritten_query']}\n\n"
            f"Sources:\n{_format_sources(state)}\n\n"
            f"Previous answer: {state['generation']}"
        )
        return {
            "generation": gateway.chat(prompt),
            "revision_attempts": state["revision_attempts"] + 1,
            "trace": ["revise_answer"],
        }

    return revise_answer
