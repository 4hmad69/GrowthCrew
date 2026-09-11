"""Competitor Analysis agent: business rules and transaction boundaries.

The CRAG graph's second consumer, per Step 8's doc. Same shape as
MarketResearchService: four independent CRAG graph invocations per
workspace, one research question per report section, built from the
workspace's business profile. Where Market Research asks broad
market-level questions, Competitor Analysis zeroes in specifically on
BusinessProfile.known_competitors - and, when none have been named yet,
asks the graph to identify likely competitors via retrieval/web search
rather than skipping the section.

Each section is a full, independent trip through the graph (rewrite,
route, retrieve, grade, generate, grade again, maybe revise) - real but
deliberately not free. force_regenerate defaults to False for exactly
that reason: nothing gets re-run, and no new tokens get spent, just
because a report was asked for again.
"""

from uuid import UUID

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from sqlalchemy.orm.exc import StaleDataError

from backend.app.agents.rag.graph import build_rag_graph
from backend.app.agents.rag.state import RagState
from backend.app.config import Settings
from backend.app.db.errors import DatabaseOperationError
from backend.app.db.models.business_profile import BusinessProfile
from backend.app.db.models.competitor_analysis import CompetitorAnalysis
from backend.app.db.repositories.business_profiles import BusinessProfileRepository
from backend.app.db.repositories.competitor_analysis import CompetitorAnalysisRepository
from backend.app.db.repositories.workspaces import WorkspaceRepository
from backend.app.embeddings.gateway import EmbeddingsGateway
from backend.app.exceptions import ResourceNotFoundError, StaleResourceError
from backend.app.llm.gateway import LLMGateway
from backend.app.services.retrieval import RetrievalService
from backend.app.websearch.gateway import WebSearchGateway

_SNIPPET_MAX_LENGTH = 320


class CompetitorAnalysisService:
    """Coordinate Competitor Analysis generation and persistence."""

    def __init__(
        self,
        session: Session,
        llm_gateway: LLMGateway,
        embeddings_gateway: EmbeddingsGateway,
        web_search_gateway: WebSearchGateway,
        settings: Settings,
    ) -> None:
        self._session = session
        self._llm = llm_gateway
        self._web_search = web_search_gateway
        self._settings = settings
        self._retrieval = RetrievalService(session, embeddings_gateway)
        self._analyses = CompetitorAnalysisRepository(session)
        self._profiles = BusinessProfileRepository(session)
        self._workspaces = WorkspaceRepository(session)

    def generate(
        self,
        workspace_id: UUID,
        *,
        force_regenerate: bool = False,
    ) -> CompetitorAnalysis:
        """Return the current report, generating one if needed.

        If a report already exists and force_regenerate is False, the
        existing record is returned without touching the LLM, retrieval,
        or web search at all - checked *before* requiring a business
        profile, so an already-generated report keeps being readable even
        if the profile it was built from is edited or removed later.
        """

        self._require_workspace(workspace_id)

        existing = self._analyses.get_by_workspace(workspace_id)
        if existing is not None and not force_regenerate:
            return existing

        profile = self._profiles.get_by_workspace(workspace_id)
        if profile is None:
            raise ResourceNotFoundError("Business profile not found.")

        sections, sources, input_tokens, output_tokens = self._run_sections(workspace_id, profile)

        analysis = (
            existing if existing is not None else CompetitorAnalysis(workspace_id=workspace_id)
        )
        self._apply_sections(analysis, sections, sources, input_tokens, output_tokens)

        if existing is None:
            self._analyses.add(analysis)

        try:
            self._session.commit()
        except StaleDataError as exc:
            self._session.rollback()
            raise StaleResourceError(
                "Competitor analysis changed while it was being regenerated."
            ) from exc
        except SQLAlchemyError as exc:
            self._session.rollback()
            raise DatabaseOperationError("Competitor analysis generation failed.") from exc

        self._session.refresh(analysis)
        return analysis

    def get(self, workspace_id: UUID) -> CompetitorAnalysis:
        """Return the current competitor analysis report for a workspace.

        Deliberately does not require a business profile to still exist -
        a generated report is a self-contained, already-written document,
        not a live view over the profile it was originally built from.
        """

        self._require_workspace(workspace_id)

        analysis = self._analyses.get_by_workspace(workspace_id)
        if analysis is None:
            raise ResourceNotFoundError("Competitor analysis has not been generated yet.")

        return analysis

    def _require_workspace(self, workspace_id: UUID) -> None:
        """Ensure the parent workspace exists."""

        workspace = self._workspaces.get(workspace_id)
        if workspace is None:
            raise ResourceNotFoundError("Workspace not found.")

    def _run_sections(
        self,
        workspace_id: UUID,
        profile: BusinessProfile,
    ) -> tuple[dict[str, str], list[dict[str, str]], int, int]:
        """Run one CRAG graph invocation per research question.

        The graph is compiled once and invoked four times - reusing it
        across sections within the same request is fine (each invoke()
        gets its own fresh state), it's only shared *across* requests
        that would be wrong, since retrieval is bound to this session.
        """

        graph = build_rag_graph(self._llm, self._retrieval, self._web_search)

        sections: dict[str, str] = {}
        sources: list[dict[str, str]] = []
        total_input_tokens = 0
        total_output_tokens = 0

        for section, query in self._build_queries(profile).items():
            result = graph.invoke(self._initial_state(workspace_id, query))
            sections[section] = result["generation"]
            total_input_tokens += result["total_input_tokens"]
            total_output_tokens += result["total_output_tokens"]

            documents = result["relevant_documents"] or result["retrieved_documents"]
            for document in documents:
                sources.append(
                    {
                        "section": section,
                        "source": document["source"],
                        "snippet": _snippet(document["content"]),
                    }
                )

        return sections, sources, total_input_tokens, total_output_tokens

    def _build_queries(self, profile: BusinessProfile) -> dict[str, str]:
        """Build one targeted research question per report section.

        Every section is centered on profile.known_competitors. When none
        have been named yet, the questions ask the graph to identify the
        most likely direct competitors itself (via retrieval and/or web
        search) rather than producing an empty or generic section.
        """

        industry = profile.industry or "its general industry"
        product = profile.product_or_service or profile.business_name
        country = profile.country or "its primary market"
        target_customer = profile.target_customer or "its typical customer"
        price_range = profile.price_range or "an unspecified price range"
        brand_tone = profile.brand_tone or "an unspecified brand tone"
        goal = profile.main_marketing_goal or "growing awareness and demand"

        if profile.known_competitors:
            competitors = ", ".join(profile.known_competitors)
            competitor_clause = f"'{profile.business_name}''s known competitors ({competitors})"
        else:
            competitor_clause = (
                f"the most likely direct competitors of '{profile.business_name}' "
                f"(none have been named yet, so identify them for '{industry}' "
                f"businesses offering '{product}' in {country})"
            )

        return {
            "competitor_overview": (
                f"Provide a detailed overview of {competitor_clause} in the "
                f"'{industry}' industry: what each one offers, roughly how "
                "large they are, and how they are positioned in the market."
            ),
            "strengths_and_weaknesses": (
                f"Compared to '{profile.business_name}' (which offers "
                f"'{product}' to {target_customer}), what are the key "
                f"strengths and weaknesses of {competitor_clause}?"
            ),
            "pricing_and_positioning": (
                f"How do {competitor_clause} price and position their "
                f"offerings, and how does that compare to '{profile.business_name}''s "
                f"price range of {price_range} and brand tone of {brand_tone}?"
            ),
            "differentiation_opportunities": (
                f"Given {competitor_clause} and a main marketing goal of "
                f"'{goal}', what concrete gaps or opportunities can "
                f"'{profile.business_name}' exploit to differentiate itself?"
            ),
        }

    def _initial_state(self, workspace_id: UUID, query: str) -> RagState:
        """Build a fresh CRAG graph state for one research question."""

        return {
            "workspace_id": workspace_id,
            "original_query": query,
            "rewritten_query": "",
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

    def _apply_sections(
        self,
        analysis: CompetitorAnalysis,
        sections: dict[str, str],
        sources: list[dict[str, str]],
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Copy generated section text plus usage metadata onto an ORM record."""

        analysis.competitor_overview = sections["competitor_overview"]
        analysis.strengths_and_weaknesses = sections["strengths_and_weaknesses"]
        analysis.pricing_and_positioning = sections["pricing_and_positioning"]
        analysis.differentiation_opportunities = sections["differentiation_opportunities"]
        analysis.sources = sources
        analysis.model_used = self._settings.llm_model
        analysis.input_tokens = input_tokens
        analysis.output_tokens = output_tokens


def _snippet(content: str) -> str:
    """Truncate a retrieved chunk's content to a citation-sized snippet."""

    if len(content) <= _SNIPPET_MAX_LENGTH:
        return content
    return content[:_SNIPPET_MAX_LENGTH].rstrip() + "..."
