"""Market Research agent: business rules and transaction boundaries.

Runs four independent CRAG graph invocations per workspace - one research
question per report section - built from the workspace's business
profile, mirroring BusinessUnderstandingService's transaction-boundary
pattern. Unlike Business Understanding, this agent's queries genuinely
need to go find external information (market data, competitor
information, current trends), which is exactly what Step 8's CRAG graph
was built for: self-correcting retrieval across the workspace's own
knowledge base and/or a live web search, not a single unsourced guess.

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
from backend.app.db.models.market_research import MarketResearch
from backend.app.db.repositories.business_profiles import BusinessProfileRepository
from backend.app.db.repositories.market_research import MarketResearchRepository
from backend.app.db.repositories.workspaces import WorkspaceRepository
from backend.app.embeddings.gateway import EmbeddingsGateway
from backend.app.exceptions import ResourceNotFoundError, StaleResourceError
from backend.app.llm.gateway import LLMGateway
from backend.app.services.retrieval import RetrievalService
from backend.app.websearch.gateway import WebSearchGateway

_SNIPPET_MAX_LENGTH = 320


class MarketResearchService:
    """Coordinate Market Research generation and persistence."""

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
        self._research = MarketResearchRepository(session)
        self._profiles = BusinessProfileRepository(session)
        self._workspaces = WorkspaceRepository(session)

    def generate(
        self,
        workspace_id: UUID,
        *,
        force_regenerate: bool = False,
    ) -> MarketResearch:
        """Return the current report, generating one if needed.

        If a report already exists and force_regenerate is False, the
        existing record is returned without touching the LLM, retrieval,
        or web search at all - checked *before* requiring a business
        profile, so an already-generated report keeps being readable even
        if the profile it was built from is edited or removed later.
        """

        self._require_workspace(workspace_id)

        existing = self._research.get_by_workspace(workspace_id)
        if existing is not None and not force_regenerate:
            return existing

        profile = self._profiles.get_by_workspace(workspace_id)
        if profile is None:
            raise ResourceNotFoundError("Business profile not found.")

        sections, sources, input_tokens, output_tokens = self._run_sections(workspace_id, profile)

        research = existing if existing is not None else MarketResearch(workspace_id=workspace_id)
        self._apply_sections(research, sections, sources, input_tokens, output_tokens)

        if existing is None:
            self._research.add(research)

        try:
            self._session.commit()
        except StaleDataError as exc:
            self._session.rollback()
            raise StaleResourceError(
                "Market research changed while it was being regenerated."
            ) from exc
        except SQLAlchemyError as exc:
            self._session.rollback()
            raise DatabaseOperationError("Market research generation failed.") from exc

        self._session.refresh(research)
        return research

    def get(self, workspace_id: UUID) -> MarketResearch:
        """Return the current market research report for a workspace.

        Deliberately does not require a business profile to still exist -
        a generated report is a self-contained, already-written document,
        not a live view over the profile it was originally built from.
        """

        self._require_workspace(workspace_id)

        research = self._research.get_by_workspace(workspace_id)
        if research is None:
            raise ResourceNotFoundError("Market research has not been generated yet.")

        return research

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
        """Build one targeted research question per report section."""

        industry = profile.industry or "its general industry"
        product = profile.product_or_service or profile.business_name
        country = profile.country or "its primary market"
        target_market = profile.target_market or "a broad audience"
        target_customer = profile.target_customer or "its typical customer"
        competitors = (
            ", ".join(profile.known_competitors)
            if profile.known_competitors
            else "no specific competitors named yet"
        )
        budget = (
            f"{profile.monthly_marketing_budget} {profile.marketing_budget_currency} per month"
            if profile.monthly_marketing_budget is not None
            else "an unspecified budget"
        )
        goal = profile.main_marketing_goal or "growing awareness and demand"
        challenges = profile.current_challenges or "no specific challenges named yet"

        return {
            "market_overview": (
                "What is the current market size, growth rate, and key "
                f"trends for the '{industry}' industry, particularly as it "
                f"relates to '{product}' in {country}?"
            ),
            "target_customer_segments": (
                "Who are the ideal target customer segments for "
                f"'{profile.business_name}', given its target market "
                f"('{target_market}') and target customer "
                f"('{target_customer}')? Include demographic and "
                "behavioral detail."
            ),
            "competitive_landscape": (
                f"Who are '{profile.business_name}''s main competitors "
                f"(known so far: {competitors}), and how is the "
                f"competitive landscape structured in the '{industry}' "
                "space?"
            ),
            "opportunities_and_risks": (
                f"Given a marketing budget of {budget}, a main marketing "
                f"goal of '{goal}', and current challenges of "
                f"'{challenges}', what marketing opportunities and risks "
                f"should '{profile.business_name}' consider?"
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
        research: MarketResearch,
        sections: dict[str, str],
        sources: list[dict[str, str]],
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Copy generated section text plus usage metadata onto an ORM record."""

        research.market_overview = sections["market_overview"]
        research.target_customer_segments = sections["target_customer_segments"]
        research.competitive_landscape = sections["competitive_landscape"]
        research.opportunities_and_risks = sections["opportunities_and_risks"]
        research.sources = sources
        research.model_used = self._settings.llm_model
        research.input_tokens = input_tokens
        research.output_tokens = output_tokens


def _snippet(content: str) -> str:
    """Truncate a retrieved chunk's content to a citation-sized snippet."""

    if len(content) <= _SNIPPET_MAX_LENGTH:
        return content
    return content[:_SNIPPET_MAX_LENGTH].rstrip() + "..."
