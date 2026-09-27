"""Marketing Strategy agent: business rules and transaction boundaries.

The first agent whose research questions are not built from the business
profile alone. Each of its four sections is a CRAG graph run seeded with
the *outputs* of Business Understanding, Market Research, Competitor
Analysis, Customer Personas, and Brand Strategy, layered on top of the
profile's own planning fields (main_marketing_goal, existing_channels,
monthly marketing budget, current_challenges). generate() hard-requires
all five prior reports to already exist for the workspace: a strategy
synthesized from a partially missing research base was judged a worse
product outcome than asking the user to finish the earlier steps first,
and it keeps query-building much simpler.

Checks run in dependency order - profile, Business Understanding, Market
Research, Competitor Analysis, Customer Personas, Brand Strategy - which
is also the order every prior report became available to a client
working through the roadmap: Personas is itself downstream of Market
Research, and Brand Strategy is downstream of Competitor Analysis and
Personas (see BrandStrategyService's own docstring). Market Research and
Competitor Analysis stay as direct, independent checks here because this
service still reads their fields itself (target_customer_segments,
opportunities_and_risks, differentiation_opportunities,
strengths_and_weaknesses) - unlike Brand Strategy, which only needs
Personas to exist and never re-checks Market Research directly.

Deliberately does NOT paste each prior report's full section text into a
query verbatim. Two of the graph's own nodes make that counterproductive:
rewrite_query is explicitly instructed to compress whatever it's given
into "a clear, self-contained search query - keep it concise", and
generate_grounded/generate_direct only ever see rewritten_query plus
retrieved sources, never original_query itself - so anything not
distilled by the rewrite step never reaches the final answer anyway.
Instead, each query folds in the prior reports' *structured* fields
(inferred_business_stage, competitive_category, key_differentiators,
brand_voice_and_tone, brand_pillars) plus a short excerpt (see _excerpt)
of the one or two most relevant narrative sections per question - the
same discipline CompetitorAnalysisService already uses for
known_competitors, extended to richer inputs. Customer Personas'
preferred_channels and pain_points are folded in the same way, but
aggregated across every persona in the set (deduplicated, comma-joined)
rather than excerpted, since they are short structured list fields, not
narrative text - see _aggregate_persona_field. That aggregation must
tolerate every persona's list fields being empty: the deterministic
"local" LLM provider used in Postgres-only tests fills every list-typed
field with [] regardless of schema (see Step 14's min_length note), so a
workspace built entirely on local-stub prerequisites has personas with
no preferred_channels or pain_points at all, and query-building must not
crash on that - it falls back to a short placeholder phrase instead.

Each section is a full, independent trip through the graph (rewrite,
route, retrieve, grade, generate, grade again, maybe revise) - real but
deliberately not free. force_regenerate defaults to False for exactly
that reason: nothing gets re-run, and no new tokens get spent, just
because a report was asked for again.
"""

from typing import Any
from uuid import UUID

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from sqlalchemy.orm.exc import StaleDataError

from backend.app.agents.rag.graph import build_rag_graph
from backend.app.agents.rag.state import RagState
from backend.app.config import Settings
from backend.app.db.errors import DatabaseOperationError
from backend.app.db.models.brand_strategy import BrandStrategy
from backend.app.db.models.business_profile import BusinessProfile
from backend.app.db.models.business_understanding import BusinessUnderstanding
from backend.app.db.models.competitor_analysis import CompetitorAnalysis
from backend.app.db.models.customer_persona_set import CustomerPersonaSet
from backend.app.db.models.market_research import MarketResearch
from backend.app.db.models.marketing_strategy import MarketingStrategy
from backend.app.db.repositories.brand_strategy import BrandStrategyRepository
from backend.app.db.repositories.business_profiles import BusinessProfileRepository
from backend.app.db.repositories.business_understanding import (
    BusinessUnderstandingRepository,
)
from backend.app.db.repositories.competitor_analysis import CompetitorAnalysisRepository
from backend.app.db.repositories.customer_persona_set import CustomerPersonaSetRepository
from backend.app.db.repositories.market_research import MarketResearchRepository
from backend.app.db.repositories.marketing_strategy import MarketingStrategyRepository
from backend.app.db.repositories.workspaces import WorkspaceRepository
from backend.app.embeddings.gateway import EmbeddingsGateway
from backend.app.exceptions import ResourceNotFoundError, StaleResourceError
from backend.app.llm.gateway import LLMGateway
from backend.app.services.retrieval import RetrievalService
from backend.app.websearch.gateway import WebSearchGateway

_SNIPPET_MAX_LENGTH = 320
_EXCERPT_MAX_LENGTH = 280


class MarketingStrategyService:
    """Coordinate Marketing Strategy generation and persistence."""

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
        self._strategies = MarketingStrategyRepository(session)
        self._profiles = BusinessProfileRepository(session)
        self._understandings = BusinessUnderstandingRepository(session)
        self._research = MarketResearchRepository(session)
        self._analyses = CompetitorAnalysisRepository(session)
        self._persona_sets = CustomerPersonaSetRepository(session)
        self._brand_strategies = BrandStrategyRepository(session)
        self._workspaces = WorkspaceRepository(session)

    def generate(
        self,
        workspace_id: UUID,
        *,
        force_regenerate: bool = False,
    ) -> MarketingStrategy:
        """Return the current strategy, generating one if needed.

        If a strategy already exists and force_regenerate is False, the
        existing record is returned without touching the LLM, retrieval,
        or web search at all - checked *before* requiring any prerequisite
        report, so an already-generated strategy keeps being readable
        even if those reports are later edited, regenerated, or removed.
        """

        self._require_workspace(workspace_id)

        existing = self._strategies.get_by_workspace(workspace_id)
        if existing is not None and not force_regenerate:
            return existing

        profile = self._profiles.get_by_workspace(workspace_id)
        if profile is None:
            raise ResourceNotFoundError("Business profile not found.")

        understanding = self._understandings.get_by_business_profile(profile.id)
        if understanding is None:
            raise ResourceNotFoundError("Business understanding has not been generated yet.")

        research = self._research.get_by_workspace(workspace_id)
        if research is None:
            raise ResourceNotFoundError("Market research has not been generated yet.")

        analysis = self._analyses.get_by_workspace(workspace_id)
        if analysis is None:
            raise ResourceNotFoundError("Competitor analysis has not been generated yet.")

        persona_set = self._persona_sets.get_by_workspace(workspace_id)
        if persona_set is None:
            raise ResourceNotFoundError("Customer personas have not been generated yet.")

        brand_strategy = self._brand_strategies.get_by_workspace(workspace_id)
        if brand_strategy is None:
            raise ResourceNotFoundError("Brand strategy has not been generated yet.")

        sections, sources, input_tokens, output_tokens = self._run_sections(
            workspace_id, profile, understanding, research, analysis, persona_set, brand_strategy
        )

        strategy = (
            existing if existing is not None else MarketingStrategy(workspace_id=workspace_id)
        )
        self._apply_sections(strategy, sections, sources, input_tokens, output_tokens)

        if existing is None:
            self._strategies.add(strategy)

        try:
            self._session.commit()
        except StaleDataError as exc:
            self._session.rollback()
            raise StaleResourceError(
                "Marketing strategy changed while it was being regenerated."
            ) from exc
        except SQLAlchemyError as exc:
            self._session.rollback()
            raise DatabaseOperationError("Marketing strategy generation failed.") from exc

        self._session.refresh(strategy)
        return strategy

    def get(self, workspace_id: UUID) -> MarketingStrategy:
        """Return the current marketing strategy for a workspace.

        Deliberately does not require the prerequisite reports to still
        exist - a generated strategy is a self-contained, already-written
        document, not a live view over the reports it originally
        synthesized.
        """

        self._require_workspace(workspace_id)

        strategy = self._strategies.get_by_workspace(workspace_id)
        if strategy is None:
            raise ResourceNotFoundError("Marketing strategy has not been generated yet.")

        return strategy

    def _require_workspace(self, workspace_id: UUID) -> None:
        """Ensure the parent workspace exists."""

        workspace = self._workspaces.get(workspace_id)
        if workspace is None:
            raise ResourceNotFoundError("Workspace not found.")

    def _run_sections(
        self,
        workspace_id: UUID,
        profile: BusinessProfile,
        understanding: BusinessUnderstanding,
        research: MarketResearch,
        analysis: CompetitorAnalysis,
        persona_set: CustomerPersonaSet,
        brand_strategy: BrandStrategy,
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

        queries = self._build_queries(
            profile, understanding, research, analysis, persona_set, brand_strategy
        )
        for section, query in queries.items():
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

    def _build_queries(
        self,
        profile: BusinessProfile,
        understanding: BusinessUnderstanding,
        research: MarketResearch,
        analysis: CompetitorAnalysis,
        persona_set: CustomerPersonaSet,
        brand_strategy: BrandStrategy,
    ) -> dict[str, str]:
        """Build one targeted research question per report section.

        Every question is centered on the profile's own planning fields,
        with structured context from Business Understanding, Customer
        Personas, and Brand Strategy folded in directly (all already
        short) and a brief excerpt of the most relevant prior narrative
        section per question (see the module docstring for why full
        sections aren't pasted in verbatim).

        content_and_messaging_pillars now draws its brand-voice context
        from Brand Strategy's brand_voice_and_tone and brand_pillars
        instead of the profile's raw brand_tone field - the same
        raw-field-to-generated-field swap Step 14 made when it replaced
        Customer Personas' "Country" line with "Existing brand tone" in
        its own prompt, now that brand tone has itself been synthesized
        into a structured deliverable.
        """

        name = profile.business_name
        goal = _clean(profile.main_marketing_goal or "growing awareness and demand")
        target_customer = _clean(profile.target_customer or "its typical customer")
        price_range = _clean(profile.price_range or "an unspecified price range")
        stage = _clean(understanding.inferred_business_stage)
        category = _clean(understanding.competitive_category)

        channels = _clean(
            ", ".join(profile.existing_channels)
            if profile.existing_channels
            else "no channels in active use yet"
        )
        budget = (
            f"{profile.monthly_marketing_budget} {profile.marketing_budget_currency} per month"
            if profile.monthly_marketing_budget is not None
            else "an unspecified budget"
        )
        challenges = _clean(profile.current_challenges or "no specific challenges named yet")
        differentiators = _clean(
            ", ".join(understanding.key_differentiators)
            if understanding.key_differentiators
            else "no clearly established differentiators yet"
        )

        segments_excerpt = _clean(_excerpt(research.target_customer_segments))
        opportunities_excerpt = _clean(_excerpt(research.opportunities_and_risks))
        differentiation_excerpt = _clean(_excerpt(analysis.differentiation_opportunities))
        strengths_weaknesses_excerpt = _clean(_excerpt(analysis.strengths_and_weaknesses))

        persona_channels = _clean(
            _aggregate_persona_field(
                persona_set.personas,
                "preferred_channels",
                fallback="no preferred channels recorded yet",
            )
        )
        persona_pain_points = _clean(
            _aggregate_persona_field(
                persona_set.personas,
                "pain_points",
                fallback="no pain points recorded yet",
            )
        )
        brand_voice = _clean(brand_strategy.brand_voice_and_tone)
        brand_pillars_joined = _clean(
            ", ".join(brand_strategy.brand_pillars)
            if brand_strategy.brand_pillars
            else "no established brand pillars yet"
        )
        positioning_excerpt = _clean(_excerpt(brand_strategy.positioning_statement))

        return {
            "recommended_channels_and_tactics": (
                f"{name}'s current business stage is: {stage}. It competes in "
                f"the {category} category and currently uses these channels: "
                f"{channels}. Given its main "
                f"marketing goal of {goal}, its target customer segments "
                f"({segments_excerpt}), its differentiation opportunities "
                f"versus competitors ({differentiation_excerpt}), and its "
                f"customer personas' preferred channels ({persona_channels}), "
                f"which specific marketing channels and tactics should {name} "
                "prioritize next?"
            ),
            "content_and_messaging_pillars": (
                f"{name}'s key differentiators are: {differentiators}. Its brand "
                f"voice and tone is {brand_voice} and its established brand "
                f"pillars are {brand_pillars_joined}. Its target customer is "
                f"{target_customer} and their common pain points include "
                f"{persona_pain_points}. Given its competitive strengths and "
                f"weaknesses versus competitors ({strengths_weaknesses_excerpt}), "
                f"what core content and messaging pillars should {name} build "
                "its marketing around?"
            ),
            "ninety_day_roadmap": (
                f"{name} currently faces these challenges ({challenges}). Given a "
                f"main marketing goal of {goal}, existing channels of {channels}, "
                f"relevant market opportunities and risks "
                f"({opportunities_excerpt}), and its brand positioning "
                f"({positioning_excerpt}), lay out a practical 90-day marketing "
                f"roadmap with concrete milestones to get {name} there."
            ),
            "budget_allocation_and_kpis": (
                f"{name} has a monthly marketing budget of {budget}, sells at "
                f"{price_range} to {target_customer}, whose preferred channels "
                f"are {persona_channels}, and its main marketing goal is {goal}. "
                f"Recommend a budget allocation across marketing channels and "
                f"the specific KPIs {name} should track to measure progress "
                "toward that goal."
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
        strategy: MarketingStrategy,
        sections: dict[str, str],
        sources: list[dict[str, str]],
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Copy generated section text plus usage metadata onto an ORM record."""

        strategy.recommended_channels_and_tactics = sections["recommended_channels_and_tactics"]
        strategy.content_and_messaging_pillars = sections["content_and_messaging_pillars"]
        strategy.ninety_day_roadmap = sections["ninety_day_roadmap"]
        strategy.budget_allocation_and_kpis = sections["budget_allocation_and_kpis"]
        strategy.sources = sources
        strategy.model_used = self._settings.llm_model
        strategy.input_tokens = input_tokens
        strategy.output_tokens = output_tokens


def _aggregate_persona_field(
    personas: list[dict[str, Any]],
    field: str,
    *,
    fallback: str,
    limit: int = 6,
) -> str:
    """Collect one list field across every persona into one comma-joined string.

    Preserves first-seen order and drops duplicates rather than
    concatenating each persona's list separately - the four sections
    below want one clean phrase per query, not a repeated list per
    persona. Falls back to `fallback` when every persona's list for
    `field` is empty, which is the normal case for a workspace built
    entirely on the deterministic "local" LLM provider: it fills every
    list-typed field with [] regardless of schema (see Step 14's
    min_length note), so a Postgres-only test's personas never actually
    have preferred_channels or pain_points to aggregate.
    """

    values: list[str] = []
    for persona in personas:
        for value in persona.get(field) or []:
            if value not in values:
                values.append(value)

    if not values:
        return fallback
    return ", ".join(values[:limit])


def _snippet(content: str) -> str:
    """Truncate a retrieved chunk's content to a citation-sized snippet."""

    if len(content) <= _SNIPPET_MAX_LENGTH:
        return content
    return content[:_SNIPPET_MAX_LENGTH].rstrip() + "..."


def _excerpt(content: str) -> str:
    """Truncate a prior report section to a short excerpt for use inside a query."""

    if len(content) <= _EXCERPT_MAX_LENGTH:
        return content
    return content[:_EXCERPT_MAX_LENGTH].rstrip() + "..."


def _clean(value: str) -> str:
    """Strip a single trailing period, if present.

    Every value _build_queries() interpolates is free text - either typed
    by a user (current_challenges) or written by a previous agent
    (inferred_business_stage, an excerpt, ...) - so it can end with its
    own period regardless of where it lands in a larger sentence. Without
    this, that period either doubles up against one _build_queries()
    itself adds, or - worse - lands mid-sentence in front of more text
    ("...snacking. category and..."), which is exactly the class of bug
    Step 10 already hit once with doubled apostrophes. Left alone if the
    value ends with an excerpt's "..." truncation marker instead, since
    that isn't a sentence-ending period.
    """

    if value.endswith("...") or not value.endswith("."):
        return value
    return value[:-1]
