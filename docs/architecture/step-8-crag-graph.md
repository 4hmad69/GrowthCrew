# Step 8 — CRAG Graph

## Purpose

The actual LangGraph workflow: rewrite -> decide -> select source ->
retrieve -> grade -> generate -> grade again -> revise -> finish. Built
generically (no Market-Research-specific assumptions) so Competitor
Analysis can reuse it later, on top of the real pgvector retrieval and
LLM gateway infrastructure Steps 5-7 built.

## Runtime architecture

```text
rewrite_query -> decide_context_need -+-> generate_direct -> END
                                       |
                                       v
                                 select_source -+-> retrieve_vectorstore -+
                                                 |                        |
                                                 +-> retrieve_web --------+
                                                                          v
                                                                 grade_documents
                                                    (no relevant docs: retry vectorstore,
                                                     then try web, then give up -> direct)
                                                                          |
                                                                          v
                                                                 generate_grounded
                                                                          |
                                                                          v
                                                                  grade_support --(not grounded)--> revise_answer -+
                                                                          |                                        |
                                                                          v                                       (loop back)
                                                                grade_usefulness --(not useful)--> revise_answer -+
                                                                          |
                                                                          v
                                                                         END
```

Loop bounds (MAX_RETRIEVAL_ATTEMPTS=2, MAX_WEB_ATTEMPTS=1,
MAX_REVISION_ATTEMPTS=2) guarantee termination - verified by running a
real invocation against an empty vectorstore and no web-search key, and
confirming it correctly exhausts every fallback before landing on a
direct answer, rather than looping forever.

## What was built

- **State & schemas** (`agents/rag/state.py`, `schemas.py`) - typed
  `RagState`, and one Pydantic schema per node's structured-output
  contract (`QueryRewrite`, `ContextNeedDecision`, `SourceSelection`,
  `RelevanceGrade`, `SupportGrade`, `UsefulnessGrade`). Every grading
  schema carries a `reasoning` field - not for display, but because
  asking a model to justify a decision measurably improves the decision
  itself, and makes a wrong one debuggable afterward.
- **`WebSearchGateway`** - wraps Tavily via `langchain-tavily` (the
  modern dedicated package, not the older `langchain_community` route
  the source repo used). One deliberate difference from `LLMGateway`/
  `EmbeddingsGateway`: no fake "local" stub. A fake LLM/embeddings
  response is obviously inert; a fake *web search result* looks exactly
  like real information but isn't - so a missing API key returns an
  empty list, matching exactly how the proven agentic-rag repo already
  treats it.
- **Decision nodes** (rewrite, decide context need, select source),
  **retrieval nodes** (vectorstore, web, per-document relevance grading),
  **generation nodes** (grounded with `[Source N]` citations, direct,
  support/usefulness grading, revision) - each built via a factory
  closing over its gateway/service dependency, since LangGraph node
  functions take only `state`.
- **Graph assembly** (`agents/rag/graph.py`) - `build_rag_graph()`
  compiles fresh per invocation, since `RetrievalService` is bound to a
  request-scoped DB session and can't be a long-lived singleton the way
  the gateways themselves are.

## Real bugs and gaps caught along the way

- `LocalStructuredRunnable` (Step 5) never had a schema with a `Literal`
  field until `SourceSelection` - its string-matching type detection
  silently failed on it. Fixed with real type introspection
  (`typing.get_origin`/`get_args`), also now correctly handling any
  `Optional`-wrapped type, not just `str`.
- `langchain-core`/`langchain-ollama` were never actually added to
  `requirements.txt` in Step 5, despite already working - a fresh
  install would have failed. Fixed here while touching the file again.
- Adding `tavily_api_key` with a `validation_alias` (to read the natural
  unprefixed `TAVILY_API_KEY` name) silently broke direct-kwarg
  construction for that one field, inconsistent with every other
  `Settings` field. Fixed with `populate_by_name=True`.
- The API key is passed explicitly to `TavilySearchAPIWrapper` rather
  than relying on `os.environ` - GrowthCrew's own `.env` resolution is
  already solid (Step 7's fix); no reason to depend on a second,
  less reliable mechanism for the same value.

## Known gaps, deliberately out of scope for this step

- Not wired into any API endpoint yet - this is the reusable engine,
  not an agent. Step 9 (Market Research) does that wiring.
- `grade_documents` grades one document per LLM call - correct (avoids a
  model rubber-stamping a whole batch as relevant because one item is),
  but not the cheapest possible approach; worth revisiting if latency
  matters once real usage exists.
- No LLM evaluation harness yet - still on the roadmap from earlier.

## Definition of done

- [x] Graph compiles and a real invocation runs end to end, verified
      manually and via `test_rag_graph.py`
- [x] Loop bounds actually terminate the graph - verified with an empty
      vectorstore and no web-search key, not just assumed from the code
- [x] Tavily API key handled explicitly and safely (never fabricates
      results when absent)
- [x] `ruff` clean, full non-integration suite green (117 passing)