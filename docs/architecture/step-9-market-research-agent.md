# Step 9 — Market Research Agent

## Purpose

Wires Step 8's CRAG graph into a real, persisted agent - the second real
agent after Business Understanding, and the first to actually need what
the graph was built for: retrieval and web search over open-ended
research questions, not synthesis from data already on hand.

## Scoping decisions

- **Four independent sections, not one query.** A report is
  `market_overview`, `target_customer_segments`, `competitive_landscape`,
  and `opportunities_and_risks`, each the output of its own full CRAG
  graph invocation over a research question built from the workspace's
  business profile. Deliberately not free: four full traversals of the
  graph (rewrite, route, retrieve, grade, generate, grade again, maybe
  revise) is real cost, real latency. `force_regenerate` defaults to
  `false` for exactly that reason - the same philosophy Business
  Understanding already established, just with a steeper price this
  time.
- **Keyed by `workspace_id`, not `profile_id`.** Unlike
  `BusinessUnderstanding`, which is 1:1 with `business_profiles` and
  cascade-deletes with it, `MarketResearch` is 1:1 with `workspaces`.
  Retrieval itself is workspace-scoped, and a generated report is a
  self-contained document, not a live view over the profile it was
  built from - `get()` deliberately does not require the profile to
  still exist, and this is verified against real Postgres foreign-key
  behavior in `test_get_survives_the_business_profile_being_deleted`.
- **Sources are built server-side, never asked of the model.** Each
  section's `sources` entries come directly from that CRAG run's own
  `relevant_documents` (or `retrieved_documents` if none were graded
  relevant), truncated to a snippet. A citation appearing in a report
  always corresponds to something genuinely retrieved.

## Runtime architecture

\```text
POST/GET /api/v1/workspaces/{id}/market-research
      |
      v
MarketResearchService
      |
      | builds 4 targeted research questions from BusinessProfile fields
      v
build_rag_graph() - compiled once, invoked 4 times (Step 8's CRAG graph)
      |
      | each invocation: rewrite -> route -> retrieve -> grade ->
      |                   generate -> grade again -> maybe revise
      v
4 section texts + sources (from each run's own relevant_documents)
+ aggregated total_input_tokens / total_output_tokens
      |
      v
MarketResearch (Postgres, 1:1 with workspaces, cascade delete)
\```

## What was built

- **Token usage threaded through the CRAG graph itself**
  (`agents/rag/state.py`, all three node modules) - `RagState` gained
  `total_input_tokens`/`total_output_tokens` (`operator.add` reducer,
  same pattern as `trace`), and every LLM-calling node switched from
  `chat()`/`structured()` to `chat_with_usage()`/`structured_with_usage()`.
  Needed because Step 8's graph only ever logged usage; persisting it
  per report required the graph to actually return it.
- **`WebSearchGateway`/`EmbeddingsGateway` exposed as real API
  dependencies** (`websearch/dependencies.py`,
  `embeddings/dependencies.py`, `main.py`) - Step 8 built
  `WebSearchGateway` but only ever used it inside the graph or tests
  directly; this step is the first thing that needs it behind an actual
  endpoint.
- **`MarketResearch` model + migration + repository** - four `Text`
  section columns, JSONB `sources` array with the same `jsonb_typeof`
  check-constraint pattern as `BusinessUnderstanding`, optimistic
  concurrency via `version_id_col`, `model_used`/`input_tokens`/
  `output_tokens` for the same per-record cost auditability.
- **Schemas**: `MarketResearchGenerateRequest` (`force_regenerate`
  flag), `MarketResearchSourceSchema`, `MarketResearchResponse`.
- **`MarketResearchService`**: `generate()` returns the existing report
  without touching the LLM, retrieval, or web search unless
  `force_regenerate=true` - checked *before* requiring a business
  profile, so an already-generated report stays readable even if the
  profile is edited or removed later. `_build_queries()` turns
  `BusinessProfile` fields into four targeted research questions,
  gracefully falling back to sensible defaults for unset fields. `get()`
  for plain reads.
- **API endpoints**: `POST`/`GET` on
  `/workspaces/{id}/market-research`, mirroring Business Understanding's
  dependency-injection and error-handling shape exactly.

## Real bugs and findings caught while building this

- The `WebSearchGateway`/`EmbeddingsGateway`-not-on-`app.state` gap
  described above was caught by checking the actual codebase before
  assuming it existed, not by reading the Step 8 doc's description of
  it - the doc said the graph was built, not that it was reachable from
  an endpoint.
- **Investigated, confirmed not a bug**: calling `generate()` with
  `force_regenerate=true` against the deterministic `local` LLM provider
  does not bump the persisted `version`. Dug into why rather than
  assuming: the local provider reproduces byte-identical text on every
  call, and SQLAlchemy correctly skips emitting an `UPDATE` (and
  therefore the version bump) when no column's value actually changed -
  proven directly in my sandbox by setting a field to its own value
  (no bump) versus a genuinely different one (bumps correctly). Against
  a real, non-deterministic model this doesn't come up, which is why
  `test_market_research_llm_integration.py`'s `force_regenerate` test
  doesn't assert on version at all - same reasoning Business
  Understanding's own LLM integration test already documented for
  content-difference assertions.
- Checked, not assumed: the new `market_researches` table's FK
  constraint name (`fk_market_researches_workspace_id_workspaces`) was
  verified directly against a real Postgres instance before treating the
  migration as safe, given Step 6 was previously bitten by a
  63-character identifier limit on a longer name. No truncation here,
  but checked rather than presumed safe by analogy.
- `model_used` reflects `Settings.llm_model` regardless of which
  provider actually ran - confirmed this is pre-existing
  `BusinessUnderstandingService` behavior (its own integration test only
  asserts the field is truthy, never a specific value) rather than a new
  defect, and left it consistent rather than "fixing" something out of
  this step's scope.

## Testing

- No fake-session unit tests for the service layer, matching this
  codebase's established convention.
- `test_market_research_integration.py`: real Postgres, the
  deterministic `local` LLM and embeddings providers, no Tavily key.
  Covers the same shape of cases as Business Understanding's Postgres
  test (create, idempotent re-fetch, `force_regenerate` reusing the same
  row, 404s for missing profile/workspace), plus two cases unique to
  this agent's design: a report surviving its source business profile
  being deleted, and `sources` genuinely reflecting a seeded knowledge
  chunk rather than being fabricated.
- `test_market_research_llm_integration.py`: real Postgres AND real
  Ollama Cloud together, gated behind both integration flags. Covers
  genuine non-trivial content across all four sections, real token usage
  persisting through the full stack, and - mirroring
  `test_retrieval_llm_integration.py`'s proof of real semantic
  understanding - a seeded chunk sharing no exact keywords with the
  business profile still getting found and cited by real embeddings and
  real relevance grading. Written and lint-verified; not run from this
  sandbox (no network to Ollama Cloud or Tavily) - flagged for Ahmad to
  run and report back, same as every other LLM integration test in this
  repo.

## Known gaps, deliberately out of scope for this step

- No per-section regeneration - `force_regenerate` always re-runs all
  four sections; regenerating just one (e.g. only
  `competitive_landscape`) is a plausible future enhancement, not
  pulled into this step's scope.
- No dedicated `/health/websearch` endpoint. `LLMGateway`/
  `EmbeddingsGateway` have health checks; `WebSearchGateway` doesn't,
  since Step 8 deliberately designed it to never raise a typed error in
  the first place (a missing key just means empty results) - there's no
  distinct "unreachable" state to report on the way there is for the
  other two gateways.
- Competitor Analysis - the CRAG graph's other intended consumer per
  Step 8's doc - not yet built. This step is the second proof the graph
  is genuinely reusable, not the last.
- Frontend "view market research report" UI not built - backend-only
  step, same split as every prior agent step.

## Definition of done

- [x] `MarketResearch` persists per workspace with optimistic
      concurrency and cascade delete from its parent workspace -
      verified against real Postgres, not just the migration file
- [x] Generation is idempotent by default; `force_regenerate` re-runs
      all four sections without creating a duplicate row
- [x] Sources are built directly from each CRAG run's own retrieved/
      relevant documents, never asked of the model
- [x] Token usage is threaded through the CRAG graph itself, not just
      logged, and persisted per record
- [x] `ruff` clean, full non-integration suite green (85 passing, 34
      deselected integration tests)
- [x] Verified against real Postgres end to end (model, service, and
      full HTTP round trip through the API) in addition to the
      permanent integration test suite (109 passing, 10 skipped
      pending real Ollama Cloud)
- [x] Real Ollama Cloud + Tavily test written and lint-verified,
      explicitly flagged as unrun pending Ahmad's local run