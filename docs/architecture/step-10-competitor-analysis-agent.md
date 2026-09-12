# Step 10 — Competitor Analysis Agent

## Purpose

Wires Step 8's CRAG graph into its second real consumer, per Step 9's
own doc. Same shape as Market Research - four independent sections, each
a full CRAG graph invocation - but zeroed in specifically on
`BusinessProfile.known_competitors` rather than the broader market
questions Market Research already covers.

## Scoping decisions

- **Four fixed sections, not one row per named competitor.**
  `known_competitors` is a variable-length list, often empty. Rather
  than a dynamic per-competitor structure with unbounded LLM cost, this
  mirrors Market Research's proven shape: `competitor_overview`,
  `strengths_and_weaknesses`, `pricing_and_positioning`,
  `differentiation_opportunities`, each one full CRAG graph invocation,
  fixed `Text` columns. `known_competitors` feeds the *questions*, not
  the schema.
- **Graceful fallback when no competitors are named yet.**
  `_build_queries()` doesn't leave a section thin or empty when
  `known_competitors` is `[]` - it asks the graph to identify the most
  likely direct competitors itself (via retrieval and/or web search),
  seeded by industry, product, and country. Verified against real
  Postgres in
  `test_generate_without_known_competitors_still_produces_all_sections`.
- **Keyed by `workspace_id`, not `profile_id`** - same reasoning as
  `MarketResearch`: a generated report is a self-contained document, not
  a live view over the profile it was built from. `get()` deliberately
  does not require the profile to still exist, verified against real
  Postgres foreign-key behavior in
  `test_get_survives_the_business_profile_being_deleted`.
- **Sources are built server-side, never asked of the model** - same
  discipline as every prior agent. Each section's `sources` entries come
  directly from that CRAG run's own `relevant_documents` (or
  `retrieved_documents` if none were graded relevant).

## Runtime architecture

```text
POST/GET /api/v1/workspaces/{id}/competitor-analysis
      |
      v
CompetitorAnalysisService
      |
      | builds 4 targeted questions from BusinessProfile fields,
      | centered on known_competitors (or a discovery fallback if empty)
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
CompetitorAnalysis (Postgres, 1:1 with workspaces, cascade delete)
```

## What was built

No new shared infrastructure was needed - Step 9 already exposed the
gateways and threaded token usage through the graph, so this step is
pure agent-shape work, same as Business Understanding's commit count.

- **`CompetitorAnalysis` model + migration + repository** - four `Text`
  section columns, JSONB `sources` array with the same `jsonb_typeof`
  check-constraint pattern as `MarketResearch`, optimistic concurrency
  via `version_id_col`, `model_used`/`input_tokens`/`output_tokens` for
  the same per-record cost auditability. `Workspace.competitor_analysis`
  relationship added alongside `market_research`.
- **Schemas**: `CompetitorAnalysisGenerateRequest` (`force_regenerate`
  flag), `CompetitorAnalysisSourceSchema`, `CompetitorAnalysisResponse`.
- **`CompetitorAnalysisService`**: `generate()` returns the existing
  report without touching the LLM, retrieval, or web search unless
  `force_regenerate=true` - checked *before* requiring a business
  profile. `_build_queries()` turns `BusinessProfile` fields into four
  targeted questions centered on `known_competitors`, with the discovery
  fallback described above. `get()` for plain reads.
- **API endpoints**: `POST`/`GET` on
  `/workspaces/{id}/competitor-analysis`, mirroring Market Research's
  dependency-injection and error-handling shape exactly.

## Real bugs and findings caught while building this

- **Doubled-apostrophe artifact in query text, found and fixed.**
  `_build_queries()` originally wrapped `business_name` in quotes and
  then appended a possessive `'s`, producing text like
  `'FitMeal''s known competitors (HelloFresh, Factor)` - confusing,
  malformed-looking punctuation, not real grammar. Traced to a
  convention mirrored from `MarketResearchService._build_queries()`,
  which has the identical artifact (`'{profile.business_name}''s main
  competitors`) already merged on `main`. Fixed in this file by dropping
  the quote-wrapping around possessives entirely; left the equivalent
  Market Research code untouched, since editing already-merged code
  from a prior step is out of this step's scope - worth a small
  follow-up cleanup there separately.
- **Investigated a real Ollama Cloud test failure rather than
  re-running blindly.** The first real-model run of
  `test_generate_produces_real_analysis_with_real_token_usage` failed
  with `LLMStructuredOutputError` (surfaced as a 503) on its first
  attempt; the other two real-Ollama-Cloud tests passed. Traced the
  cause: a single `generate()` call fans out into many real structured-
  output calls (query rewrite, context-routing decision, source
  selection, per-document relevance grading, support/usefulness
  grading) across four independent graph traversals, and
  `llm/retry.py`'s `is_transient_error()` deliberately does not retry a
  schema-validation miss - only network-style failures (timeouts,
  429/502/503/504) - on the reasoning that retrying a bad model output
  would just waste money on a call that won't succeed. One malformed
  JSON response from the real model, anywhere in that whole chain, is
  enough to fail the request outright. This is a property of the shared
  CRAG graph and LLM structured-output layer from Steps 7-9, equally
  present in Market Research, not something introduced by this step's
  code - confirmed by finding the identical query-construction pattern
  already merged there. Ahmad re-ran the specific failing test after the
  apostrophe fix above and confirmed all three real-Ollama-Cloud tests
  now pass; whether that fix, or simple non-determinism in a single
  real network call, explains the difference isn't fully separable from
  one re-run, and is noted below as a known gap rather than claimed as
  a proven fix.
- **Checked, not assumed**: the new `competitor_analyses` table's FK
  constraint name (`fk_competitor_analyses_workspace_id_workspaces`, 46
  characters) was verified directly against a real Postgres instance
  before treating the migration as safe, given Step 6 was previously
  bitten by a 63-character identifier limit on a longer name.
- **Confirmed pre-existing, not introduced here**: `alembic check`
  reports drift on `knowledge_chunks.chunk_metadata`'s server default.
  Verified this exists on unmodified `main` (stashed this step's changes
  entirely and re-ran the check) - predates Step 10, not conflated with
  this step's migration.
- A `frontend/tests/test_onboarding_flow_integration.py` failure
  (`AppTest script run timed out after 3(s)`) appeared in one full
  `pytest -q` run during this step. Confirmed unrelated: introduced in a
  single old commit untouched by any of this step's changes, and the
  timeout is Streamlit's own internal default for a real
  server-backed UI test, not something a backend code change would
  trigger. Left out of this step's scope; worth its own look separately
  if it recurs.

## Testing

- No fake-session unit tests for the service layer, matching this
  codebase's established convention.
- `test_competitor_analysis_integration.py`: real Postgres, the
  deterministic `local` LLM and embeddings providers, no Tavily key.
  Covers the same shape of cases as Market Research's Postgres test
  (create, idempotent re-fetch, `force_regenerate` reusing the same row,
  404s for missing profile/workspace, a report surviving its source
  business profile being deleted, `sources` genuinely reflecting a
  seeded knowledge chunk), plus one case unique to this agent's design:
  generation still produces all four complete sections when
  `known_competitors` is empty, proving the discovery-fallback decision
  above actually holds.
- `test_competitor_analysis_llm_integration.py`: real Postgres AND real
  Ollama Cloud together, gated behind both integration flags. Covers
  genuine non-trivial content across all four sections, real token usage
  persisting through the full stack, `force_regenerate` succeeding
  end to end, and - mirroring `test_retrieval_llm_integration.py`'s
  proof of real semantic understanding - a seeded chunk sharing no
  exact keywords with the business profile still getting found and
  cited by real embeddings and real relevance grading. Run by Ahmad
  against real Ollama Cloud: all three tests pass (see findings above
  for the one failure hit and resolved along the way).

## Known gaps, deliberately out of scope for this step

- No per-section regeneration - `force_regenerate` always re-runs all
  four sections, same limitation Market Research already has.
- The shared CRAG graph's structured-output calls have no retry margin
  for a schema-validation miss specifically (only for network-style
  failures) - a single bad JSON response from the real model anywhere
  in a request's internal call chain fails the whole request. This
  affects Market Research equally and predates this step; revisiting
  the retry policy for this failure mode is a plausible follow-up, not
  pulled into this step's scope.
- The identical doubled-apostrophe query-construction artifact this
  step fixed locally still exists in `MarketResearchService._build_queries()`
  on `main` - a small, safe cleanup left for a separate follow-up rather
  than edited here.
- Frontend "view competitor analysis report" UI not built - backend-only
  step, same split as every prior agent step.

## Definition of done

- [x] `CompetitorAnalysis` persists per workspace with optimistic
      concurrency and cascade delete from its parent workspace -
      verified against real Postgres, not just the migration file
- [x] Generation is idempotent by default; `force_regenerate` re-runs
      all four sections without creating a duplicate row
- [x] Generation succeeds and produces all four complete sections even
      when no competitors have been named yet
- [x] Sources are built directly from each CRAG run's own retrieved/
      relevant documents, never asked of the model
- [x] `ruff` clean, full non-integration suite green (85 passing, 46
      deselected integration tests)
- [x] Verified against real Postgres end to end (model, service, and
      full HTTP round trip through the API) in addition to the
      permanent integration test suite (118 passing, 13 skipped pending
      real Ollama Cloud)
- [x] Real Ollama Cloud test run by Ahmad: all three tests pass (121
      passing, 10 skipped, with both integration flags set)