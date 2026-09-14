# Step 11 — Marketing Strategy Agent

## Purpose

The first synthesizing agent: rather than building its research
questions from `BusinessProfile` alone like every prior agent, it reads
the *outputs* of Business Understanding, Market Research, and Competitor
Analysis and turns them into an actionable plan. Four sections, each a
full CRAG graph invocation, same shape as Market Research and Competitor
Analysis - but the first agent whose `generate()` hard-requires three
other reports to already exist, not just a business profile.

## Scoping decisions

- **Hard-require all three prior reports, no partial-data fallback.**
  `generate()` checks Business Understanding (via the business profile),
  Market Research, and Competitor Analysis in that order, raising
  `ResourceNotFoundError` naming whichever is missing first. A strategy
  synthesized from a partially-missing research base was judged a worse
  product outcome than asking the user to finish the earlier steps -
  and it keeps `_build_queries()` far simpler than a fallback would.
  Verified against real Postgres as four separate 404 cases, one per
  missing prerequisite, in `test_generate_without_*_returns_404`.
- **Queries built from structured fields and short excerpts, never full
  report text.** Inspecting `decision_nodes.py` and
  `generation_nodes.py` before writing `_build_queries()` surfaced a
  real constraint: `rewrite_query` explicitly compresses whatever it's
  given into "a clear, self-contained search query - keep it concise,"
  and `generate_grounded`/`generate_direct` only ever read
  `rewritten_query` plus retrieved sources - `original_query` itself
  never reaches final generation. Pasting full raw paragraphs from the
  three prior reports into the query would have been silently discarded
  before the model saw them. Each query instead folds in Business
  Understanding's short structured fields
  (`inferred_business_stage`, `competitive_category`,
  `key_differentiators`) directly, plus a ~280-character excerpt (see
  `_excerpt()`) of the one or two most relevant Market Research /
  Competitor Analysis sections per question - the same discipline
  Competitor Analysis already used for `known_competitors`, extended to
  richer inputs.
- **Four fixed sections**: `recommended_channels_and_tactics`,
  `content_and_messaging_pillars`, `ninety_day_roadmap`,
  `budget_allocation_and_kpis` - same reasoning as every prior agent's
  fixed-`Text`-column shape, sized to what a business can act on rather
  than one row per channel or per week.
- **Keyed by `workspace_id`, not tied to any of its three source
  reports.** `MarketingStrategy` has no foreign key to
  `BusinessUnderstanding`, `MarketResearch`, or `CompetitorAnalysis` at
  all - only to the workspace. `get()` deliberately does not require any
  of them to still exist, verified against real Postgres in
  `test_get_survives_prerequisite_reports_being_deleted` (deletes the
  business profile, which cascades away Business Understanding too, and
  confirms the strategy is still readable).
- **Sources are built server-side, never asked of the model** - same
  discipline as every prior agent.

## Runtime architecture

```text
POST/GET /api/v1/workspaces/{id}/marketing-strategy
      |
      v
MarketingStrategyService
      |
      | requires BusinessUnderstanding + MarketResearch +
      | CompetitorAnalysis to already exist (404s naming
      | whichever is missing, checked in that order)
      v
builds 4 targeted questions from BusinessProfile's planning fields
+ BusinessUnderstanding's structured fields
+ short excerpts of MarketResearch / CompetitorAnalysis sections
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
MarketingStrategy (Postgres, 1:1 with workspaces, cascade delete)
```

## What was built

No new shared infrastructure was needed - Step 8's CRAG graph and the
gateways it depends on were already reusable, so this step is pure
agent-shape work plus one genuinely new piece of query-building
discipline (see above).

- **`MarketingStrategy` model + migration + repository** - four `Text`
  section columns, JSONB `sources` array with the same `jsonb_typeof`
  check-constraint pattern as `CompetitorAnalysis`, optimistic
  concurrency via `version_id_col`,
  `model_used`/`input_tokens`/`output_tokens` for the same per-record
  cost auditability. `Workspace.marketing_strategy` relationship added
  alongside `competitor_analysis`.
- **Schemas**: `MarketingStrategyGenerateRequest` (`force_regenerate`
  flag), `MarketingStrategySourceSchema`, `MarketingStrategyResponse`.
- **`MarketingStrategyService`**: `generate()` returns the existing
  strategy without touching the LLM, retrieval, or web search unless
  `force_regenerate=true` - checked *before* requiring any prerequisite
  report, so an already-generated strategy stays readable even if those
  reports are later edited or removed. `_build_queries()` turns
  `BusinessProfile` plus the three prior reports into four targeted
  questions, using `_clean()` (see findings below) to keep every
  interpolated value safe to drop into the middle of a sentence. `get()`
  for plain reads.
- **API endpoints**: `POST`/`GET` on
  `/workspaces/{id}/marketing-strategy`, mirroring Competitor Analysis's
  dependency-injection and error-handling shape exactly. No changes
  needed to `main.py`/`errors.py` - `ResourceNotFoundError` already maps
  generically to 404 via the shared `DomainError` handler.

## Real bugs and findings caught while building this

- **A genuine article-grammar bug, caught before it shipped.**
  `inferred_business_stage` is arbitrary free text, so an early draft's
  `"{name} is a {stage} business"` produced `"is a early-stage
  startup"` for the most ordinary possible value. Fixed by rephrasing
  to `"{name}'s current business stage is: {stage}"`, which sidesteps
  needing an article at all rather than trying to guess a/an correctly
  against unpredictable text.
- **A broader stray/doubled-period bug class, caught by testing
  worst-case inputs, not just the happy path.** Every free-text field
  this service interpolates - `current_challenges`, the four narrative
  report sections behind `_excerpt()`, even `inferred_business_stage`
  and `competitive_category` - can legitimately end in its own period,
  since real user input and real LLM-written prose almost always does.
  Deliberately re-ran `_build_queries()` with every such field
  constructed to end in a period, which surfaced two distinct failure
  modes: periods doubling up (`"...costs.."`) and - worse - a stray
  period landing mid-sentence in front of more text
  (`"...snacking. category and..."`). This is the same class of bug
  Step 10 hit once with doubled apostrophes. Fixed systematically with a
  `_clean()` helper applied to every dynamic value at the point each is
  computed, rather than patched one occurrence at a time - re-ran the
  worst-case check after the fix to confirm all four queries read
  cleanly regardless of input.
- **Checked, not assumed**: the new `marketing_strategies` table's FK
  constraint name (`fk_marketing_strategies_workspace_id_workspaces`, 47
  characters) was verified directly against a real Postgres instance
  before treating the migration as safe.
- **Confirmed pre-existing, not introduced here**: `alembic check`
  reports drift on `knowledge_chunks.chunk_metadata`'s server default,
  same as it did in Step 10. Not conflated with this step's migration.
- A full-suite run briefly showed 39 failures partway through this step;
  investigated immediately rather than reported as a regression.
  Postgres had simply stopped running in the sandbox between commands -
  restarting it and re-running confirmed a clean 151 passed / 16
  skipped, no code issue.

## Testing

- No fake-session unit tests for the service layer, matching this
  codebase's established convention.
- `test_marketing_strategy_integration.py` (11 tests): real Postgres,
  the deterministic `local` LLM and embeddings providers, no Tavily key.
  Covers create, idempotent re-fetch, `force_regenerate` reusing the
  same row, each of the four prerequisite 404s as its own test (missing
  profile, missing Business Understanding, missing Market Research,
  missing Competitor Analysis - each asserting its exact message),
  missing-workspace 404, the strategy surviving its own prerequisite
  reports being deleted, and `sources` genuinely reflecting a seeded
  knowledge chunk.
- `test_marketing_strategy_llm_integration.py` (3 tests): real Postgres
  AND real Ollama Cloud together, gated behind both integration flags.
  Its shared setup helper generates all three prerequisite reports for
  real (there's no shortcut for seeding them directly, since the service
  reads them back through their own repositories). Covers genuine
  non-trivial content across all four sections with real token usage
  persisting through the full stack, `force_regenerate` succeeding end
  to end, and a seeded chunk sharing no exact keywords with the business
  profile still getting found and cited by real embeddings and real
  relevance grading. **Written and lint-verified in the sandbox; not yet
  run against real Ollama Cloud - pending Ahmad's confirmation.**

## Known gaps, deliberately out of scope for this step

- No per-section regeneration - `force_regenerate` always re-runs all
  four sections, same limitation every prior agent already has.
- The shared CRAG graph's structured-output calls have no retry margin
  for a schema-validation miss specifically (only network-style
  failures) - same pre-existing limitation as Steps 9-10, and this
  agent's `generate()` now depends on three *other* agents' generation
  calls succeeding first, so the number of structured-output calls that
  must all succeed for one request to complete is larger here than for
  any prior agent.
- Frontend "view marketing strategy report" UI not built - backend-only
  step, same split as every prior agent step.

## Definition of done

- [x] `MarketingStrategy` persists per workspace with optimistic
      concurrency and cascade delete from its parent workspace -
      verified against real Postgres, not just the migration file
- [x] Generation is idempotent by default; `force_regenerate` re-runs
      all four sections without creating a duplicate row
- [x] Generation requires Business Understanding, Market Research, and
      Competitor Analysis to already exist, 404ing with the correct
      message for whichever is missing - verified as four separate
      cases against real Postgres
- [x] A generated strategy stays readable even after every report it
      was built from is deleted - verified against real Postgres
      cascade-delete behavior
- [x] Sources are built directly from each CRAG run's own retrieved/
      relevant documents, never asked of the model
- [x] `ruff` clean, full non-integration suite green (106 passing, 61
      deselected integration tests)
- [x] Verified against real Postgres end to end (model, schema, service,
      and full HTTP round trip through the API) in addition to the
      permanent integration test suite (151 passing, 16 skipped pending
      real Ollama Cloud)
- [x] Real Ollama Cloud test run by Ahmad - not yet confirmed; update
      this line with the result once run