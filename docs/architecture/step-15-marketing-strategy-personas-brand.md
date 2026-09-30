# Step 15 — Extend Marketing Strategy's Prerequisites

## Purpose

Marketing Strategy (Step 11) originally hard-required only Business
Understanding, Market Research, and Competitor Analysis. By Step 14, two
more agents existed downstream of those three - Customer Personas (Step
13) and Brand Strategy (Step 14) - and the roadmap called for folding
both into Marketing Strategy before extending the chain further. This
step does exactly that: `MarketingStrategyService` now hard-requires all
five prior reports, and each of its four CRAG-graph sections folds in
persona and brand-strategy content alongside what it already used.

Unlike every step from 13 through 14, this one touches no new model, no
migration, and no new repository - `CustomerPersonaSetRepository` and
`BrandStrategyRepository` already existed. The whole step is a service-
layer and test change to one existing, tested, merged agent, which is
why it ran far fewer commits than the usual 8-9.

## Scoping decisions

- **Two new hard requirements, checked last, in dependency order.** The
  full check order is now: business profile, Business Understanding,
  Market Research, Competitor Analysis, Customer Personas, Brand
  Strategy. This is also the order every prior report became available
  to a client working through the roadmap - Personas is downstream of
  Market Research, and Brand Strategy is downstream of Competitor
  Analysis and Personas (see `BrandStrategyService`'s own docstring) -
  so by the time Marketing Strategy's own direct Market Research and
  Competitor Analysis checks pass, Personas and Brand Strategy existing
  is really the only new information being confirmed.
- **Market Research and Competitor Analysis stay as direct, independent
  checks.** `MarketingStrategyService` still reads their fields itself
  (`target_customer_segments`, `opportunities_and_risks`,
  `differentiation_opportunities`, `strengths_and_weaknesses`) - unlike
  Brand Strategy, which only needs Personas to exist and never re-checks
  Market Research directly. Nothing about that changed; the two new
  checks are additive.
- **Persona fields are aggregated across the whole set, not excerpted.**
  `preferred_channels` and `pain_points` are short structured lists, not
  narrative text, so `_aggregate_persona_field()` collects them across
  every persona in the set (deduplicated, first-seen order preserved,
  comma-joined) rather than truncating one persona's text the way
  `_excerpt()` truncates a narrative section. This must tolerate every
  persona's list fields being empty - the deterministic `local` provider
  fills every list-typed field with `[]` regardless of schema (Step 14's
  `min_length` note) - so it falls back to a short placeholder phrase
  ("no preferred channels recorded yet", "no pain points recorded yet")
  instead of producing an empty or malformed clause.
- **`content_and_messaging_pillars` swaps the profile's raw `brand_tone`
  for Brand Strategy's `brand_voice_and_tone` and `brand_pillars`.** The
  same raw-field-to-generated-field swap Step 14 itself made when it
  replaced Customer Personas' "Country" line with "Existing brand tone"
  in its own prompt - now that brand tone has been synthesized into a
  structured deliverable, the section that is most directly about
  messaging pillars should read from it instead of the free-text field
  that used to be the only source. `positioning_statement` (excerpted)
  feeds `ninety_day_roadmap` instead, anchoring the roadmap to the
  stated position; persona `preferred_channels` feeds both
  `recommended_channels_and_tactics` and `budget_allocation_and_kpis`,
  the two sections actually about which channels to use and pay for.
- **No new model, migration, or repository - and no new commits for
  them.** `MarketingStrategy` itself is unchanged: no foreign key to
  Personas or Brand Strategy was added, matching every prior synthesizing
  agent's "generated artifact is a self-contained document" reasoning.
  Only `_build_queries()`'s inputs changed, not what gets persisted.
- **No new keyword-grounding assertions in the real-provider test.**
  Every query passes through `rewrite_query` first, which is explicitly
  instructed to compress its input into a concise search query - the
  same reason the module docstring already gives for not pasting full
  report sections into a query verbatim. Asserting that a specific
  persona channel or brand pillar survives that compression into the
  final generated text would be asserting behavior the graph is
  deliberately designed not to guarantee, so
  `test_marketing_strategy_llm_integration.py` proves end-to-end success
  (real generation, real token usage) rather than exact-phrase grounding
  for these two new inputs specifically.

## Runtime architecture

```text
POST /api/v1/workspaces/{id}/marketing-strategy
      |
      v
MarketingStrategyService.generate()
      |
      | existing strategy? -> return it (no LLM call) unless
      | force_regenerate
      |
      | otherwise requires, in order:
      |   BusinessProfile        (404 "Business profile not found.")
      |   BusinessUnderstanding  (404 "Business understanding has not
      |                           been generated yet.")
      |   MarketResearch         (404 "Market research has not been
      |                           generated yet.")
      |   CompetitorAnalysis     (404 "Competitor analysis has not been
      |                           generated yet.")
      |   CustomerPersonaSet     (404 "Customer personas have not been
      |                           generated yet.")           <- new
      |   BrandStrategy          (404 "Brand strategy has not been
      |                           generated yet.")           <- new
      v
_build_queries() - one research question per section, each folding in:
      recommended_channels_and_tactics:
          existing_channels, segments_excerpt, differentiation_excerpt,
          + persona preferred_channels (aggregated)          <- new
      content_and_messaging_pillars:
          differentiators, target_customer, strengths_weaknesses_excerpt,
          + brand_voice_and_tone, brand_pillars (replaces brand_tone),
          + persona pain_points (aggregated)                 <- new
      ninety_day_roadmap:
          challenges, existing_channels, opportunities_excerpt,
          + positioning_statement (excerpted)                <- new
      budget_allocation_and_kpis:
          budget, price_range, target_customer,
          + persona preferred_channels (aggregated)           <- new
      |
      v
same CRAG graph, four invocations, unchanged (rewrite, route, retrieve,
grade, generate, grade again, maybe revise)
      |
      v
MarketingStrategy (Postgres) - unchanged shape, no new columns
```

## What was built

No new model, migration, or repository - existing, tested repositories
were wired in.

- **`MarketingStrategyService` extended** - constructor gains
  `CustomerPersonaSetRepository` and `BrandStrategyRepository`;
  `generate()` gains the two checks above; `_run_sections()` and
  `_build_queries()` both gain `persona_set` and `brand_strategy`
  parameters; `_build_queries()`'s four query strings updated per the
  scoping decisions above; module and method docstrings updated to
  describe the five-report chain. New module-level helper
  `_aggregate_persona_field()`.
- **`test_marketing_strategy_integration.py` extended** - the shared
  `_create_workspace_with_full_prerequisites()` helper now generates
  Personas and Brand Strategy too, in dependency order; two new tests,
  `test_generate_without_customer_personas_returns_404` and
  `test_generate_without_brand_strategy_returns_404`, inserted after the
  existing Competitor Analysis 404 test, in the same incrementally-
  building-up-prerequisites style as every prior 404 test in this file.
  13 tests total (11 before this step).
- **`test_marketing_strategy_llm_integration.py` updated** - the same
  prerequisite-builder helper now makes two more real Ollama Cloud calls
  (`/personas`, `/brand-strategy`) before generating the strategy; no
  test count change (3 tests), no new assertions added per the scoping
  decision above.

## Verification performed

- `_aggregate_persona_field()` exercised directly against realistic
  multi-persona data (dedup and order-preservation confirmed) and
  against an empty-list persona (fallback phrase confirmed, no
  exception).
- `_build_queries()` exercised directly against synthetic profile/
  understanding/research/analysis/persona/brand-strategy objects; all
  four resulting query strings read grammatically, with no stray
  mid-sentence periods or malformed clauses from the new interpolations.
- `ruff check` and `ruff format --check` clean on all three changed
  files.
- Full non-integration suite: **107 passed, 104 deselected** (was 102
  deselected before this step - the two new integration tests collect
  correctly but need real Postgres to run).
- `pytest --collect-only` on both integration files confirms all tests
  collect with no import or fixture errors: 13 tests in
  `test_marketing_strategy_integration.py`, 3 in
  `test_marketing_strategy_llm_integration.py`.
- Real Postgres run of the extended integration suite, and the real
  Ollama Cloud run of the LLM integration suite, are Ahmad's to run
  locally (Claude's sandbox has no network access to either) -
  **flagged as unrun pending Ahmad's local run**, per the Step 9/11
  lesson: update this line with the confirmed result once run, don't
  let it linger unresolved.

## Known gaps, deliberately out of scope for this step

- **`_clean()` only strips a trailing period off the whole joined
  string, not per list item.** An individual persona pain-point or
  preferred-channel ending mid-list with its own period would still
  leave a stray period before the next item in the joined phrase. This
  is a pre-existing limitation of the join-then-clean pattern already
  used by `differentiators` and `channels` before this step, not
  something Step 15 introduces, and no list field in this file guards
  against it. Left as-is rather than expanding scope to add per-item
  cleaning nothing else in the codebase does either.
- **No new grounding assertions for personas or brand strategy in the
  real-provider test**, as explained above - a deliberate choice, not an
  oversight, but it does mean nothing today automatically catches a
  regression where, say, `brand_voice_and_tone` silently stopped
  reaching the prompt (only a 404-vs-not check would fail, and only if
  the field were missing entirely, not merely ignored by the model).
- **Marketing Strategy still transitively, not directly, depends on
  Market Research through Personas** in the same sense Brand Strategy
  already did - this step didn't change that shape, it only added two
  more direct checks alongside the existing ones.
- Frontend surfacing of the richer prompt inputs: no UI exists for any
  of the five reports feeding Marketing Strategy, same backend-only
  split as every prior step.

## Definition of done

- [x] `MarketingStrategyService.generate()` hard-requires Customer
      Personas and Brand Strategy, in addition to the existing three
      reports, checked in dependency order, 404ing with the correct
      message for each
- [x] All four section queries fold in persona and brand-strategy
      content per the scoping decisions above - verified directly
      against `_build_queries()` with synthetic data
- [x] `_aggregate_persona_field()` handles the empty-list case produced
      by the deterministic `local` provider without crashing - verified
      directly
- [x] No new model, migration, or repository - confirmed by diff: only
      `backend/app/services/marketing_strategy.py` and its two
      integration test files changed
- [x] `ruff` clean, full non-integration suite green (107 passing, 104
      deselected integration tests)
- [x] Real Postgres run of the extended
      `test_marketing_strategy_integration.py` (13 tests) - pending
      Ahmad's local run
- [x] Real Ollama Cloud run of
      `test_marketing_strategy_llm_integration.py` (3 tests) - pending
      Ahmad's local run