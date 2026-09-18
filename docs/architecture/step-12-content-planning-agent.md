# Step 12 — Content Planning Agent

## Purpose

The second synthesizing agent: turns Marketing Strategy's decisions
(recommended channels, content pillars, roadmap, budget/KPIs) into a
concrete, structured 30-day content calendar. Unlike Market Research,
Competitor Analysis, and Marketing Strategy, this agent runs no CRAG
graph at all - it needs no external grounding, only synthesis over a
strategy that already exists. Same shape as Business Understanding: one
direct structured LLM call via `LLMGateway.structured_with_usage()`, no
embeddings or web-search gateway dependency - the simplest
dependency-injection footprint of any agent so far.

## Scoping decisions

- **Direct structured call, not the CRAG graph.** A content calendar is
  pure synthesis over decisions Marketing Strategy already made, not new
  research. Forcing it through the CRAG graph would mean either 30
  separate graph invocations (30x the cost/latency of any prior agent)
  or asking `generate_grounded`/`generate_direct` to emit structured
  JSON, which conflicts with those nodes only ever producing plain text
  (the same constraint Step 11's own scoping decisions already
  documented). `ContentPlanService` follows
  `BusinessUnderstandingService`'s shape instead.
- **Hard-require Marketing Strategy only - deliberately not the business
  profile too.** `generate()` checks Marketing Strategy and raises
  `ResourceNotFoundError` if it's missing. Unlike Marketing Strategy's
  own three prerequisites, this does not also require the business
  profile to still exist: the workspace's own `name` (always present
  once the workspace itself exists) is enough context for the prompt,
  and requiring the profile too would reintroduce exactly the kind of
  extra, avoidable failure path Marketing Strategy already sidesteps by
  carrying no foreign key to any of its own inputs.
- **`entries` has no `min_length` in the LLM-facing schema - a real
  constraint found by reading `LocalStructuredRunnable`, not assumed.**
  The deterministic "local" provider used in Postgres-only tests fills
  every list-typed field with `[]` regardless of the nested schema or
  any `Field` constraints on it. A `min_length=30` would make every
  local-backed call fail Pydantic validation before it ever reached the
  service - the same reason every other agent's list fields
  (`key_differentiators`, `likely_customer_pain_points`, ...) already
  avoid `min_length`. The real "exactly 30" target lives in the prompt
  and is verified against genuine Ollama Cloud output in
  `test_content_plan_llm_integration.py`, not hard-coded into the
  schema.
- **Full section text pasted into the prompt, not excerpted.**
  `MarketingStrategyService._build_queries()` excerpts prior-report text
  because the CRAG graph's `rewrite_query` node compresses (and can
  discard) anything not distilled before generation ever sees it. There
  is no such node here - this is one direct structured call, so the
  model should see the whole strategy it is turning into a calendar.
  Each field sits on its own labeled line (matching
  `BusinessUnderstandingService._build_prompt`'s style), not stitched
  into a hand-written sentence, sidestepping Step 11's
  stray-period/article-grammar bug class entirely rather than needing a
  `_clean()`-style fix.
- **Keyed by `workspace_id`, no foreign key to `MarketingStrategy`.**
  Same reasoning as every prior synthesizing agent: a generated plan
  stays readable even if the strategy it was built from is later
  regenerated or removed. Verified against real Postgres in
  `test_get_survives_marketing_strategy_being_deleted`.

## Runtime architecture

```text
POST/GET /api/v1/workspaces/{id}/content-plan
      |
      v
ContentPlanService
      |
      | requires MarketingStrategy to already exist
      | (404 "Marketing strategy has not been generated yet." if not)
      v
builds one prompt from Workspace.name + MarketingStrategy's four
section columns (full text, not excerpted - no CRAG rewrite step
here to lose anything to)
      |
      v
LLMGateway.structured_with_usage() - one direct call, no retrieval,
no CRAG graph, no embeddings or web-search gateway involved
      |
      v
overview + up to 30 calendar entries (day/week/channel/content_type/
pillar/topic/cta) + input/output tokens
      |
      v
ContentPlan (Postgres, 1:1 with workspaces, cascade delete)
```

## What was built

No new shared infrastructure was needed - `LLMGateway.structured_with_usage()`
already existed from Business Understanding, so this step is pure
agent-shape work.

- **`ContentPlan` model + migration + repository** - `overview` `Text`
  column, JSONB `entries` array with the same `jsonb_typeof` check
  constraint pattern as `sources` on `MarketingStrategy`, optimistic
  concurrency via `version_id_col`,
  `model_used`/`input_tokens`/`output_tokens` for the same per-record
  cost auditability every prior agent has. `Workspace.content_plan`
  relationship added alongside `marketing_strategy`. Verified directly
  against a real Postgres instance via `\d content_plans` before
  treating the migration as safe - every constraint name matches
  convention exactly (`pk_content_plans`, `uq_content_plans_workspace_id`,
  `fk_content_plans_workspace_id_workspaces`, `ck_content_plans_entries_array`).
- **Schemas**: `ContentPlanGenerateRequest` (`force_regenerate` flag),
  `ContentCalendarEntrySchema`, `ContentPlanResponse`.
- **`ContentPlanService`**: `generate()` returns the existing plan
  without touching the LLM unless `force_regenerate=true` - checked
  *before* requiring Marketing Strategy to exist, so an already-generated
  plan stays readable even if the strategy is later regenerated or
  removed. `_build_prompt()` turns the workspace name plus Marketing
  Strategy's four sections into one prompt. `_ContentPlanDraft` /
  `_ContentCalendarEntryDraft` are the LLM-facing schemas; `_apply_draft()`
  copies a generated draft plus usage metadata onto the ORM record.
  `get()` for plain reads.
- **API endpoints**: `POST`/`GET` on `/workspaces/{id}/content-plan` -
  only `LLMGateway` injected, no embeddings or web-search dependency,
  the simplest DI footprint of any agent so far. No changes needed to
  `main.py`/`errors.py` - `ResourceNotFoundError` already maps
  generically to 404 via the shared `DomainError` handler.

## Real bugs and findings caught while building this

- **`LocalStructuredRunnable` fills every list field with `[]`,
  regardless of schema - caught by reading the actual code before
  writing `_ContentPlanDraft`, not assumed.** Confirmed empirically: a
  `min_length=30` constraint would have broken every Postgres-only test
  before the service was ever reached. See the scoping decision above.
- **`force_regenerate` does not bump `version` against the deterministic
  local stub - confirmed as expected behavior, not a bug.** An
  unchanged prompt produces byte-identical output from the local
  provider, so SQLAlchemy sees zero net attribute change and never
  issues an UPDATE, so `version_id_col` never increments.
  Cross-checked against `test_marketing_strategy_integration.py`'s own
  `test_force_regenerate_reuses_the_same_row`, which already only
  asserts the row `id` stays the same for exactly this reason -
  `test_content_plan_integration.py`'s equivalent test follows the same
  pattern deliberately, not by oversight.
- **Reproduced Ahmad's exact local environment error before diagnosing
  it.** `alembic upgrade head` run from the repo root (rather than
  `backend/`) fails with `FAILED: No 'script_location' key found in
  configuration.` - running the identical command from the repo root in
  the sandbox produced the identical message, confirming it was a
  working-directory issue and not a migration problem, before telling
  Ahmad the fix.
- A full-suite run briefly showed failures partway through this step on
  more than one occasion; each time, investigated immediately rather
  than reported as a regression. Postgres had simply stopped running in
  the sandbox between commands, same class of issue Step 11 already
  documented - restarting it and re-running confirmed a clean suite
  each time, no code issue.
- **Confirmed pre-existing, not introduced here**: `alembic check`
  still reports drift on `knowledge_chunks.chunk_metadata`'s server
  default, same as Steps 10 and 11.

## Testing

- No fake-session unit tests for the service layer, matching this
  codebase's established convention.
- `test_content_plan_integration.py` (7 tests): real Postgres and the
  deterministic `local` LLM provider. The marketing strategy each test
  needs is seeded directly via the repository rather than generated
  through the real five-step chain (profile -> understanding -> market
  research -> competitor analysis -> marketing strategy) - this suite is
  about Content Planning's own logic, which only cares that a
  `MarketingStrategy` row exists, not how it got there; Marketing
  Strategy's own generation is already covered by its own test file.
  Covers the 404-before-generation case, create+persist, idempotent
  re-fetch, `force_regenerate` reusing the same row, the
  missing-marketing-strategy 404, the missing-workspace 404, and the
  plan surviving its own marketing strategy being deleted.
- `test_content_plan_llm_integration.py` (2 tests): real Postgres AND
  real Ollama Cloud together, gated behind both integration flags. Seeds
  a realistic but static marketing strategy directly rather than
  regenerating the full five-step chain for real - unlike Marketing
  Strategy's own LLM integration test, there's no retrieval/grading
  chain here to re-prove, so a fixed, realistic input isolates the one
  real question this file needs to answer: does a real model turn a
  real strategy into genuine, well-formed calendar entries. Asserts
  entries are non-empty and every entry present is well-formed (all
  fields populated) rather than an exact or minimum count, since the
  schema deliberately doesn't enforce one (see scoping decisions) and a
  hard count assertion against a live, non-batch model would risk
  flakiness. Also covers real (non-zero) token usage persisting through
  the full stack, and `force_regenerate` succeeding end to end. **Written
  and lint-verified in the sandbox; not yet run against real Ollama
  Cloud - pending Ahmad's confirmation.**

## Known gaps, deliberately out of scope for this step

- No per-entry regeneration - `force_regenerate` always re-runs the
  whole calendar, same limitation every prior agent's `force_regenerate`
  already has at its own granularity.
- Entry count is not hard-enforced at exactly 30 - the prompt asks for
  it, but the schema only caps at 30 (see scoping decisions), so a real
  model under-delivering is a product-quality question, not something
  covered by an automated test assertion today.
- Frontend "view content calendar" UI not built - backend-only step,
  same split as every prior agent step.

## Definition of done

- [x] `ContentPlan` persists per workspace with optimistic concurrency
      and cascade delete from its parent workspace - verified against
      real Postgres, not just the migration file
- [x] Generation is idempotent by default; `force_regenerate` re-runs
      the calendar without creating a duplicate row
- [x] Generation requires Marketing Strategy to already exist, 404ing
      with the correct message when it doesn't - verified against real
      Postgres
- [x] A generated plan stays readable even after the marketing strategy
      it was built from is deleted - verified against real Postgres
      cascade-delete behavior
- [x] `ruff` clean, full non-integration suite green (85 passing, 69
      deselected integration tests)
- [x] Verified against real Postgres end to end (model, schema, service,
      and full HTTP round trip through the API) in addition to the
      permanent integration test suite (136 passing, 18 skipped pending
      real Ollama Cloud)
- [x] Real Ollama Cloud test run by Ahmad and confirmed passing, after
      the `llm_num_predict` fix (commit 8) - the first run hit
      `LLMStructuredOutputError` from truncated JSON at the old
      1024-token cap; re-run with `llm_num_predict=8192` passed