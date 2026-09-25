# Step 14 — Brand Strategy Agent

## Purpose

The fourth synthesizing agent: turns Business Understanding's
differentiators, Competitor Analysis's four researched sections, and
Customer Personas' overview and persona set into a structured brand
positioning deliverable - an overview, a canonical positioning
statement, a value proposition, brand voice and tone guidance, a short
set of brand pillars, and a handful of candidate taglines. Like Customer
Personas and Content Planning, this agent runs no CRAG graph - it needs
no external grounding, only synthesis over reports that already did
their own grounding. One direct structured LLM call via
`LLMGateway.structured_with_usage()`, no embeddings or web-search
gateway dependency.

Before this step, "brand strategy" did not exist as its own deliverable
anywhere in the codebase: `brand_tone` was a single free-text field on
the onboarding `BusinessProfile`, fed as input into other agents'
prompts but never synthesized into a positioning statement, value
proposition, or voice guidance of its own - exactly the scope gap the
project knowledge file names. This step turns it into a structured,
reusable object that Step 15 (extending Marketing Strategy's
prerequisites) will consume directly.

## Scoping decisions

- **Direct structured call, not the CRAG graph.** Brand Strategy is pure
  synthesis over research and personas that already exist. Same
  reasoning, and same shape, as Customer Personas: `BrandStrategyService`
  follows `CustomerPersonasService` / `ContentPlanService`, not the CRAG
  agents.
- **Hard-require the business profile, Business Understanding,
  Competitor Analysis, and Customer Personas, in that order.** The
  roadmap's own framing - "needs to know the customer and the
  competition before it can position against them" - names Competitor
  Analysis and Personas directly; Business Understanding is required for
  the same "no partial-data fallback" reasoning every prior synthesizing
  agent used, and because it is keyed by `profile_id`, requiring it
  already implies the profile exists.
- **Consequence for the dependency graph is deeper than the roadmap's
  three names.** Customer Personas is itself downstream of Market
  Research (Step 13), so generating a brand strategy transitively
  requires Market Research to have existed at some point too.
  `BrandStrategyService` does not re-check Market Research directly,
  though - it only needs the Personas record to exist right now, for the
  same reason Personas does not re-check Business Understanding's
  freshness: a generated artifact is a self-contained document, not a
  live view over its inputs. Step 16 (orchestration) must sequence Brand
  Strategy after both Competitor Analysis and the Market-Research-then-
  Personas leg.
- **Checks, in order:** workspace, then an existing brand strategy
  (returned immediately unless `force_regenerate`), then business
  profile, then Business Understanding, then Competitor Analysis, then
  Customer Personas. The existing-record check deliberately runs
  *before* any prerequisite is required, so a generated strategy stays
  readable even after the reports it was built from are regenerated or
  deleted.
- **`brand_pillars` and `tagline_options` have no `min_length` - a real
  constraint, not an assumption.** Same reasoning as every prior agent's
  list fields: the deterministic "local" provider used in Postgres-only
  tests fills every list-typed field with `[]` regardless of the schema,
  so a `min_length` would make every local-backed call fail Pydantic
  validation before it reached the service. The real "3-5 pillars, 3-5
  taglines" target lives in the prompt; the schema only caps the worst
  case (6 and 5 respectively). A unit test runs the draft schema through
  the real local-provider gateway so a constraint that would break every
  Postgres test fails immediately instead.
- **Full report text pasted into the prompt, not excerpted.** Same
  reasoning as Customer Personas: there is no CRAG rewrite step to lose
  anything to, and the model should see the whole competitive and
  customer picture it is positioning against. Each field sits on its own
  labeled line, and blank optional profile fields are omitted entirely
  rather than printed as `None`.
- **Flat string and list fields, not a nested object schema.** Unlike
  Customer Personas, which needed a child `PersonaSchema` because each
  persona carries nine sub-fields, Brand Strategy's output is a single
  positioning document - four `Text` fields plus two short JSONB string
  arrays, each with its own `jsonb_typeof(...) = 'array'` check
  constraint (matching Business Understanding's two-array precedent
  rather than Content Planning's or Customer Personas' single array).
- **Table and model naming.** The table holds one row per workspace
  (`brand_strategies`, model `BrandStrategy`), matching the
  plural-of-the-model-noun convention (`content_plans`,
  `marketing_strategies`). The service and API keep the agent's name
  (`BrandStrategyService`, `/brand-strategy`).
- **Keyed by `workspace_id`, no foreign key to any source report.** Same
  reasoning as every prior synthesizing agent. Verified against real
  Postgres in
  `test_brand_strategy_survives_prerequisite_reports_being_deleted`.

## Runtime architecture

```text
POST/GET /api/v1/workspaces/{id}/brand-strategy
      |
      v
BrandStrategyService
      |
      | existing brand strategy? -> return it (no LLM call) unless
      | force_regenerate
      |
      | otherwise requires, in order:
      |   BusinessProfile        (404 "Business profile not found.")
      |   BusinessUnderstanding  (404 "Business understanding has not
      |                           been generated yet.")
      |   CompetitorAnalysis     (404 "Competitor analysis has not been
      |                           generated yet.")
      |   CustomerPersonaSet     (404 "Customer personas have not been
      |                           generated yet.")
      v
builds one prompt from the profile's positioning-relevant fields
(including brand_tone), Business Understanding's summary /
differentiators / pain points, all four Competitor Analysis sections in
full, and Customer Personas' overview plus every persona rendered as
labeled lines
      |
      v
LLMGateway.structured_with_usage() - one direct call, no retrieval, no
CRAG graph, no embeddings or web-search gateway involved
      |
      v
overview + positioning_statement + value_proposition +
brand_voice_and_tone + brand_pillars (asked for 3-5) + tagline_options
(asked for 3-5) + input/output tokens
      |
      v
BrandStrategy (Postgres, 1:1 with workspaces, cascade delete)
```

## What was built

No new shared infrastructure was needed -
`LLMGateway.structured_with_usage()` already existed, so this step is
pure agent-shape work, same as Step 13.

- **`BrandStrategy` model + migration + repository** - four `Text`
  columns (`overview`, `positioning_statement`, `value_proposition`,
  `brand_voice_and_tone`), two JSONB string-array columns
  (`brand_pillars`, `tagline_options`) each with its own
  `jsonb_typeof` check constraint, optimistic concurrency via
  `version_id_col`, `model_used`/`input_tokens`/`output_tokens` for
  per-record cost auditability. `Workspace.brand_strategy` relationship
  added alongside `customer_persona_set`, and the model registered in
  `db/models/__init__.py`. Migration `20260924_0010` (revises
  `20260919_0009`) verified up, down, and up again against real
  Postgres, and inspected with `\d brand_strategies` - every constraint
  name matches convention (`pk_brand_strategies`,
  `uq_brand_strategies_workspace_id`,
  `fk_brand_strategies_workspace_id_workspaces`,
  `ck_brand_strategies_brand_pillars_array`,
  `ck_brand_strategies_tagline_options_array`).
- **Schemas**: `BrandStrategyGenerateRequest` (`force_regenerate` flag),
  `BrandStrategyResponse`. No nested object schema needed, unlike
  Customer Personas - `brand_pillars` and `tagline_options` are plain
  `list[str]`.
- **`BrandStrategyService`**: `generate()` and `get()` as described
  above. `_build_prompt()` turns the four source records into one
  prompt - the profile section swaps Customer Personas' `Country` line
  for `Existing brand tone`, since brand tone is the more relevant
  profile field for a positioning agent. `_BrandStrategyDraft` is the
  LLM-facing schema; a unit test pins its content-field names against
  `BrandStrategyResponse`'s so the two cannot drift apart. `_apply_draft()`
  copies a generated draft plus usage metadata onto the ORM record.
  Transaction handling matches every prior agent: `StaleDataError` maps
  to `StaleResourceError`, other database errors to
  `DatabaseOperationError`.
- **API endpoints**: `POST`/`GET` on `/workspaces/{id}/brand-strategy` -
  only `LLMGateway` injected. No changes needed to
  `main.py`/`errors.py`; `ResourceNotFoundError` and `StaleResourceError`
  already map generically through the shared `DomainError` handler.

## Real bugs and findings caught while building this

- **A throwaway sanity-check script, not committed code, made a wrong
  assumption about `version_id_col`.** While manually verifying the API
  round trip before writing the Postgres integration tests, a one-off
  script asserted `force_regenerate` always bumps `version`. It doesn't:
  the deterministic `local` stub produces byte-identical output for an
  unchanged prompt, so SQLAlchemy sees no net attribute change and
  `version_id_col` only increments on an actual `UPDATE` - the exact
  behavior Customer Personas' and Content Planning's own
  `test_force_regenerate_reuses_the_same_row` already document. No
  service code was wrong; only the throwaway script's assertion was,
  and it was never committed.
- **Directly confirmed, rather than assumed, that the LLM integration
  test's seeded scenario actually grounds.** Before committing
  `test_brand_strategy_llm_integration.py`, `_build_prompt()` was called
  in the sandbox against the exact seeded `CompetitorAnalysis` and
  `CustomerPersonaSet` records, and the resulting prompt text was
  checked against both keyword lists. This catches a class of bug the
  Postgres-only tests can't: a typo or mismatch between the seeded text
  and the keywords used to judge the real model's grounding, which would
  make the real-Ollama-Cloud assertion pass or fail for the wrong
  reason.
- Postgres stopped running in the sandbox between commands more than
  once during this step, same class of issue Steps 11 through 13
  documented; restarting it and re-verifying confirmed a clean suite
  each time, no code issue, and no data loss (the test database and its
  tables survived every restart).
- **Confirmed pre-existing, not introduced here**: `alembic check` still
  reports drift only on `knowledge_chunks.chunk_metadata`'s server
  default, same as Steps 10 through 13. Nothing is reported for
  `brand_strategies`.

## Testing

- No fake-session unit tests for the service layer, matching this
  codebase's established convention.
- `test_brand_strategy_schemas.py` (6 tests): request defaults and
  strictness, response validation from an ORM record, the empty-list
  case the local stub produces, and rejection of a response missing a
  required field or carrying an unexpected one.
- `test_brand_strategy_draft.py` (5 tests): the LLM-facing draft's
  content fields are a subset of the response schema's fields; a stored
  draft validates as the API schema; the real local-provider gateway can
  fill the draft; the brand-pillar cap and the empty-positioning-
  statement rejection.
- `test_brand_strategy_integration.py` (15 tests): real Postgres and the
  deterministic `local` stubs. Prerequisite reports are produced through
  the real endpoints (profile -> understanding -> competitor analysis,
  and separately profile -> market research -> personas), like Customer
  Personas' tests. Covers the 404 before generation, create+persist with
  the exact response fields, idempotent re-fetch, an LLM-call-counting
  test proving only `force_regenerate` spends tokens again,
  `force_regenerate` reusing the same row, a 404 for each of the four
  missing prerequisites in order plus a missing workspace, the brand
  strategy surviving deletion of its source reports, a forced
  regeneration failing without damaging the existing strategy, cascade
  delete from the workspace, and two direct `_build_prompt()` tests
  (report text present, blank fields omitted).
- `test_brand_strategy_llm_integration.py` (2 tests): real Postgres AND
  real Ollama Cloud together, gated behind both integration flags.
  Seeds a realistic profile, Business Understanding, Competitor
  Analysis, and Customer Personas for a specialty coffee subscription,
  with a deliberately distinctive and non-overlapping competitor
  differentiation opportunity ("roast-date transparency") and persona
  language ("home barista", "espresso"), so a model that ignored the
  competitor research or the persona and wrote generic positioning would
  visibly fail a check that both are represented. Also asserts real
  (non-zero) token usage that persists, output below `llm_num_predict`,
  and that the overview, positioning statement, value proposition,
  brand voice, pillars, and taglines are all non-empty. Pillar and
  tagline counts are printed (`pytest -s`) rather than asserted, since
  the schema deliberately does not enforce one. Its grounding keywords
  were confirmed, in the sandbox, to actually appear in the prompt built
  from the seeded records (see "Real bugs and findings" above) - this
  proves the test logic is sound, not the real model's behavior.
  **Confirmed passing against real Ollama Cloud** (`gpt-oss:120b-cloud`)
  by Ahmad: `2 passed`. The real model produced 5 brand pillars (Roast-to-
  Order Freshness, Transparent Origin Naming, Direct-Trade Relationships,
  Crafted for Home Baristas, Premium Yet Approachable) and 5 taglines
  (Freshness Delivered; From Farm to Cup in Days; Taste the Day It Was
  Roasted; Your Roast, Your Schedule; Coffee as Fresh as It Gets), both at
  the schema's cap. The positioning statement itself named the seeded
  roast-date/48-hour differentiation directly ("guarantees beans roasted
  within 48 hours of delivery... print the exact roast date and farm
  name on every bag") and the seeded persona ("urban young professionals
  who brew at home" plus the "Crafted for Home Baristas" pillar) - both
  grounding assertions passed. 1659 input / 1844 output tokens, well
  under the 8192 cap.

## Known gaps, deliberately out of scope for this step

- No per-field regeneration - `force_regenerate` always re-runs the
  whole strategy, same limitation every prior agent has at its own
  granularity.
- Brand pillar and tagline counts are not hard-enforced at exactly 3-5 -
  the prompt asks for it and the schema caps the worst case, so a real
  model drifting outside that range is a product-quality question, not
  something an automated assertion covers today. The real-provider test
  prints both counts so drift can be observed.
- Nothing consumes the brand strategy yet. Step 15 (extending Marketing
  Strategy's prerequisites) is where it starts feeding another agent;
  until then this is a standalone deliverable, same as Personas was
  before this step.
- A forced regeneration needs the source reports again: once any of
  them is deleted, `force_regenerate` returns 404 (leaving the existing
  strategy untouched) rather than regenerating from nothing. Deliberate,
  and covered by a test.
- The full report text is pasted into the prompt with no input-size
  guard, matching every prior direct-call agent.
- The Postgres-only integration tests still build their prerequisites
  before the `try` that guarantees cleanup, so a failure during setup
  can leak a workspace - the same inherited pattern as every prior
  integration file, this step's included. Fixing it would be a
  cross-cutting test-hygiene change, not part of this step.
- Frontend "view brand strategy" UI not built - backend-only step, same
  split as every prior agent step.

## Definition of done

- [x] `BrandStrategy` persists per workspace with optimistic concurrency
      and cascade delete from its parent workspace - verified against
      real Postgres (`\d`, direct ORM checks, and the permanent
      integration suite), not just the migration file
- [x] Migration applies, reverses, and re-applies cleanly against real
      Postgres; `alembic check` reports only the pre-existing
      `knowledge_chunks` drift
- [x] Generation is idempotent by default and makes no LLM call on a
      repeat request; `force_regenerate` re-runs the generation without
      creating a duplicate row
- [x] Generation requires the business profile, Business Understanding,
      Competitor Analysis, and Customer Personas in that order, 404ing
      with the correct message for each - verified against real
      Postgres
- [x] A generated brand strategy stays readable after the reports it
      was built from are deleted, and a failed forced regeneration
      leaves it untouched - verified against real Postgres
- [x] The competitor research, personas, and profile fields reach the
      prompt, and blank optional fields are omitted - verified directly
      against the prompt builder
- [x] `ruff` clean, full non-integration suite green (107 passing, 102
      deselected integration tests)
- [x] Verified against real Postgres end to end (model, schema, service,
      and full HTTP round trip through the API) in addition to the
      permanent integration suite (80 passing, 22 skipped pending real
      Ollama Cloud)
- [x] Real Ollama Cloud test run by Ahmad and confirmed passing:
      `2 passed` (`gpt-oss:120b-cloud`), 5 pillars, 5 taglines, 1659
      input / 1844 output tokens against the 8192 cap; positioning
      statement and pillars named the seeded roast-date differentiation
      and home-barista persona directly