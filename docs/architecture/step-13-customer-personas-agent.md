# Step 13 — Customer Personas Agent

## Purpose

The third synthesizing agent: turns Market Research's researched customer
segments and Business Understanding's inferred pain points into a small
set of concrete, structured customer personas (name, segment, summary,
demographics, goals, pain points, buying triggers, objections, preferred
channels) plus a short overview naming the primary persona. Like Content
Planning, this agent runs no CRAG graph - it needs no external grounding,
only synthesis over reports that already exist. One direct structured
LLM call via `LLMGateway.structured_with_usage()`, no embeddings or
web-search gateway dependency.

Before this step, "who is the customer" existed only as
`target_customer_segments`, one narrative section inside the Market
Research report, and as free-text fields on the onboarding profile. This
step turns it into a structured, reusable object that later agents (Brand
Strategy, then Marketing Strategy) can consume directly.

## Scoping decisions

- **Direct structured call, not the CRAG graph.** Personas are pure
  synthesis over research that already happened. Same reasoning, and same
  shape, as Content Planning: `CustomerPersonasService` follows
  `BusinessUnderstandingService` / `ContentPlanService`, not the CRAG
  agents.
- **Hard-require Business Understanding and Market Research (and,
  through Business Understanding, the business profile).** The roadmap
  described this agent two incompatible ways - as synthesizing over
  Market Research's `target_customer_segments`, and as requiring "only
  the business profile". Reading a Market Research report necessarily
  means one exists, so the second description could not hold. The
  decision: require both reports, because without the researched segments
  the model would be inventing customers rather than turning researched
  segments into usable personas - the same "no partial-data fallback"
  reasoning Marketing Strategy used for its own prerequisites. Business
  Understanding is keyed by `profile_id` (not `workspace_id`), so
  requiring it already implies the profile exists; unlike Content
  Planning's deliberate decision to avoid also requiring the profile,
  this adds no extra, avoidable failure path.
- **Consequence for the dependency graph.** Personas is downstream of
  Market Research, not an independent sibling of Competitor Analysis as
  the roadmap first sketched. It is independent *of Competitor Analysis*
  (neither reads the other), but it cannot be generated until Business
  Understanding and Market Research both exist. Step 14 (Brand Strategy)
  and Step 16 (orchestration) must order the chain accordingly.
- **Checks, in order:** workspace, then an existing persona set (returned
  immediately unless `force_regenerate`), then business profile, then
  Business Understanding, then Market Research. The existing-set check
  deliberately runs *before* any prerequisite is required, so a generated
  persona set stays readable even after the reports it was built from are
  regenerated or deleted.
- **`personas` has no `min_length`, and neither do its list fields - a
  real constraint, not an assumption.** The deterministic "local"
  provider used in Postgres-only tests fills every list-typed field with
  `[]` (and strings with a short snippet of the prompt) regardless of the
  schema. A `min_length` on the persona list or on any list inside a
  persona would make every local-backed call fail Pydantic validation
  before it reached the service. The real "exactly 3 personas, 3-5 items
  per list" target lives in the prompt; the schema only caps the worst
  case (`personas` at 5, each list at 6, plus per-field string caps).
  A unit test runs the draft schema through the real local-provider
  gateway so a constraint that would break every Postgres test fails
  immediately instead.
- **Full report text pasted into the prompt, not excerpted.** Same
  reasoning as Content Planning: there is no CRAG rewrite step to lose
  anything to, and the model should see every researched segment it is
  turning into a persona. Each field sits on its own labeled line, and
  blank optional profile fields are omitted entirely rather than printed
  as `None`.
- **Persona shape stored as a JSONB array, not a child table.** Personas
  are always read and regenerated together as one unit, and downstream
  agents consume the whole set. Same pattern as `ContentPlan.entries`,
  with a `jsonb_typeof(personas) = 'array'` check constraint.
- **Keyed by `workspace_id`, no foreign key to any source report.** Same
  reasoning as every prior synthesizing agent. Verified against real
  Postgres in `test_persona_set_survives_prerequisite_reports_being_deleted`.
- **Table and model naming.** The table holds one row per workspace with
  an array of personas, so it is named for the set (`customer_persona_sets`,
  model `CustomerPersonaSet`), matching the plural-of-the-model-noun
  convention (`content_plans`, `marketing_strategies`). The service and
  API keep the agent's name (`CustomerPersonasService`, `/personas`).

## Runtime architecture

```text
POST/GET /api/v1/workspaces/{id}/personas
      |
      v
CustomerPersonasService
      |
      | existing persona set? -> return it (no LLM call) unless
      | force_regenerate
      |
      | otherwise requires, in order:
      |   BusinessProfile      (404 "Business profile not found.")
      |   BusinessUnderstanding (404 "Business understanding has not
      |                          been generated yet.")
      |   MarketResearch       (404 "Market research has not been
      |                          generated yet.")
      v
builds one prompt from the profile's target fields, Business
Understanding's summary / differentiators / pain points, and Market
Research's full target_customer_segments text
      |
      v
LLMGateway.structured_with_usage() - one direct call, no retrieval,
no CRAG graph, no embeddings or web-search gateway involved
      |
      v
overview + up to 5 personas (asked for exactly 3) + input/output tokens
      |
      v
CustomerPersonaSet (Postgres, 1:1 with workspaces, cascade delete)
```

## What was built

No new shared infrastructure was needed - `LLMGateway.structured_with_usage()`
already existed, so this step is pure agent-shape work.

- **`CustomerPersonaSet` model + migration + repository** - `overview`
  `Text` column, JSONB `personas` array with a `jsonb_typeof` check
  constraint, optimistic concurrency via `version_id_col`,
  `model_used`/`input_tokens`/`output_tokens` for per-record cost
  auditability. `Workspace.customer_persona_set` relationship added
  alongside `content_plan`, and the model registered in
  `db/models/__init__.py`. Migration `20260919_0009` (revises
  `20260916_0008`) verified up, down, and up again against real Postgres,
  and inspected with `\d customer_persona_sets` - every constraint name
  matches convention (`pk_customer_persona_sets`,
  `uq_customer_persona_sets_workspace_id`,
  `fk_customer_persona_sets_workspace_id_workspaces`,
  `ck_customer_persona_sets_personas_array`).
- **Schemas**: `CustomerPersonaSetGenerateRequest` (`force_regenerate`
  flag), `PersonaSchema`, `CustomerPersonaSetResponse`. Unknown fields are
  rejected on requests and on stored personas, so drift between stored
  JSON and the schema surfaces immediately.
- **`CustomerPersonasService`**: `generate()` and `get()` as described
  above. `_build_prompt()` turns the three source records into one
  prompt. `_PersonaDraft` / `_CustomerPersonasDraft` are the LLM-facing
  schemas; a unit test pins `_PersonaDraft`'s field names to
  `PersonaSchema` so the two cannot drift apart. `_apply_draft()` copies a
  generated draft plus usage metadata onto the ORM record. Transaction
  handling matches Content Planning: `StaleDataError` maps to
  `StaleResourceError`, other database errors to `DatabaseOperationError`.
- **API endpoints**: `POST`/`GET` on `/workspaces/{id}/personas` - only
  `LLMGateway` injected. No changes needed to `main.py`/`errors.py`;
  `ResourceNotFoundError` and `StaleResourceError` already map generically
  through the shared `DomainError` handler.

## Real bugs and findings caught while building this

- **The roadmap contradicted itself about this agent's prerequisites.**
  See the scoping decision above; resolved by reading the actual data
  model (Business Understanding is keyed by `profile_id`) rather than
  picking one sentence over the other.
- **The Postgres integration fixture has to stub embeddings, not just the
  LLM.** The first run of `test_customer_personas_integration.py` failed 8
  tests with "Failed to connect to Ollama". Personas itself needs only the
  LLM gateway, but the tests build the Market Research prerequisite
  through its real endpoint, which runs the CRAG graph and therefore the
  embeddings gateway. Fixed by matching Marketing Strategy's fixture
  (`embeddings_provider="local"`); the fixture docstring records why.
- **A wrong test expectation, not a service bug.** One test deleted both
  Business Understanding and Market Research and asserted the "Market
  research" 404. The service correctly reports Business Understanding
  first, since it is checked first. The test was corrected; the service
  was not touched.
- **Mutation-checked the permanent tests.** The service was temporarily
  broken four ways - never returning the existing set, checking
  prerequisites before returning it, dropping the market research from
  the prompt, and printing `None` for blank fields - and each was caught
  by the intended test. Notably, "never return the existing set" is *not*
  caught by the plain idempotency test (the row id and version are
  unchanged under the deterministic stub); only
  `test_only_force_regenerate_calls_the_llm_again`, which counts real
  gateway calls, catches it. That test is what guards the token-cost
  guarantee.
- **The local stub cannot prove the segments reach the model.** It
  returns `personas == []` whatever it is given, so no API-level test can
  distinguish a prompt that carries the market research from one that
  silently drops it. Two tests call `_build_prompt()` directly on known
  report text instead.
- **A setup failure leaks a workspace.** When the 8 tests above failed,
  each left a workspace behind, because the prerequisite-building helper
  runs before the test's `try`/`finally` cleanup. Removed manually. The
  same pattern exists in every prior integration file and in this step's
  Postgres-only file (see known gaps); only the real-provider test file
  seeds inside its `try`.
- Postgres briefly stopped running in the sandbox between commands again,
  same class of issue Steps 11 and 12 documented; restarting it and
  re-running confirmed a clean suite, no code issue.
- **Confirmed pre-existing, not introduced here**: `alembic check` still
  reports drift on `knowledge_chunks.chunk_metadata`'s server default,
  same as Steps 10 through 12. Nothing is reported for
  `customer_persona_sets`.

## Testing

- No fake-session unit tests for the service layer, matching this
  codebase's established convention.
- `test_customer_persona_set_schemas.py` (6 tests): request defaults and
  strictness, response validation from an ORM record with JSONB personas,
  the empty-list case the local stub produces, and rejection of missing
  or unexpected persona fields. Also spot-checked against a JSONB round
  trip through real Postgres.
- `test_customer_personas_draft.py` (5 tests): the LLM-facing draft and
  the API schema have identical persona fields; a stored draft validates
  as the API schema; the real local-provider gateway can fill the draft;
  the persona cap and the empty-overview rejection.
- `test_customer_personas_integration.py` (14 tests): real Postgres and
  the deterministic `local` stubs. Prerequisite reports are produced
  through the real endpoints (profile -> understanding -> market
  research), like Marketing Strategy's tests. Covers the 404 before
  generation, create+persist with the exact response fields, idempotent
  re-fetch, an LLM-call-counting test proving only `force_regenerate`
  spends tokens again, `force_regenerate` reusing the same row, a 404 for
  each missing prerequisite plus a missing workspace, the persona set
  surviving deletion of its source reports, a forced regeneration
  failing without damaging the existing set, cascade delete from the
  workspace, and two direct `_build_prompt()` tests (report text present,
  blank fields omitted).
- `test_customer_personas_llm_integration.py` (2 tests): real Postgres
  AND real Ollama Cloud together, gated behind both integration flags.
  Seeds a realistic profile, Business Understanding, and Market Research
  with deliberately distinctive segments (night-shift nurses, amateur
  marathon runners, new parents), so a model that ignored the research
  and wrote stock personas would visibly fail a check that at least two
  of the three appear. Also asserts real (non-zero) token usage that
  persists, output below `llm_num_predict`, and every persona present is
  fully populated. The persona count is printed (`pytest -s`) rather
  than asserted, since the schema deliberately does not enforce one.
  Its assertions were first exercised in the sandbox against a scripted
  stand-in for the model (one good response accepted; five deliberately
  bad ones - generic personas, no personas, output at the token cap, an
  empty list field, only one segment grounded - each rejected with a
  clear message), which proved the test logic, not the real model's
  behavior. **Confirmed passing against real Ollama Cloud**
  (`gpt-oss:120b-cloud`) by Ahmad: `2 passed`. The real model produced
  exactly 3 personas - Ayesha the Night Nurse, Bilal the Runner, Sara
  the New Mom - one per seeded segment, at 1619 input / 3642 output
  tokens (well under the 8192 cap).

## Known gaps, deliberately out of scope for this step

- No per-persona regeneration - `force_regenerate` always re-runs the
  whole set, same limitation every prior agent has at its own granularity.
- Persona count is not hard-enforced at exactly 3 - the prompt asks for
  it and the schema caps at 5, so a real model drifting from 3 is a
  product-quality question, not something an automated assertion covers
  today. The real-provider test prints the count so it can be observed.
- Nothing consumes personas yet. Step 14 (Brand Strategy) and Step 15
  (extending Marketing Strategy's prerequisites) are where they start
  feeding other agents; until then this is a standalone deliverable.
- A forced regeneration needs the source reports again: once they are
  deleted, `force_regenerate` returns 404 (leaving the existing set
  untouched) rather than regenerating from nothing. Deliberate, and
  covered by a test.
- The full report text is pasted into the prompt with no input-size
  guard, matching Content Planning.
- The Postgres-only integration tests still build their prerequisites
  before the `try` that guarantees cleanup, so a failure during setup can
  leak a workspace - the same pattern as every prior integration file.
  Fixing it would be a cross-cutting test-hygiene change, not part of
  this step.
- Frontend "view customer personas" UI not built - backend-only step,
  same split as every prior agent step.

## Definition of done

- [x] `CustomerPersonaSet` persists per workspace with optimistic
      concurrency and cascade delete from its parent workspace -
      verified against real Postgres (`\d`, direct ORM checks, and the
      permanent integration suite), not just the migration file
- [x] Migration applies, reverses, and re-applies cleanly against real
      Postgres; `alembic check` reports only the pre-existing
      `knowledge_chunks` drift
- [x] Generation is idempotent by default and makes no LLM call on a
      repeat request; `force_regenerate` re-runs the generation without
      creating a duplicate row
- [x] Generation requires the business profile, Business Understanding,
      and Market Research in that order, 404ing with the correct message
      for each - verified against real Postgres
- [x] A generated persona set stays readable after the reports it was
      built from are deleted, and a failed forced regeneration leaves it
      untouched - verified against real Postgres
- [x] The researched segments, pain points, and profile fields reach the
      prompt, and blank optional fields are omitted - verified directly
      against the prompt builder
- [x] `ruff` clean, full non-integration suite green (117 passing, 86
      deselected integration tests)
- [x] Verified against real Postgres end to end (model, schema, service,
      and full HTTP round trip through the API) in addition to the
      permanent integration suite (183 passing, 20 skipped pending real
      Ollama Cloud)
- [x] Real Ollama Cloud test run by Ahmad and confirmed passing - record
      the persona count, input/output token usage, and pass/fail here
      **and** in the Testing section above, then remove the "not yet
      run" sentence there