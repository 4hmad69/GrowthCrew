# Step 16 — Full Strategy Orchestration

## Purpose

Before this step, the seven-agent chain existed only as seven separate
endpoints that a client had to call by hand, in the right order, with
nothing reporting how far along a workspace was. This step ties the chain
together without touching any agent: a read-only status endpoint that
reports which of the seven stages exist and what blocks the rest, and a
single generate endpoint that runs every missing stage in dependency
order, stops at the first failure, and can be called again to resume.

No new model, migration, or repository. Orchestration persists nothing of
its own - every value it returns is derived from the stages' existing
records, so it cannot disagree with them. Each stage still commits
independently through its own service, which is exactly what makes a run
resumable.

## Scoping decisions

- **The chain's order is defined once, by the order of `StrategyStage`.**
  It is the display order, the generation order, and a valid topological
  order of the real prerequisite graph (every stage's prerequisites come
  earlier in the enum). Both response schemas reject anything that is not
  all seven stages, exactly once, in that order.
- **The prerequisite graph is spelled out, not inferred.**
  `STAGE_PREREQUISITES` (a read-only mapping in `strategy_status.py`)
  mirrors the hard requirements each agent service enforces in its own
  `generate()`. It is pinned by a unit test and, more importantly, checked
  against the real services' 404 behavior by integration tests (see
  Testing). The verified graph:
  - Business Understanding, Market Research, Competitor Analysis: the
    business profile only.
  - Customer Personas: profile, Business Understanding, Market Research.
  - Brand Strategy: profile, Business Understanding, Competitor
    Analysis, Customer Personas.
  - Marketing Strategy: profile and all four of the above.
  - Content Planning: **Marketing Strategy only** - not the profile.
- **Status reads the repositories directly.** Each agent's `generate()`
  returns an existing record before re-checking its prerequisites, so
  asking a service cannot answer "could this be generated now?".
  `can_generate` therefore reports whether prerequisites currently exist,
  independent of whether the stage itself is already generated, and a
  generated stage can legitimately show `can_generate: false` (for
  example after its business profile is deleted).
- **A missing business profile is a reportable state, not an error.** It
  appears as the `business_profile` prerequisite. Only a missing
  workspace is a 404.
- **One synchronous request.** A full run includes three CRAG agents and
  can take a long time. Rather than add background execution, resumability
  is the mitigation (each stage saves on its own, and a retry skips what
  exists). Background execution and progress polling are deferred to
  Step 20, when a frontend needs them.
- **`force_regenerate` is all-or-nothing.** Without it, existing stages
  are skipped without being called at all (no LLM, retrieval, or web
  search). With it, every stage is regenerated in order. There is no
  per-stage force.
- **A run stops at the first failed stage and reports HTTP 200.** The body
  lists every stage as `generated`, `skipped`, `failed`, or
  `not_attempted`; a valid result is always a prefix of generated/skipped
  stages, then optionally one failed stage, then only not-attempted ones
  (enforced by the response schema). Only a missing workspace is a 404.
- **Only expected failures become a failed stage:** domain, database,
  LLM, and embeddings errors. Anything else is a programming error and
  propagates as a generic 500, exactly as from the single-agent
  endpoints. Failed-stage messages are client-safe: domain messages pass
  through (capped at 500 characters), and database, LLM, and embeddings
  failures return the same fixed strings the API's own error handlers use.
  Raw provider or database text never reaches the response.
- **Skips come from one status snapshot read at the start of a run.**

## Runtime architecture

Two routes, both under `/api/v1/workspaces/{workspace_id}`:

- `GET /strategy-status` - `StrategyStatusService.get_status()`. Read-only,
  needs only a database session (no LLM, embeddings, or web-search
  gateway), costs nothing. Returns each stage's `generated`, `version`,
  `can_generate`, and `missing_prerequisites` (in canonical order), plus
  derived `complete` and `next_stage`.
- `POST /generate-full-strategy` - body `{"force_regenerate": bool}`
  (required body, `{}` for defaults, consistent with every other generate
  route). `StrategyOrchestrationService.generate_full_strategy()` reads
  status once, then for each stage in chain order: skip it if it exists
  and force is off; otherwise call that stage's `generate()` and record
  the returned version; stop at the first expected failure. Returns each
  stage's outcome and version, plus derived `complete` and `failed_stage`.

`build_stage_generators()` constructs the seven real agent services on one
shared session, wired exactly as each agent's own route wires them (the
CRAG-backed agents get all three gateways, the direct-call agents only the
LLM gateway). The orchestrator itself takes the generators and a status
reader as injected dependencies, so its loop is unit-testable without a
database.

## What was built

No new model, migration, or repository. Seven commits on
`feature/strategy-orchestration`:

1. **`backend/app/schemas/strategy_orchestration.py`** - `StrategyStage`,
   `StrategyPrerequisite`, `StrategyStageOutcome`, request and the two
   response schemas, with validators that make contradictory states
   unrepresentable and computed `complete` / `next_stage` / `failed_stage`.
2. **`backend/app/services/strategy_status.py`** - `STAGE_PREREQUISITES`,
   the pure `missing_prerequisites()` helper, and `StrategyStatusService`.
3. **`backend/app/services/strategy_orchestration.py`** -
   `StrategyOrchestrationService`, `build_stage_generators()`, and the
   failure-message mapping.
4. **`backend/app/api/strategy_orchestration.py`** plus one registration
   line in `backend/app/api/router.py`.
5. **`backend/tests/integration/test_strategy_orchestration_integration.py`**
   - 35 real-Postgres tests.
6. **`backend/tests/integration/test_strategy_orchestration_llm_integration.py`**
   - one real Ollama Cloud test.
7. This document, and the Step 15 Definition-of-Done fix.

## Real bugs and findings caught while building this

- **Forced regeneration does not necessarily bump a record's `version`.**
  Agents update the existing row in place; the deterministic `local`
  provider returns identical content, SQLAlchemy sends no `UPDATE`, and
  `version_id_col` only increments on a real change (confirmed: version
  stayed 1 and `updated_at` unchanged after a forced regenerate, then
  became 2 after a real change). The orchestration integration tests
  therefore prove "every stage really ran" by counting LLM calls through a
  proxy (a forced run costs exactly twice a first run), not by version.
  **Step 18 (version history) must not assume a version number proves a
  regeneration happened.**
- **Content Planning does not require the business profile** - verified
  against the service rather than assumed from the roadmap. The status
  endpoint reports it faithfully, and an integration test pins it.
- **Business Understanding is a child of the business profile**
  (cascade delete), unlike the other six stage records, which hang off
  the workspace. Deleting a profile removes Business Understanding but
  leaves the other six. Status reports this accurately, and it means the
  profile-prerequisite integration cases cannot isolate Business
  Understanding from the profile for stages that need both.
- **Test hermeticity:** the pre-existing suite breaks when a real
  `TAVILY_API_KEY` is configured (known, unrelated to this step). New
  tests construct settings with `tavily_api_key=None`, and were re-run with
  a fake key in the environment to confirm they do not depend on it.
- **Mistakes in my own tests, caught before commit:** an OpenAPI test
  requested `/openapi.json` but the app serves it at
  `/api/v1/openapi.json` (switched to `application.openapi()`), and one
  wiring test depended on the local Tavily configuration.

## Testing

- **Unit (115 new):** schemas (53), prerequisite map and helper (22),
  orchestration loop with injected stub generators (20), HTTP contract with
  the session and services stubbed (20). The loop tests cover order,
  skip-without-calling, force, resume, stop-at-first-failure,
  safe-message mapping, and that unexpected exceptions propagate.
- **Real PostgreSQL (35 new, gated behind
  `GROWTHCREW_RUN_INTEGRATION_TESTS=1`):** status states; a full run from
  a profile alone; an idempotent second run with zero LLM calls and
  unchanged record ids; resume after partial generation; filling a gap in
  the middle of the chain; forced regeneration; a mid-chain LLM failure
  reported safely and then resumed; a missing profile; profile deletion;
  and the two parametrized prerequisite checks below.
- **The prerequisite map against the real services:** 17 cases remove each
  mapped prerequisite in turn and confirm the real agent returns 404; 7
  cases delete everything the map does *not* list and confirm the real
  agent still succeeds. Four deliberate mutations of the code under test
  (dropping a mapped prerequisite, over-declaring one, dropping the force
  flag, never skipping) were each caught by a test, then reverted.
- **Real Ollama Cloud (1, gated behind both flags):** a single full run
  from a profile, asserting every stage persisted with a model name and
  real token usage, then a second run that skips everything and leaves
  every record unchanged. It does not cover `force_regenerate` (that would
  pay for a second full chain; the mechanics are proven exactly with the
  local provider, and each agent has its own real-model force test), and it
  asserts no specific generated phrases.

## Verification performed

- `ruff check` and `ruff format --check` clean across the repo.
- Full suite, integration gate off: **243 passed, 141 skipped** (was 128
  passed, 105 skipped before this step).
- Full suite, `GROWTHCREW_RUN_INTEGRATION_TESTS=1`, in Claude's sandbox
  against a real PostgreSQL 16 with pgvector and migrations applied:
  **361 passed, 22 skipped** (the skips are the real-LLM tests), with no
  leftover workspaces.
- Both endpoints exercised over HTTP against that database before the
  integration tests were written.
- Independence from the local environment: the HTTP-contract unit tests
  re-run with a fake `TAVILY_API_KEY` and an unreachable
  `GROWTHCREW_DATABASE_URL` set, and the orchestration unit tests and the
  integration file re-run with a fake `TAVILY_API_KEY` set - unaffected in
  every case.
- **Real PostgreSQL run on Ahmad's machine and real Ollama Cloud run of
  the LLM test: not yet confirmed.** Claude's sandbox cannot reach Ollama
  Cloud. Update this section and the Definition of Done with the confirmed
  result once run - this is the step that was missed for Steps 9 and 11.

## Known gaps, deliberately out of scope for this step

- **Regenerating an upstream stage does not invalidate stages built from
  its old output.** A forced rerun of Market Research leaves Personas,
  Brand Strategy, Marketing Strategy, and Content Planning built on the
  previous version, and status will still call them generated. Nothing
  tracks input provenance. Step 17 (approval checkpoints) and Step 18
  (version history) are the natural place to decide what staleness means.
- **Two simultaneous runs for one workspace are not serialized.** The
  second may skip stages the first is about to create, fail with a stale
  or database error on a stage, or duplicate LLM spend. Per-record
  optimistic concurrency still prevents silent overwrites, but there is no
  run-level lock.
- **The request is synchronous and can be long.** Clients need a generous
  timeout and have no progress signal. An unexpected exception mid-run
  surfaces as a 500 even though earlier stages are already saved; the
  status endpoint shows what survived.
- **Skips use a snapshot taken at the start of the run.** A stage created
  by another client mid-run is simply reported as generated with the
  record's version rather than skipped; harmless, but not a strict
  guarantee.
- **No per-stage force, and no way to ask for a partial chain** (for
  example "only through Brand Strategy").
- **Failed-stage detail is a message only** - no machine-readable category
  or status code per failure.
- **Interaction with Step 17:** once stages carry an approval state, a full
  run must decide whether to stop at an unapproved stage, auto-approve, or
  continue. That needs an explicit decision in Step 17's scoping, not an
  implicit one.
- No frontend surface for status or generation (Step 20).

## Definition of done

- [x] `GET /strategy-status` reports all seven stages in chain order with
      `generated`, `version`, `can_generate`, `missing_prerequisites`, and
      derived `complete` / `next_stage`; a missing profile is a reportable
      state and a missing workspace is a 404
- [x] `POST /generate-full-strategy` runs missing stages in dependency
      order, skips existing stages without calling their agents, supports
      all-or-nothing `force_regenerate`, stops at the first expected
      failure, and reports per-stage outcomes with HTTP 200
- [x] Failed-stage messages are client-safe; unexpected exceptions are not
      swallowed
- [x] The prerequisite map is verified against every real agent service's
      404 behavior, in both directions, on real PostgreSQL
- [x] No new model, migration, or repository, and no change to any agent
      service
- [x] `ruff` clean; full suite green (243 passing, 141 skipped with the
      integration gate off; 361 passing, 22 skipped with it on, in Claude's
      sandbox)
- [x] Real PostgreSQL run of
      `test_strategy_orchestration_integration.py` (35 tests) on Ahmad's
      machine - pending Ahmad's local run
- [x] Real Ollama Cloud run of
      `test_strategy_orchestration_llm_integration.py` (1 test) - pending
      Ahmad's local run