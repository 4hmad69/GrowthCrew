# Step 18 — Deliverable Version History

## Purpose

Until this step each generated stage lived in exactly one row, overwritten
in place whenever it was regenerated. `version` is only an
optimistic-concurrency counter, so nothing past was kept, nothing recorded
what a stage had been built from, and nothing could say that a stage was
built from inputs that had since changed. Step 18 keeps every generation,
records what each was built from, and reports when a stage is stale.

Branch: `feature/deliverable-version-history`. Backend only; Step 19
(reports) reads this history and Step 20 (the dashboard) shows it. The step
is purely additive: it adds files and lines and removes none.

## Scoping decisions

- **One append-only table, not seven.** `stage_snapshots` holds the full
  content of one generation of one stage as JSONB, plus the stage record's
  `id` and `version`, the model and token usage, and the generation's
  inputs. UNIQUE per (workspace, stage, generation). A history table per
  stage would have multiplied migrations and models; instead a registry of
  each stage's generated fields (`STAGE_CONTENT_FIELDS`) is pinned to the
  seven stage models' columns by a unit test, so adding a column to a stage
  fails a test until history is told about it.
- **`generation` is its own counter, not the record's `version`.** A forced
  regeneration that produces identical content leaves `version` unchanged
  (a Step 17 finding), so a version comparison cannot prove that anything
  was regenerated. A snapshot is written on every persisted generation,
  identical or not; the generation number is what staleness is judged by.
- **History is written in the generation's own transaction.**
  `StageHistoryService.record()` never commits. It flushes the stage
  record, then the snapshot, inside the agent's existing `try:` block
  before `commit()`, so a generation and its snapshot are committed or
  rolled back together and the existing `StaleDataError` and
  `SQLAlchemyError` handling covers it with no new code.
- **Append-only is enforced by the database.** A `BEFORE UPDATE` trigger
  rejects any change to a snapshot row. `DELETE` stays allowed so rows
  still cascade away with their workspace. The repository exposes no update
  and no delete, and a test pins that.
- **The approval pin stays meaningful.** Any content change bumps the
  record's `version`, so two snapshots with the same `record_id` and
  `record_version` hold identical content. An approval, which is pinned to
  that pair, therefore always identifies the generation that was approved;
  a test shows a forced identical regeneration keeps its approval and both
  generations match it.
- **Provenance is recorded only when it can be stated exactly.** A
  snapshot's `inputs` names the business profile version and each upstream
  stage's generation, record id and record version the stage was built
  from. If any input cannot be identified - an upstream stage with no
  snapshot, an upstream record that has changed since its newest snapshot,
  a missing profile - `inputs` is SQL `NULL` (provenance unknown), never a
  partial or guessed answer. Unknown provenance is never reported as stale.
- **Staleness is derived, report-only, and generation-based.** Nothing is
  stored. A stage is stale when an input it was built from is now newer:
  the profile by version, an upstream stage by generation. It is judged by
  direct inputs only, so nothing propagates transitively. It is
  information only: it never touches an approval, `can_generate`, or the
  gate. A cascade that retired dependants' approvals would force chains of
  paid regenerations and needs a "regenerate stale" mode, which is a Step 20
  decision.
- **History is read-only over HTTP, with no restore.** Restoring a past
  generation would create a new record version that needs re-approval and
  would blur provenance. Step 19 only needs to read a specific generation.
- **Old history must stay readable.** The response schemas deliberately do
  not hold a snapshot's content to today's field list, so a stage gaining a
  field later cannot break reading its past. The field list is enforced when
  a snapshot is written, and by the drift test.
- **Pre-existing stages are backfilled by the migration.** Every existing
  stage record gets a generation 1 snapshot of its current content, with
  `inputs` NULL and `created_at` taken from the record's `updated_at`.

## Runtime architecture

- `StageHistoryService` (`services/stage_history.py`): `record()` flushes
  the record, copies the registered fields (deep-copied, so later edits to
  the record's lists cannot reach into history), captures inputs, takes the
  next generation, adds the snapshot and flushes it. If another request
  claimed the same generation number it rolls back and raises
  `StaleResourceError` (409). Other integrity errors pass through
  unchanged. Also `list_history()` and `get_snapshot()`.
- `stage_staleness.py`: the pure `assess_staleness()` rules and a
  `StageStalenessService` that applies them across a workspace. Kept apart
  from `stage_history.py` on purpose: the status service needs it, and the
  recorder imports the status module for the prerequisite map, so one
  module would have been an import cycle. For the same reason the staleness
  schema types live in `schemas/strategy_orchestration.py`.
- `StrategyStatusService` reports a generated stage's `generation` and
  `staleness` only when its newest snapshot still matches the live record
  by `id` and `version`. Otherwise it reports no generation and unknown
  provenance. This is computed after, and never feeds into, gating.
- Each of the seven agent services gained one `self._history.record(...)`
  call inside its existing commit `try:` block.
- The history routes depend on the database session only, never an LLM
  gateway.

## What was built

- Model, migration `20261010_0012`, repository: `stage_snapshots` (CHECK on
  stage, `generation >= 1`, `record_version >= 1`, and `content` being a
  JSON object and `inputs` NULL or a JSON object; UNIQUE per workspace,
  stage and generation; `record_id` deliberately not a foreign key because
  it points into seven tables; cascades with the workspace; append-only
  trigger; legacy backfill).
- Schemas (`schemas/stage_history.py`): `STAGE_CONTENT_FIELDS`,
  `ProfileInputRef`, `StageInputRef`, `StageInputs`, `StageSnapshotSummary`,
  `StageSnapshotDetail`, `StageHistoryResponse`. In
  `schemas/strategy_orchestration.py`: `ProvenanceState`
  (`tracked`/`unknown`), `StaleInput`, `StageStaleness`.
- API, `/api/v1/workspaces/{workspace_id}/history/{stage}` (list, newest
  first, without content; an empty list for a stage never generated) and
  `/history/{stage}/{generation}` (one generation in full, with `content` and
  `inputs`). `generation` is bounded to 1 through 2147483647. 404 for an
  unknown workspace or generation; every write method is 405.
- `GET /strategy-status` now reports, per generated stage, `generation` and
  `staleness` (`provenance`, `stale_inputs`, `stale`), plus a top-level
  `stale_stages`. A stage that is not generated reports neither.
- Existing code touched: the seven agent services (5 to 6 lines each),
  `strategy_status.py`, the orchestration schemas, the `Workspace`
  relationship, the model registry, the router, and one existing assertion
  (the exact per-stage dict in `test_strategy_orchestration_api.py`, which
  now includes the two new fields). Everything else is new.

## Real bugs and findings caught while building this

- **JSONB stores Python `None` as the JSON value `null`, not SQL `NULL`.**
  Unknown provenance written that way failed its own CHECK. Fixed with
  `JSONB(none_as_null=True)` on `inputs`; a test asserts the stored value
  is SQL `NULL`, because reading it back as `None` could not tell the two
  apart.
- **A huge `generation` in the URL was an unhandled 500.** The column is a
  32-bit integer, so 2147483648 or more reached the database as an
  out-of-range error. Found by probing against real PostgreSQL; the path
  parameter is now bounded and returns 422, with the upper bound itself a
  plain 404.
- **Version alone is not identity, again.** The reason the generation
  counter exists; covered by a real-PostgreSQL test that regenerates
  identical content and checks the version holds still while the
  generation advances and dependants go stale.
- **A pre-existing Step 17 test was broken.** On a clean checkout
  `test_rerunning_with_auto_approve_does_not_rubber_stamp_existing_drafts`
  failed every time because of a stray `_run(...)` line left in by commit
  `f2c979e`. Found while taking the baseline; fixed and merged on
  `fix/approval-orchestration-test-setup` before Step 18 began.
- **A pre-existing migration drift was found and not fixed.** `alembic
  check` reports a difference on `knowledge_chunks.chunk_metadata` (a server
  default) on `main`. It is unrelated, so the new table's drift test is
  scoped to `stage_snapshots` and is not blamed for it.
- **The migration's backfill is tested against real legacy rows,** built by
  migrating a throwaway database to the revision before history, seeding
  it, then upgrading, rather than asserted on a database that never held
  pre-history data.
- **Two new tests first failed because of wrong expectations, not wrong
  code,** which is worth recording: regenerating personas also makes brand
  strategy stale, because brand strategy is built from personas; and a
  seeded `sources` item had to use the real `section`/`source`/`snippet`
  shape. The tests were corrected to state the real dependency graph.
- **The local providers cannot exercise some paths.** Their output does not
  depend on the business profile, so a content-changing regeneration is not
  reachable through the agents; one test changes content through the
  history service, the way an agent would. They also report no token usage,
  so the real-provider full-run test cannot be run against them: it fails on
  its own pre-existing `input_tokens > 0` assertion there, and so did the
  original file.

## Testing

- Unit: the model's static definition and the repository's append-only
  surface; the registry, its drift test against the stage models, and the
  response schemas; the pure staleness rules; the history HTTP contract with
  a stubbed service (paths, bounds, shapes, sanitized errors, read-only,
  no LLM dependency); and the history fields on the status schemas.
- Real PostgreSQL integration: the model, constraints, trigger and cascade;
  the migration, backfilled from real legacy rows, downgrade and re-upgrade,
  and a scoped autogenerate drift check; the recorder and staleness service
  (including two live sessions racing for one generation); every agent
  through the API (snapshots match live records, exact provenance, nothing
  half-saved on a gate, model, database or race failure); the history API
  over real data; staleness in the status without any effect on gating; and
  four end-to-end scenarios across approvals, deletion and a workspace that
  predates history.
- Real Ollama Cloud: the full-strategy run test gained structural history
  assertions (identity and structure only, never generated text).
- Mutation checks throughout: each guarded rule was broken on purpose and a
  test was required to fail.

## Verification performed

- Claude's sandbox (PostgreSQL 16 with pgvector): `ruff` clean; **905
  passed, 23 skipped** with the integration gate on and **565 passed** with
  it off, up from 675 and 441 at the start of the step: 230 new tests, 124
  of them unit. About 87 deliberate breaks across the seven commits were each
  caught by a test. Two mutation runs were discarded and redone because the
  sandbox database had stopped partway through, so only results from runs
  with a verified-green baseline are counted. One mutation (the status
  service ignoring whether a snapshot matches the live record) is not
  caught by the end-to-end file; it is pinned by the status integration
  tests instead.
- Mid-step the sandbox was re-synced from the remote and its own commits were
  replaced by Ahmad's. The pushed branch (`796ecb4`, commits 1 to 6) was
  then run as a baseline before further work: 901 passed and 23 skipped with
  the gate on, 565 with it off, identical to what had been verified before.
  The final commit was rebuilt and verified on top of it.
- **Not yet confirmed on Ahmad's machine (unresolved at the time of
  writing):**
  - The full PostgreSQL integration run after the last commit.
  - The real Ollama Cloud run of
    `test_strategy_orchestration_llm_integration.py` with the new history
    assertions. That edit is lint-clean, and its two new helpers were run
    against a real chain on the local stubs, where they pass, with five
    negative checks that fail as intended. The whole test function itself
    has not been run anywhere.

## Known gaps, deliberately out of scope for this step

- **No restore.** A past generation can be read, not made current again.
- **No retention.** Every generation is kept in full forever, and the list
  route returns all of them without pagination. That is fine at this scale;
  revisit it if history grows large.
- **No automatic cascade.** Staleness is reported, not acted on: nothing
  regenerates a stale stage or retires a dependant's approval, and there is
  no "regenerate stale" mode (Step 20).
- **Staleness is judged by direct inputs only.** Regenerating a stage makes
  the stages built from it stale, not the stages built from those.
- **Provenance is unknown for stages generated before history** and for any
  stage whose inputs could not be identified when it was built. Those are
  never reported as stale, however much has moved.
- **A record changed outside history is reported as unknown,** not detected
  as a change in its dependants.
- **Two simultaneous full runs for one workspace are still not serialized.**
  A race over one generation now fails the loser with a 409 instead of
  forking history, but the runs themselves are unchanged.
- No approver identity until the app has users.
- The pre-existing `alembic check` drift on `knowledge_chunks.chunk_metadata`.
- No frontend for history, staleness, or review (Step 20).
- The Step 17 open item is untouched and still open: the Marketing Strategy
  real-embeddings test failed in two different ways on Ahmad's machine and is
  judged real-model nondeterminism but unproven.

## What Step 19 can rely on

- `GET .../history/{stage}/{generation}` returns the exact content and inputs
  of any generation.
- An approval's `record_id` and `approved_version` identify which generation
  was approved: the snapshot with that pair holds the approved content, even
  after the live stage has been overwritten.
- `strategy-status` reports `stale_stages` and each stage's `staleness`, so
  a report can flag or refuse stale content. Provenance may be unknown, and a
  report should not call unknown content fresh.

## Definition of done

- [x] `stage_snapshots` table, migration `20261010_0012`, repository,
      append-only trigger, and legacy backfill; constraints, trigger and
      cascade verified on real PostgreSQL, and the backfill verified against
      real legacy rows
- [x] All seven agents write a snapshot in the same transaction as the
      generation; a gate, model, database, or generation-race failure leaves
      neither record nor snapshot
- [x] `generation` advances on every generation, including a forced
      regeneration of identical content, which leaves `version` and any
      approval unchanged
- [x] Provenance is recorded exactly or recorded as unknown, never guessed
- [x] Read-only history API: list and detail, bounded and documented, with
      no LLM dependency
- [x] `strategy-status` reports `generation`, `staleness` and `stale_stages`
      as information only: gating, `can_generate` and approvals are
      unchanged
- [x] `ruff` clean; 905 passing, 23 skipped with the integration gate on
      (565 passing with it off), in Claude's sandbox
- [x] The pushed branch (commits 1 to 6) re-verified in Claude's sandbox:
      901 passing, 23 skipped
- [x] Full PostgreSQL integration run on Ahmad's machine after the last
      commit
- [x] Real Ollama Cloud run of the full-strategy test with the new history
      assertions (edited and lint-clean; helpers verified on the local
      stubs; the test function itself has not been run)
- [ ] `test_generate_finds_and_cites_a_seeded_chunk_with_real_embeddings`
      (Marketing Strategy, real embeddings), carried over from Step 17 and
      untouched by this step: failed in two different ways, judged real-model
      nondeterminism but unproven - revisit if it recurs