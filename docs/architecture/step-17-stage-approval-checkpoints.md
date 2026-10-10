# Step 17 — Per-Stage Human Approval Checkpoints

## Purpose

Until this step, the seven-stage chain ran on existence alone: any stage
could build on any other as soon as it existed, whether or not a person
had looked at it. Step 17 adds the human checkpoint the original product
vision described. A person reviews a generated stage and approves it, and
only then can the stages that build on it be generated.

Branch: `feature/stage-approval-checkpoints`. Backend only; the review UI
is Step 20.

## Scoping decisions

- **Approvals live in their own table, not a column on each stage.**
  `stage_approvals` holds one row per (workspace, stage), pinned to the
  exact stage record that was reviewed: the record's `id` **and** its
  `version`. A stage counts as approved only while its current record still
  matches both, so regenerating a stage retires its approval with no reset
  logic in any of the seven agent services. Approving never touches the
  artifact's own `version` or `updated_at` (a test proves it).
- **The record id is part of the pin, not just the version.** Deleting a
  stage and generating it again restarts at version 1. Matching on version
  alone would treat that new, unreviewed record as approved. A dedicated
  test pins this trap.
- **The gate is always on, in the four downstream agents** (Customer
  Personas, Brand Strategy, Marketing Strategy, Content Planning). Each
  `generate()` checks that every stage it builds on is approved, after its
  existing prerequisite-exists checks and before any prompt or LLM call. A
  blocked generation returns `409` and costs nothing. The three stages that
  need only the business profile are never gated.
- **Order of checks:** a missing prerequisite is still a 404, reported
  before approval is considered. An existing record is still returned
  without re-checking anything; `force_regenerate` is gated like a first
  generation.
- **Full runs pause at checkpoints.** `generate-full-strategy` stops at the
  first stage whose prerequisites await approval and reports it as the new
  `awaiting_approval` outcome (HTTP 200, naming the stages to approve) -
  not as a failure. The opt-in `auto_approve` request flag keeps a run going
  without a person: it approves each stage the run **generated**, recorded
  as `source=auto`, and never a stage it merely skipped.
- **No approver identity.** The application has no users yet, so approvals
  record `human` or `auto`, not who.
- **Staleness cascade is deliberately not solved.** Regenerating an
  upstream stage retires that stage's own approval, but stages built from
  its old output keep theirs. Provenance belongs to Step 18.

## Runtime architecture

- `StrategyRecordLoader` (`services/strategy_records.py`) is the one place
  that finds a workspace's seven stage records and decides which approvals
  are still current, so status and approval can never disagree.
- `StageApprovalService` (`services/stage_approval.py`) approves, revokes,
  reads, and exposes `require_prerequisites_approved()`, which raises
  `ApprovalRequiredError` (a `ResourceConflictError`, so it maps to 409).
- `unapproved_prerequisites()` sits beside `STAGE_PREREQUISITES` in
  `services/strategy_status.py`, so the gate and the status endpoint share
  one prerequisite map.
- Approval rules: only the version the reviewer looked at can be approved
  (stale version: 409); approving is idempotent; a person approving an
  auto-approved stage upgrades it to `human`, and auto never downgrades a
  human approval; revoking is idempotent and also clears stale rows; two
  simultaneous approvals settle to one via the unique constraint.

## What was built

- Model, migration `20261003_0011`, repository: `stage_approvals`
  (CHECK on stage, source, and `approved_version >= 1`; UNIQUE per
  workspace and stage; `record_id` deliberately not a foreign key because
  it points into seven tables; cascades with the workspace).
- Schemas: `ApprovalSource`, `StageApprovalRequest` (strict integer
  `version`), `StageApprovalResponse`; `StageApprovalState`
  (`draft`/`approved`) in the orchestration schemas.
- API, `/api/v1/workspaces/{workspace_id}/approvals/{stage}`: `POST`
  approve (body `{"version": N}`; 404 workspace or stage not generated, 409
  stale version), `GET` read the current approval (404 if not approved),
  `DELETE` revoke (204, idempotent). These routes need no LLM gateway.
- `GET /strategy-status` now reports, per stage, `approval`
  (`draft`/`approved`/none) and `unapproved_prerequisites`, plus top-level
  `approved` and `next_to_approve`. `can_generate` now means: every
  prerequisite exists **and** is approved.
- `generate-full-strategy`: `auto_approve` (strict boolean, default false),
  per-stage `auto_approved`, `awaiting_approval` outcome and
  `awaiting_approval_stage`.
- The `409` is documented in the OpenAPI spec of the four gated routes.

## Real bugs and findings caught while building this

- **Version alone is not identity.** Delete-and-regenerate restarts at
  version 1; found while designing the pin and covered by a real-PostgreSQL
  test so it cannot regress.
- **Turning the gate on broke 90 existing tests across 10 files**, which
  was the expected blast radius: every test that built the chain by plain
  generation. Fixed at the helper level (prerequisite helpers approve what
  they generate; Step 16's `_run` helper uses `auto_approve`) and by
  rewriting tests that had assumed an ungated chain, including a full
  human pause-and-resume loop.
- **Status must not claim more than the services allow.** `can_generate`
  changed meaning only together with the gate; a test compares it with what
  a forced generate then does, for every stage across four approval states.
- **With the deterministic local provider, a forced regeneration of an
  unchanged stage leaves its version alone**, so a human approval survives
  a forced auto run (tested). Behavior with a real provider, whose token
  counts differ run to run, is not asserted here.
- **A real-provider test, not a mock, was needed for the full run:** see
  Verification below.

## Testing

- Unit: schemas, the pure prerequisite and message rules, the HTTP
  contract with a stubbed service, orchestration with stub generators and a
  stub approver, and the OpenAPI contract of the gated routes.
- Real PostgreSQL integration: repository and constraints; the service
  (idempotency, version pinning, retirement on change, the recreate trap,
  races simulated by interleaving two sessions); the HTTP API end to end
  including zero LLM calls; auto-approval in full runs (skipped stages never
  approved, edited stages never rubber-stamped, failure and resume); and the
  gate per agent (409 message, nothing created, zero model calls,
  one-at-a-time unblocking, force gated, 404 before 409, stale approval
  blocks, status agrees with reality).
- Real Ollama Cloud: the five LLM test files were updated to approve their
  prerequisites or run with `auto_approve`.

## Verification performed

- Claude's sandbox (PostgreSQL 16 with pgvector): `ruff` clean; 675 passed,
  23 skipped with the integration gate on; 441 passed with it off. Gate
  and approval logic were mutation-tested: each deliberate break was caught
  by a test.
- Real Ollama Cloud run on Ahmad's machine (the five LLM test files, 37m38s):
  **9 of 10 passed**, including the full-strategy run with `auto_approve`
  and every force-regenerate, Personas, Brand Strategy, and Content Plan
  test. **1 failed:**
  `test_marketing_strategy_llm_integration.py::test_generate_finds_and_cites_a_seeded_chunk_with_real_embeddings`
  - the generation succeeded (HTTP 200, so the approval gate was passed)
  but the seeded chunk `internal:strategy-notes` was not among the cited
  sources. That assertion depends on real embeddings and real relevance
  grading, and Step 17 changes nothing in retrieval or the CRAG graph. It
  is **unresolved**: whether this is a regression or pre-existing is not
  known, because Step 11's doc records no earlier real-provider pass of
  this test. Ahmad's run also had a Tavily key configured.
  A second, single-test run of it failed earlier, in the Market Research
  prerequisite step, with a 503 from `LLMStructuredOutputError` (the real
  model returned output that did not fit the schema). Market Research is
  not gated and no approval code had run for it. The test therefore failed
  two different ways on two runs, which points to real-model
  nondeterminism rather than the gate, but it does not explain the first
  failure, so the original question stays open.

## Known gaps, deliberately out of scope for this step

- **No cascade invalidation.** Approving an upstream stage after
  regenerating it does not force its dependants to regenerate, and a
  dependant built on the old version keeps its approval (Step 18).
- No approver identity until the app has users.
- `auto_approve` is per run, not per stage.
- A plain run without `auto_approve` stops at Customer Personas on a fresh
  workspace until Business Understanding and Market Research are approved;
  that is the design, not a defect.
- Two simultaneous runs for one workspace are still not serialized.
- Real-provider tests are nondeterministic. The Marketing Strategy
  retrieval test (five real stages plus real embeddings) is the weakest
  and has failed in two different ways; revisit it if it recurs.
- No frontend for review or approval (Step 20).

## Definition of done

- [x] `stage_approvals` table, migration, repository; constraints verified
      on real PostgreSQL, including cascade with the workspace
- [x] Approve / read / revoke routes with version-pinned approval, human
      source only over HTTP, and zero LLM calls
- [x] An approval retires itself when its stage record changes, including
      delete-and-regenerate at the same version
- [x] Personas, Brand Strategy, Marketing Strategy, and Content Planning
      refuse to generate (including `force_regenerate`) until prerequisite
      stages are approved: 409 naming the stages, nothing created, no LLM
      call
- [x] `strategy-status` reports approval state, `unapproved_prerequisites`,
      and a `can_generate` that agrees with what generation does
- [x] `generate-full-strategy` pauses at checkpoints (`awaiting_approval`)
      and supports opt-in `auto_approve` that never approves skipped stages
- [x] `ruff` clean; 675 passing, 23 skipped with the integration gate on
      (441 passing with it off), in Claude's sandbox
- [x] Real Ollama Cloud run of the five LLM test files: 9 of 10 passed
- [ ] `test_generate_finds_and_cites_a_seeded_chunk_with_real_embeddings`
      (Marketing Strategy, real embeddings) failed on Ahmad's machine in two
      different ways across two runs (seeded chunk not cited; then a
      structured-output error in Market Research). Judged real-model
      nondeterminism, not the gate, but unproven - revisit if it recurs
- [x] Full PostgreSQL integration run on Ahmad's machine after the last
      commit: 675 passed, 23 skipped; failures limited to the known
      Tavily-key tests 