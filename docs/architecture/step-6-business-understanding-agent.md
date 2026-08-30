# Step 6 — Business Understanding Agent

## Purpose

Step 6 builds the first real agent: Business Understanding. It reads an
existing business profile (from Step 4's onboarding) and produces a
genuine synthesis - a narrative summary plus inferred attributes never
asked for directly on the form - rather than the fixed CRAG-style
retrieval graph the later research agents will need.

## Scoping decision

The original plan for this step was "port the agentic-rag graph + build
the first agent" in one go. Split instead: the CRAG graph (retrieve ->
grade -> generate -> grade again -> revise) is built for open-ended
research questions, which fits Market Research and Competitor Analysis
(Step 7) - agents that genuinely need to go find external information.
Business Understanding doesn't retrieve anything; it synthesizes from
data already on hand. Forcing it through the full RAG graph would have
been over-engineering. This step proves the gateway + persistence
pattern end to end; Step 7 is where pgvector and the RAG graph actually
show up.

## What the agent actually adds

Not a restatement of form fields - genuine inference:
- A narrative summary
- Inferred business stage (e.g. "early-stage startup")
- Inferred competitive category
- Inferred key differentiators
- Inferred likely customer pain points

## Runtime architecture

```text
POST/GET /api/v1/workspaces/{id}/business-profile/understanding
      |
      v
BusinessUnderstandingService
      |
      | builds prompt from BusinessProfile fields
      v
LLMGateway.structured_with_usage()
      |
      v
BusinessUnderstanding (Postgres, 1:1 with business_profiles, cascade delete)
```

## What was built

- **`BusinessUnderstanding` model + migration**: 1:1 with `business_profiles`
  (cascade delete), JSONB array fields for differentiators/pain points
  with the same `jsonb_typeof` check-constraint pattern as onboarding,
  optimistic concurrency via `version_id_col`, `model_used`/
  `input_tokens`/`output_tokens` for per-record cost auditability
- **Schemas**: `BusinessUnderstandingGenerateRequest` (`force_regenerate`
  flag) and `BusinessUnderstandingResponse`
- **`LLMGateway` extended**: added `chat_with_usage()` /
  `structured_with_usage()` alongside the existing `chat()`/`structured()`
  - the existing methods only *logged* usage, but the service needs it
  *returned* to persist per-record. Zero behavior change to the existing
  methods; all of Step 5's gateway tests pass unchanged.
- **`BusinessUnderstandingService`**: `generate()` returns the existing
  record without calling the LLM unless `force_regenerate=true` -
  generation costs real tokens, so nothing regenerates just because it
  was asked for again. `get()` for plain reads.
- **API endpoints**: `POST`/`GET` on
  `/workspaces/{id}/business-profile/understanding`, mirroring the
  onboarding endpoints' dependency-injection and error-handling shape
  exactly

## Bugs caught while building this

- FK constraint name exceeded Postgres's 63-character identifier limit
  (`fk_business_understandings_business_profile_id_business_profiles`) -
  caught by actually running the migration, not just writing it. Fixed
  by shortening the column to `profile_id`.
- Placeholder code mistakenly left in mid-edit (a stub function that
  would have raised `NotImplementedError` at runtime) - caught before it
  was ever saved, by reviewing the diff before writing the file.

## Testing

- No fake-session unit tests for the service layer - matching this
  codebase's existing convention (`BusinessProfileService` has none
  either): mocking SQLAlchemy transaction/constraint behavior
  convincingly is low-value: the real-Postgres integration suite is
  where this logic is actually tested.
- `test_business_understanding_integration.py`: real Postgres, the
  deterministic `local` LLM provider, gated behind
  `GROWTHCREW_RUN_INTEGRATION_TESTS`. Covers persistence and the API
  contract: create, idempotent re-fetch, `force_regenerate` reusing the
  same row, 404s for missing profile/workspace.
- `test_business_understanding_llm_integration.py`: real Postgres AND
  real Ollama Cloud together, gated behind both integration flags at
  once. Covers what only a real model can prove: genuine non-trivial
  synthesis and real token usage persisting through the full stack.
  Deliberately does not assert regenerated content differs or that
  version bumps - temperature is fixed at 0, so a real model can
  legitimately reproduce near-identical output for an unchanged prompt;
  asserting a difference would be a flaky test failing on correct code.

## Known gaps, deliberately out of scope for this step

- No retrieval/RAG - this agent works purely from the onboarding profile
  already on hand
- No website scraping, even though `website` is a captured field -
  clearly flagged as a future enhancement, not pulled into this step's
  scope
- Frontend "review the AI's understanding" UI not built yet - Step 6 is
  backend-only, same split as Steps 3/4

## Definition of done

- [x] `BusinessUnderstanding` persists with optimistic concurrency and
      cascade delete from its parent business profile
- [x] Generation is idempotent by default; `force_regenerate` re-runs it
      without creating a duplicate row
- [x] Token usage is captured and persisted per record, not just logged
- [x] `ruff` clean, full non-integration suite green
- [x] Verified against real Postgres (persistence, API contract) and
      real Ollama Cloud (genuine synthesis, real usage)