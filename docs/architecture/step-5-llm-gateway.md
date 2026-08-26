# Step 5 — LLM Gateway Foundation

## Purpose

Step 5 establishes a single, typed gateway that every future agent (Step 6
onward) imports instead of touching a provider SDK or retry logic
directly. Infrastructure only, no agents yet - same discipline as Step 2's
database foundation before any business tables existed.

## Runtime architecture

```text
Future agents (Step 6+)
      |
      v
LLMGateway (backend/app/llm/gateway.py)
      |
      | chat() / structured()
      v
ChatOllama (langchain-ollama) -> local Ollama daemon -> Ollama Cloud
                                   (or the "local" stub provider for tests)
```

## What was built

- **Settings**: `llm_provider`, `llm_model`, `llm_base_url`,
  `llm_request_timeout_seconds`, `llm_num_predict`, `llm_retry_attempts`,
  `llm_retry_initial_delay_seconds`, `llm_cost_per_1k_input_tokens`,
  `llm_cost_per_1k_output_tokens` - env-driven via the `GROWTHCREW_`
  prefix, defaults matching the proven agentic-rag config
  (`gpt-oss:120b-cloud` over `http://localhost:11434`)
- **Typed errors** (`backend/app/llm/errors.py`): `LLMGatewayError` ->
  `LLMProviderUnavailableError`, `LLMResponseError` ->
  `LLMStructuredOutputError`
- **Retry/backoff** (`backend/app/llm/retry.py`): full-jitter exponential
  backoff, transient-vs-permanent error classification, exhausted
  transient failures raise `LLMProviderUnavailableError`
- **Structured-output robustness** (`backend/app/llm/structured.py`):
  native structured output -> JSON-only-prompt fallback -> typed error.
  Deliberately does *not* fall back to a fabricated guess in production,
  unlike the source agentic-rag repo - `LocalStructuredRunnable` exists
  only for the `local` test/offline provider
- **`LLMGateway`** (`backend/app/llm/gateway.py`): ties retry and
  structured-output together behind `chat()`/`structured()`, captures
  token usage via LangChain callbacks (verified working for both the
  native and JSON-fallback structured paths against real Ollama Cloud),
  logs provider/model/tokens/estimated cost/duration per call
- **`/api/v1/health/llm`**: mirrors the existing database health pattern
  exactly, wired through a new `handle_llm_error` exception handler (503)

## Deliberate improvements over the source agentic-rag repo

- Full-jitter backoff instead of fixed-jitter (spreads retries more
  evenly, standard AWS-recommended strategy)
- Fixed a latent bug where int/float schema fields would crash the local
  structured-output fallback
- Removed the schema-name-specific guessing fallback in favor of a typed
  error - GrowthCrew will have many agent-specific schemas this module
  can't know about in advance; a fabricated guess is worse than failing
  loudly

## Testing

- Unit tests for every module using fakes/the local provider - no real
  Ollama Cloud needed for CI (settings, retry, structured-output, gateway,
  health - 5 files)
- One real end-to-end test against actual Ollama Cloud
  (`test_llm_gateway_integration.py`), gated behind its own
  `GROWTHCREW_RUN_LLM_INTEGRATION_TESTS` flag, kept separate from the
  Postgres integration flag since the two external dependencies are
  independent

## Known gaps, deliberately out of scope for this step

- No LLM evaluation harness yet
- No hybrid search / reranker (comes with the agentic-rag port to
  pgvector, not this step)
- Per-call temperature override not exposed (fixed at 0 for now; add when
  an agent actually needs it)

## Definition of done

- [x] `LLMGateway` wraps `ChatOllama` with retry/backoff and
      structured-output fallback
- [x] Token usage and estimated cost are logged per call
- [x] `/api/v1/health/llm` reports gateway reachability
- [x] Settings are env-driven via the existing `GROWTHCREW_` prefix
- [x] `ruff` clean, `pytest` green, unit tests need no real network