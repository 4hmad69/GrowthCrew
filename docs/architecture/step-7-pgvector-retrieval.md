# Step 7 — pgvector Retrieval Infrastructure

## Purpose

Step 7 builds the retrieval plumbing Step 2 provisioned but nothing had
used yet: embeddings, persistent vector storage, and workspace-scoped
similarity search. Infrastructure only, same discipline as every
foundation step before it - no Market Research agent, no CRAG graph.
Those come in Steps 8 and 9, once real retrieval exists to build on.

## Scoping decision

The original plan bundled "port the agentic-rag graph" with "build
Market Research" into one step. Split further: the CRAG graph (rewrite
-> decide -> retrieve -> grade -> generate -> grade again -> revise) is
a control-flow problem. Retrieval is a data-plumbing problem - embeddings,
storage, similarity search. Building the graph before real retrieval
existed would have meant mocking the most important part of it.

## Runtime architecture

```text
Future agents (Step 8+)
      |
      v
RetrievalService (backend/app/services/retrieval.py)
      |
      | add_chunk / add_chunks / search / reindex
      v
EmbeddingsGateway  ---->  KnowledgeChunkRepository
      |                          |
      v                          v
OllamaEmbeddings          Postgres (pgvector, HNSW index,
(nomic-embed-text)         cosine distance, workspace-scoped)
```

## What was built

- **`KnowledgeChunk`** - workspace-scoped, `vector(768)` column (matches
  `nomic-embed-text`'s real dimension, verified before building the
  schema rather than guessed), HNSW index using `vector_cosine_ops`,
  cascade delete from its parent workspace. No optimistic concurrency -
  chunks are write-once/read-many; a content change is delete-and-reinsert,
  not an in-place edit.
- **`EmbeddingsGateway`** - mirrors `LLMGateway`'s shape: provider
  selection (`ollama` / `local`), retry via the same `invoke_with_retries`
  used for chat calls, a deterministic hash-based `local` stub for tests.
- **`RetrievalService`** - `add_chunk`/`add_chunks`/`search`/`reindex`.
  `search()` deliberately has no similarity-score threshold - filtering
  by actual relevance is an LLM grading step's job (the CRAG graph's
  future `grade_documents` node), not a raw cosine-distance cutoff.
- **`/api/v1/health/embeddings`** - mirrors the LLM health check, plus
  one thing unique to embeddings: verifies the *returned dimension*
  matches what's configured, since a provider/model change returning a
  different width would otherwise only surface as a confusing insert
  failure later.

## A real, documented-not-guessed detail

Verified via web search before building the schema: `nomic-embed-text`
produces 768-dimensional embeddings, and its model card documents task
prefixes as *required*, not optional - `search_document: ` for indexed
content, `search_query: ` for queries. Skipping these silently degrades
retrieval quality without ever raising an error. `EmbeddingsGateway`
applies them automatically; callers never need to know they exist.

## Bugs and gaps caught along the way

- A foreign-key constraint name exceeded Postgres's 63-character limit -
  caught by running the migration, not just writing it.
- `backend/app/db/models/__init__.py`, the central model registry Alembic
  relies on, never had `BusinessUnderstanding` added when Step 6 shipped
  it - a real, pre-existing gap, fixed here since models were already
  being touched.
- `Settings`' `.env` loading resolved relative to whatever directory a
  command happened to run from, which silently broke when `alembic`
  (run from `backend/`) needed the `.env` file that lives at the project
  root. Fixed with an absolute, CWD-independent path.
- The nomic-embed-text prefix requirement itself (see above) - caught
  before it ever shipped ungrounded, by verifying the model's documented
  usage instead of assuming embeddings work the same everywhere.

## Testing

- Unit tests for the embeddings gateway (both providers, retry behavior,
  the required prefixes) - no real Ollama Cloud needed for CI.
- `test_retrieval_integration.py` - real Postgres, deterministic `local`
  provider, gated behind `GROWTHCREW_RUN_INTEGRATION_TESTS`. Proves
  mechanics: persistence, workspace isolation, `limit`, `reindex`.
- `test_retrieval_llm_integration.py` - real Postgres AND real Ollama
  Cloud together. Proves quality, not just mechanics: a query sharing
  zero exact words with the correct chunk still has to rank it first.

## Known gaps, deliberately out of scope for this step

- No text chunking/splitting - `add_chunk`/`add_chunks` treat one call as
  one chunk; splitting long documents is the caller's concern once a real
  ingestion pipeline (web scraping, document upload) exists
- No ingestion API endpoint - nothing outside tests calls `RetrievalService`
  yet; that wiring happens when an agent actually needs it
- No similarity-score threshold on `search()` - relevance filtering is an
  LLM grading concern, deferred to the CRAG graph

## Definition of done

- [x] `KnowledgeChunk` persists with correct dimension and a working HNSW
      cosine-distance index, verified against real Postgres
- [x] Search is correctly scoped per workspace - verified no cross-workspace
      leakage, both mechanically (local provider) and semantically (real
      Ollama Cloud)
- [x] Required nomic-embed-text task prefixes applied automatically,
      transparent to callers
- [x] `/api/v1/health/embeddings` catches a dimension mismatch, not just
      unreachability
- [x] `ruff` clean, full non-integration suite green