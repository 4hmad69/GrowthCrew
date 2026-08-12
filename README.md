
# GrowthCrew

GrowthCrew is an AI-powered marketing department for small businesses and
early-stage startups.

> Your AI marketing team for planning, creating, launching, and improving
> campaigns.

## Current milestone

The repository currently implements Step 3: Workspace and Business Profile
Backend.

Implemented:

- FastAPI backend
- Streamlit foundation
- PostgreSQL 17
- pgvector
- SQLAlchemy ORM foundation
- Alembic migrations
- Workspace persistence
- Business-profile persistence
- Optimistic concurrency
- Soft workspace archival
- Typed REST contracts
- Unit tests
- PostgreSQL integration tests
- Ruff linting and formatting

Ollama Cloud remains the approved first LLM provider and will be integrated
during Step 5.

## Current architecture

```text
Streamlit
    |
    v
FastAPI
    |
    v
Application Services
    |
    v
Repositories
    |
    v
SQLAlchemy
    |
    v
PostgreSQL + pgvector