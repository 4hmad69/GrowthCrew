# GrowthCrew

GrowthCrew is an AI-powered marketing department for small businesses and
early-stage startups.

> Your AI marketing team for planning, creating, launching, and improving
> campaigns.

## Current milestone

The repository currently implements Step 2: Database Foundation.

Available functionality:

* FastAPI backend
* Streamlit frontend
* PostgreSQL 17 development service
* pgvector extension
* SQLAlchemy engine and session foundation
* Alembic migration infrastructure
* Process health endpoint
* Database readiness endpoint
* Typed frontend API client
* Unit, contract, and database integration tests
* Ruff formatting and linting

Ollama Cloud is the approved first LLM provider and will be integrated in
Step 5.

## Architecture

```text
Browser
   |
   v
Streamlit
   |
   | HTTP
   v
FastAPI
   |
   v
SQLAlchemy + Psycopg
   |
   v
PostgreSQL + pgvector