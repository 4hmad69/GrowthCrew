# Step 2 Database Foundation

## Purpose

Step 2 establishes the PostgreSQL, pgvector, SQLAlchemy, Alembic, and
database-readiness foundation required by later GrowthCrew domain features.

No business tables are introduced during this milestone.

## Runtime architecture

```text
Streamlit
    |
    | HTTP
    v
FastAPI
    |
    v
SQLAlchemy Engine and Connection Pool
    |
    v
Psycopg 3
    |
    v
PostgreSQL 17 with pgvector