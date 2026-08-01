# Step 1 Foundation Architecture

## Purpose

Step 1 proves that the GrowthCrew frontend and backend can run locally,
communicate through a typed HTTP contract, load environment configuration,
handle failures safely, and pass automated quality checks.

## Runtime architecture

```text
User's browser
      |
      v
Streamlit server
      |
      | HTTP GET /api/v1/health
      v
FastAPI server