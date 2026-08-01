# GrowthCrew

GrowthCrew is an AI-powered marketing department for small businesses and
early-stage startups.

Tagline:

> Your AI marketing team for planning, creating, launching, and improving campaigns.

## Current milestone

This repository currently implements Step 1: Project Foundation.

Available functionality:

- FastAPI backend
- Streamlit frontend
- Typed health endpoint
- Frontend-to-backend connectivity
- Environment-based configuration
- Central unexpected-error handling
- Automated tests
- Ruff linting and formatting

No marketing agents or LLM calls are implemented yet.

Ollama Cloud is the approved first LLM provider and will be integrated in Step 5.

## Architecture

```text
Browser
   |
   v
Streamlit frontend
   |
   | HTTP
   v
FastAPI backend