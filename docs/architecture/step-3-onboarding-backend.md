# Step 3 — Onboarding Backend Architecture

## Purpose

Step 3 introduces GrowthCrew's first persistent product-domain entities:

- Workspace
- Business Profile

The Streamlit onboarding UI remains deferred until Step 4.

## Domain relationship

```text
Workspace
    |
    | one-to-zero-or-one
    v
BusinessProfile