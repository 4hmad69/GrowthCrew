# Step 4 — Onboarding UI Architecture

## Purpose

Step 4 delivers the Streamlit UI that lets a user create a Workspace and
Business Profile through the existing Step 3 API, replacing direct API
calls with a guided onboarding flow. No new backend logic is introduced —
this step only consumes the existing `/api/v1/workspaces` and
`/api/v1/business-profiles` endpoints.

## Runtime architecture

```text
User's browser
      |
      v
Streamlit onboarding pages
      |
      | onboarding_api.py (httpx, via shared backend_http.py)
      v
FastAPI (Step 3 endpoints)
      |
      v
Service layer -> Repository layer -> PostgreSQL
```

## Screen flow

```text
Home
  |
  v
New Workspace (name)
  |
  v
Business Profile Form (business name, website, product/service, industry,
country, target customer, price range, monthly budget + currency, main
goal, existing channels, brand tone, known competitors, current
challenges, additional instructions)
  |
  v
Review & Confirm (read-only summary of entered data)
  |
  v
Submit -> POST business profile
  |
  v
Success (Review screen, with "Start another workspace" reset)
```

## State handling

- Form state lives in `OnboardingState` (a dataclass), stored under one key
  in `st.session_state` and accessed only through
  `frontend/state/onboarding_state.py` — no raw `st.session_state` access
  in the view modules
- The flow advances via `advance_to()` + `st.rerun()`; step order is
  `new_workspace -> business_profile -> review`

## Validation & error UX

- Client-side validation runs through the same `BusinessProfileCreateRequest`
  the API client sends, including the budget/currency pairing rule and
  currency-uppercasing, mirroring the backend's own validator
- 409 (duplicate profile) surfaces as `OnboardingConflictError` with the
  backend's message shown directly
- 422 responses surface as `OnboardingValidationError`, carrying the raw
  FastAPI field-error list so the UI can report "field: message" instead
  of a generic banner

## Frontend files (as built)

```text
frontend/
├── services/
│   ├── backend_http.py      # shared HTTP plumbing (added during Step 4;
│   │                          extracted from api_client.py so onboarding_api.py
│   │                          didn't duplicate connection/error handling)
│   ├── api_client.py         # unchanged public API, now delegates to backend_http.py
│   └── onboarding_api.py     # workspace + business-profile client
├── state/
│   └── onboarding_state.py
└── views/
    ├── new_workspace.py
    ├── business_profile_form.py
    └── review_confirm.py
```

## Testing strategy (as built)

- Unit tests for `OnboardingApiClient` via `httpx.MockTransport`
  (`test_onboarding_api.py`): success, connection failure, 409, 422
- Unit tests for the business-profile form's pure logic
  (`test_business_profile_form.py`): budget/currency pairing, comma-list
  parsing, error-message formatting — no Streamlit runtime needed
- One real end-to-end test (`test_onboarding_flow_integration.py`),
  gated behind `GROWTHCREW_RUN_INTEGRATION_TESTS=1` like the Step 3
  backend integration tests: drives the actual UI via
  `streamlit.testing.v1.AppTest` against the real FastAPI app (in-process
  over `httpx.ASGITransport`) and real Postgres, then cleans up the row
  it created

## Definition of done

- [x] User can create a workspace and business profile end-to-end
      through the UI without touching the API directly
- [x] Optimistic-concurrency conflicts are handled gracefully in the UI
      (duplicate-profile 409 surfaces as a clear message, not a crash)
- [x] Client-side and server-side validation stay in sync — the frontend
      request model mirrors the backend's Pydantic constraints and
      validators exactly
- [x] `ruff` clean, `pytest` green
- [x] Home page milestone caption updated from the stale "Step 2" text
      to Step 4