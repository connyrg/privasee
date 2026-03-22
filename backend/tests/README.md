# PrivaSee Backend — Testing Strategy

## Overview

The test suite has three levels, each with a distinct responsibility:

**Unit tests** (`tests/unit/`, `@pytest.mark.unit`)
Test a single function or class in complete isolation. All external I/O (HTTP,
filesystem, databases) is either mocked or avoided entirely. Real library code
(PyMuPDF, Faker, dateutil) is exercised — only the system boundary is mocked.
Unit tests run in milliseconds and should be run after every save.

**Integration tests** (`tests/integration/`, `@pytest.mark.integration`)
Test a full API endpoint from HTTP request to HTTP response. The FastAPI app
runs in-process via `httpx.AsyncClient + ASGITransport`. Session storage
(UCSessionManager) and Databricks model serving are replaced with
`MagicMock(spec=SessionManager)` and `MOCK_DATABRICKS=True`. The goal is to
verify that the endpoint orchestrates its dependencies correctly — status codes,
response shapes, call ordering, error propagation — without any live services.

**Contract tests** (`tests/contracts/`, `@pytest.mark.contract`)
Verify that `MockDatabricksClient` (the in-process Databricks stand-in) produces
responses that conform to the schema the real Databricks endpoint is expected
to return (`DATABRICKS_RESPONSE_SCHEMA`, schema version 3.0.0).
A mismatch here means integration tests are giving false confidence:
the mock passes but the real endpoint would fail.

---

## Running tests locally

All commands must be run from the `backend/` directory.

```bash
# Fast feedback during development — unit + contract only, no I/O
make test-fast

# Unit tests only
make test-unit

# Integration tests only (tests the full API request/response cycle)
make test-integration

# Contract tests only (run when mock or Databricks schema changes)
make test-contract

# Full suite with coverage report — run before every commit
make test-all
```

`make test-all` enforces a minimum coverage of 70% and writes an HTML report
to `coverage_html/`. Open `coverage_html/index.html` in a browser to see which
lines are untested.

---

## When to run contract tests

Contract tests must be run in two situations:

1. **MockDatabricksClient is updated** — if `tests/contracts/mock_databricks_client.py`
   changes its output shape (new fields, renamed fields, different types), the
   contract tests will catch mismatches before they reach production.

2. **The Databricks model's `predict` return schema changes** — when the real
   model is retrained or its output contract is updated, the contract tests
   act as the specification: update them first to reflect the new schema, then
   update the mock to match.

Running only unit and integration tests after a schema change will give you a
fully green suite while the mock silently diverges from reality.

---

## Adding new tests

Use this decision rule to choose the right level:

| Question | Answer | Level |
|---|---|---|
| Does it test a single function with no external dependencies? | Yes | **unit** |
| Does it test an API endpoint behaviour end-to-end? | Yes | **integration** |
| Does it verify a mock matches a real service schema? | Yes | **contract** |

Practical examples:
- New helper in `session_manager.py` → unit test in `tests/unit/test_session_manager.py`
- New endpoint `POST /api/export` → integration tests in `tests/integration/test_export_endpoint.py`
- New field added to the Databricks response → contract test update in `tests/contracts/`
- New field added to the masking payload → integration test update in `tests/integration/test_approve_and_mask_endpoint.py`
- New endpoint `POST /api/sessions/{id}/verify` → integration tests in `tests/integration/test_verify_endpoint.py` (not yet written — should cover: 200 with text extraction, 404 on missing masked.pdf, empty entity list edge case)

Every test must carry one of `@pytest.mark.unit`, `@pytest.mark.integration`,
or `@pytest.mark.contract` so the Makefile targets select the right subset.

---

## Known limitations

**Integration tests use a mock Databricks client.** The `MOCK_DATABRICKS=True`
flag causes `POST /api/process` to call `_mock_entities()` instead of the real
Databricks Model Serving endpoint. This means integration tests cannot catch:

- Bugs in the Databricks PyFunc model's `predict` method
- Runtime errors in the model's UC file-fetch logic (`_fetch_original_file`)
- Performance regressions in the model
- Authentication or network failures in the Databricks workspace

These are caught by the end-to-end test script (`backend/scripts/e2e_upload_test.py`),
which must be run against a live environment with real Databricks credentials.

**`POST /api/sessions/{id}/verify` reports score = 100 for scanned PDFs.** The endpoint
extracts text using PyMuPDF's text layer. Image-only (scanned) PDFs have no text layer,
so every entity appears "masked" even if no redaction was applied. This is a known
limitation documented in the endpoint's docstring and surfaced as a tooltip in the UI.
