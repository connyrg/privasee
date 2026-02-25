"""
Integration tests for POST /api/process.

Uses the `client` fixture (httpx.AsyncClient + ASGITransport) and
`override_databricks_dependency`.  Entity extraction is tested with
MOCK_DATABRICKS=true (monkeypatched at module level) so no real Databricks
endpoint is required.

TODO: implement tests covering:
  - 200 response with MOCK_DATABRICKS=true → entities list is non-empty
  - 200 response: returned entities each have id, entity_type, bounding_box,
    original_text, replacement_text, page_number
  - 404 when the session_id is not found in the mock SessionManager
  - 501 when sm.get_session() raises NotImplementedError
  - 503 when _session_manager is None (not configured)
  - 503 when DATABRICKS_MODEL_ENDPOINT is empty and MOCK_DATABRICKS=false
  - 504 when the Databricks HTTP call times out (mock httpx.TimeoutException)
  - 502 when Databricks returns a non-200 status
  - 502 when the Databricks response cannot be parsed
  - field_definitions with zero items → 422 (pydantic min_length validation)
"""

import pytest


@pytest.mark.integration
def test_placeholder():
    """Remove once real tests are in place."""
    assert True
