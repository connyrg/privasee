"""
Integration tests for POST /api/upload.

Uses the `client` fixture (httpx.AsyncClient + ASGITransport) and
`override_databricks_dependency` to replace the SessionManager with a mock
so no UC volume or Databricks workspace is needed.

TODO: implement tests covering:
  - 200 response for a valid PDF upload → returns session_id, filename, file_size
  - 200 response for a valid PNG upload
  - 400 for an unsupported file extension (e.g. .txt)
  - 413 for a file that exceeds MAX_FILE_SIZE_MB
  - 503 when _session_manager is None (not configured)
  - 501 when sm.create_session() raises NotImplementedError
  - 503 when sm.create_session() raises a generic exception
  - session_id in the response is a valid UUID string
  - preview_url in the response starts with "/api/files/uploads/"
"""

import pytest


@pytest.mark.integration
def test_placeholder():
    """Remove once real tests are in place."""
    assert True
