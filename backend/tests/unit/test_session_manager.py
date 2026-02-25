"""
Unit tests for SessionManager.

All tests mock the Databricks SDK WorkspaceClient so no live workspace is
needed.  Once the SessionManager is fully implemented (NotImplementedError
removed), these stubs should be expanded to verify UC volume interactions.

TODO: implement tests covering:
  - create_session() calls self._client.files.upload() with the correct path
  - create_session() returns a SessionData with the given session_id
  - get_session() returns None when the UC path does not exist
  - get_session() deserialises session.json into a SessionData correctly
  - update_session() reads, merges, and writes session.json atomically
  - delete_session() calls self._client.files.delete() for every artefact
  - save_file() uploads bytes and returns the full UC volume path
  - get_file() raises FileNotFoundError for a missing path
  - save_entities() / get_entities() round-trip the entity list losslessly
"""

import pytest


@pytest.mark.unit
def test_placeholder():
    """Remove once real tests are in place."""
    assert True
