"""
Integration tests for POST /api/approve-and-mask.

Uses the `client` fixture (httpx.AsyncClient + ASGITransport) and
`override_databricks_dependency`.  The masking pipeline (_apply_masking_sync)
is exercised with PNG inputs (no poppler dependency) using `test_image_bytes`
and `sample_entities` from conftest.py.

TODO: implement tests covering:
  - 200 response for a valid PNG session + approved entity IDs
    → returns masked_pdf_url, entities_masked count
  - masked_pdf_url follows the pattern /api/files/output/{session_id}_masked.pdf
  - entities_masked equals the number of IDs in approved_entity_ids
  - 400 when approved_entity_ids is empty
  - 400 when no stored entities match the approved IDs
  - 400 when no entities are stored in the session at all
  - 404 when the session_id is not found
  - 404 when the original file is not found in UC storage (FileNotFoundError)
  - 422 when the original file is a .docx (masking not supported)
  - 501 when sm.get_session() raises NotImplementedError and no
    updated_entities provided
  - updated_entities replacement_text overrides are applied to entities_to_mask
  - Masking via run_in_threadpool completes without blocking the event loop
"""

import pytest


@pytest.mark.integration
def test_placeholder():
    """Remove once real tests are in place."""
    assert True
