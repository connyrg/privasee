"""
Integration tests for POST /api/approve-and-mask.

Uses the `client` fixture (httpx.AsyncClient + ASGITransport) and
`override_databricks_dependency`.  The masking pipeline (_apply_masking_sync)
is patched in every test so no real image/PDF processing (and therefore no
poppler or OpenCV dependency) is required at this layer — those behaviours
are covered by the MaskingService unit tests.

Session storage calls (get_session, get_file, get_entities, save_file,
update_session) are all handled via the mock_session_manager from conftest.
"""

import pytest
from unittest.mock import MagicMock, patch

import app.main as main_module


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Minimal valid PDF bytes returned by the patched masking helper.
_FAKE_PDF_BYTES = b"%PDF-1.4\n1 0 obj<</Type/Catalog>>endobj\n%%EOF"

# Two representative stored entity dicts (shape matches model_dump() output).
_ENTITY_1 = {
    "id": "entity-1",
    "entity_type": "Full Name",
    "original_text": "John Smith",
    "replacement_text": "Jane Doe",
    "bounding_box": [0.05, 0.08, 0.45, 0.025],
    "confidence": 0.95,
    "approved": True,
    "page_number": 1,
}

_ENTITY_2 = {
    "id": "entity-2",
    "entity_type": "Email",
    "original_text": "john@example.com",
    "replacement_text": "[REDACTED]",
    "bounding_box": [0.05, 0.15, 0.45, 0.025],
    "confidence": 0.95,
    "approved": True,
    "page_number": 1,
}


def _make_session(filename: str = "document.png") -> MagicMock:
    """Return a mock SessionData object with a known filename."""
    mock_session = MagicMock()
    mock_session.filename = filename
    return mock_session


# ===========================================================================
# Group 1 — Happy path
# ===========================================================================


@pytest.mark.integration
async def test_approve_and_mask_returns_masked_pdf_url(
    client, override_databricks_dependency
):
    """200 response with masked_pdf_url, original_pdf_url, and entities_masked."""
    sm = override_databricks_dependency
    sm.get_session.return_value = _make_session()
    sm.get_file.return_value = b"fake image bytes"
    sm.get_entities.return_value = [_ENTITY_1]

    with patch("app.main._apply_masking_sync", return_value=_FAKE_PDF_BYTES):
        response = await client.post(
            "/api/approve-and-mask",
            json={
                "session_id": "test-session-id",
                "approved_entity_ids": ["entity-1"],
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == "test-session-id"
    assert "masked_pdf_url" in body
    assert "masked" in body["masked_pdf_url"]
    assert "original_pdf_url" in body
    assert body["entities_masked"] == 1


@pytest.mark.integration
async def test_approve_and_mask_only_redacts_approved_entities(
    client, override_databricks_dependency
):
    """Only the entity IDs listed in approved_entity_ids are passed to masking."""
    sm = override_databricks_dependency
    sm.get_session.return_value = _make_session()
    sm.get_file.return_value = b"fake image bytes"
    sm.get_entities.return_value = [_ENTITY_1, _ENTITY_2]

    captured: list = []

    def _capture(file_bytes, ext, entities):
        captured.extend(entities)
        return _FAKE_PDF_BYTES

    with patch("app.main._apply_masking_sync", side_effect=_capture):
        response = await client.post(
            "/api/approve-and-mask",
            json={
                "session_id": "test-session-id",
                "approved_entity_ids": ["entity-1"],  # only entity-1
            },
        )

    assert response.status_code == 200
    assert len(captured) == 1, f"Expected 1 entity passed to masking, got {len(captured)}"
    assert captured[0]["id"] == "entity-1"


@pytest.mark.integration
async def test_approve_and_mask_applies_entity_updates(
    client, override_databricks_dependency
):
    """replacement_text from updated_entities overrides the stored value."""
    sm = override_databricks_dependency
    sm.get_session.return_value = _make_session()
    sm.get_file.return_value = b"fake image bytes"
    sm.get_entities.return_value = [{**_ENTITY_1, "replacement_text": "Original Text"}]

    captured: list = []

    def _capture(file_bytes, ext, entities):
        captured.extend(entities)
        return _FAKE_PDF_BYTES

    with patch("app.main._apply_masking_sync", side_effect=_capture):
        response = await client.post(
            "/api/approve-and-mask",
            json={
                "session_id": "test-session-id",
                "approved_entity_ids": ["entity-1"],
                "updated_entities": [
                    {
                        "id": "entity-1",
                        "entity_type": "Full Name",
                        "original_text": "John Smith",
                        "replacement_text": "Updated Text",
                        "bounding_box": [0.05, 0.08, 0.45, 0.025],
                        "confidence": 0.95,
                        "approved": True,
                        "page_number": 1,
                    }
                ],
            },
        )

    assert response.status_code == 200
    assert len(captured) == 1
    assert captured[0]["replacement_text"] == "Updated Text", (
        f"Expected 'Updated Text', got: {captured[0]['replacement_text']!r}"
    )


@pytest.mark.integration
async def test_approve_and_mask_updates_status_to_completed(
    client, override_databricks_dependency
):
    """update_session must be called with status='completed' after successful masking."""
    sm = override_databricks_dependency
    sm.get_session.return_value = _make_session()
    sm.get_file.return_value = b"fake image bytes"
    sm.get_entities.return_value = [_ENTITY_1]

    with patch("app.main._apply_masking_sync", return_value=_FAKE_PDF_BYTES):
        response = await client.post(
            "/api/approve-and-mask",
            json={
                "session_id": "test-session-id",
                "approved_entity_ids": ["entity-1"],
            },
        )

    assert response.status_code == 200
    sm.update_session.assert_called_with("test-session-id", status="completed")


# ===========================================================================
# Group 2 — Error responses
# ===========================================================================


@pytest.mark.integration
async def test_approve_and_mask_returns_404_for_unknown_session(
    client, override_databricks_dependency
):
    """404 when get_session returns None for an unrecognised session_id."""
    sm = override_databricks_dependency
    sm.get_session.return_value = None

    response = await client.post(
        "/api/approve-and-mask",
        json={
            "session_id": "unknown-session",
            "approved_entity_ids": ["entity-1"],
        },
    )

    assert response.status_code == 404
    body = response.json()
    assert "error" in body
    assert "unknown-session" in body["error"]


@pytest.mark.integration
async def test_approve_and_mask_returns_404_when_original_file_not_found(
    client, override_databricks_dependency
):
    """404 when the original file is missing in UC storage (FileNotFoundError)."""
    sm = override_databricks_dependency
    sm.get_session.return_value = _make_session()
    sm.get_file.side_effect = FileNotFoundError("original.png not found")

    response = await client.post(
        "/api/approve-and-mask",
        json={
            "session_id": "test-session-id",
            "approved_entity_ids": ["entity-1"],
        },
    )

    assert response.status_code == 404
    body = response.json()
    assert "error" in body


@pytest.mark.integration
async def test_approve_and_mask_returns_400_when_no_approved_entities_match(
    client, override_databricks_dependency
):
    """400 when none of the approved_entity_ids match entities stored in the session."""
    sm = override_databricks_dependency
    sm.get_session.return_value = _make_session()
    sm.get_file.return_value = b"fake image bytes"
    sm.get_entities.return_value = [_ENTITY_1]  # entity-1 is stored

    response = await client.post(
        "/api/approve-and-mask",
        json={
            "session_id": "test-session-id",
            "approved_entity_ids": ["non-existent-id"],
        },
    )

    assert response.status_code == 400
    body = response.json()
    assert "error" in body


# ===========================================================================
# Group 3 — Storage failure (non-fatal)
# ===========================================================================


@pytest.mark.integration
async def test_approve_and_mask_save_output_failure_is_nonfatal(
    client, override_databricks_dependency
):
    """A save_file failure when persisting the masked PDF must not prevent a 200
    response — the error is logged but the endpoint still returns successfully."""
    sm = override_databricks_dependency
    sm.get_session.return_value = _make_session()
    sm.get_file.return_value = b"fake image bytes"
    sm.get_entities.return_value = [_ENTITY_1]
    sm.save_file.side_effect = Exception("Storage unavailable")

    with patch("app.main._apply_masking_sync", return_value=_FAKE_PDF_BYTES):
        response = await client.post(
            "/api/approve-and-mask",
            json={
                "session_id": "test-session-id",
                "approved_entity_ids": ["entity-1"],
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["entities_masked"] == 1
