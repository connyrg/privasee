"""
Integration tests for POST /api/approve-and-mask.

Uses the `client` fixture (httpx.AsyncClient + ASGITransport) and
`override_databricks_dependency` (MOCK_DATABRICKS=True, no masking endpoint).

In mock mode the masking step is skipped — the endpoint filters entities,
applies user edits, and updates the session status to "completed" without
calling Databricks.  Tests that need to verify which entities would be sent
to Databricks set DATABRICKS_MASKING_ENDPOINT via monkeypatch and mock the
httpx call.

Session storage calls (get_session, get_file, get_entities, update_session)
are all handled via the mock_session_manager from conftest.
"""

import json

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

import app.main as main_module

# Two representative stored entity dicts (shape matches model_dump() output).
_ENTITY_1 = {
    "id": "entity-1",
    "entity_type": "Full Name",
    "original_text": "John Smith",
    "replacement_text": "Jane Doe",
    "confidence": 0.95,
    "approved": True,
    "occurrences": [
        {"page_number": 1, "original_text": "John Smith", "bounding_boxes": [[0.05, 0.08, 0.45, 0.025]]},
    ],
}

_ENTITY_2 = {
    "id": "entity-2",
    "entity_type": "Email",
    "original_text": "john@example.com",
    "replacement_text": "[REDACTED]",
    "confidence": 0.95,
    "approved": True,
    "occurrences": [
        {"page_number": 1, "original_text": "john@example.com", "bounding_boxes": [[0.05, 0.15, 0.45, 0.025]]},
    ],
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
    client, override_databricks_dependency, monkeypatch
):
    """Only the entity IDs listed in approved_entity_ids are sent to masking."""
    sm = override_databricks_dependency
    sm.get_session.return_value = _make_session()
    sm.get_file.return_value = b"fake image bytes"
    sm.get_entities.return_value = [_ENTITY_1, _ENTITY_2]

    monkeypatch.setattr(main_module, "MOCK_DATABRICKS", False)
    monkeypatch.setattr(main_module, "DATABRICKS_MASKING_ENDPOINT", "https://fake-masking/invocations")

    captured_payloads: list = []

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()

    async def _capture_post(url, **kw):
        captured_payloads.append(kw.get("json", {}))
        return mock_resp

    mock_http = AsyncMock()
    mock_http.post = _capture_post

    with patch("app.main.httpx.AsyncClient") as MockClient:
        MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_http)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)
        response = await client.post(
            "/api/approve-and-mask",
            json={
                "session_id": "test-session-id",
                "approved_entity_ids": ["entity-1"],  # only entity-1
            },
        )

    assert response.status_code == 200
    assert len(captured_payloads) == 1
    sent_entities = json.loads(captured_payloads[0]["dataframe_records"][0]["entities_to_mask"])
    assert len(sent_entities) == 1
    assert sent_entities[0]["id"] == "entity-1"


@pytest.mark.integration
async def test_approve_and_mask_applies_entity_updates(
    client, override_databricks_dependency, monkeypatch
):
    """replacement_text from updated_entities overrides the stored value before sending to Databricks."""
    sm = override_databricks_dependency
    sm.get_session.return_value = _make_session()
    sm.get_file.return_value = b"fake image bytes"
    sm.get_entities.return_value = [{**_ENTITY_1, "replacement_text": "Original Text"}]

    monkeypatch.setattr(main_module, "MOCK_DATABRICKS", False)
    monkeypatch.setattr(main_module, "DATABRICKS_MASKING_ENDPOINT", "https://fake-masking/invocations")

    captured_payloads: list = []

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()

    async def _capture_post(url, **kw):
        captured_payloads.append(kw.get("json", {}))
        return mock_resp

    mock_http = AsyncMock()
    mock_http.post = _capture_post

    with patch("app.main.httpx.AsyncClient") as MockClient:
        MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_http)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)
        response = await client.post(
            "/api/approve-and-mask",
            json={
                "session_id": "test-session-id",
                "approved_entity_ids": ["entity-1"],
                "updated_entities": [
                    {
                        "id": "entity-1",
                        "replacement_text": "Updated Text",
                    }
                ],
            },
        )

    assert response.status_code == 200
    assert len(captured_payloads) == 1
    sent_entities = json.loads(captured_payloads[0]["dataframe_records"][0]["entities_to_mask"])
    assert len(sent_entities) == 1
    assert sent_entities[0]["replacement_text"] == "Updated Text", (
        f"Expected 'Updated Text', got: {sent_entities[0]['replacement_text']!r}"
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
async def test_approve_and_mask_returns_503_when_masking_decisions_fail(
    client, override_databricks_dependency
):
    """503 when save_masking_decisions raises — audit persistence failure is not silent."""
    sm = override_databricks_dependency
    sm.get_session.return_value = _make_session()
    sm.get_file.return_value = b"fake image bytes"
    sm.get_entities.return_value = [_ENTITY_1]
    sm.save_masking_decisions.side_effect = Exception("Storage unavailable")

    response = await client.post(
        "/api/approve-and-mask",
        json={
            "session_id": "test-session-id",
            "approved_entity_ids": ["entity-1"],
        },
    )

    assert response.status_code == 503


@pytest.mark.integration
async def test_approve_and_mask_final_status_update_failure_is_nonfatal(
    client, override_databricks_dependency
):
    """A failure of the final update_session('completed') call must not prevent a 200 —
    masking already succeeded and masked.pdf is in UC, so the status update is best-effort."""
    sm = override_databricks_dependency
    sm.get_session.return_value = _make_session()
    sm.get_file.return_value = b"fake image bytes"
    sm.get_entities.return_value = [_ENTITY_1]
    sm.update_session.side_effect = Exception("Storage unavailable")

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
