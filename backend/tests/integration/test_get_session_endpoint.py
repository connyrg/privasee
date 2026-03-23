"""
Integration tests for GET /api/sessions/{session_id}.

Uses the `client` fixture (httpx.AsyncClient + ASGITransport) and
`override_databricks_dependency` (mock UCSessionManager).
"""

from unittest.mock import MagicMock

import pytest

from app.models import SessionData


def _make_session(
    session_id: str = "test-session-id",
    filename: str = "document.pdf",
    status: str = "uploaded",
    file_size: int = 12345,
    error_message: str = None,
) -> MagicMock:
    mock = MagicMock(spec=SessionData)
    mock.session_id = session_id
    mock.filename = filename
    mock.status = status
    mock.file_size = file_size
    mock.error_message = error_message
    return mock


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


# ===========================================================================
# Happy path
# ===========================================================================


@pytest.mark.integration
async def test_get_session_returns_200_for_uploaded_session(
    client, override_databricks_dependency
):
    """200 with session_id, filename, and status for an existing session."""
    sm = override_databricks_dependency
    sm.get_session.return_value = _make_session(status="uploaded")

    response = await client.get("/api/sessions/test-session-id")

    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == "test-session-id"
    assert body["filename"] == "document.pdf"
    assert body["status"] == "uploaded"


@pytest.mark.integration
async def test_get_session_returns_entities_when_awaiting_review(
    client, override_databricks_dependency
):
    """Entities are loaded and returned when status is awaiting_review."""
    sm = override_databricks_dependency
    sm.get_session.return_value = _make_session(status="awaiting_review")
    sm.get_entities.return_value = [_ENTITY_1]

    response = await client.get("/api/sessions/test-session-id")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "awaiting_review"
    assert body["entity_count"] == 1
    assert body["entities"][0]["id"] == "entity-1"


@pytest.mark.integration
async def test_get_session_has_masked_output_when_completed(
    client, override_databricks_dependency
):
    """has_masked_output is True when status is completed."""
    sm = override_databricks_dependency
    sm.get_session.return_value = _make_session(status="completed")
    sm.get_entities.return_value = [_ENTITY_1]

    response = await client.get("/api/sessions/test-session-id")

    assert response.status_code == 200
    body = response.json()
    assert body["has_masked_output"] is True


@pytest.mark.integration
async def test_get_session_no_entities_loaded_when_uploaded(
    client, override_databricks_dependency
):
    """get_entities is NOT called for sessions still in 'uploaded' status."""
    sm = override_databricks_dependency
    sm.get_session.return_value = _make_session(status="uploaded")

    response = await client.get("/api/sessions/test-session-id")

    assert response.status_code == 200
    sm.get_entities.assert_not_called()
    assert response.json()["entity_count"] == 0


@pytest.mark.integration
async def test_get_session_returns_error_message_when_status_is_error(
    client, override_databricks_dependency
):
    """error_message from session metadata is surfaced in the response."""
    sm = override_databricks_dependency
    sm.get_session.return_value = _make_session(
        status="error", error_message="Databricks timed out"
    )

    response = await client.get("/api/sessions/test-session-id")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "error"
    assert body["error_message"] == "Databricks timed out"


@pytest.mark.integration
async def test_get_session_survives_missing_entities_json(
    client, override_databricks_dependency
):
    """A FileNotFoundError from get_entities is tolerated — returns empty entity list."""
    sm = override_databricks_dependency
    sm.get_session.return_value = _make_session(status="awaiting_review")
    sm.get_entities.side_effect = FileNotFoundError("entities.json not found")

    response = await client.get("/api/sessions/test-session-id")

    assert response.status_code == 200
    body = response.json()
    assert body["entity_count"] == 0
    assert body["entities"] == []


@pytest.mark.integration
async def test_get_session_returns_503_on_unexpected_entities_error(
    client, override_databricks_dependency
):
    """503 when get_entities raises an unexpected error (not FileNotFoundError)."""
    sm = override_databricks_dependency
    sm.get_session.return_value = _make_session(status="awaiting_review")
    sm.get_entities.side_effect = Exception("Storage error")

    response = await client.get("/api/sessions/test-session-id")

    assert response.status_code == 503


# ===========================================================================
# Error responses
# ===========================================================================


@pytest.mark.integration
async def test_get_session_returns_404_for_unknown_session(
    client, override_databricks_dependency
):
    """404 when the session does not exist."""
    sm = override_databricks_dependency
    sm.get_session.return_value = None

    response = await client.get("/api/sessions/unknown-session")

    assert response.status_code == 404
    assert "error" in response.json()


@pytest.mark.integration
async def test_get_session_returns_503_on_storage_failure(
    client, override_databricks_dependency
):
    """503 when get_session raises an unexpected exception."""
    sm = override_databricks_dependency
    sm.get_session.side_effect = Exception("Storage unavailable")

    response = await client.get("/api/sessions/test-session-id")

    assert response.status_code == 503
