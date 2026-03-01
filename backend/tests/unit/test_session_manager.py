"""
Unit tests for UCSessionManager.

All HTTP calls are mocked — no live Databricks workspace is required.
requests.get / requests.put / requests.delete are patched at the import
site inside app.session_manager so every test is fully isolated.
"""

from __future__ import annotations

import json
import re
from unittest.mock import MagicMock, patch

import pytest
from requests import HTTPError

from app.session_manager import UCSessionManager

# ---------------------------------------------------------------------------
# Constants used throughout
# ---------------------------------------------------------------------------

TEST_HOST = "https://test-workspace.azuredatabricks.net"
TEST_TOKEN = "test-token-12345"
TEST_VOLUME = "/Volumes/test_catalog/test_schema/sessions"


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def session_manager_instance() -> UCSessionManager:
    return UCSessionManager(
        databricks_host=TEST_HOST,
        token=TEST_TOKEN,
        volume_path=TEST_VOLUME,
    )


# ---------------------------------------------------------------------------
# Response-builder helpers (not fixtures — plain functions)
# ---------------------------------------------------------------------------


def _ok_response(body: dict | None = None) -> MagicMock:
    """Mock a 200 OK response whose .json() returns *body*."""
    body = body if body is not None else {}
    mock = MagicMock()
    mock.status_code = 200
    mock.content = json.dumps(body).encode()
    mock.json.return_value = body
    mock.raise_for_status.return_value = None
    return mock


def _ok_bytes_response(content: bytes) -> MagicMock:
    """Mock a 200 OK response whose .content is raw bytes."""
    mock = MagicMock()
    mock.status_code = 200
    mock.content = content
    mock.raise_for_status.return_value = None
    return mock


def _not_found_response() -> MagicMock:
    """Mock a 404 response that does NOT raise on raise_for_status."""
    mock = MagicMock()
    mock.status_code = 404
    mock.raise_for_status.return_value = None
    return mock


def _http_error_response(status_code: int) -> MagicMock:
    """Mock a response that raises HTTPError on raise_for_status."""
    mock = MagicMock()
    mock.status_code = status_code
    mock.raise_for_status.side_effect = HTTPError(
        f"HTTP {status_code}", response=mock
    )
    return mock


def _malformed_json_response() -> MagicMock:
    """Mock a 200 response whose .json() raises JSONDecodeError."""
    mock = MagicMock()
    mock.status_code = 200
    mock.raise_for_status.return_value = None
    mock.json.side_effect = json.JSONDecodeError("Expecting value", "not-json", 0)
    return mock


# ===========================================================================
# Group 1 — create_session
# ===========================================================================


@pytest.mark.unit
def test_create_session_returns_valid_uuid(session_manager_instance):
    with patch("app.session_manager.requests.put") as mock_put:
        mock_put.return_value = _ok_response()
        session_id = session_manager_instance.create_session("document.pdf")

    uuid_v4_re = re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
        re.IGNORECASE,
    )
    assert uuid_v4_re.match(session_id), f"Not a valid UUID v4: {session_id!r}"


@pytest.mark.unit
def test_create_session_writes_metadata_json(session_manager_instance):
    with patch("app.session_manager.requests.put") as mock_put:
        mock_put.return_value = _ok_response()
        session_manager_instance.create_session("report.pdf")

    mock_put.assert_called_once()
    url = mock_put.call_args.args[0]
    assert "metadata.json" in url

    body = json.loads(mock_put.call_args.kwargs["data"])
    assert "session_id" in body
    assert body["original_filename"] == "report.pdf"
    assert "created_at" in body
    assert body["status"] == "uploaded"


@pytest.mark.unit
def test_create_session_sets_correct_auth_header(session_manager_instance):
    with patch("app.session_manager.requests.put") as mock_put:
        mock_put.return_value = _ok_response()
        session_manager_instance.create_session("file.pdf")

    headers = mock_put.call_args.kwargs["headers"]
    assert headers["Authorization"] == f"Bearer {TEST_TOKEN}"


# ===========================================================================
# Group 2 — save_entities
# ===========================================================================


@pytest.mark.unit
def test_save_entities_writes_to_correct_path(session_manager_instance):
    session_id = "abc-session-123"
    with patch("app.session_manager.requests.put") as mock_put:
        mock_put.return_value = _ok_response()
        session_manager_instance.save_entities(session_id, [])

    url = mock_put.call_args.args[0]
    assert f"/sessions/{session_id}/entities.json" in url


@pytest.mark.unit
def test_save_entities_includes_all_required_fields(session_manager_instance):
    session_id = "abc-session-123"
    entities = [{"id": "e1", "original_text": "John"}]

    with patch("app.session_manager.requests.put") as mock_put:
        mock_put.return_value = _ok_response()
        session_manager_instance.save_entities(session_id, entities)

    body = json.loads(mock_put.call_args.kwargs["data"])
    assert "session_id" in body
    assert "saved_at" in body
    assert "status" in body
    assert "entities" in body


@pytest.mark.unit
def test_save_entities_sets_status_awaiting_review(session_manager_instance):
    session_id = "abc-session-123"

    with patch("app.session_manager.requests.put") as mock_put:
        mock_put.return_value = _ok_response()
        session_manager_instance.save_entities(session_id, [])

    body = json.loads(mock_put.call_args.kwargs["data"])
    assert body["status"] == "awaiting_review"


@pytest.mark.unit
def test_save_entities_preserves_bounding_boxes(session_manager_instance):
    session_id = "abc-session-123"
    bbox = [0.0841, 0.1044, 0.1344, 0.0142]
    entities = [
        {
            "id": "e1",
            "entity_type": "Full Name",
            "original_text": "John Smith",
            "bounding_box": bbox,
        }
    ]

    with patch("app.session_manager.requests.put") as mock_put:
        mock_put.return_value = _ok_response()
        session_manager_instance.save_entities(session_id, entities)

    body = json.loads(mock_put.call_args.kwargs["data"])
    assert body["entities"][0]["bounding_box"] == bbox


# ===========================================================================
# Group 3 — get_entities
# ===========================================================================


@pytest.mark.unit
def test_get_entities_reads_from_correct_path(session_manager_instance):
    session_id = "abc-session-123"

    with patch("app.session_manager.requests.get") as mock_get:
        mock_get.return_value = _ok_response({"entities": []})
        session_manager_instance.get_entities(session_id)

    url = mock_get.call_args.args[0]
    assert f"/sessions/{session_id}/entities.json" in url


@pytest.mark.unit
def test_get_entities_returns_entities_list(session_manager_instance):
    session_id = "abc-session-123"
    payload = {
        "session_id": session_id,
        "status": "awaiting_review",
        "entities": [{"id": "e1"}],
    }

    with patch("app.session_manager.requests.get") as mock_get:
        mock_get.return_value = _ok_response(payload)
        result = session_manager_instance.get_entities(session_id)

    assert isinstance(result, list)
    assert result == [{"id": "e1"}]


@pytest.mark.unit
def test_get_entities_raises_on_404(session_manager_instance):
    session_id = "missing-session-xyz"

    with patch("app.session_manager.requests.get") as mock_get:
        mock_get.return_value = _not_found_response()

        with pytest.raises(Exception) as exc_info:
            session_manager_instance.get_entities(session_id)

    # Must include the session_id so callers can diagnose which session failed
    assert session_id in str(exc_info.value)
    # Must NOT be a raw requests.HTTPError — should be a meaningful FileNotFoundError
    assert not isinstance(exc_info.value, HTTPError)


@pytest.mark.unit
def test_get_entities_raises_on_malformed_json(session_manager_instance):
    session_id = "abc-session-123"

    with patch("app.session_manager.requests.get") as mock_get:
        mock_get.return_value = _malformed_json_response()

        with pytest.raises(Exception) as exc_info:
            session_manager_instance.get_entities(session_id)

    # Message must include session_id so the caller knows which session is broken
    assert session_id in str(exc_info.value)


# ===========================================================================
# Group 4 — save_file and get_file
# ===========================================================================


@pytest.mark.unit
def test_save_file_sends_bytes_as_request_body(session_manager_instance):
    session_id = "abc-session-123"
    filename = "original.pdf"
    binary_data = b"%PDF-1.4 \x00\x01\x02 sample binary content"

    with patch("app.session_manager.requests.put") as mock_put:
        mock_put.return_value = _ok_response()
        session_manager_instance.save_file(session_id, filename, binary_data)

    sent_data = mock_put.call_args.kwargs["data"]
    assert sent_data == binary_data
    assert isinstance(sent_data, bytes)


@pytest.mark.unit
def test_get_file_returns_bytes(session_manager_instance):
    session_id = "abc-session-123"
    filename = "original.pdf"
    expected = b"%PDF-1.4 \x00\x01\x02 sample binary content"

    with patch("app.session_manager.requests.get") as mock_get:
        mock_get.return_value = _ok_bytes_response(expected)
        result = session_manager_instance.get_file(session_id, filename)

    assert result == expected
    assert isinstance(result, bytes)


@pytest.mark.unit
def test_file_path_uses_session_id_as_folder(session_manager_instance):
    session_id = "abc-session-123"
    filename = "original.pdf"

    with patch("app.session_manager.requests.get") as mock_get:
        mock_get.return_value = _ok_bytes_response(b"data")
        session_manager_instance.get_file(session_id, filename)

    url = mock_get.call_args.args[0]
    assert f"/sessions/{session_id}/{filename}" in url


# ===========================================================================
# Group 5 — update_status
# ===========================================================================


@pytest.mark.unit
def test_update_status_reads_then_writes_metadata(session_manager_instance):
    session_id = "abc-session-123"
    existing_metadata = {
        "session_id": session_id,
        "original_filename": "report.pdf",
        "created_at": "2024-06-01T12:00:00+00:00",
        "status": "uploaded",
        "extra_field": "must_be_preserved",
    }

    with patch("app.session_manager.requests.get") as mock_get, \
            patch("app.session_manager.requests.put") as mock_put:
        mock_get.return_value = _ok_response(existing_metadata)
        mock_put.return_value = _ok_response()

        session_manager_instance.update_status(session_id, "processing")

    # Both GET (read) and PUT (write) must have been called
    assert mock_get.called
    assert mock_put.called

    # The written body must carry the new status …
    written = json.loads(mock_put.call_args.kwargs["data"])
    assert written["status"] == "processing"
    # … and must preserve all pre-existing fields
    assert written["original_filename"] == "report.pdf"
    assert written["created_at"] == "2024-06-01T12:00:00+00:00"
    assert written["extra_field"] == "must_be_preserved"


@pytest.mark.unit
def test_update_status_rejects_invalid_status(session_manager_instance):
    # ValueError must be raised before any HTTP call is made
    with pytest.raises(ValueError, match="invalid_status"):
        session_manager_instance.update_status("abc-session-123", "invalid_status")


# ===========================================================================
# Group 6 — update_session
# ===========================================================================


@pytest.mark.unit
def test_update_session_merges_kwargs_into_metadata(session_manager_instance):
    """update_session must read metadata, merge all kwargs, and write back."""
    session_id = "abc-session-123"
    existing_metadata = {
        "session_id": session_id,
        "original_filename": "report.pdf",
        "created_at": "2024-06-01T12:00:00+00:00",
        "status": "uploaded",
        "extra_field": "must_be_preserved",
    }

    with patch("app.session_manager.requests.get") as mock_get, \
            patch("app.session_manager.requests.put") as mock_put:
        mock_get.return_value = _ok_response(existing_metadata)
        mock_put.return_value = _ok_response()

        session_manager_instance.update_session(
            session_id,
            status="processing",
            field_definitions=[{"name": "Full Name"}],
        )

    # Both GET (read) and PUT (write) must have been called
    assert mock_get.called
    assert mock_put.called

    written = json.loads(mock_put.call_args.kwargs["data"])
    # New kwargs must be written
    assert written["status"] == "processing"
    assert written["field_definitions"] == [{"name": "Full Name"}]
    # Pre-existing fields must be preserved
    assert written["original_filename"] == "report.pdf"
    assert written["created_at"] == "2024-06-01T12:00:00+00:00"
    assert written["extra_field"] == "must_be_preserved"


@pytest.mark.unit
def test_update_session_rejects_invalid_status(session_manager_instance):
    """ValueError must be raised before any HTTP call when status is invalid."""
    with pytest.raises(ValueError, match="invalid_status"):
        session_manager_instance.update_session("abc-session-123", status="invalid_status")


@pytest.mark.unit
def test_update_session_accepts_no_status_kwarg(session_manager_instance):
    """update_session with no status kwarg must merge other kwargs without error."""
    session_id = "abc-session-123"
    existing_metadata = {"session_id": session_id, "status": "uploaded"}

    with patch("app.session_manager.requests.get") as mock_get, \
            patch("app.session_manager.requests.put") as mock_put:
        mock_get.return_value = _ok_response(existing_metadata)
        mock_put.return_value = _ok_response()

        session_manager_instance.update_session(
            session_id, field_definitions=[{"name": "Email"}]
        )

    written = json.loads(mock_put.call_args.kwargs["data"])
    assert written["field_definitions"] == [{"name": "Email"}]
    assert written["status"] == "uploaded"  # unchanged


# ===========================================================================
# Group 7 — get_file 404 handling
# ===========================================================================


@pytest.mark.unit
def test_get_file_raises_file_not_found_on_404(session_manager_instance):
    """A 404 from the Files API must surface as FileNotFoundError, not HTTPError."""
    session_id = "abc-session-123"
    filename = "original.pdf"

    with patch("app.session_manager.requests.get") as mock_get:
        mock_get.return_value = _not_found_response()

        with pytest.raises(FileNotFoundError) as exc_info:
            session_manager_instance.get_file(session_id, filename)

    assert session_id in str(exc_info.value)
    assert filename in str(exc_info.value)
