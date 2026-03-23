"""
Unit tests for UCSessionManager.

All HTTP calls are mocked — no live Databricks workspace is required.
requests.request is patched at the import site inside app.session_manager
so every test is fully isolated.
"""

from __future__ import annotations

import json
import re
from unittest.mock import MagicMock, call, patch

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
    with patch("app.session_manager.requests.request") as mock_request:
        mock_request.return_value = _ok_response()
        session_id = session_manager_instance.create_session("document.pdf")

    uuid_v4_re = re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
        re.IGNORECASE,
    )
    assert uuid_v4_re.match(session_id), f"Not a valid UUID v4: {session_id!r}"


@pytest.mark.unit
def test_create_session_writes_metadata_json(session_manager_instance):
    with patch("app.session_manager.requests.request") as mock_request:
        mock_request.return_value = _ok_response()
        session_manager_instance.create_session("report.pdf")

    mock_request.assert_called_once()
    # requests.request(method, url, ...) — url is the second positional arg
    url = mock_request.call_args.args[1]
    assert "metadata.json" in url

    body = json.loads(mock_request.call_args.kwargs["data"])
    assert "session_id" in body
    assert body["original_filename"] == "report.pdf"
    assert "created_at" in body
    assert body["status"] == "uploaded"


@pytest.mark.unit
def test_create_session_sets_correct_auth_header(session_manager_instance):
    with patch("app.session_manager.requests.request") as mock_request:
        mock_request.return_value = _ok_response()
        session_manager_instance.create_session("file.pdf")

    headers = mock_request.call_args.kwargs["headers"]
    assert headers["Authorization"] == f"Bearer {TEST_TOKEN}"


# ===========================================================================
# Group 2 — save_entities
# ===========================================================================


@pytest.mark.unit
def test_save_entities_writes_to_correct_path(session_manager_instance):
    session_id = "abc-session-123"
    with patch("app.session_manager.requests.request") as mock_request:
        mock_request.return_value = _ok_response()
        session_manager_instance.save_entities(session_id, [])

    url = mock_request.call_args.args[1]
    assert f"/sessions/{session_id}/entities.json" in url


@pytest.mark.unit
def test_save_entities_includes_all_required_fields(session_manager_instance):
    session_id = "abc-session-123"
    entities = [{"id": "e1", "original_text": "John"}]

    with patch("app.session_manager.requests.request") as mock_request:
        mock_request.return_value = _ok_response()
        session_manager_instance.save_entities(session_id, entities)

    body = json.loads(mock_request.call_args.kwargs["data"])
    assert "session_id" in body
    assert "saved_at" in body
    assert "status" in body
    assert "entities" in body


@pytest.mark.unit
def test_save_entities_sets_status_awaiting_review(session_manager_instance):
    session_id = "abc-session-123"

    with patch("app.session_manager.requests.request") as mock_request:
        mock_request.return_value = _ok_response()
        session_manager_instance.save_entities(session_id, [])

    body = json.loads(mock_request.call_args.kwargs["data"])
    assert body["status"] == "awaiting_review"


@pytest.mark.unit
def test_save_entities_preserves_occurrences(session_manager_instance):
    session_id = "abc-session-123"
    bbox = [0.0841, 0.1044, 0.1344, 0.0142]
    entities = [
        {
            "id": "e1",
            "entity_type": "Full Name",
            "original_text": "John Smith",
            "occurrences": [
                {"page_number": 1, "original_text": "John Smith", "bounding_boxes": [bbox]},
            ],
        }
    ]

    with patch("app.session_manager.requests.request") as mock_request:
        mock_request.return_value = _ok_response()
        session_manager_instance.save_entities(session_id, entities)

    body = json.loads(mock_request.call_args.kwargs["data"])
    assert body["entities"][0]["occurrences"][0]["bounding_boxes"] == [bbox]


# ===========================================================================
# Group 3 — get_entities
# ===========================================================================


@pytest.mark.unit
def test_get_entities_reads_from_correct_path(session_manager_instance):
    session_id = "abc-session-123"

    with patch("app.session_manager.requests.request") as mock_request:
        mock_request.return_value = _ok_response({"entities": []})
        session_manager_instance.get_entities(session_id)

    url = mock_request.call_args.args[1]
    assert f"/sessions/{session_id}/entities.json" in url


@pytest.mark.unit
def test_get_entities_returns_entities_list(session_manager_instance):
    session_id = "abc-session-123"
    payload = {
        "session_id": session_id,
        "status": "awaiting_review",
        "entities": [{"id": "e1"}],
    }

    with patch("app.session_manager.requests.request") as mock_request:
        mock_request.return_value = _ok_response(payload)
        result = session_manager_instance.get_entities(session_id)

    assert isinstance(result, list)
    assert result == [{"id": "e1"}]


@pytest.mark.unit
def test_get_entities_raises_on_404(session_manager_instance):
    session_id = "missing-session-xyz"

    with patch("app.session_manager.requests.request") as mock_request:
        mock_request.return_value = _not_found_response()

        with pytest.raises(Exception) as exc_info:
            session_manager_instance.get_entities(session_id)

    # Must include the session_id so callers can diagnose which session failed
    assert session_id in str(exc_info.value)
    # Must NOT be a raw requests.HTTPError — should be a meaningful FileNotFoundError
    assert not isinstance(exc_info.value, HTTPError)


@pytest.mark.unit
def test_get_entities_raises_on_malformed_json(session_manager_instance):
    session_id = "abc-session-123"

    with patch("app.session_manager.requests.request") as mock_request:
        mock_request.return_value = _malformed_json_response()

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

    with patch("app.session_manager.requests.request") as mock_request:
        mock_request.return_value = _ok_response()
        session_manager_instance.save_file(session_id, filename, binary_data)

    sent_data = mock_request.call_args.kwargs["data"]
    assert sent_data == binary_data
    assert isinstance(sent_data, bytes)


@pytest.mark.unit
def test_get_file_returns_bytes(session_manager_instance):
    session_id = "abc-session-123"
    filename = "original.pdf"
    expected = b"%PDF-1.4 \x00\x01\x02 sample binary content"

    with patch("app.session_manager.requests.request") as mock_request:
        mock_request.return_value = _ok_bytes_response(expected)
        result = session_manager_instance.get_file(session_id, filename)

    assert result == expected
    assert isinstance(result, bytes)


@pytest.mark.unit
def test_file_path_uses_session_id_as_folder(session_manager_instance):
    session_id = "abc-session-123"
    filename = "original.pdf"

    with patch("app.session_manager.requests.request") as mock_request:
        mock_request.return_value = _ok_bytes_response(b"data")
        session_manager_instance.get_file(session_id, filename)

    url = mock_request.call_args.args[1]
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

    with patch("app.session_manager.requests.request") as mock_request:
        # First call is GET (read), second is PUT (write)
        mock_request.side_effect = [_ok_response(existing_metadata), _ok_response()]
        session_manager_instance.update_status(session_id, "processing")

    assert mock_request.call_count == 2
    get_call, put_call = mock_request.call_args_list
    assert get_call.args[0] == "get"
    assert put_call.args[0] == "put"

    written = json.loads(put_call.kwargs["data"])
    assert written["status"] == "processing"
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

    with patch("app.session_manager.requests.request") as mock_request:
        mock_request.side_effect = [_ok_response(existing_metadata), _ok_response()]
        session_manager_instance.update_session(
            session_id,
            status="processing",
            field_definitions=[{"name": "Full Name"}],
        )

    assert mock_request.call_count == 2
    get_call, put_call = mock_request.call_args_list
    assert get_call.args[0] == "get"
    assert put_call.args[0] == "put"

    written = json.loads(put_call.kwargs["data"])
    assert written["status"] == "processing"
    assert written["field_definitions"] == [{"name": "Full Name"}]
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

    with patch("app.session_manager.requests.request") as mock_request:
        mock_request.side_effect = [_ok_response(existing_metadata), _ok_response()]
        session_manager_instance.update_session(
            session_id, field_definitions=[{"name": "Email"}]
        )

    _, put_call = mock_request.call_args_list
    written = json.loads(put_call.kwargs["data"])
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

    with patch("app.session_manager.requests.request") as mock_request:
        mock_request.return_value = _not_found_response()

        with pytest.raises(FileNotFoundError) as exc_info:
            session_manager_instance.get_file(session_id, filename)

    assert session_id in str(exc_info.value)
    assert filename in str(exc_info.value)


# ===========================================================================
# Group 8 — get_session
# ===========================================================================


@pytest.mark.unit
def test_get_session_returns_none_on_404(session_manager_instance):
    """get_session must return None (not raise) when metadata.json is missing."""
    with patch("app.session_manager.requests.request") as mock_request:
        mock_request.return_value = _not_found_response()
        result = session_manager_instance.get_session("missing-session")

    assert result is None


@pytest.mark.unit
def test_get_session_returns_session_data_on_200(session_manager_instance):
    """get_session returns a populated SessionData when metadata.json exists."""
    session_id = "abc-session-123"
    metadata = {
        "session_id": session_id,
        "original_filename": "report.pdf",
        "file_size": 12345,
        "status": "awaiting_review",
    }

    with patch("app.session_manager.requests.request") as mock_request:
        mock_request.return_value = _ok_response(metadata)
        result = session_manager_instance.get_session(session_id)

    assert result is not None
    assert result.session_id == session_id
    assert result.filename == "report.pdf"
    assert result.file_size == 12345
    assert result.status == "awaiting_review"


@pytest.mark.unit
def test_get_session_reads_from_correct_path(session_manager_instance):
    """get_session must request metadata.json from the correct UC path."""
    session_id = "abc-session-123"

    with patch("app.session_manager.requests.request") as mock_request:
        mock_request.return_value = _not_found_response()
        session_manager_instance.get_session(session_id)

    url = mock_request.call_args.args[1]
    assert f"/sessions/{session_id}/metadata.json" in url


@pytest.mark.unit
def test_get_session_raises_on_malformed_json(session_manager_instance):
    """get_session raises ValueError (not HTTPError) when metadata.json is invalid JSON."""
    session_id = "abc-session-123"

    with patch("app.session_manager.requests.request") as mock_request:
        mock_request.return_value = _malformed_json_response()

        with pytest.raises(ValueError) as exc_info:
            session_manager_instance.get_session(session_id)

    assert session_id in str(exc_info.value)


@pytest.mark.unit
def test_get_session_preserves_error_message(session_manager_instance):
    """error_message from metadata is surfaced on the returned SessionData."""
    session_id = "abc-session-123"
    metadata = {
        "session_id": session_id,
        "original_filename": "file.pdf",
        "status": "error",
        "error_message": "Databricks timed out",
    }

    with patch("app.session_manager.requests.request") as mock_request:
        mock_request.return_value = _ok_response(metadata)
        result = session_manager_instance.get_session(session_id)

    assert result.error_message == "Databricks timed out"


# ===========================================================================
# Group 9 — delete_session
# ===========================================================================


@pytest.mark.unit
def test_delete_session_reads_metadata_then_deletes_files(session_manager_instance):
    """delete_session GETs metadata.json first, then DELETEs all candidate files."""
    session_id = "abc-session-123"
    metadata = {"session_id": session_id, "original_filename": "report.pdf"}

    responses = [_ok_response(metadata)] + [_ok_response() for _ in range(5)]

    with patch("app.session_manager.requests.request") as mock_request:
        mock_request.side_effect = responses
        session_manager_instance.delete_session(session_id)

    assert mock_request.call_count == 6  # 1 GET + 5 DELETEs

    get_call = mock_request.call_args_list[0]
    assert get_call.args[0] == "get"

    delete_calls = mock_request.call_args_list[1:]
    for c in delete_calls:
        assert c.args[0] == "delete"


@pytest.mark.unit
def test_delete_session_uses_correct_extension_from_metadata(session_manager_instance):
    """delete_session reads the original filename extension to delete original.{ext}."""
    session_id = "abc-session-123"
    metadata = {"session_id": session_id, "original_filename": "scan.png"}

    responses = [_ok_response(metadata)] + [_ok_response() for _ in range(5)]

    with patch("app.session_manager.requests.request") as mock_request:
        mock_request.side_effect = responses
        session_manager_instance.delete_session(session_id)

    delete_urls = [c.args[1] for c in mock_request.call_args_list[1:]]
    assert any("original.png" in url for url in delete_urls)


@pytest.mark.unit
def test_delete_session_ignores_404_on_missing_files(session_manager_instance):
    """404 responses for individual files must be silently skipped."""
    session_id = "abc-session-123"
    metadata = {"session_id": session_id, "original_filename": "report.pdf"}

    # metadata GET succeeds; all DELETEs return 404
    responses = [_ok_response(metadata)] + [_not_found_response() for _ in range(5)]

    with patch("app.session_manager.requests.request") as mock_request:
        mock_request.side_effect = responses
        # Must not raise even though every DELETE is a 404
        session_manager_instance.delete_session(session_id)


@pytest.mark.unit
def test_delete_session_falls_back_to_pdf_when_metadata_unreadable(session_manager_instance):
    """If metadata.json cannot be read, delete_session falls back to original.pdf."""
    session_id = "abc-session-123"

    from requests.exceptions import ConnectionError as ReqConnError
    import app.session_manager as sm_module

    # _request retries ConnectionError up to _MAX_RETRIES times, so we need
    # _MAX_RETRIES+1 ConnectionErrors to exhaust all attempts, then 5 DELETE ok responses.
    conn_errors = [ReqConnError("down")] * (sm_module._MAX_RETRIES + 1)
    delete_responses = [_ok_response() for _ in range(5)]

    with patch("app.session_manager.requests.request") as mock_request, \
         patch.object(sm_module, "_RETRY_DELAYS", (0, 0, 0)):
        mock_request.side_effect = conn_errors + delete_responses
        session_manager_instance.delete_session(session_id)

    delete_urls = [c.args[1] for c in mock_request.call_args_list[sm_module._MAX_RETRIES + 1:]]
    assert any("original.pdf" in url for url in delete_urls)


# ===========================================================================
# Group 10 — save_masking_decisions
# ===========================================================================


@pytest.mark.unit
def test_save_masking_decisions_writes_to_correct_path(session_manager_instance):
    session_id = "abc-session-123"

    with patch("app.session_manager.requests.request") as mock_request:
        mock_request.return_value = _ok_response()
        session_manager_instance.save_masking_decisions(session_id, [], set())

    url = mock_request.call_args.args[1]
    assert f"/sessions/{session_id}/masking_decisions.json" in url


@pytest.mark.unit
def test_save_masking_decisions_marks_approved_flag_correctly(session_manager_instance):
    """Entities in approved_ids get approved=True; others get approved=False."""
    session_id = "abc-session-123"
    entities = [
        {"id": "e1", "original_text": "Alice"},
        {"id": "e2", "original_text": "Bob"},
    ]
    approved_ids = {"e1"}

    with patch("app.session_manager.requests.request") as mock_request:
        mock_request.return_value = _ok_response()
        session_manager_instance.save_masking_decisions(session_id, entities, approved_ids)

    body = json.loads(mock_request.call_args.kwargs["data"])
    by_id = {e["id"]: e for e in body["entities"]}
    assert by_id["e1"]["approved"] is True
    assert by_id["e2"]["approved"] is False


@pytest.mark.unit
def test_save_masking_decisions_includes_timestamp(session_manager_instance):
    session_id = "abc-session-123"

    with patch("app.session_manager.requests.request") as mock_request:
        mock_request.return_value = _ok_response()
        session_manager_instance.save_masking_decisions(session_id, [], set())

    body = json.loads(mock_request.call_args.kwargs["data"])
    assert "decided_at" in body
    assert "session_id" in body


# ===========================================================================
# Group 11 — _request retry logic
# ===========================================================================


@pytest.mark.unit
def test_request_retries_on_transient_status_codes(session_manager_instance):
    """_request retries on 429/500/502/503/504 and returns the eventual success."""
    from unittest.mock import patch as _patch
    import app.session_manager as sm_module

    # Simulate: first call → 503, second call → 200
    transient = MagicMock()
    transient.status_code = 503
    success = _ok_response({"result": "ok"})

    with _patch("app.session_manager.requests.request") as mock_req, \
         _patch.object(sm_module, "_RETRY_DELAYS", (0, 0, 0)):
        mock_req.side_effect = [transient, success]
        resp = session_manager_instance._request("get", "https://host/path")

    assert resp.status_code == 200
    assert mock_req.call_count == 2


@pytest.mark.unit
def test_request_returns_transient_response_after_max_retries(session_manager_instance):
    """After _MAX_RETRIES, _request returns the final transient response (doesn't raise).
    Only network errors (ConnectionError) cause _request to raise after exhausting retries.
    """
    import app.session_manager as sm_module

    transient = MagicMock()
    transient.status_code = 503

    with patch("app.session_manager.requests.request") as mock_req, \
         patch.object(sm_module, "_RETRY_DELAYS", (0, 0, 0)):
        mock_req.return_value = transient
        resp = session_manager_instance._request("get", "https://host/path")

    # Should have tried _MAX_RETRIES + 1 = 4 times and returned the last response
    assert mock_req.call_count == sm_module._MAX_RETRIES + 1
    assert resp.status_code == 503


@pytest.mark.unit
def test_request_raises_after_max_network_errors(session_manager_instance):
    """_request raises when every attempt fails with a network error (ConnectionError)."""
    import app.session_manager as sm_module
    from requests.exceptions import ConnectionError as ReqConnError

    with patch("app.session_manager.requests.request") as mock_req, \
         patch.object(sm_module, "_RETRY_DELAYS", (0, 0, 0)):
        mock_req.side_effect = ReqConnError("down")
        with pytest.raises(Exception):
            session_manager_instance._request("get", "https://host/path")

    assert mock_req.call_count == sm_module._MAX_RETRIES + 1


@pytest.mark.unit
def test_request_retries_on_connection_error(session_manager_instance):
    """Network errors (ConnectionError) trigger the retry loop."""
    import app.session_manager as sm_module
    from requests.exceptions import ConnectionError as ReqConnError

    success = _ok_response()

    with patch("app.session_manager.requests.request") as mock_req, \
         patch.object(sm_module, "_RETRY_DELAYS", (0, 0, 0)):
        mock_req.side_effect = [ReqConnError("down"), success]
        resp = session_manager_instance._request("get", "https://host/path")

    assert resp.status_code == 200
    assert mock_req.call_count == 2


@pytest.mark.unit
def test_request_does_not_retry_on_404(session_manager_instance):
    """404 is not a transient error — _request must return immediately without retrying."""
    not_found = _not_found_response()

    with patch("app.session_manager.requests.request") as mock_req:
        mock_req.return_value = not_found
        resp = session_manager_instance._request("get", "https://host/path")

    assert resp.status_code == 404
    assert mock_req.call_count == 1
