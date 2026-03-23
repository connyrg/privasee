"""
Integration tests for POST /api/process.

The endpoint now returns 202 immediately and runs entity extraction as a
background task.  All Databricks/UC interactions happen after the response
is sent.  With ASGITransport the background task completes before the
awaited `client.post(...)` call returns, so assertions on mock side-effects
(save_entities, update_session) remain synchronous from the test's perspective.

Happy-path tests use MOCK_DATABRICKS=True (set by override_databricks_dependency).
Error-path tests for Databricks failures patch httpx.AsyncClient to simulate
network errors and then verify that the session is marked with status='error'.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from httpx import AsyncClient, ASGITransport

import app.main as main_module
from app.main import app


# ---------------------------------------------------------------------------
# Shared request payload
# ---------------------------------------------------------------------------

VALID_PAYLOAD = {
    "session_id": "test-session-id",
    "field_definitions": [
        {
            "name": "Full Name",
            "description": "Person's full name",
        },
    ],
}


# ===========================================================================
# Group 1 — Happy path (MOCK_DATABRICKS=True)
# ===========================================================================


@pytest.mark.integration
async def test_process_returns_202_accepted(client, override_databricks_dependency):
    """POST /api/process must return 202 with session_id and status='processing'."""
    response = await client.post("/api/process", json=VALID_PAYLOAD)

    assert response.status_code == 202
    body = response.json()
    assert body["session_id"] == "test-session-id"
    assert body["status"] == "processing"


@pytest.mark.integration
async def test_process_saves_entities_to_uc(client, override_databricks_dependency):
    """Entities returned by the mock must be persisted via save_entities (background task)."""
    sm = override_databricks_dependency

    response = await client.post("/api/process", json=VALID_PAYLOAD)

    assert response.status_code == 202
    sm.save_entities.assert_called_once()
    saved_session_id = sm.save_entities.call_args.args[0]
    saved_entities = sm.save_entities.call_args.args[1]
    assert saved_session_id == "test-session-id"
    assert isinstance(saved_entities, list) and len(saved_entities) >= 1


@pytest.mark.integration
async def test_process_updates_session_status_to_processing_then_awaiting_review(
    client, override_databricks_dependency
):
    """update_session must be called with status='processing' before status='awaiting_review'."""
    sm = override_databricks_dependency

    response = await client.post("/api/process", json=VALID_PAYLOAD)

    assert response.status_code == 202
    calls = sm.update_session.call_args_list
    statuses = [c.kwargs.get("status") for c in calls]
    assert "processing" in statuses, (
        f"Expected 'processing' in update_session calls; got: {statuses}"
    )
    assert "awaiting_review" in statuses, (
        f"Expected 'awaiting_review' in update_session calls; got: {statuses}"
    )
    assert statuses.index("processing") < statuses.index("awaiting_review"), (
        "Expected 'processing' to be set before 'awaiting_review'"
    )


# ===========================================================================
# Group 2 — Session / storage errors
# ===========================================================================


@pytest.mark.integration
async def test_process_returns_404_for_unknown_session(
    client, override_databricks_dependency
):
    """404 when get_session returns None for an unrecognised session_id."""
    sm = override_databricks_dependency
    sm.get_session.return_value = None

    response = await client.post("/api/process", json=VALID_PAYLOAD)

    assert response.status_code == 404
    body = response.json()
    assert "error" in body
    assert "test-session-id" in body["error"]


@pytest.mark.integration
async def test_process_returns_501_when_get_session_raises_not_implemented(
    client, override_databricks_dependency
):
    """501 when get_session raises NotImplementedError (storage not configured)."""
    sm = override_databricks_dependency
    sm.get_session.side_effect = NotImplementedError

    response = await client.post("/api/process", json=VALID_PAYLOAD)

    assert response.status_code == 501
    body = response.json()
    assert "error" in body


@pytest.mark.integration
async def test_process_returns_503_when_session_manager_not_configured(monkeypatch):
    """503 when _session_manager is None (Databricks credentials not provided)."""
    monkeypatch.setattr(main_module, "_session_manager", None)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        response = await ac.post("/api/process", json=VALID_PAYLOAD)

    assert response.status_code == 503
    body = response.json()
    assert "error" in body


# ===========================================================================
# Group 3 — Databricks errors (MOCK_DATABRICKS=False)
#
# Errors now happen asynchronously in the background task.  The HTTP response
# is always 202; failures are reflected by status='error' written to the
# session via update_session.
# ===========================================================================


@pytest.mark.integration
async def test_process_sets_session_error_when_databricks_endpoint_not_configured(
    client, override_databricks_dependency, monkeypatch
):
    """Background task sets status='error' when MOCK_DATABRICKS=False and endpoint is empty."""
    sm = override_databricks_dependency
    monkeypatch.setattr(main_module, "MOCK_DATABRICKS", False)
    monkeypatch.setattr(main_module, "DATABRICKS_MODEL_ENDPOINT", "")

    response = await client.post("/api/process", json=VALID_PAYLOAD)

    assert response.status_code == 202
    calls = sm.update_session.call_args_list
    statuses = [c.kwargs.get("status") for c in calls]
    assert "error" in statuses, f"Expected 'error' status set; got: {statuses}"
    error_call = next(c for c in calls if c.kwargs.get("status") == "error")
    assert error_call.kwargs.get("error_message")


@pytest.mark.integration
async def test_process_sets_session_error_on_databricks_timeout(
    client, override_databricks_dependency, monkeypatch
):
    """Background task sets status='error' when the Databricks HTTP call times out."""
    sm = override_databricks_dependency
    monkeypatch.setattr(main_module, "MOCK_DATABRICKS", False)
    monkeypatch.setattr(
        main_module, "DATABRICKS_MODEL_ENDPOINT", "https://fake.databricks/endpoint"
    )

    mock_http = AsyncMock()
    mock_http.post.side_effect = httpx.TimeoutException("timed out")

    with patch("app.main.httpx.AsyncClient") as MockClient:
        MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_http)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)
        response = await client.post("/api/process", json=VALID_PAYLOAD)

    assert response.status_code == 202
    calls = sm.update_session.call_args_list
    statuses = [c.kwargs.get("status") for c in calls]
    assert "error" in statuses, f"Expected 'error' status set; got: {statuses}"


@pytest.mark.integration
async def test_process_sets_session_error_on_databricks_non_200_status(
    client, override_databricks_dependency, monkeypatch
):
    """Background task sets status='error' when Databricks returns a non-200 HTTP status."""
    sm = override_databricks_dependency
    monkeypatch.setattr(main_module, "MOCK_DATABRICKS", False)
    monkeypatch.setattr(
        main_module, "DATABRICKS_MODEL_ENDPOINT", "https://fake.databricks/endpoint"
    )

    mock_db_response = MagicMock()
    mock_db_response.status_code = 503
    mock_db_response.text = "Service Unavailable"

    mock_http = AsyncMock()
    mock_http.post.side_effect = httpx.HTTPStatusError(
        "503 Service Unavailable",
        request=MagicMock(),
        response=mock_db_response,
    )

    with patch("app.main.httpx.AsyncClient") as MockClient:
        MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_http)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)
        response = await client.post("/api/process", json=VALID_PAYLOAD)

    assert response.status_code == 202
    calls = sm.update_session.call_args_list
    statuses = [c.kwargs.get("status") for c in calls]
    assert "error" in statuses, f"Expected 'error' status set; got: {statuses}"


@pytest.mark.integration
async def test_process_sets_session_error_when_databricks_response_cannot_be_parsed(
    client, override_databricks_dependency, monkeypatch
):
    """Background task sets status='error' when Databricks response JSON is malformed."""
    sm = override_databricks_dependency
    monkeypatch.setattr(main_module, "MOCK_DATABRICKS", False)
    monkeypatch.setattr(
        main_module, "DATABRICKS_MODEL_ENDPOINT", "https://fake.databricks/endpoint"
    )

    mock_db_response = MagicMock()
    mock_db_response.raise_for_status = MagicMock()
    mock_db_response.json.return_value = {
        "predictions": [{"entities": [{"invalid_field": True}]}]
    }

    mock_http = AsyncMock()
    mock_http.post.return_value = mock_db_response

    with patch("app.main.httpx.AsyncClient") as MockClient:
        MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_http)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)
        response = await client.post("/api/process", json=VALID_PAYLOAD)

    assert response.status_code == 202
    calls = sm.update_session.call_args_list
    statuses = [c.kwargs.get("status") for c in calls]
    assert "error" in statuses, f"Expected 'error' status set; got: {statuses}"


# ===========================================================================
# Group 4 — Input validation
# ===========================================================================


@pytest.mark.integration
async def test_process_returns_503_when_initial_session_update_fails(
    client, override_databricks_dependency
):
    """503 when the pre-task update_session (status=processing + field_definitions) fails.
    Storage failure here means the whole workflow is broken — do not silently continue."""
    sm = override_databricks_dependency
    sm.update_session.side_effect = Exception("Storage unavailable")

    response = await client.post("/api/process", json=VALID_PAYLOAD)

    assert response.status_code == 503
    assert "error" in response.json()


@pytest.mark.integration
async def test_process_rejects_empty_field_definitions(
    client, override_databricks_dependency
):
    """422 when field_definitions is an empty list (pydantic min_length=1)."""
    payload = {
        "session_id": "test-session-id",
        "field_definitions": [],
    }

    response = await client.post("/api/process", json=payload)

    assert response.status_code == 422
    body = response.json()
    assert "detail" in body
