"""
Integration tests for POST /api/process.

Uses the `client` fixture (httpx.AsyncClient + ASGITransport) and
`override_databricks_dependency`.  Entity extraction is tested with
MOCK_DATABRICKS=true (monkeypatched at module level) so no real Databricks
endpoint is required.

Databricks failure tests (504, 502) temporarily disable MOCK_DATABRICKS and
patch `httpx.AsyncClient` directly so no live endpoint is contacted.
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
async def test_process_returns_entity_list(client, override_databricks_dependency):
    """With MOCK_DATABRICKS=True the response must include a non-empty entity list
    and every entity must carry the required fields."""
    response = await client.post("/api/process", json=VALID_PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == "test-session-id"
    assert body["total_entities"] >= 1
    assert len(body["entities"]) >= 1

    entity = body["entities"][0]
    for field in (
        "id",
        "entity_type",
        "original_text",
        "replacement_text",
        "bounding_box",
        "page_number",
    ):
        assert field in entity, f"Entity missing required field '{field}'"


@pytest.mark.integration
async def test_process_saves_entities_to_uc(client, override_databricks_dependency):
    """Entities returned by the mock must be persisted via save_entities."""
    sm = override_databricks_dependency

    response = await client.post("/api/process", json=VALID_PAYLOAD)

    assert response.status_code == 200
    sm.save_entities.assert_called_once()
    saved_session_id = sm.save_entities.call_args.args[0]
    saved_entities = sm.save_entities.call_args.args[1]
    assert saved_session_id == "test-session-id"
    assert isinstance(saved_entities, list) and len(saved_entities) >= 1


@pytest.mark.integration
async def test_process_updates_session_status_to_processing_then_ready(
    client, override_databricks_dependency
):
    """update_session must be called with status='processing' before status='ready'."""
    sm = override_databricks_dependency

    response = await client.post("/api/process", json=VALID_PAYLOAD)

    assert response.status_code == 200
    calls = sm.update_session.call_args_list
    statuses = [c.kwargs.get("status") for c in calls]
    assert "processing" in statuses, (
        f"Expected 'processing' in update_session calls; got: {statuses}"
    )
    assert "ready" in statuses, (
        f"Expected 'ready' in update_session calls; got: {statuses}"
    )
    assert statuses.index("processing") < statuses.index("ready"), (
        "Expected 'processing' to be set before 'ready'"
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
# ===========================================================================


@pytest.mark.integration
async def test_process_returns_503_when_databricks_endpoint_not_configured(
    client, override_databricks_dependency, monkeypatch
):
    """503 when MOCK_DATABRICKS=False and DATABRICKS_MODEL_ENDPOINT is empty."""
    monkeypatch.setattr(main_module, "MOCK_DATABRICKS", False)
    monkeypatch.setattr(main_module, "DATABRICKS_MODEL_ENDPOINT", "")

    response = await client.post("/api/process", json=VALID_PAYLOAD)

    assert response.status_code == 503
    body = response.json()
    assert "error" in body


@pytest.mark.integration
async def test_process_returns_504_on_databricks_timeout(
    client, override_databricks_dependency, monkeypatch
):
    """504 when the Databricks HTTP call times out (httpx.TimeoutException)."""
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

    assert response.status_code == 504
    body = response.json()
    assert "error" in body


@pytest.mark.integration
async def test_process_returns_502_on_databricks_non_200_status(
    client, override_databricks_dependency, monkeypatch
):
    """502 when Databricks returns a non-200 HTTP status (httpx.HTTPStatusError)."""
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

    assert response.status_code == 502
    body = response.json()
    assert "error" in body


@pytest.mark.integration
async def test_process_returns_502_when_databricks_response_cannot_be_parsed(
    client, override_databricks_dependency, monkeypatch
):
    """502 when the Databricks response JSON cannot be mapped to Entity objects."""
    monkeypatch.setattr(main_module, "MOCK_DATABRICKS", False)
    monkeypatch.setattr(
        main_module, "DATABRICKS_MODEL_ENDPOINT", "https://fake.databricks/endpoint"
    )

    # Entities with missing required fields → pydantic ValidationError during parsing
    mock_db_response = MagicMock()
    mock_db_response.raise_for_status = MagicMock()  # no-op
    mock_db_response.json.return_value = {
        "predictions": [{"entities": [{"invalid_field": True}]}]
    }

    mock_http = AsyncMock()
    mock_http.post.return_value = mock_db_response

    with patch("app.main.httpx.AsyncClient") as MockClient:
        MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_http)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)
        response = await client.post("/api/process", json=VALID_PAYLOAD)

    assert response.status_code == 502
    body = response.json()
    assert "error" in body


# ===========================================================================
# Group 4 — Input validation
# ===========================================================================


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
