"""
Integration tests for DELETE /api/sessions/{session_id}.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from app.models import SessionData


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _session(session_id: str = "sess-abc-123", filename: str = "doc.pdf") -> SessionData:
    return SessionData(
        session_id=session_id,
        filename=filename,
        file_size=1024,
        status="completed",
    )


# ===========================================================================
# Happy path
# ===========================================================================


@pytest.mark.integration
async def test_delete_session_returns_204(
    client: AsyncClient,
    override_databricks_dependency,
):
    """A valid session delete returns 204 No Content."""
    sm = override_databricks_dependency
    sm.get_session.return_value = _session()

    response = await client.delete("/api/sessions/sess-abc-123")

    assert response.status_code == 204
    assert response.content == b""


@pytest.mark.integration
async def test_delete_session_calls_delete_session_on_manager(
    client: AsyncClient,
    override_databricks_dependency,
):
    """The endpoint delegates to sm.delete_session with the correct session_id."""
    sm = override_databricks_dependency
    sm.get_session.return_value = _session(session_id="my-session")

    await client.delete("/api/sessions/my-session")

    sm.delete_session.assert_called_once_with("my-session")


# ===========================================================================
# Error paths
# ===========================================================================


@pytest.mark.integration
async def test_delete_session_404_when_session_not_found(
    client: AsyncClient,
    override_databricks_dependency,
):
    """Returns 404 when the session does not exist."""
    sm = override_databricks_dependency
    sm.get_session.return_value = None

    response = await client.delete("/api/sessions/nonexistent")

    assert response.status_code == 404
    assert "not found" in response.json()["error"].lower()


@pytest.mark.integration
async def test_delete_session_503_when_get_session_raises(
    client: AsyncClient,
    override_databricks_dependency,
):
    """Returns 503 when get_session raises an unexpected error."""
    sm = override_databricks_dependency
    sm.get_session.side_effect = Exception("storage down")

    response = await client.delete("/api/sessions/sess-abc-123")

    assert response.status_code == 503


@pytest.mark.integration
async def test_delete_session_503_when_delete_raises(
    client: AsyncClient,
    override_databricks_dependency,
):
    """Returns 503 when delete_session raises an unexpected error."""
    sm = override_databricks_dependency
    sm.get_session.return_value = _session()
    sm.delete_session.side_effect = Exception("delete failed")

    response = await client.delete("/api/sessions/sess-abc-123")

    assert response.status_code == 503
