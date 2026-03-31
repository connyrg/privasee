"""
Integration tests for GET /api/files/{folder}/{filename}.

URL conventions:
  /api/files/uploads/{session_id}.{ext}         → original uploaded file
  /api/files/output/{session_id}_masked.pdf      → masked output

The endpoint validates the folder, extracts the session_id from the filename,
looks up the session in storage, and serves the file bytes.
"""

import uuid
from unittest.mock import MagicMock

import pytest

from app.models import SessionData

_VALID_SESSION_ID = "a1b2c3d4-e5f6-4789-abcd-ef0123456789"


def _make_session(filename: str = "document.pdf") -> MagicMock:
    mock = MagicMock(spec=SessionData)
    mock.session_id = _VALID_SESSION_ID
    mock.filename = filename
    mock.status = "completed"
    return mock


# ===========================================================================
# Happy path — uploads folder (original files)
# ===========================================================================


@pytest.mark.integration
async def test_serve_original_pdf_returns_200(client, override_databricks_dependency):
    """GET /api/files/uploads/{session_id}.pdf returns the original file bytes."""
    sm = override_databricks_dependency
    sm.get_session.return_value = _make_session("document.pdf")
    sm.get_file.return_value = b"%PDF-1.4 fake content"

    response = await client.get(f"/api/files/uploads/{_VALID_SESSION_ID}.pdf")

    assert response.status_code == 200
    assert response.content == b"%PDF-1.4 fake content"
    assert response.headers["content-type"] == "application/pdf"


@pytest.mark.integration
async def test_serve_original_png_returns_correct_content_type(
    client, override_databricks_dependency
):
    """PNG uploads are served with image/png content-type."""
    sm = override_databricks_dependency
    sm.get_session.return_value = _make_session("scan.png")
    sm.get_file.return_value = b"\x89PNG\r\n\x1a\n fake png"

    response = await client.get(f"/api/files/uploads/{_VALID_SESSION_ID}.png")

    assert response.status_code == 200
    assert "image/png" in response.headers["content-type"]


@pytest.mark.integration
async def test_serve_original_uses_extension_from_session_filename(
    client, override_databricks_dependency
):
    """get_file is called with the extension from session.filename, not from the URL."""
    sm = override_databricks_dependency
    sm.get_session.return_value = _make_session("scan.png")
    sm.get_file.return_value = b"fake bytes"

    await client.get(f"/api/files/uploads/{_VALID_SESSION_ID}.png")

    sm.get_file.assert_called_once_with(_VALID_SESSION_ID, "original.png")


# ===========================================================================
# Happy path — output folder (masked files)
# ===========================================================================


@pytest.mark.integration
async def test_serve_masked_pdf_returns_200(client, override_databricks_dependency):
    """GET /api/files/output/{session_id}_masked.pdf returns the masked file."""
    sm = override_databricks_dependency
    sm.get_session.return_value = _make_session("document.pdf")
    sm.get_file.return_value = b"%PDF-1.4 masked content"

    response = await client.get(f"/api/files/output/{_VALID_SESSION_ID}_masked.pdf")

    assert response.status_code == 200
    assert response.content == b"%PDF-1.4 masked content"
    sm.get_file.assert_called_once_with(_VALID_SESSION_ID, "masked.pdf")


# ===========================================================================
# Error responses
# ===========================================================================


@pytest.mark.integration
async def test_serve_file_returns_400_for_invalid_folder(
    client, override_databricks_dependency
):
    """400 when folder is not 'uploads' or 'output'."""
    response = await client.get(f"/api/files/secrets/{_VALID_SESSION_ID}.pdf")

    assert response.status_code == 400


@pytest.mark.integration
async def test_serve_file_returns_400_for_non_uuid_filename(
    client, override_databricks_dependency
):
    """400 when the filename does not contain a valid UUID."""
    response = await client.get("/api/files/uploads/not-a-uuid.pdf")

    assert response.status_code == 400


@pytest.mark.integration
async def test_serve_file_returns_404_for_unknown_session(
    client, override_databricks_dependency
):
    """404 when get_session returns None."""
    sm = override_databricks_dependency
    sm.get_session.return_value = None

    response = await client.get(f"/api/files/uploads/{_VALID_SESSION_ID}.pdf")

    assert response.status_code == 404


@pytest.mark.integration
async def test_serve_file_returns_404_when_file_not_in_storage(
    client, override_databricks_dependency
):
    """404 when the file itself is missing from UC (FileNotFoundError)."""
    sm = override_databricks_dependency
    sm.get_session.return_value = _make_session("document.pdf")
    sm.get_file.side_effect = FileNotFoundError("original.pdf not found")

    response = await client.get(f"/api/files/uploads/{_VALID_SESSION_ID}.pdf")

    assert response.status_code == 404


@pytest.mark.integration
async def test_serve_file_returns_503_on_storage_error(
    client, override_databricks_dependency
):
    """503 when get_session raises an unexpected exception."""
    sm = override_databricks_dependency
    sm.get_session.side_effect = Exception("Storage down")

    response = await client.get(f"/api/files/uploads/{_VALID_SESSION_ID}.pdf")

    assert response.status_code == 503
