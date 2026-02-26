"""
Integration tests for POST /api/upload.

Uses the ``client`` fixture (httpx.AsyncClient + ASGITransport) and the
``mock_session_manager`` fixture from conftest.py.  The Databricks dependency
override is applied automatically through the ``client`` → ``override_databricks_dependency``
→ ``mock_session_manager`` fixture chain, so no real UC volume or Databricks
workspace is required.

Error-response conventions used by this app
-------------------------------------------
HTTP exceptions raised inside endpoint code go through the custom
``http_exception_handler`` in main.py, which serialises them as::

    {"error": "<detail string>", "status_code": <int>}

FastAPI's built-in RequestValidationError handler (422) uses the standard::

    {"detail": [{"loc": ..., "msg": ..., "type": ...}]}

Tests check against these actual response shapes.
"""

from __future__ import annotations

import io

import pytest
from httpx import AsyncClient


# ===========================================================================
# Group 1 — Happy path uploads
# ===========================================================================


@pytest.mark.integration
async def test_upload_valid_pdf_returns_200_with_session_id(
    client: AsyncClient,
    mock_session_manager,
    test_pdf_bytes: bytes,
):
    """A valid PDF upload returns 200 with a non-empty session_id and preview_url."""
    response = await client.post(
        "/api/upload",
        files={"file": ("document.pdf", io.BytesIO(test_pdf_bytes), "application/pdf")},
    )

    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert data["session_id"]              # must be non-empty
    assert "preview_url" in data
    assert data["preview_url"].startswith("/api/files/uploads/")


@pytest.mark.integration
async def test_upload_valid_png_returns_200(
    client: AsyncClient,
    mock_session_manager,
    test_image_bytes: bytes,
):
    """A valid PNG image upload returns 200 and echoes the filename."""
    response = await client.post(
        "/api/upload",
        files={"file": ("photo.png", io.BytesIO(test_image_bytes), "image/png")},
    )

    assert response.status_code == 200
    assert response.json()["filename"] == "photo.png"


@pytest.mark.integration
async def test_upload_valid_jpg_returns_200(
    client: AsyncClient,
    mock_session_manager,
    test_image_bytes: bytes,
):
    """A valid JPG image upload returns 200."""
    response = await client.post(
        "/api/upload",
        files={"file": ("scan.jpg", io.BytesIO(test_image_bytes), "image/jpeg")},
    )

    assert response.status_code == 200


@pytest.mark.integration
async def test_upload_stores_file_in_uc(
    client: AsyncClient,
    mock_session_manager,
    test_pdf_bytes: bytes,
):
    """
    After a successful upload the raw file bytes must have been passed to
    ``save_file`` on the session manager exactly once, unchanged.
    """
    await client.post(
        "/api/upload",
        files={"file": ("report.pdf", io.BytesIO(test_pdf_bytes), "application/pdf")},
    )

    mock_session_manager.save_file.assert_called_once()
    # Positional args: (session_id, stored_filename, file_bytes)
    saved_bytes = mock_session_manager.save_file.call_args.args[2]
    assert saved_bytes == test_pdf_bytes


@pytest.mark.integration
async def test_upload_creates_session_with_filename(
    client: AsyncClient,
    mock_session_manager,
    test_pdf_bytes: bytes,
):
    """
    ``create_session`` must be called with the original filename present in
    the metadata dict so the session record correctly identifies the file.
    """
    await client.post(
        "/api/upload",
        files={"file": ("my_document.pdf", io.BytesIO(test_pdf_bytes), "application/pdf")},
    )

    mock_session_manager.create_session.assert_called_once()
    # Positional args: (session_id, metadata_dict)
    metadata = mock_session_manager.create_session.call_args.args[1]
    assert metadata["filename"] == "my_document.pdf"


# ===========================================================================
# Group 2 — Validation rejections
# ===========================================================================


@pytest.mark.integration
async def test_upload_rejects_non_pdf_non_image(
    client: AsyncClient,
    mock_session_manager,
):
    """Uploading a .exe file must be rejected with 400 and a message listing allowed types."""
    response = await client.post(
        "/api/upload",
        files={"file": ("malware.exe", io.BytesIO(b"MZ\x90\x00"), "application/octet-stream")},
    )

    assert response.status_code == 400
    body = response.json()
    assert "error" in body
    # The error must mention what is allowed — callers should be able to self-serve
    error_text = body["error"]
    assert "Accepted" in error_text or "Unsupported" in error_text


@pytest.mark.integration
async def test_upload_rejects_file_over_10mb(
    client: AsyncClient,
    mock_session_manager,
):
    """A file larger than MAX_FILE_SIZE_MB must be rejected with 413."""
    eleven_mb = b"\x00" * (11 * 1024 * 1024)

    response = await client.post(
        "/api/upload",
        files={"file": ("big.pdf", io.BytesIO(eleven_mb), "application/pdf")},
    )

    assert response.status_code == 413
    body = response.json()
    assert "error" in body
    assert body["error"]  # must be non-empty; never expose a raw traceback


@pytest.mark.integration
async def test_upload_rejects_missing_file(
    client: AsyncClient,
    mock_session_manager,
):
    """
    POST without any file field must return 422 (FastAPI request validation).
    The response uses FastAPI's standard ``{"detail": [...]}`` envelope.
    """
    response = await client.post("/api/upload")

    assert response.status_code == 422
    body = response.json()
    # FastAPI validation errors always use the "detail" key
    assert "detail" in body


@pytest.mark.integration
async def test_upload_rejects_empty_filename(
    client: AsyncClient,
    mock_session_manager,
    test_pdf_bytes: bytes,
):
    """
    A file uploaded with an empty filename is rejected before the endpoint
    logic runs: Starlette's multipart parser cannot construct a valid
    UploadFile from a part with filename="" and returns 422.

    Note: the 400 path inside the endpoint (ext not in ALLOWED_EXTENSIONS)
    is never reached in this case; the rejection happens at the framework
    validation layer, so the response uses FastAPI's standard
    ``{"detail": [...]}`` envelope.
    """
    response = await client.post(
        "/api/upload",
        files={"file": ("", io.BytesIO(test_pdf_bytes), "application/pdf")},
    )

    # 422 — framework-level validation failure before endpoint code runs
    assert response.status_code == 422
    body = response.json()
    assert "detail" in body  # FastAPI validation error envelope


# ===========================================================================
# Group 3 — Storage failure handling
# ===========================================================================


@pytest.mark.integration
async def test_upload_returns_503_when_uc_unavailable(
    client: AsyncClient,
    mock_session_manager,
    test_pdf_bytes: bytes,
):
    """
    If ``create_session`` raises an unexpected exception the endpoint must
    return 503 with a user-friendly error message — not the raw exception
    text or a Python traceback.
    """
    mock_session_manager.create_session.side_effect = Exception(
        "UC connectivity failure"
    )

    response = await client.post(
        "/api/upload",
        files={"file": ("document.pdf", io.BytesIO(test_pdf_bytes), "application/pdf")},
    )

    assert response.status_code == 503
    body = response.json()
    assert "error" in body

    error_text = body["error"]
    assert error_text                               # non-empty
    assert "UC connectivity failure" not in error_text   # no raw exception leakage
    assert "traceback" not in error_text.lower()        # no stack trace


@pytest.mark.integration
async def test_upload_returns_503_when_save_file_fails(
    client: AsyncClient,
    mock_session_manager,
    test_pdf_bytes: bytes,
):
    """
    If ``save_file`` raises an unexpected exception the endpoint must return
    503, wrapping the error in a user-friendly message.
    """
    mock_session_manager.save_file.side_effect = Exception("Databricks write error")

    response = await client.post(
        "/api/upload",
        files={"file": ("document.pdf", io.BytesIO(test_pdf_bytes), "application/pdf")},
    )

    assert response.status_code == 503
    body = response.json()
    assert "error" in body

    error_text = body["error"]
    assert error_text                                    # non-empty
    assert "Databricks write error" not in error_text   # no raw exception leakage
    assert "traceback" not in error_text.lower()
