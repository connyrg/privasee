"""
PrivaSee FastAPI Backend — orchestration layer.

This module manages the user-facing workflow.  It does NOT perform any
document intelligence — all OCR and entity extraction is delegated to the
Databricks Model Serving endpoint.  Masking is the sole processing step
that runs in-process, using the unchanged MaskingService from the PoC.

Environment variables (see .env.template):
    DATABRICKS_HOST             Workspace URL
    DATABRICKS_TOKEN            Personal access token
    DATABRICKS_MODEL_ENDPOINT   Model Serving invocation URL
    UC_VOLUME_PATH              /Volumes/catalog/schema/sessions
    ALLOWED_ORIGINS             Comma-separated list of CORS origins
    MOCK_DATABRICKS             "true" → skip Databricks, return fake entities
    MAX_FILE_SIZE_MB            Upload size cap (default 10)
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from app.models import (
    ApprovalRequest,
    ApprovalResponse,
    DatabricksProcessRequest,
    DatabricksProcessResponse,
    Entity,
    ErrorResponse,
    HealthResponse,
    ProcessRequest,
    ProcessResponse,
    SessionInfo,
    UploadResponse,
)
from app.services.masking_service import MaskingService
from app.session_manager import SessionManager, UCSessionManager

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Settings (all sourced from environment)
# ---------------------------------------------------------------------------

DATABRICKS_HOST: str = os.getenv("DATABRICKS_HOST", "")
DATABRICKS_TOKEN: str = os.getenv("DATABRICKS_TOKEN", "")
DATABRICKS_MODEL_ENDPOINT: str = os.getenv("DATABRICKS_MODEL_ENDPOINT", "")
UC_VOLUME_PATH: str = os.getenv("UC_VOLUME_PATH", "")

ALLOWED_ORIGINS: List[str] = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
    if o.strip()
]

MOCK_DATABRICKS: bool = os.getenv("MOCK_DATABRICKS", "false").lower() in ("true", "1", "yes")
MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
MAX_FILE_SIZE: int = MAX_FILE_SIZE_MB * 1024 * 1024

# File types accepted at upload
ALLOWED_EXTENSIONS: frozenset = frozenset({".pdf", ".png", ".jpg", ".jpeg", ".docx"})

# Extension → MIME type
CONTENT_TYPE_MAP: Dict[str, str] = {
    ".pdf":  "application/pdf",
    ".png":  "image/png",
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PrivaSee API",
    description="Document de-identification — orchestration layer",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Service initialisation
# ---------------------------------------------------------------------------

masking_service = MaskingService()

# SessionManager is initialised lazily so the app still starts without
# Databricks credentials (useful for local testing with MOCK_DATABRICKS=true).
_session_manager: Optional[SessionManager] = None

if DATABRICKS_HOST and DATABRICKS_TOKEN and UC_VOLUME_PATH:
    _session_manager = UCSessionManager(
        databricks_host=DATABRICKS_HOST,
        token=DATABRICKS_TOKEN,
        volume_path=UC_VOLUME_PATH,
    )
    logger.info("UCSessionManager initialised")
else:
    logger.warning(
        "DATABRICKS_HOST / DATABRICKS_TOKEN / UC_VOLUME_PATH not fully configured. "
        "Session storage unavailable. Set MOCK_DATABRICKS=true for local testing."
    )


def _require_session_manager() -> SessionManager:
    """
    Return the SessionManager or raise 503 if it was not configured.

    Called at the start of every endpoint that needs UC storage.
    """
    if _session_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Session storage is not configured. "
                "Set DATABRICKS_HOST, DATABRICKS_TOKEN, and UC_VOLUME_PATH, "
                "or enable MOCK_DATABRICKS=true for local development."
            ),
        )
    return _session_manager


# ---------------------------------------------------------------------------
# Mock entity generation (MOCK_DATABRICKS=true)
# ---------------------------------------------------------------------------

# Representative fake values for the most common field types.
# Each entry is (original_text, replacement_text).
_MOCK_FIELD_DATA: Dict[str, tuple] = {
    "full name":         ("John Smith",                 "Jane Doe"),
    "name":              ("John Smith",                 "Jane Doe"),
    "first name":        ("John",                       "Jane"),
    "last name":         ("Smith",                      "Doe"),
    "date of birth":     ("01/15/1985",                 "07/22/1990"),
    "dob":               ("01/15/1985",                 "07/22/1990"),
    "ssn":               ("123-45-6789",                "987-65-4321"),
    "social security":   ("123-45-6789",                "987-65-4321"),
    "email":             ("john.smith@example.com",     "j.doe@example.org"),
    "phone":             ("(555) 123-4567",             "(555) 987-6543"),
    "address":           ("123 Main Street, Springfield", "456 Oak Ave, Shelbyville"),
    "employer":          ("Acme Corporation",           "Globex Industries"),
    "company":           ("Acme Corporation",           "Globex Industries"),
    "job title":         ("Senior Analyst",             "Principal Consultant"),
    "license":           ("D1234567",                   "X9876543"),
    "passport":          ("AB1234567",                  "CD9876543"),
    "credit card":       ("4111 1111 1111 1111",        "5500 0000 0000 0004"),
    "bank account":      ("0001234567890",              "0009876543210"),
}


def _mock_entities(session_id: str, field_definitions: list) -> List[Entity]:
    """
    Return a plausible entity list without calling Databricks.

    Generates one entity per FieldDefinition, placed at a deterministic
    (but visually spread-out) bounding box position.
    """
    entities: List[Entity] = []
    for i, field_def in enumerate(field_definitions):
        name = field_def.name if hasattr(field_def, "name") else field_def.get("name", "Unknown")
        lookup_key = name.lower()

        # Find the best matching mock entry
        match = None
        for key, pair in _MOCK_FIELD_DATA.items():
            if key in lookup_key or lookup_key in key:
                match = pair
                break
        if match is None:
            match = (f"Sample {name}", f"[{name.upper()}_REDACTED]")

        original, replacement = match

        entities.append(
            Entity(
                id=f"{session_id}_mock_{i}",
                entity_type=name,
                original_text=original,
                replacement_text=replacement,
                # Distribute boxes vertically so they don't overlap in the preview
                bounding_box=[0.05, 0.08 + i * 0.07, 0.45, 0.025],
                confidence=0.95,
                approved=True,
                page_number=1,
            )
        )
    return entities


# ---------------------------------------------------------------------------
# In-process masking helper
# ---------------------------------------------------------------------------

def _apply_masking_sync(
    file_bytes: bytes,
    original_ext: str,
    entities_to_mask: List[Dict[str, Any]],
) -> bytes:
    """
    Apply visual redactions and return the resulting PDF bytes.

    Runs synchronously (intended to be called via run_in_threadpool).

    For image files (PNG / JPG) the image is masked directly and then
    wrapped in a single-page PDF.  For PDFs each page is converted to an
    image, masked, and the pages are re-assembled into a PDF.
    DOCX masking is not yet supported.

    Args:
        file_bytes:       Raw bytes of the original document.
        original_ext:     File extension including the dot, e.g. ".pdf".
        entities_to_mask: List of entity dicts (model_dump output).

    Returns:
        PDF bytes of the masked document.

    Raises:
        HTTPException 422  if the file type cannot be masked.
        HTTPException 501  if pdf2image / poppler is not installed.
    """
    ext = original_ext.lower()

    if ext == ".docx":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Masking DOCX files is not yet supported. Convert to PDF first.",
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, f"input{ext}")
        with open(input_path, "wb") as fh:
            fh.write(file_bytes)

        if ext == ".pdf":
            try:
                from pdf2image import convert_from_path  # type: ignore
            except ImportError as exc:
                raise HTTPException(
                    status_code=status.HTTP_501_NOT_IMPLEMENTED,
                    detail=(
                        "PDF masking requires poppler. "
                        "Install it with: brew install poppler (macOS) "
                        "or apt-get install poppler-utils (Linux)."
                    ),
                ) from exc

            images = convert_from_path(input_path, dpi=200)
            masked_image_paths: List[str] = []

            for page_num, img in enumerate(images, start=1):
                page_png = os.path.join(tmpdir, f"page_{page_num}.png")
                img.save(page_png, "PNG")

                masked_png = os.path.join(tmpdir, f"masked_page_{page_num}.png")
                page_entities = [
                    e for e in entities_to_mask
                    if e.get("page_number", 1) == page_num
                ]
                masking_service.apply_masks(page_png, page_entities, masked_png)
                masked_image_paths.append(masked_png)

            # Combine masked page images back into a PDF using PIL
            from PIL import Image as PILImage  # type: ignore

            pil_images = [PILImage.open(p).convert("RGB") for p in masked_image_paths]
            masked_pdf_path = os.path.join(tmpdir, "masked.pdf")
            pil_images[0].save(
                masked_pdf_path,
                save_all=True,
                append_images=pil_images[1:],
                format="PDF",
            )
            with open(masked_pdf_path, "rb") as fh:
                return fh.read()

        elif ext in (".png", ".jpg", ".jpeg"):
            masked_img_path = os.path.join(tmpdir, f"masked{ext}")
            masking_service.apply_masks(input_path, entities_to_mask, masked_img_path)

            # Wrap the masked image in a single-page PDF
            from PIL import Image as PILImage  # type: ignore

            img = PILImage.open(masked_img_path).convert("RGB")
            masked_pdf_path = os.path.join(tmpdir, "masked.pdf")
            img.save(masked_pdf_path, format="PDF")
            with open(masked_pdf_path, "rb") as fh:
                return fh.read()

        else:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Unsupported file type for masking: '{ext}'",
            )


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc: Exception):
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error"},
    )


# ---------------------------------------------------------------------------
# GET /api/health
# ---------------------------------------------------------------------------

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """
    Liveness / readiness probe.

    Returns the API version, whether mock mode is active, and whether the
    Databricks endpoint and UC volume are configured.  Useful for confirming
    the right runtime configuration is running after deployment.
    """
    return HealthResponse(
        status="ok",
        version="2.0.0",
        mock_databricks=MOCK_DATABRICKS,
        databricks_endpoint_configured=bool(DATABRICKS_MODEL_ENDPOINT),
        uc_volume_configured=bool(UC_VOLUME_PATH),
    )


# ---------------------------------------------------------------------------
# POST /api/upload
# ---------------------------------------------------------------------------

@app.post("/api/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Accept a document upload, validate it, and create a UC session.

    Accepted types: PDF, PNG, JPG, JPEG, DOCX (up to MAX_FILE_SIZE_MB).
    Does NOT trigger processing — that is a separate POST /api/process call.

    Returns a session_id that the frontend uses for all subsequent calls.
    """
    sm = _require_session_manager()

    # --- Validate file type ---
    filename = file.filename or ""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file type '{ext or '(none)'}'. "
                f"Accepted: {sorted(ALLOWED_EXTENSIONS)}"
            ),
        )

    # --- Read and validate size ---
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File too large ({len(contents) / 1_048_576:.1f} MB). "
                f"Maximum allowed: {MAX_FILE_SIZE_MB} MB."
            ),
        )

    # Stored under the session directory with a stable name so endpoints
    # can find it without knowing the original extension at call time.
    stored_filename = f"original{ext}"

    try:
        session_id = sm.create_session(filename)
        sm.save_file(session_id, stored_filename, contents)
    except NotImplementedError:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Session storage is not yet implemented. Check server logs.",
        )
    except Exception as exc:
        logger.error("Failed to upload %s: %s", filename, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to store the uploaded file. Check Databricks connectivity.",
        )

    logger.info("Uploaded %s → session %s (%d bytes)", filename, session_id, len(contents))

    # The frontend hardcodes the original URL as /api/files/uploads/{session_id}.pdf
    # (App.jsx line 48).  The file-serving endpoint looks up the session to find
    # the actual stored filename, so the URL extension does not need to match.
    return UploadResponse(
        session_id=session_id,
        filename=filename,
        file_size=len(contents),
        page_count=1,   # accurate page count requires PDF parsing — deferred to processing
        preview_url=f"/api/files/uploads/{session_id}{ext}",
        message="File uploaded successfully",
    )


# ---------------------------------------------------------------------------
# POST /api/process
# ---------------------------------------------------------------------------

@app.post("/api/process", response_model=ProcessResponse)
async def process_document(request: ProcessRequest):
    """
    Trigger entity extraction for an uploaded document.

    Saves the field definitions to the UC session, then delegates extraction
    to the Databricks Model Serving endpoint (or mock).  Persists the
    returned entities to the session and returns them to the frontend.

    The Databricks call has a 60-second timeout.  If MOCK_DATABRICKS=true
    a hard-coded entity list is returned instead — essential for Phase 3
    validation before the Databricks endpoint is deployed.
    """
    sm = _require_session_manager()

    # --- Verify session exists ---
    try:
        session = sm.get_session(request.session_id)
    except NotImplementedError:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Session storage is not yet implemented.",
        )
    except Exception as exc:
        logger.error("get_session(%s) failed: %s", request.session_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to read session from storage.",
        )

    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {request.session_id}",
        )

    # --- Persist field definitions so masking can retrieve them later ---
    try:
        sm.update_session(
            request.session_id,
            status="processing",
            field_definitions=[f.model_dump() for f in request.field_definitions],
        )
    except NotImplementedError:
        pass  # Non-fatal — proceed without persisting
    except Exception as exc:
        logger.warning("Could not update field_definitions for %s: %s", request.session_id, exc)

    # --- Entity extraction ---
    if MOCK_DATABRICKS:
        logger.info(
            "MOCK_DATABRICKS=true — returning mock entities for session %s",
            request.session_id,
        )
        entities = _mock_entities(request.session_id, request.field_definitions)

    else:
        if not DATABRICKS_MODEL_ENDPOINT:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "DATABRICKS_MODEL_ENDPOINT is not configured. "
                    "Set the environment variable or enable MOCK_DATABRICKS=true."
                ),
            )

        db_request = DatabricksProcessRequest(
            session_id=request.session_id,
            field_definitions=request.field_definitions,
        )

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    DATABRICKS_MODEL_ENDPOINT,
                    json=db_request.to_mlflow_payload(),
                    headers={
                        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
                        "Content-Type": "application/json",
                    },
                )
            response.raise_for_status()
            raw: Dict[str, Any] = response.json()

        except httpx.TimeoutException:
            logger.error(
                "Databricks endpoint timed out for session %s", request.session_id
            )
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=(
                    "Entity extraction timed out (> 60 s). "
                    "The document may be complex or the endpoint is under load. "
                    "Try again or reduce the number of pages."
                ),
            )
        except httpx.HTTPStatusError as exc:
            logger.error(
                "Databricks returned HTTP %s for session %s: %s",
                exc.response.status_code,
                request.session_id,
                exc.response.text,
            )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=(
                    f"Databricks Model Serving returned HTTP {exc.response.status_code}. "
                    "Check that the endpoint is deployed and the token is valid."
                ),
            )
        except httpx.RequestError as exc:
            logger.error(
                "Could not reach Databricks endpoint for session %s: %s",
                request.session_id,
                exc,
            )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=(
                    "Could not reach the Databricks Model Serving endpoint. "
                    "Verify DATABRICKS_MODEL_ENDPOINT and network connectivity."
                ),
            )

        try:
            db_response = DatabricksProcessResponse.from_mlflow_response(raw)
            entities = db_response.entities
        except Exception as exc:
            logger.error(
                "Failed to parse Databricks response for session %s: %s — raw: %s",
                request.session_id,
                exc,
                raw,
            )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Could not parse the entity list returned by Databricks.",
            )

    # --- Persist entities to UC session ---
    try:
        sm.save_entities(request.session_id, [e.model_dump() for e in entities])
        sm.update_session(request.session_id, status="ready")
    except NotImplementedError:
        pass  # Non-fatal — entities are returned to the frontend regardless
    except Exception as exc:
        logger.warning(
            "Could not persist entities for session %s: %s", request.session_id, exc
        )

    suffix = " (mock)" if MOCK_DATABRICKS else ""
    logger.info(
        "Processed session %s — %d entities found%s",
        request.session_id,
        len(entities),
        suffix,
    )

    return ProcessResponse(
        session_id=request.session_id,
        entities=entities,
        total_entities=len(entities),
        message=f"Found {len(entities)} entities{suffix}",
    )


# ---------------------------------------------------------------------------
# POST /api/approve-and-mask
# ---------------------------------------------------------------------------

@app.post("/api/approve-and-mask", response_model=ApprovalResponse)
async def approve_and_mask(request: ApprovalRequest):
    """
    Apply visual redactions to the document and store the masked output.

    1. Reads the original file from UC storage.
    2. Reads the full entity list from UC (falls back to updated_entities
       from the request if storage is not yet available).
    3. Filters to the entity IDs the user approved.
    4. Applies any replacement-text overrides the user made in the ReviewTable.
    5. Calls MaskingService in a thread pool (CPU-bound, synchronous).
    6. Saves the masked PDF back to UC storage.
    7. Updates the session status to "completed".
    """
    sm = _require_session_manager()
    t0 = time.monotonic()

    # --- Load original file ---
    try:
        session = sm.get_session(request.session_id)
    except NotImplementedError:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Session storage is not yet implemented.",
        )
    except Exception as exc:
        logger.error("get_session(%s) failed: %s", request.session_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to read session from storage.",
        )

    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {request.session_id}",
        )

    # Determine the original file's extension so we can read the right file
    # and pass the correct format hint to the masking helper.
    original_ext = Path(session.filename).suffix.lower() if session.filename else ".pdf"
    stored_original = f"original{original_ext}"

    try:
        original_bytes = sm.get_file(request.session_id, stored_original)
    except NotImplementedError:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Session storage is not yet implemented.",
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Original file not found in session {request.session_id}.",
        )
    except Exception as exc:
        logger.error(
            "get_file(%s, %s) failed: %s", request.session_id, stored_original, exc, exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to read the original file from storage.",
        )

    # --- Load stored entities ---
    stored_entities: List[Dict[str, Any]] = []
    try:
        stored_entities = sm.get_entities(request.session_id)
    except NotImplementedError:
        # Storage not implemented yet — fall back to updated_entities from request
        if request.updated_entities:
            stored_entities = [e.model_dump() for e in request.updated_entities]
        else:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    "No entities are available in session storage and no updated_entities "
                    "were provided in the request. Process the document first."
                ),
            )
    except Exception as exc:
        logger.error(
            "get_entities(%s) failed: %s", request.session_id, exc, exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to read entities from storage.",
        )

    if not stored_entities:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No entities found in session. Run POST /api/process first.",
        )

    # --- Filter to approved entity IDs ---
    approved_ids = set(request.approved_entity_ids)
    entities_to_mask: List[Dict[str, Any]] = [
        e for e in stored_entities if e.get("id") in approved_ids
    ]

    if not entities_to_mask:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "None of the provided entity IDs matched entities in the session. "
                "Ensure approved_entity_ids contains valid entity IDs from /api/process."
            ),
        )

    # --- Apply user edits to replacement_text ---
    if request.updated_entities:
        updates: Dict[str, str] = {
            e.id: e.replacement_text for e in request.updated_entities
        }
        for entity in entities_to_mask:
            if entity.get("id") in updates:
                entity["replacement_text"] = updates[entity["id"]]

    # --- Apply masking in a thread pool (blocking I/O + CPU) ---
    logger.info(
        "Masking %d entities for session %s", len(entities_to_mask), request.session_id
    )
    try:
        masked_pdf_bytes: bytes = await run_in_threadpool(
            _apply_masking_sync,
            original_bytes,
            original_ext,
            entities_to_mask,
        )
    except HTTPException:
        raise  # propagate 422 / 501 from _apply_masking_sync
    except Exception as exc:
        logger.error("Masking failed for session %s: %s", request.session_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to apply redactions to the document.",
        )

    # --- Save masked PDF to UC ---
    try:
        sm.save_file(request.session_id, "masked.pdf", masked_pdf_bytes)
        sm.update_session(request.session_id, status="completed")
    except NotImplementedError:
        pass  # Non-fatal — the masked PDF is returned in the response
    except Exception as exc:
        logger.warning(
            "Could not persist masked PDF for session %s: %s", request.session_id, exc
        )

    elapsed = round(time.monotonic() - t0, 2)
    logger.info(
        "Masking complete for session %s — %d entities, %.2f s",
        request.session_id,
        len(entities_to_mask),
        elapsed,
    )

    return ApprovalResponse(
        session_id=request.session_id,
        # The frontend hardcodes the original URL during upload (App.jsx:48),
        # but we return it here too for completeness and API clients that use it.
        original_pdf_url=f"/api/files/uploads/{request.session_id}{original_ext}",
        masked_pdf_url=f"/api/files/output/{request.session_id}_masked.pdf",
        entities_masked=len(entities_to_mask),
        message="Masked PDF generated successfully",
    )


# ---------------------------------------------------------------------------
# GET /api/files/{folder}/{filename}
# ---------------------------------------------------------------------------

@app.get("/api/files/{folder}/{filename}")
async def serve_file(folder: str, filename: str):
    """
    Serve document files stored in the UC volume.

    URL conventions (maintained for frontend compatibility):
        /api/files/uploads/{session_id}{ext}   → original uploaded file
        /api/files/output/{session_id}_masked.pdf → masked output

    The session_id is extracted from the filename so the endpoint can look
    up the correct file in UC storage regardless of the URL extension.

    Security: only the "uploads" and "output" folder tokens are accepted;
    filenames are never used as raw filesystem paths.
    """
    sm = _require_session_manager()

    # --- Validate folder ---
    if folder not in ("uploads", "output"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid folder '{folder}'. Allowed: uploads, output.",
        )

    # --- Extract session_id and stored filename from URL ---
    stem = Path(filename).stem   # e.g. "abc123" or "abc123_masked"

    if folder == "uploads":
        # URL pattern: {session_id}.{ext}
        session_id = stem
        stored_filename_prefix = "original"
    else:
        # folder == "output", URL pattern: {session_id}_masked.pdf
        if stem.endswith("_masked"):
            session_id = stem[: -len("_masked")]
        else:
            # Fallback: treat everything before the first dot as session_id
            session_id = stem
        stored_filename_prefix = "masked"

    # --- Validate session_id looks like a UUID (basic sanity check) ---
    if not session_id or len(session_id) < 8:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not parse a valid session ID from filename '{filename}'.",
        )

    # --- Look up session to determine original extension ---
    try:
        session = sm.get_session(session_id)
    except NotImplementedError:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Session storage is not yet implemented.",
        )
    except Exception as exc:
        logger.error("get_session(%s) failed: %s", session_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to read session from storage.",
        )

    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )

    # Determine the exact stored filename
    if stored_filename_prefix == "original":
        original_ext = Path(session.filename).suffix.lower() if session.filename else ".pdf"
        stored_name = f"original{original_ext}"
        content_type = CONTENT_TYPE_MAP.get(original_ext, "application/octet-stream")
    else:
        stored_name = "masked.pdf"
        content_type = "application/pdf"

    # --- Read file bytes from UC ---
    try:
        file_bytes = sm.get_file(session_id, stored_name)
    except NotImplementedError:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Session storage is not yet implemented.",
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File '{stored_name}' not found for session {session_id}.",
        )
    except Exception as exc:
        logger.error(
            "get_file(%s, %s) failed: %s", session_id, stored_name, exc, exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to read file from storage.",
        )

    return Response(
        content=file_bytes,
        media_type=content_type,
        headers={
            # inline so iframes display the file rather than downloading it
            "Content-Disposition": f'inline; filename="{filename}"',
            "Content-Length": str(len(file_bytes)),
        },
    )


# ---------------------------------------------------------------------------
# GET /api/sessions/{session_id}
# ---------------------------------------------------------------------------

@app.get("/api/sessions/{session_id}", response_model=SessionInfo)
async def get_session_info(session_id: str):
    """Return public metadata for a session (no file paths or entity details)."""
    sm = _require_session_manager()

    try:
        session = sm.get_session(session_id)
    except NotImplementedError:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Session storage is not yet implemented.",
        )
    except Exception as exc:
        logger.error("get_session(%s) failed: %s", session_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to read session from storage.",
        )

    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )

    return SessionInfo(
        session_id=session.session_id,
        filename=session.filename,
        file_size=session.file_size,
        status=session.status,
        entity_count=len(session.entities),
        has_masked_output=session.status == "completed",
    )


# ---------------------------------------------------------------------------
# DELETE /api/sessions/{session_id}
# ---------------------------------------------------------------------------

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and all its artefacts from UC storage."""
    sm = _require_session_manager()

    try:
        session = sm.get_session(session_id)
    except NotImplementedError:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Session storage is not yet implemented.",
        )
    except Exception as exc:
        logger.error("get_session(%s) failed: %s", session_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to read session from storage.",
        )

    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )

    try:
        sm.delete_session(session_id)
    except NotImplementedError:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Session storage is not yet implemented.",
        )
    except Exception as exc:
        logger.error("delete_session(%s) failed: %s", session_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to delete session from storage.",
        )

    logger.info("Session deleted: %s", session_id)
    return {"message": "Session deleted successfully", "session_id": session_id}


# ---------------------------------------------------------------------------
# Dev server entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
