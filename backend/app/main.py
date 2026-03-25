"""
PrivaSee FastAPI Backend — orchestration layer.

This module manages the user-facing workflow.  No document processing runs
in-process — OCR/entity extraction and masking are both delegated to
Databricks Model Serving endpoints.

Environment variables (see .env.template):
    DATABRICKS_HOST               Workspace URL
    DATABRICKS_TOKEN              Personal access token
    DATABRICKS_MODEL_ENDPOINT     Model Serving invocation URL (document intelligence)
    DATABRICKS_MASKING_ENDPOINT   Model Serving invocation URL (masking model)
    UC_VOLUME_PATH                /Volumes/catalog/schema/sessions
    ALLOWED_ORIGINS               Comma-separated list of CORS origins
    MOCK_DATABRICKS               "true" → skip Databricks, return fake entities
    MAX_FILE_SIZE_MB              Upload size cap (default 10)

Deploy:
export http_proxy="" && export https_proxy="" && rsconnect deploy fastapi  --server  https://sds-posit-connect-prod.int.corp.sun/ --api-key $POSIT_CONNECT_API_KEY -p venv/bin/python --entrypoint app.main:app . --insecure  --exclude venv/
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from app.config_manager import ConfigManager
from app.models import (
    ApprovalRequest,
    ApprovalResponse,
    ConfigDetail,
    ConfigSummary,
    DatabricksProcessRequest,
    DatabricksProcessResponse,
    Entity,
    EntityVerifyResult,
    Occurrence,
    ErrorResponse,
    FieldDefinition,
    HealthResponse,
    ProcessAcceptedResponse,
    ProcessRequest,
    ProcessResponse,
    SaveConfigRequest,
    SessionInfo,
    SystemTemplateDetail,
    SystemTemplateSummary,
    UploadResponse,
    VerifyRequest,
    VerifyResponse,
)
from app.session_manager import UCSessionManager

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,  # Posit Connect captures stdout, not stderr
    force=True,  # override any handlers Uvicorn already installed
)
logger = logging.getLogger(__name__)
# Dedicated audit logger — records key workflow transitions and all failures.
# Can be routed to a separate handler via logging config (e.g. grep "privasee.audit").
audit_logger = logging.getLogger("privasee.audit")

# ---------------------------------------------------------------------------
# Settings (all sourced from environment)
# ---------------------------------------------------------------------------

DATABRICKS_HOST: str = os.getenv("DATABRICKS_HOST", "")
DATABRICKS_TOKEN: str = os.getenv("DATABRICKS_TOKEN", "")
DATABRICKS_MODEL_ENDPOINT: str = os.getenv("DATABRICKS_MODEL_ENDPOINT", "")
DATABRICKS_MASKING_ENDPOINT: str = os.getenv("DATABRICKS_MASKING_ENDPOINT", "")
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

# SessionManager is initialised lazily so the app still starts without
# Databricks credentials (useful for local testing with MOCK_DATABRICKS=true).
_session_manager: Optional[UCSessionManager] = None
_config_manager: Optional[ConfigManager] = None

if DATABRICKS_HOST and DATABRICKS_TOKEN and UC_VOLUME_PATH:
    _session_manager = UCSessionManager(
        databricks_host=DATABRICKS_HOST,
        token=DATABRICKS_TOKEN,
        volume_path=UC_VOLUME_PATH,
    )
    _config_manager = ConfigManager(
        databricks_host=DATABRICKS_HOST,
        token=DATABRICKS_TOKEN,
        sessions_volume_path=UC_VOLUME_PATH,
    )
    logger.info("UCSessionManager and ConfigManager initialised")
else:
    logger.warning(
        "DATABRICKS_HOST / DATABRICKS_TOKEN / UC_VOLUME_PATH not fully configured. "
        "Session storage unavailable. Set MOCK_DATABRICKS=true for local testing."
    )


def _require_config_manager() -> ConfigManager:
    if _config_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Config storage is not configured. Set DATABRICKS_HOST, DATABRICKS_TOKEN, and UC_VOLUME_PATH.",
        )
    return _config_manager


def _require_session_manager() -> UCSessionManager:
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
# System templates (hardcoded — no UC volume needed)
# ---------------------------------------------------------------------------

_SYSTEM_TEMPLATES: List[Dict[str, Any]] = [
    {
        "key": "common_pii",
        "template_name": "Common PII",
        "description": "Standard fields for Australian healthcare document de-identification",
        "field_definitions": [
            {
                "name": "Full Name",
                "description": "Patient or person's full name, including first and last name and any middle names",
                "strategy": "Fake Data",
            },
            {
                "name": "Date of Birth",
                "description": "Date of birth in any format (e.g., 01/01/1980, 1 January 1980, DOB: 01-01-1980)",
                "strategy": "Fake Data",
            },
            {
                "name": "Medicare Number",
                "description": "Australian Medicare card number, typically 10-11 digits, sometimes with a reference number suffix",
                "strategy": "Black Out",
            },
            {
                "name": "Physical Address",
                "description": "Full residential or mailing address including street number, street name, suburb, state, and postcode",
                "strategy": "Fake Data",
            },
        ],
    },
]

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

        field_strategy = (
            field_def.strategy.value if hasattr(field_def, "strategy") else "Fake Data"
        )
        entities.append(
            Entity(
                id=f"{session_id}_mock_{i}",
                entity_type=name,
                original_text=original,
                replacement_text=replacement,
                confidence=0.95,
                approved=True,
                strategy=field_strategy,
                occurrences=[
                    Occurrence(
                        page_number=1,
                        original_text=original,
                        # Distribute boxes vertically so they don't overlap in the preview
                        bounding_boxes=[[0.05, 0.08 + i * 0.07, 0.45, 0.025]],
                    )
                ],
            )
        )
    return entities


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
        databricks_masking_endpoint_configured=bool(DATABRICKS_MASKING_ENDPOINT),
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
    audit_logger.info("UPLOAD session=%s file=%r bytes=%d", session_id, filename, len(contents))

    # Count pages for PDFs using PyMuPDF (already a dependency).
    page_count = 1
    if ext == ".pdf":
        try:
            import fitz
            doc = fitz.open(stream=contents, filetype="pdf")
            page_count = doc.page_count
            doc.close()
        except Exception as exc:
            logger.warning("Could not count pages for %s: %s", filename, exc)

    return UploadResponse(
        session_id=session_id,
        filename=filename,
        file_size=len(contents),
        page_count=page_count,
        preview_url=f"/api/files/uploads/{session_id}{ext}",
        message="File uploaded successfully",
    )


# ---------------------------------------------------------------------------
# POST /api/process  (fire-and-forget — returns 202 immediately)
# ---------------------------------------------------------------------------

async def _process_background(
    session_id: str,
    field_definitions: List[FieldDefinition],
    sm: UCSessionManager,
) -> None:
    """
    Background task: call Databricks, persist entities, update session status.

    Runs after the 202 response has been sent to the client.  All blocking
    UCSessionManager calls are wrapped in asyncio.to_thread so they don't
    stall the event loop.
    """
    _t_start = time.monotonic()
    audit_logger.info("PROCESS_START session=%s fields=%d", session_id, len(field_definitions))

    if MOCK_DATABRICKS:
        logger.info("MOCK_DATABRICKS=true — using mock entities for session %s", session_id)
        entities = _mock_entities(session_id, field_definitions)
    else:
        if not DATABRICKS_MODEL_ENDPOINT:
            err = "DATABRICKS_MODEL_ENDPOINT is not configured."
            logger.error(err)
            audit_logger.error("PROCESS_FAILURE session=%s reason=%r", session_id, err)
            try:
                await asyncio.to_thread(sm.update_session, session_id, status="error", error_message=err)
            except Exception as exc:
                logger.error("Failed to mark session %s as error after missing endpoint: %s", session_id, exc)
            return

        db_request = DatabricksProcessRequest(
            session_id=session_id,
            field_definitions=field_definitions,
        )
        logger.info("Calling Databricks endpoint for session %s", session_id)
        _t_http = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                response = await client.post(
                    DATABRICKS_MODEL_ENDPOINT,
                    json=db_request.to_mlflow_payload(),
                    headers={
                        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
                        "Content-Type": "application/json",
                    },
                )
            response.raise_for_status()
            _http_elapsed = time.monotonic() - _t_http
            logger.info(
                "Databricks HTTP call complete for session %s: %.1fs, response %d bytes",
                session_id, _http_elapsed, len(response.content),
            )
            raw: Dict[str, Any] = response.json()
        except httpx.TimeoutException as exc:
            err = "Entity extraction timed out. The document may be complex or the endpoint is under load."
            logger.error("Databricks timed out for session %s after %.1fs: %s", session_id, time.monotonic() - _t_http, exc)
            audit_logger.error("PROCESS_FAILURE session=%s reason=timeout", session_id)
            try:
                await asyncio.to_thread(sm.update_session, session_id, status="error", error_message=err)
            except Exception as ue:
                logger.error("Failed to mark session %s as error after timeout: %s", session_id, ue)
            return
        except (httpx.HTTPStatusError, httpx.RequestError) as exc:
            err = f"Databricks request failed: {exc}"
            logger.error("Databricks error for session %s after %.1fs: %s", session_id, time.monotonic() - _t_http, exc)
            audit_logger.error("PROCESS_FAILURE session=%s reason=%r", session_id, str(exc))
            try:
                await asyncio.to_thread(sm.update_session, session_id, status="error", error_message=err)
            except Exception as ue:
                logger.error("Failed to mark session %s as error after HTTP error: %s", session_id, ue)
            return

        logger.info(
            "Databricks raw response keys for session %s: %s",
            session_id,
            list(raw.keys()),
        )

        # Check directly for a model-level error in the raw response before
        # attempting to parse entities.  This handles the case where the model's
        # predict() outer try/except caught an internal error (e.g. vision API
        # 504) and returned {"status": "error", "error_message": "..."} so the
        # user sees the real message rather than a generic parse-failure string.
        _predictions = raw.get("predictions") or raw.get("dataframe_records")
        _record = _predictions[0] if isinstance(_predictions, list) and _predictions else raw
        if isinstance(_record, dict) and _record.get("status") == "error":
            err = _record.get("error_message") or "Entity extraction failed."
            logger.error("Model reported error for session %s: %s", session_id, err)
            audit_logger.error("PROCESS_FAILURE session=%s reason=model_error err=%r", session_id, err)
            try:
                await asyncio.to_thread(sm.update_session, session_id, status="error", error_message=err)
            except Exception as ue:
                logger.error("Failed to mark session %s as error after model error: %s", session_id, ue)
            return

        try:
            db_response = DatabricksProcessResponse.from_mlflow_response(raw)
            entities = db_response.entities
            logger.info("Parsed %d entities for session %s", len(entities), session_id)
        except Exception as exc:
            err = f"Entity extraction failed: {exc}"
            logger.error("Parse error for session %s: %s — raw: %s", session_id, exc, raw)
            audit_logger.error("PROCESS_FAILURE session=%s reason=parse_error err=%r", session_id, str(exc))
            try:
                await asyncio.to_thread(sm.update_session, session_id, status="error", error_message=err)
            except Exception as ue:
                logger.error("Failed to mark session %s as error after parse failure: %s", session_id, ue)
            return

    # --- Persist entities (mock only) and mark session ready for review ---
    # Real path: the Databricks model already wrote merged entities.json to UC
    # via _write_to_uc_volume — saving again would overwrite with unmerged data.
    if MOCK_DATABRICKS:
        try:
            await asyncio.to_thread(sm.save_entities, session_id, [e.model_dump() for e in entities])
        except Exception as exc:
            logger.error("Could not persist mock entities for session %s: %s", session_id, exc)
            try:
                await asyncio.to_thread(
                    sm.update_session, session_id, status="error",
                    error_message=f"Failed to save extraction results: {exc}",
                )
            except Exception as ue:
                logger.error("Failed to mark session %s as error after entity save failure: %s", session_id, ue)
            return

    _t_status = time.monotonic()
    try:
        await asyncio.to_thread(sm.update_session, session_id, status="awaiting_review")
        logger.info("Session %s status update to awaiting_review took %.1fs", session_id, time.monotonic() - _t_status)
    except Exception as exc:
        logger.error(
            "Could not update session %s to 'awaiting_review' after %.1fs: %s — "
            "attempting to mark as error so the user is not left waiting.",
            session_id, time.monotonic() - _t_status, exc, exc_info=True,
        )
        audit_logger.error("PROCESS_FAILURE session=%s reason=status_update_failed", session_id)
        try:
            await asyncio.to_thread(
                sm.update_session, session_id,
                status="error",
                error_message="Extraction completed but session update failed. Please reset and try again.",
            )
        except Exception as ue:
            logger.error(
                "Also failed to mark session %s as error after awaiting_review update failure: %s",
                session_id, ue,
            )
        return

    _total = time.monotonic() - _t_start
    suffix = " (mock)" if MOCK_DATABRICKS else ""
    logger.info("Background processing done for session %s — %d entities, total %.1fs%s", session_id, len(entities), _total, suffix)
    audit_logger.info("PROCESS_COMPLETE session=%s entities=%d total_s=%.1f mock=%s", session_id, len(entities), _total, MOCK_DATABRICKS)


@app.post("/api/process", response_model=ProcessAcceptedResponse, status_code=202)
async def process_document(request: ProcessRequest, background_tasks: BackgroundTasks):
    """
    Trigger entity extraction for an uploaded document.

    Returns 202 immediately and runs the Databricks call in the background.
    Poll GET /api/sessions/{session_id} until status is 'awaiting_review' (done)
    or 'error' (failed).  The response will include entities once ready.
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

    # --- Persist field definitions and set status to processing ---
    try:
        await asyncio.to_thread(
            sm.update_session,
            request.session_id,
            status="processing",
            field_definitions=[f.model_dump() for f in request.field_definitions],
        )
    except NotImplementedError:
        pass
    except Exception as exc:
        logger.error("Could not update session %s before processing: %s", request.session_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to update session state in storage.",
        )

    # --- Schedule background extraction and return immediately ---
    background_tasks.add_task(
        _process_background,
        request.session_id,
        request.field_definitions,
        sm,
    )

    return ProcessAcceptedResponse(session_id=request.session_id)


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
    5. Delegates masking to the Databricks MaskingModel endpoint, which writes
       masked.pdf back to UC directly.
    6. Updates the session status to "completed".
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
        for entity in stored_entities:
            if entity.get("id") in updates:
                entity["replacement_text"] = updates[entity["id"]]
        entities_to_mask = [e for e in stored_entities if e.get("id") in approved_ids]

    # --- Persist audit record before masking (captured even if masking fails) ---
    try:
        sm.save_masking_decisions(request.session_id, stored_entities, approved_ids)
    except Exception as exc:
        logger.error(
            "Could not save masking decisions for %s: %s", request.session_id, exc, exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to persist masking decisions to storage.",
        )

    # --- Delegate masking to Databricks ---
    logger.info(
        "Masking %d entities for session %s", len(entities_to_mask), request.session_id
    )
    audit_logger.info("MASK_START session=%s entities=%d", request.session_id, len(entities_to_mask))

    if not MOCK_DATABRICKS and not DATABRICKS_MASKING_ENDPOINT:
        audit_logger.error("MASK_FAILURE session=%s reason=endpoint_not_configured", request.session_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "DATABRICKS_MASKING_ENDPOINT is not configured. "
                "Set the environment variable or enable MOCK_DATABRICKS=true."
            ),
        )

    if MOCK_DATABRICKS and not DATABRICKS_MASKING_ENDPOINT:
        # Local dev / test mode: skip real masking, just update status.
        logger.info("Mock mode: skipping masking for session %s", request.session_id)
    else:
        # Production: POST to the Databricks MaskingModel endpoint.
        # The model reads the original file from UC, applies redactions,
        # and writes masked.pdf back to UC — no local save_file needed.
        payload = {
            "dataframe_records": [
                {
                    "session_id": request.session_id,
                    "entities_to_mask": json.dumps(entities_to_mask),
                }
            ]
        }
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    DATABRICKS_MASKING_ENDPOINT,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
                        "Content-Type": "application/json",
                    },
                )
                resp.raise_for_status()
        except httpx.TimeoutException:
            audit_logger.error("MASK_FAILURE session=%s reason=timeout", request.session_id)
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Masking request to Databricks timed out.",
            )
        except httpx.HTTPStatusError as exc:
            logger.error(
                "Databricks masking endpoint returned %s for session %s",
                exc.response.status_code, request.session_id,
            )
            audit_logger.error(
                "MASK_FAILURE session=%s reason=http_error status=%d",
                request.session_id, exc.response.status_code,
            )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Databricks masking endpoint returned an error.",
            )
        except Exception as exc:
            logger.error(
                "Masking via Databricks failed for session %s: %s",
                request.session_id, exc, exc_info=True,
            )
            audit_logger.error("MASK_FAILURE session=%s reason=%r", request.session_id, str(exc))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to apply redactions via Databricks.",
            )

    # Update session status — non-fatal: masking succeeded and masked.pdf is
    # already in UC. Status being stuck at 'awaiting_review' has no visible
    # impact on the user (frontend uses the API response, not stored status).
    try:
        sm.update_session(request.session_id, status="completed")
    except Exception as exc:
        logger.error(
            "Could not update session %s to 'completed' after successful masking: %s",
            request.session_id, exc, exc_info=True,
        )
        audit_logger.error("MASK_FAILURE session=%s reason=status_update_failed", request.session_id)

    elapsed = round(time.monotonic() - t0, 2)
    logger.info(
        "Masking complete for session %s — %d entities, %.2f s",
        request.session_id,
        len(entities_to_mask),
        elapsed,
    )
    audit_logger.info(
        "MASK_COMPLETE session=%s entities_masked=%d elapsed_s=%.2f",
        request.session_id, len(entities_to_mask), elapsed,
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

    # --- Validate session_id is a well-formed UUID ---
    try:
        uuid.UUID(session_id)
    except (ValueError, AttributeError):
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
        session = await asyncio.to_thread(sm.get_session, session_id)
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

    # Load entities from entities.json when the session has completed extraction.
    # get_session() only reads metadata.json, so we must fetch entities separately.
    entities: List[Entity] = []
    if session.status in ("awaiting_review", "completed"):
        try:
            raw_entities = await asyncio.to_thread(sm.get_entities, session_id)
            entities = [Entity(**e) for e in raw_entities]
        except FileNotFoundError:
            logger.debug("entities.json not yet present for session %s — returning empty list", session_id)
        except Exception as exc:
            logger.error("Could not load entities for session %s: %s", session_id, exc, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Failed to load entities from storage.",
            )

    return SessionInfo(
        session_id=session.session_id,
        filename=session.filename,
        file_size=session.file_size,
        status=session.status,
        entity_count=len(entities),
        has_masked_output=session.status == "completed",
        entities=entities,
        error_message=session.error_message,
    )


# ---------------------------------------------------------------------------
# DELETE /api/sessions/{session_id}
# ---------------------------------------------------------------------------

@app.delete("/api/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(session_id: str):
    """
    Delete all UC volume artefacts for a session.

    Removes metadata.json, entities.json, original.{ext}, and masked.pdf.
    Files that were never created are silently skipped.
    Returns 204 on success, 404 if the session does not exist.
    """
    sm = _require_session_manager()

    try:
        session = sm.get_session(session_id)
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
    except Exception as exc:
        logger.error("delete_session(%s) failed: %s", session_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to delete session artefacts from storage.",
        )

    logger.info("Deleted session %s", session_id)
    audit_logger.info("DELETE session=%s", session_id)


# ---------------------------------------------------------------------------
# Verify masking
# ---------------------------------------------------------------------------

@app.post("/api/sessions/{session_id}/verify", response_model=VerifyResponse)
async def verify_session(session_id: str, request: VerifyRequest):
    """
    Verify that entities were successfully masked in the output PDF.

    Retrieves masked.pdf from UC storage, extracts its text layer using
    PyMuPDF, and checks case-insensitively whether each entity's
    original_text still appears in the extracted text.

    Note: image-only (scanned) PDFs have no text layer, so all entities
    will appear as masked regardless of actual redaction quality.
    """
    sm = _require_session_manager()

    try:
        pdf_bytes = sm.get_file(session_id, "masked.pdf")
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Masked PDF not found for session: {session_id}",
        )
    except Exception as exc:
        logger.error("get_file(%s, masked.pdf) failed: %s", session_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to retrieve masked PDF from storage.",
        )

    import fitz  # noqa: PLC0415 — deferred to avoid top-level cost
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        extracted_text = "".join(page.get_text() for page in doc)
        doc.close()
    except Exception as exc:
        logger.error("Text extraction failed for session %s: %s", session_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to extract text from masked PDF.",
        )

    extracted_lower = extracted_text.lower()
    results: List[EntityVerifyResult] = []
    for entity in request.entities:
        texts_to_check = {entity.original_text.lower()}
        if entity.occurrences:
            for occ in entity.occurrences:
                if occ.original_text:
                    texts_to_check.add(occ.original_text.lower())
        masked = not any(t in extracted_lower for t in texts_to_check)
        results.append(EntityVerifyResult(
            id=entity.id,
            original_text=entity.original_text,
            masked=masked,
        ))

    total = len(results)
    masked_count = sum(1 for r in results if r.masked)
    score = round((masked_count / total * 100) if total > 0 else 100.0, 1)

    logger.info("Verify session %s: %d/%d masked (score=%.1f)", session_id, masked_count, total, score)

    return VerifyResponse(
        session_id=session_id,
        score=score,
        masked_count=masked_count,
        total=total,
        entities=results,
    )


# ---------------------------------------------------------------------------
# System templates
# ---------------------------------------------------------------------------

@app.get("/api/templates", response_model=List[SystemTemplateSummary])
async def list_system_templates():
    """List all built-in system templates (no auth required)."""
    return [
        SystemTemplateSummary(
            key=t["key"],
            template_name=t["template_name"],
            description=t["description"],
            field_count=len(t["field_definitions"]),
        )
        for t in _SYSTEM_TEMPLATES
    ]


@app.get("/api/templates/{key}", response_model=SystemTemplateDetail)
async def get_system_template(key: str):
    """Load a built-in system template by its key."""
    tmpl = next((t for t in _SYSTEM_TEMPLATES if t["key"] == key), None)
    if tmpl is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Template not found: {key}",
        )
    return SystemTemplateDetail(
        key=tmpl["key"],
        template_name=tmpl["template_name"],
        description=tmpl["description"],
        field_count=len(tmpl["field_definitions"]),
        field_definitions=[FieldDefinition(**f) for f in tmpl["field_definitions"]],
    )


# ---------------------------------------------------------------------------
# Config management
# ---------------------------------------------------------------------------

@app.post("/api/configs", response_model=ConfigSummary, status_code=status.HTTP_201_CREATED)
async def save_config(request: SaveConfigRequest):
    """Save a named set of field definitions. Overwrites if the same name already exists."""
    cm = _require_config_manager()
    try:
        key = cm.save_config(
            config_name=request.config_name,
            field_definitions=[f.model_dump() for f in request.field_definitions],
        )
    except Exception as exc:
        logger.error("Failed to save config %r: %s", request.config_name, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to save config to storage.",
        )
    return ConfigSummary(
        config_name=request.config_name,
        key=key,
        saved_at=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/api/configs", response_model=List[ConfigSummary])
async def list_configs():
    """List all saved configs (name + key + timestamp, no field definitions)."""
    cm = _require_config_manager()
    try:
        return cm.list_configs()
    except Exception as exc:
        logger.error("Failed to list configs: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to retrieve configs from storage.",
        )


@app.get("/api/configs/{key}", response_model=ConfigDetail)
async def get_config(key: str):
    """Load a saved config by its key (includes field definitions)."""
    cm = _require_config_manager()
    try:
        config = cm.get_config(key)
    except Exception as exc:
        logger.error("Failed to load config %r: %s", key, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to retrieve config from storage.",
        )
    if config is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Config not found: {key}",
        )
    return ConfigDetail(**config)


# ---------------------------------------------------------------------------
# Static frontend (Posit Connect / production)
# ---------------------------------------------------------------------------
# Mount a pre-built static frontend bundle if one is present.
# This is only used for the legacy React frontend (frontend/).  The primary
# Dash frontend (frontend_dash/) is deployed separately to Posit Connect and
# does not use this mount.
#
# The directory would be populated by:
#   cp -r frontend/dist backend/static
#
# The mount is conditional so it has no effect when the static/ directory is
# absent (the default for a plain checkout or Dash-only deployment).
#
# MUST be registered last — FastAPI resolves routes in registration order, so
# all /api/* routes above take precedence over this catch-all mount.

_STATIC_DIR = Path(__file__).parent.parent / "static"
if _STATIC_DIR.exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="frontend")
    logger.info("Serving frontend from %s", _STATIC_DIR)


# ---------------------------------------------------------------------------
# Dev server entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    path, port = '', 8000 # declare uvicorn arguments

    # When running in Posit Workbench, pass port to rserver-url to determine the root path
    # See https://docs.posit.co/ide/server-pro/user/vs-code/guide/posit-workbench-extension.html#fastapi for details
    if 'RS_SERVER_URL' in os.environ and os.environ['RS_SERVER_URL']:
        import subprocess
        path = subprocess.run(f'echo $(/usr/lib/rstudio-server/bin/rserver-url -l {port})', stdout=subprocess.PIPE, shell=True).stdout.decode().strip()

    uvicorn.run("app.main:app", host="0.0.0.0", root_path = path, port = 8000, reload=True)
