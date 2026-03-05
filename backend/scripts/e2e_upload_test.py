"""
E2E Test — PrivaSee full document workflow

Covers the three main API endpoints in sequence:
    1. POST /api/upload          — store document in UC volume
    2. POST /api/process         — extract PII entities (requires MOCK_DATABRICKS=true
                                   on the server, or a live Databricks model endpoint)
    3. POST /api/approve-and-mask — apply redactions and store masked PDF in UC volume

Test cases:
    • Digital PDF        — text embedded as PDF text objects (E1, E6)
    • Multi-page PDF     — 2-page PDF (E4)
    • PNG upload         — image input, masked output is always PDF (E5)
    • Partial approval   — approve only a subset of entities (E2)
    • Reprocess session  — call /api/process twice on same session (E3)
    • Concurrent sessions — two sessions in parallel, verify isolation (E7)
    • Scanned PDF        — image-only page (placeholder; not yet implemented)

Usage:
    cd backend
    python scripts/e2e_upload_test.py

Prerequisites:
    - .env populated with DATABRICKS_HOST, DATABRICKS_TOKEN, UC_VOLUME_PATH
    - Server running with MOCK_DATABRICKS=true (or a live Databricks endpoint):
        MOCK_DATABRICKS=true uvicorn app.main:app --reload
"""

from __future__ import annotations

import io
import json
import os
import sys
import threading

import fitz  # pymupdf
import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv()

DATABRICKS_HOST = os.getenv("DATABRICKS_HOST", "").rstrip("/")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN", "")
UC_VOLUME_PATH = os.getenv("UC_VOLUME_PATH", "").rstrip("/")
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")

FILES_API = "/api/2.0/fs/files"

# Field definitions sent to /api/process — chosen to match text in the dummy PDFs
# so mock entities have recognisable names even when MOCK_DATABRICKS=true.
FIELD_DEFINITIONS = [
    {"name": "Full Name",  "description": "Person's full name",  "strategy": "Fake Data"},
    {"name": "Email",      "description": "Email address",        "strategy": "Fake Data"},
    {"name": "Phone",      "description": "Phone number",         "strategy": "Fake Data"},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check(condition: bool, msg: str) -> None:
    if not condition:
        print(f"  FAIL: {msg}")
        raise AssertionError(msg)
    print(f"  OK   {msg}")


def _db_headers() -> dict:
    return {"Authorization": f"Bearer {DATABRICKS_TOKEN}"}


def _db_url(uc_path: str) -> str:
    return f"{DATABRICKS_HOST}{FILES_API}{uc_path}"


def _uc_path(session_id: str, filename: str) -> str:
    return f"{UC_VOLUME_PATH}/{session_id}/{filename}"


def _uc_exists(session_id: str, filename: str) -> tuple[bool, int]:
    """Return (exists, http_status) for a file in the UC volume."""
    resp = requests.get(_db_url(_uc_path(session_id, filename)), headers=_db_headers(), timeout=30)
    return resp.status_code == 200, resp.status_code


def _uc_read_json(session_id: str, filename: str) -> dict:
    """Read and parse a JSON file directly from the UC volume."""
    resp = requests.get(_db_url(_uc_path(session_id, filename)), headers=_db_headers(), timeout=30)
    resp.raise_for_status()
    return resp.json()


def _check_entity_structure(entity: dict, idx: int) -> None:
    """Assert an entity dict has all required fields with correct types."""
    for field in ("id", "entity_type", "original_text", "replacement_text"):
        _check(
            field in entity and isinstance(entity[field], str),
            f"entity[{idx}].{field} is a string",
        )
    bbox = entity.get("bounding_box", [])
    _check(
        isinstance(bbox, list) and len(bbox) == 4,
        f"entity[{idx}].bounding_box has 4 values",
    )
    _check(
        isinstance(entity.get("page_number", 0), int) and entity.get("page_number", 0) >= 1,
        f"entity[{idx}].page_number >= 1",
    )


# ---------------------------------------------------------------------------
# Document factories
# ---------------------------------------------------------------------------

def make_digital_pdf() -> bytes:
    """Digital PDF: text embedded as real PDF text objects."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4
    page.insert_text((72, 100), "PrivaSee E2E upload test — digital document")
    page.insert_text((72, 130), "Name: Jane Doe")
    page.insert_text((72, 160), "Email: jane@example.com")
    page.insert_text((72, 190), "Phone: +44 7700 900123")
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def make_multipage_pdf() -> bytes:
    """Two-page PDF for testing multi-page handling."""
    doc = fitz.open()
    p1 = doc.new_page(width=595, height=842)
    p1.insert_text((72, 100), "Page 1 — PrivaSee E2E multi-page test")
    p1.insert_text((72, 130), "Name: Alice Smith")
    p2 = doc.new_page(width=595, height=842)
    p2.insert_text((72, 100), "Page 2 — continuation")
    p2.insert_text((72, 130), "Email: alice@example.com")
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def make_png() -> bytes:
    """Single-page PNG rasterised from a fitz document."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 100), "PrivaSee E2E PNG test")
    page.insert_text((72, 130), "Name: Bob Jones")
    pix = page.get_pixmap(dpi=150)
    png_bytes = pix.tobytes("png")
    doc.close()
    return png_bytes


def make_scanned_pdf() -> bytes:
    """
    Scanned PDF: text rasterised to a pixel image then embedded in the PDF.
    There is no text layer — OCR is required to extract text.

    TODO: implement scanned PDF E2E test case.
    """
    raise NotImplementedError("Scanned PDF test case not yet implemented")


# ---------------------------------------------------------------------------
# Step: POST /api/upload
# ---------------------------------------------------------------------------

def step_upload(file_bytes: bytes, filename: str) -> str:
    print("  [upload] POST /api/upload ...")
    ext = os.path.splitext(filename)[1].lower()
    mime = "image/png" if ext == ".png" else "application/pdf"

    resp = requests.post(
        f"{API_BASE_URL}/api/upload",
        files={"file": (filename, io.BytesIO(file_bytes), mime)},
        timeout=30,
    )
    _check(resp.status_code == 200, f"HTTP {resp.status_code} — body: {resp.text[:200]}")
    body = resp.json()
    session_id = body.get("session_id", "")
    _check(bool(session_id), f"session_id present: {session_id!r}")
    _check(body.get("filename") == filename, f"filename echoed: {body.get('filename')!r}")
    _check(body.get("file_size") == len(file_bytes), f"file_size matches: {body.get('file_size')}")

    # Artefacts in UC
    stored_name = f"original{ext}"
    exists, code = _uc_exists(session_id, stored_name)
    _check(exists, f"{stored_name} in UC (HTTP {code})")
    exists, code = _uc_exists(session_id, "metadata.json")
    _check(exists, f"metadata.json in UC (HTTP {code})")

    # metadata.json content
    meta = _uc_read_json(session_id, "metadata.json")
    _check(meta.get("session_id") == session_id, "metadata.session_id matches")
    _check(meta.get("original_filename") == filename, "metadata.original_filename matches")
    _check(meta.get("status") == "uploaded", "metadata.status='uploaded' after upload")

    print(f"  INFO session_id={session_id}")
    return session_id


# ---------------------------------------------------------------------------
# Step: GET /api/sessions/{session_id} — status assertion (E6)
# ---------------------------------------------------------------------------

def step_check_status(session_id: str, expected: str) -> None:
    resp = requests.get(f"{API_BASE_URL}/api/sessions/{session_id}", timeout=10)
    _check(resp.status_code == 200, f"GET /api/sessions → HTTP {resp.status_code}")
    body = resp.json()
    _check(
        body.get("status") == expected,
        f"session status='{body.get('status')}' expected='{expected}'",
    )


# ---------------------------------------------------------------------------
# Step: GET /api/files — file-serving endpoints
# ---------------------------------------------------------------------------

def step_check_file_serving(session_id: str, original_ext: str = ".pdf") -> None:
    """Verify original and masked files are served with the correct content-type."""
    # Original file
    url = f"{API_BASE_URL}/api/files/uploads/{session_id}{original_ext}"
    resp = requests.get(url, timeout=30)
    _check(resp.status_code == 200, f"GET /api/files/uploads → HTTP {resp.status_code}")
    expected_ct = "image/png" if original_ext == ".png" else "application/pdf"
    _check(expected_ct in resp.headers.get("Content-Type", ""), f"original Content-Type is {expected_ct}")
    _check(len(resp.content) > 0, "original file response has content")

    # Masked PDF — always PDF regardless of input type
    masked_url = f"{API_BASE_URL}/api/files/output/{session_id}_masked.pdf"
    resp = requests.get(masked_url, timeout=30)
    _check(resp.status_code == 200, f"GET /api/files/output → HTTP {resp.status_code}")
    _check("application/pdf" in resp.headers.get("Content-Type", ""), "masked Content-Type is application/pdf")
    _check(len(resp.content) > 0, "masked file response has content")


# ---------------------------------------------------------------------------
# Step: POST /api/process
# ---------------------------------------------------------------------------

def step_process(session_id: str) -> list[dict]:
    print("  [process] POST /api/process ...")
    payload = {"session_id": session_id, "field_definitions": FIELD_DEFINITIONS}
    resp = requests.post(f"{API_BASE_URL}/api/process", json=payload, timeout=60)

    if resp.status_code == 503:
        body = resp.json()
        if "DATABRICKS_MODEL_ENDPOINT" in body.get("error", ""):
            print("  SKIP process — server has MOCK_DATABRICKS=false and no model endpoint configured")
            print("         Restart the server with MOCK_DATABRICKS=true to test this step.")
            return []

    _check(resp.status_code == 200, f"HTTP {resp.status_code} — body: {resp.text[:300]}")
    body = resp.json()
    entities = body.get("entities", [])
    _check(isinstance(entities, list), "entities is a list")
    _check(len(entities) > 0, f"at least one entity returned ({len(entities)} total)")

    # Validate structure of every entity
    for i, entity in enumerate(entities):
        _check_entity_structure(entity, i)

    # entities.json should now exist in UC
    exists, code = _uc_exists(session_id, "entities.json")
    _check(exists, f"entities.json in UC (HTTP {code})")

    print(f"  INFO {len(entities)} entities found")
    return entities


# ---------------------------------------------------------------------------
# Step: POST /api/approve-and-mask
# ---------------------------------------------------------------------------

def step_approve_and_mask(
    session_id: str,
    entities: list[dict],
    approved_ids: list[str] | None = None,
) -> None:
    """
    Call approve-and-mask.

    If approved_ids is None, all entities are approved.
    Otherwise only the supplied IDs are approved (E2: partial approval).
    """
    print("  [mask] POST /api/approve-and-mask ...")

    if not entities:
        print("  SKIP approve-and-mask — no entities from process step")
        return

    if approved_ids is None:
        approved_ids = [e["id"] for e in entities]

    payload = {
        "session_id": session_id,
        "approved_entity_ids": approved_ids,
        "updated_entities": entities,
    }
    resp = requests.post(f"{API_BASE_URL}/api/approve-and-mask", json=payload, timeout=60)
    _check(resp.status_code == 200, f"HTTP {resp.status_code} — body: {resp.text[:300]}")
    body = resp.json()
    _check(
        body.get("entities_masked") == len(approved_ids),
        f"entities_masked={body.get('entities_masked')} expected={len(approved_ids)}",
    )
    _check("masked_pdf_url" in body, "masked_pdf_url present in response")

    exists, code = _uc_exists(session_id, "masked.pdf")
    _check(exists, f"masked.pdf in UC (HTTP {code})")

    print(f"  INFO {body.get('entities_masked')} entities masked")


# ---------------------------------------------------------------------------
# Step: cleanup
# ---------------------------------------------------------------------------

def step_cleanup(session_id: str, ext: str = ".pdf") -> None:
    print("  [cleanup] Removing session artefacts from UC ...")
    for fname in (f"original{ext}", "metadata.json", "entities.json", "masked.pdf"):
        url = _db_url(_uc_path(session_id, fname))
        resp = requests.delete(url, headers=_db_headers(), timeout=30)
        # 200/204 = deleted, 404 = never existed (fine if an earlier step was skipped)
        _check(resp.status_code in (200, 204, 404), f"DELETE {fname} → HTTP {resp.status_code}")


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def run_test_case(label: str, file_bytes: bytes, filename: str) -> bool:
    """E1 / E4 / E5 — full upload → process → approve all → file-serve → cleanup."""
    print(f"\n{'─' * 60}")
    print(f"Test: {label}")
    print(f"{'─' * 60}")
    session_id = None
    ext = os.path.splitext(filename)[1].lower()
    try:
        _check(len(file_bytes) > 0, f"document built ({len(file_bytes):,} bytes)")
        session_id = step_upload(file_bytes, filename)
        step_check_status(session_id, "uploaded")                    # E6: after upload

        entities = step_process(session_id)
        if entities:
            step_check_status(session_id, "awaiting_review")         # E6: after process

        step_approve_and_mask(session_id, entities)
        if entities:
            step_check_status(session_id, "completed")               # E6: after mask
            step_check_file_serving(session_id, ext)

        step_cleanup(session_id, ext)
        print("  ✅ PASSED")
        return True
    except AssertionError:
        if session_id:
            print(f"  INFO: artefacts left in UC for inspection")
            print(f"        {UC_VOLUME_PATH}/{session_id}/")
        print("  ❌ FAILED")
        return False


def run_partial_approval_test() -> bool:
    """E2: Approve only the first entity; verify entities_masked equals 1."""
    print(f"\n{'─' * 60}")
    print("Test: Partial Approval (approve first entity only)")
    print(f"{'─' * 60}")
    session_id = None
    try:
        session_id = step_upload(make_digital_pdf(), "e2e_partial.pdf")
        entities = step_process(session_id)
        if not entities:
            print("  SKIP — no entities (process step skipped)")
            step_cleanup(session_id)
            return True
        _check(len(entities) >= 2, f"need >=2 entities for partial test (got {len(entities)})")
        step_approve_and_mask(session_id, entities, approved_ids=[entities[0]["id"]])
        step_cleanup(session_id)
        print("  ✅ PASSED")
        return True
    except AssertionError:
        if session_id:
            print(f"  INFO: {UC_VOLUME_PATH}/{session_id}/")
        print("  ❌ FAILED")
        return False


def run_reprocess_test() -> bool:
    """E3: Process the same session twice; verify entities.json is overwritten not appended."""
    print(f"\n{'─' * 60}")
    print("Test: Reprocess Same Session")
    print(f"{'─' * 60}")
    session_id = None
    try:
        session_id = step_upload(make_digital_pdf(), "e2e_reprocess.pdf")
        entities_first = step_process(session_id)
        if not entities_first:
            print("  SKIP — process step unavailable")
            step_cleanup(session_id)
            return True

        entities_second = step_process(session_id)
        _check(
            len(entities_second) == len(entities_first),
            f"second process returns same count ({len(entities_first)})",
        )

        stored = _uc_read_json(session_id, "entities.json")
        _check(
            len(stored.get("entities", [])) == len(entities_second),
            "entities.json count matches second process result (overwritten, not appended)",
        )

        step_cleanup(session_id)
        print("  ✅ PASSED")
        return True
    except AssertionError:
        if session_id:
            print(f"  INFO: {UC_VOLUME_PATH}/{session_id}/")
        print("  ❌ FAILED")
        return False


def run_concurrent_sessions_test() -> bool:
    """E7: Two sessions uploaded in parallel; verify artefacts are isolated."""
    print(f"\n{'─' * 60}")
    print("Test: Concurrent Sessions (isolation)")
    print(f"{'─' * 60}")
    pdf_bytes = make_digital_pdf()
    session_ids: dict[str, str] = {}
    errors: dict[str, str] = {}

    def worker(label: str) -> None:
        try:
            sid = step_upload(pdf_bytes, f"e2e_concurrent_{label}.pdf")
            session_ids[label] = sid
        except AssertionError as exc:
            errors[label] = str(exc)

    threads = [threading.Thread(target=worker, args=(str(i),)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    try:
        for label, err in errors.items():
            _check(False, f"worker {label} failed: {err}")

        _check(len(session_ids) == 2, "both workers produced session IDs")
        sid_a, sid_b = session_ids["0"], session_ids["1"]
        _check(sid_a != sid_b, f"distinct session IDs: {sid_a[:8]}… vs {sid_b[:8]}…")

        # Each session's artefacts reference their own session_id
        for label, sid in session_ids.items():
            exists, code = _uc_exists(sid, "metadata.json")
            _check(exists, f"session {label} metadata.json exists (HTTP {code})")
            meta = _uc_read_json(sid, "metadata.json")
            _check(meta.get("session_id") == sid, f"session {label} metadata.session_id is correct")

        for sid in session_ids.values():
            step_cleanup(sid)

        print("  ✅ PASSED")
        return True
    except AssertionError:
        for sid in session_ids.values():
            print(f"  INFO: {UC_VOLUME_PATH}/{sid}/")
        print("  ❌ FAILED")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def preflight() -> None:
    missing = [v for v in ("DATABRICKS_HOST", "DATABRICKS_TOKEN", "UC_VOLUME_PATH") if not os.getenv(v)]
    if missing:
        sys.exit(f"Missing env vars: {', '.join(missing)}\nCopy .env.template → .env and fill in values.")
    print(f"API_BASE_URL  : {API_BASE_URL}")
    print(f"UC_VOLUME_PATH: {UC_VOLUME_PATH}")

    try:
        health = requests.get(f"{API_BASE_URL}/api/health", timeout=5).json()
        mock = health.get("mock_databricks", False)
        print(f"MOCK_DATABRICKS: {mock} (server-side)")
        if not mock and not health.get("databricks_endpoint_configured"):
            print("  WARN: MOCK_DATABRICKS=false and no model endpoint — process step will be skipped")
    except requests.ConnectionError:
        sys.exit(f"Cannot connect to {API_BASE_URL} — is the server running?")


def main() -> None:
    preflight()

    results: list[tuple[str, bool | None]] = []

    try:
        results.append(("Digital PDF", run_test_case(
            label="Digital PDF (text layer — no OCR needed)",
            file_bytes=make_digital_pdf(),
            filename="e2e_digital.pdf",
        )))

        results.append(("Multi-page PDF", run_test_case(
            label="Multi-page PDF (2 pages)",
            file_bytes=make_multipage_pdf(),
            filename="e2e_multipage.pdf",
        )))

        results.append(("PNG upload", run_test_case(
            label="PNG upload (image → masked PDF output)",
            file_bytes=make_png(),
            filename="e2e_image.png",
        )))

        results.append(("Partial approval", run_partial_approval_test()))
        results.append(("Reprocess session", run_reprocess_test()))
        results.append(("Concurrent sessions", run_concurrent_sessions_test()))

        # Scanned PDF — placeholder
        print(f"\n{'─' * 60}")
        print("Test: Scanned PDF (image-only — OCR required)")
        print(f"{'─' * 60}")
        print("  SKIP — not yet implemented (see make_scanned_pdf())")
        results.append(("Scanned PDF", None))

    except requests.ConnectionError:
        sys.exit(f"\nLost connection to {API_BASE_URL} mid-run.")

    # Summary
    print(f"\n{'═' * 60}")
    print("Summary")
    print(f"{'═' * 60}")
    all_passed = True
    for name, passed in results:
        if passed is None:
            print(f"  ⏭  SKIPPED  {name}")
        elif passed:
            print(f"  ✅ PASSED   {name}")
        else:
            print(f"  ❌ FAILED   {name}")
            all_passed = False

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
