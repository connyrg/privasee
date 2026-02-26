"""
E2E Test — PrivaSee full document workflow

Covers the three main API endpoints in sequence:
    1. POST /api/upload          — store document in UC volume
    2. POST /api/process         — extract PII entities (requires MOCK_DATABRICKS=true
                                   on the server, or a live Databricks model endpoint)
    3. POST /api/approve-and-mask — apply redactions and store masked PDF in UC volume

Two test cases:
    • Digital PDF  — text embedded as PDF text objects (machine-readable)
    • Scanned PDF  — image-only page (placeholder; not yet implemented)

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


# ---------------------------------------------------------------------------
# PDF factories
# ---------------------------------------------------------------------------

def make_digital_pdf() -> bytes:
    """
    Digital PDF: text is embedded as real PDF text objects.
    Text is directly selectable/extractable — no OCR required.
    """
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4
    page.insert_text((72, 100), "PrivaSee E2E upload test — digital document")
    page.insert_text((72, 130), "Name: Jane Doe")
    page.insert_text((72, 160), "Email: jane@example.com")
    page.insert_text((72, 190), "Phone: +44 7700 900123")
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


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

def step_upload(pdf_bytes: bytes, filename: str) -> str:
    print("  [upload] POST /api/upload ...")
    resp = requests.post(
        f"{API_BASE_URL}/api/upload",
        files={"file": (filename, io.BytesIO(pdf_bytes), "application/pdf")},
        timeout=30,
    )
    _check(resp.status_code == 200, f"HTTP {resp.status_code} — body: {resp.text[:200]}")
    body = resp.json()
    session_id = body.get("session_id", "")
    _check(bool(session_id), f"session_id present: {session_id}")
    _check(body.get("filename") == filename, f"filename echoed: {body.get('filename')}")
    _check(body.get("file_size") == len(pdf_bytes), f"file_size matches: {body.get('file_size')}")

    exists, code = _uc_exists(session_id, "original.pdf")
    _check(exists, f"original.pdf in UC (HTTP {code})")

    exists, code = _uc_exists(session_id, "metadata.json")
    _check(exists, f"metadata.json in UC (HTTP {code})")

    print(f"  INFO session_id={session_id}")
    return session_id


# ---------------------------------------------------------------------------
# Step: POST /api/process
# ---------------------------------------------------------------------------

def step_process(session_id: str) -> list[dict]:
    print("  [process] POST /api/process ...")
    payload = {
        "session_id": session_id,
        "field_definitions": FIELD_DEFINITIONS,
    }
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

    # entities.json should now exist in UC
    exists, code = _uc_exists(session_id, "entities.json")
    _check(exists, f"entities.json in UC (HTTP {code})")

    print(f"  INFO {len(entities)} entities found")
    return entities


# ---------------------------------------------------------------------------
# Step: POST /api/approve-and-mask
# ---------------------------------------------------------------------------

def step_approve_and_mask(session_id: str, entities: list[dict]) -> None:
    print("  [mask] POST /api/approve-and-mask ...")

    if not entities:
        print("  SKIP approve-and-mask — no entities from process step")
        return

    approved_ids = [e["id"] for e in entities]
    payload = {
        "session_id": session_id,
        "approved_entity_ids": approved_ids,
        # Pass entities as updated_entities so the endpoint has a fallback
        # if get_entities is not available on the session manager.
        "updated_entities": entities,
    }
    resp = requests.post(f"{API_BASE_URL}/api/approve-and-mask", json=payload, timeout=60)
    _check(resp.status_code == 200, f"HTTP {resp.status_code} — body: {resp.text[:300]}")
    body = resp.json()
    _check(body.get("entities_masked", 0) > 0, f"entities_masked={body.get('entities_masked')}")
    _check("masked_pdf_url" in body, "masked_pdf_url present in response")

    # masked.pdf should now exist in UC
    exists, code = _uc_exists(session_id, "masked.pdf")
    _check(exists, f"masked.pdf in UC (HTTP {code})")

    print(f"  INFO {body.get('entities_masked')} entities masked")


# ---------------------------------------------------------------------------
# Step: cleanup
# ---------------------------------------------------------------------------

def step_cleanup(session_id: str) -> None:
    print("  [cleanup] Removing session artefacts from UC ...")
    for fname in ("original.pdf", "metadata.json", "entities.json", "masked.pdf"):
        url = _db_url(_uc_path(session_id, fname))
        resp = requests.delete(url, headers=_db_headers(), timeout=30)
        # 200/204 = deleted, 404 = never existed (fine if an earlier step was skipped)
        _check(resp.status_code in (200, 204, 404), f"DELETE {fname} → HTTP {resp.status_code}")


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def run_test_case(label: str, pdf_bytes: bytes, filename: str) -> bool:
    """Run a full upload → process → approve-and-mask cycle. Returns True on pass."""
    print(f"\n{'─' * 60}")
    print(f"Test: {label}")
    print(f"{'─' * 60}")
    session_id = None
    try:
        _check(len(pdf_bytes) > 0, f"PDF built ({len(pdf_bytes):,} bytes)")
        session_id = step_upload(pdf_bytes, filename)
        entities = step_process(session_id)
        step_approve_and_mask(session_id, entities)
        step_cleanup(session_id)
        print("  ✅ PASSED")
        return True
    except AssertionError:
        if session_id:
            print(f"  INFO: artefacts left in UC for inspection")
            print(f"        {UC_VOLUME_PATH}/{session_id}/")
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

    # Check server is reachable and report MOCK_DATABRICKS mode
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

    results: list[tuple[str, bool]] = []

    try:
        results.append(("Digital PDF", run_test_case(
            label="Digital PDF (text layer — no OCR needed)",
            pdf_bytes=make_digital_pdf(),
            filename="e2e_digital.pdf",
        )))

        # Scanned PDF test case — placeholder
        print(f"\n{'─' * 60}")
        print("Test: Scanned PDF (image-only — OCR required)")
        print(f"{'─' * 60}")
        print("  SKIP — not yet implemented (see make_scanned_pdf() in this script)")
        results.append(("Scanned PDF", None))  # None = skipped

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
