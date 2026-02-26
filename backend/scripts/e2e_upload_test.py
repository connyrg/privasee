"""
E2E Upload Test — PrivaSee

Verifies that POST /api/upload stores the document in the Unity Catalog volume.

Usage:
    cd backend
    python scripts/e2e_upload_test.py

Prerequisites:
    - .env file populated with DATABRICKS_HOST, DATABRICKS_TOKEN, UC_VOLUME_PATH
    - FastAPI server running (default: http://localhost:8000)
      cd backend && uvicorn app.main:app --reload
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

DUMMY_FILENAME = "e2e_test.pdf"


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


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def step_preflight() -> None:
    print("[0/5] Preflight checks ...")
    missing = [v for v in ("DATABRICKS_HOST", "DATABRICKS_TOKEN", "UC_VOLUME_PATH") if not os.getenv(v)]
    if missing:
        sys.exit(f"  FAIL: missing env vars: {', '.join(missing)}\n  Copy .env.template → .env and fill in values.")
    print(f"  OK   API_BASE_URL={API_BASE_URL}")
    print(f"  OK   UC_VOLUME_PATH={UC_VOLUME_PATH}")


def step_create_pdf() -> bytes:
    print("[1/5] Creating dummy PDF in memory ...")
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4
    page.insert_text((72, 100), "PrivaSee E2E upload test")
    page.insert_text((72, 130), "Name: Jane Doe")
    page.insert_text((72, 160), "Email: jane@example.com")
    pdf_bytes = doc.tobytes()
    doc.close()
    _check(len(pdf_bytes) > 0, f"PDF created ({len(pdf_bytes)} bytes)")
    return pdf_bytes


def step_upload(pdf_bytes: bytes) -> str:
    print("[2/5] POST /api/upload ...")
    resp = requests.post(
        f"{API_BASE_URL}/api/upload",
        files={"file": (DUMMY_FILENAME, io.BytesIO(pdf_bytes), "application/pdf")},
        timeout=30,
    )
    _check(resp.status_code == 200, f"HTTP {resp.status_code} (expected 200) — body: {resp.text[:200]}")
    body = resp.json()
    session_id = body.get("session_id", "")
    _check(bool(session_id), f"session_id present: {session_id}")
    _check(body.get("filename") == DUMMY_FILENAME, f"filename={body.get('filename')}")
    _check(body.get("file_size") == len(pdf_bytes), f"file_size={body.get('file_size')}")
    print(f"  INFO session_id={session_id}")
    return session_id


def step_verify_file(session_id: str, pdf_bytes: bytes) -> None:
    print("[3/5] Verifying original.pdf in UC volume ...")
    url = _db_url(_uc_path(session_id, "original.pdf"))
    resp = requests.get(url, headers=_db_headers(), timeout=30)
    _check(resp.status_code == 200, f"GET original.pdf → HTTP {resp.status_code}")
    _check(resp.content == pdf_bytes, f"File bytes match ({len(resp.content)} bytes)")


def step_verify_metadata(session_id: str) -> None:
    print("[4/5] Verifying metadata.json in UC volume ...")
    url = _db_url(_uc_path(session_id, "metadata.json"))
    resp = requests.get(url, headers=_db_headers(), timeout=30)
    _check(resp.status_code == 200, f"GET metadata.json → HTTP {resp.status_code}")
    try:
        meta = resp.json()
    except json.JSONDecodeError:
        _check(False, f"metadata.json is not valid JSON: {resp.text[:200]}")
        return
    _check(meta.get("status") == "uploaded", f"status={meta.get('status')}")
    _check("session_id" in meta, f"session_id field present")


def step_cleanup(session_id: str) -> None:
    print("[5/5] Cleaning up session artefacts from UC ...")
    for fname in ("original.pdf", "metadata.json"):
        url = _db_url(_uc_path(session_id, fname))
        resp = requests.delete(url, headers=_db_headers(), timeout=30)
        # 200 or 404 are both acceptable (file may not exist if an earlier step failed)
        _check(resp.status_code in (200, 204, 404), f"DELETE {fname} → HTTP {resp.status_code}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    session_id = None
    try:
        step_preflight()
        pdf_bytes = step_create_pdf()
        session_id = step_upload(pdf_bytes)
        step_verify_file(session_id, pdf_bytes)
        step_verify_metadata(session_id)
        step_cleanup(session_id)
        print("\n✅ E2E upload test PASSED")
    except AssertionError:
        if session_id:
            print(f"\n  INFO: artefacts left in UC for inspection — session_id={session_id}")
            print(f"  Path: {UC_VOLUME_PATH}/{session_id}/")
        print("\n❌ E2E upload test FAILED")
        sys.exit(1)
    except requests.ConnectionError:
        print(f"\n  FAIL: could not connect to {API_BASE_URL} — is the server running?")
        print("\n❌ E2E upload test FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
