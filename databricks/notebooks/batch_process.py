# Databricks notebook source
# MAGIC %md
# MAGIC # PrivaSee — Batch Document Processing
# MAGIC
# MAGIC Processes all documents in an input UC volume folder through the full
# MAGIC PrivaSee pipeline (OCR → entity extraction → masking) and writes masked
# MAGIC outputs to an output UC volume folder.
# MAGIC
# MAGIC ## Widgets
# MAGIC | Widget | Description |
# MAGIC |--------|-------------|
# MAGIC | `input_folder` | UC volume path containing documents to process |
# MAGIC | `output_folder` | UC volume path where masked files will be written |
# MAGIC | `config_path` | UC volume path to a saved PrivaSee config JSON file |
# MAGIC | `di_endpoint` | Document Intelligence model serving endpoint URL |
# MAGIC | `masking_endpoint` | Masking model serving endpoint URL |
# MAGIC
# MAGIC ## Notes
# MAGIC - Supported file types: PDF, PNG, JPG, JPEG
# MAGIC - Each file creates a session under `sessions_volume_path` (kept for audit)
# MAGIC - Masked output is written as `{original_stem}_masked.pdf` in `output_folder`
# MAGIC - If a file has no entities, it is recorded as `no_entities` and no output file is written
# MAGIC - Run as a Databricks Job for scheduled/unattended processing

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 — Widgets

# COMMAND ----------

dbutils.widgets.text(
    "input_folder",
    "/Volumes/datascience_dev_bronze_sandbox/privasee/batch_input",
    "Input folder (UC path)",
)
dbutils.widgets.text(
    "output_folder",
    "/Volumes/datascience_dev_bronze_sandbox/privasee/batch_output",
    "Output folder (UC path)",
)
dbutils.widgets.text(
    "config_path",
    "/Volumes/datascience_dev_bronze_sandbox/privasee/configs/default.json",
    "Config file (UC path to saved config JSON)",
)
dbutils.widgets.text(
    "di_endpoint",
    "",
    "Document Intelligence endpoint URL",
)
dbutils.widgets.text(
    "masking_endpoint",
    "",
    "Masking endpoint URL",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 — Configuration

# COMMAND ----------

import json
import os
import time
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("privasee.batch")

# --- Read widgets ---
INPUT_FOLDER        = dbutils.widgets.get("input_folder").rstrip("/")
OUTPUT_FOLDER       = dbutils.widgets.get("output_folder").rstrip("/")
CONFIG_PATH         = dbutils.widgets.get("config_path")
DI_ENDPOINT         = dbutils.widgets.get("di_endpoint").rstrip("/")
MASKING_ENDPOINT    = dbutils.widgets.get("masking_endpoint").rstrip("/")
# SESSIONS_VOLUME     = dbutils.widgets.get("sessions_volume_path").rstrip("/")
SESSIONS_VOLUME     = "/Volumes/datascience_dev_bronze_sandbox/ds_document_deidentification/sessions"

# Databricks token — available automatically in all notebook/job contexts
DATABRICKS_HOST  = spark.conf.get("spark.databricks.workspaceUrl")
if not DATABRICKS_HOST.startswith("https://"):
    DATABRICKS_HOST = f"https://{DATABRICKS_HOST}"
DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}

print("✅ Configuration loaded")
print(f"   Input folder  : {INPUT_FOLDER}")
print(f"   Output folder : {OUTPUT_FOLDER}")
print(f"   Config path   : {CONFIG_PATH}")
print(f"   Sessions path : {SESSIONS_VOLUME}")
print(f"   DI endpoint   : {DI_ENDPOINT or '(not set)'}")
print(f"   Masking ep    : {MASKING_ENDPOINT or '(not set)'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 — Load Field Definitions from Config

# COMMAND ----------

def load_field_definitions(config_path: str) -> List[Dict[str, Any]]:
    """
    Read a PrivaSee config JSON from a UC volume path and return its
    field_definitions list.

    The config file format matches what ConfigManager.save_config() writes:
        {
            "config_name": "...",
            "key": "...",
            "saved_at": "...",
            "field_definitions": [
                {"name": "Full Name", "description": "...", "strategy": "Fake Data"},
                ...
            ]
        }
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            "Save a config from the PrivaSee UI first, then copy its UC path here."
        )

    field_defs = config.get("field_definitions", [])
    if not field_defs:
        raise ValueError(f"Config at {config_path} contains no field_definitions.")

    print(f"✅ Loaded config '{config.get('config_name', '(unnamed)')}' "
          f"with {len(field_defs)} field definition(s):")
    for fd in field_defs:
        print(f"   • {fd['name']} [{fd.get('strategy', 'Fake Data')}]: {fd.get('description', '')}")

    return field_defs


FIELD_DEFINITIONS = load_field_definitions(CONFIG_PATH)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 — Validate Inputs and Create Output Folder

# COMMAND ----------

# Validate endpoints
if not DI_ENDPOINT:
    raise ValueError("di_endpoint widget is empty — set the Document Intelligence endpoint URL.")
if not MASKING_ENDPOINT:
    raise ValueError("masking_endpoint widget is empty — set the Masking endpoint URL.")

# Validate input folder
try:
    input_entries = dbutils.fs.ls(INPUT_FOLDER)
except Exception as exc:
    raise FileNotFoundError(f"Input folder not accessible: {INPUT_FOLDER}\n{exc}")

# Collect supported files
input_files = [
    e.name for e in input_entries
    if not e.isDir() and Path(e.name).suffix.lower() in SUPPORTED_EXTENSIONS
]
if not input_files:
    dbutils.notebook.exit(f"No supported files found in {INPUT_FOLDER}. Supported: {SUPPORTED_EXTENSIONS}")

print(f"✅ Found {len(input_files)} file(s) to process:")
for f in input_files:
    print(f"   • {f}")

# Create output folder if it doesn't exist
try:
    dbutils.fs.ls(OUTPUT_FOLDER)
except Exception:
    dbutils.fs.mkdirs(OUTPUT_FOLDER)
    print(f"✅ Created output folder: {OUTPUT_FOLDER}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 — Helper Functions

# COMMAND ----------

def _files_api_headers() -> Dict[str, str]:
    return {"Authorization": f"Bearer {DATABRICKS_TOKEN}"}


def _files_api_url(uc_path: str) -> str:
    return f"{DATABRICKS_HOST}/api/2.0/fs/files{uc_path}"


def create_session(filename: str) -> str:
    """Create a new session in the UC sessions volume and return the session_id."""
    session_id = str(uuid.uuid4())
    metadata = {
        "session_id": session_id,
        "original_filename": filename,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "uploaded",
        "source": "batch_notebook",
    }
    url = _files_api_url(f"{SESSIONS_VOLUME}/{session_id}/metadata.json")
    resp = requests.request(
        "put", url,
        headers=_files_api_headers(),
        data=json.dumps(metadata),
    )
    resp.raise_for_status()
    return session_id


def save_file_to_session(session_id: str, filename: str, file_bytes: bytes) -> None:
    """Upload file bytes to the UC session directory."""
    ext = Path(filename).suffix.lower()
    stored_name = f"original{ext}"
    url = _files_api_url(f"{SESSIONS_VOLUME}/{session_id}/{stored_name}")
    resp = requests.request(
        "put", url,
        headers=_files_api_headers(),
        data=file_bytes,
    )
    resp.raise_for_status()


def call_di_endpoint(session_id: str, field_definitions: List[Dict]) -> Dict[str, Any]:
    """
    Call the Document Intelligence model serving endpoint synchronously.

    The model reads the original file from UC directly (by session_id),
    runs OCR + entity extraction, writes entities.json to UC, and returns
    the result in the response body — no polling required.
    """
    payload = {
        "dataframe_records": [
            {
                "session_id": session_id,
                "field_definitions": field_definitions,
            }
        ]
    }
    headers = {
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
        "Content-Type": "application/json",
    }
    resp = requests.post(
        DI_ENDPOINT,
        headers=headers,
        json=payload,
        timeout=600,  # allow up to 10 min for large/scanned documents
    )
    resp.raise_for_status()
    predictions = resp.json().get("predictions", [])
    if not predictions:
        raise ValueError("Document Intelligence endpoint returned empty predictions.")
    return predictions[0]


def call_masking_endpoint(session_id: str, entities: List[Dict]) -> None:
    """
    Call the Masking model serving endpoint synchronously.

    The model reads the original file from UC, applies redactions, and
    writes masked.pdf back to the UC session directory.
    """
    payload = {
        "dataframe_records": [
            {
                "session_id": session_id,
                "entities_to_mask": json.dumps(entities),
            }
        ]
    }
    headers = {
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
        "Content-Type": "application/json",
    }
    resp = requests.post(
        MASKING_ENDPOINT,
        headers=headers,
        json=payload,
        timeout=600,
    )
    resp.raise_for_status()


def copy_masked_to_output(session_id: str, original_filename: str, output_folder: str) -> str:
    """
    Read masked.pdf from the UC session directory and write it to output_folder.
    Returns the output file path.
    """
    stem = Path(original_filename).stem
    output_filename = f"{stem}_masked.pdf"
    output_path = f"{output_folder}/{output_filename}"

    # Read masked.pdf from UC session via Files API
    read_url = _files_api_url(f"{SESSIONS_VOLUME}/{session_id}/masked.pdf")
    resp = requests.request("get", read_url, headers=_files_api_headers(), timeout=60)
    resp.raise_for_status()
    masked_bytes = resp.content

    # Write to output folder via Files API
    write_url = _files_api_url(output_path)
    resp = requests.request(
        "put", write_url,
        headers=_files_api_headers(),
        data=masked_bytes,
    )
    resp.raise_for_status()
    return output_path


def update_session_status(session_id: str, status: str) -> None:
    """Update the status field in metadata.json (best-effort, non-fatal)."""
    try:
        read_url = _files_api_url(f"{SESSIONS_VOLUME}/{session_id}/metadata.json")
        resp = requests.request("get", read_url, headers=_files_api_headers(), timeout=10)
        resp.raise_for_status()
        metadata = resp.json()
        metadata["status"] = status
        metadata["updated_at"] = datetime.now(timezone.utc).isoformat()
        write_url = _files_api_url(f"{SESSIONS_VOLUME}/{session_id}/metadata.json")
        requests.request("put", write_url, headers=_files_api_headers(),
                         data=json.dumps(metadata), timeout=10)
    except Exception as exc:
        logger.warning("Could not update session %s status to %s: %s", session_id, status, exc)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6 — Process Files

# COMMAND ----------

results = []

for i, filename in enumerate(input_files, start=1):
    file_path = f"{INPUT_FOLDER}/{filename}"
    print(f"\n{'='*70}")
    print(f"[{i}/{len(input_files)}] Processing: {filename}")
    print(f"{'='*70}")
    t_file_start = time.time()

    result = {
        "filename": filename,
        "session_id": None,
        "entities_found": 0,
        "entities_masked": 0,
        "output_path": None,
        "status": None,
        "error": None,
        "elapsed_s": None,
    }

    try:
        # --- Read file from input folder ---
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        print(f"  ✅ Read {len(file_bytes):,} bytes")

        # --- Create session ---
        session_id = create_session(filename)
        result["session_id"] = session_id
        print(f"  ✅ Session created: {session_id}")

        # --- Upload file to UC session ---
        save_file_to_session(session_id, filename, file_bytes)
        print(f"  ✅ File saved to UC session")

        # --- Document Intelligence ---
        print(f"  ⏳ Calling Document Intelligence endpoint...")
        t_di = time.time()
        di_result = call_di_endpoint(session_id, FIELD_DEFINITIONS)
        print(f"  ✅ DI completed in {time.time() - t_di:.1f}s")

        if di_result.get("status") == "error":
            raise RuntimeError(f"DI endpoint error: {di_result.get('error_message')}")

        entities = di_result.get("entities", [])
        result["entities_found"] = len(entities)
        print(f"  ✅ {len(entities)} entit{'y' if len(entities) == 1 else 'ies'} found")

        if not entities:
            result["status"] = "no_entities"
            print(f"  ℹ️  No entities found — skipping masking")
            update_session_status(session_id, "completed")
        else:
            # --- Masking ---
            print(f"  ⏳ Calling Masking endpoint...")
            t_mask = time.time()
            call_masking_endpoint(session_id, entities)
            print(f"  ✅ Masking completed in {time.time() - t_mask:.1f}s")
            result["entities_masked"] = len(entities)

            # --- Copy output ---
            output_path = copy_masked_to_output(session_id, filename, OUTPUT_FOLDER)
            result["output_path"] = output_path
            result["status"] = "completed"
            print(f"  ✅ Masked file written: {output_path}")
            update_session_status(session_id, "completed")

    except Exception as exc:
        logger.error("Failed to process %s: %s", filename, exc, exc_info=True)
        result["status"] = "error"
        result["error"] = str(exc)
        print(f"  ❌ Error: {exc}")
        if result["session_id"]:
            update_session_status(result["session_id"], "error")

    result["elapsed_s"] = round(time.time() - t_file_start, 1)
    results.append(result)
    print(f"  ⏱  Elapsed: {result['elapsed_s']}s")

print(f"\n{'='*70}")
print(f"Batch complete — {len(results)} file(s) processed")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7 — Results Summary

# COMMAND ----------

completed = [r for r in results if r["status"] == "completed"]
no_entities = [r for r in results if r["status"] == "no_entities"]
errors = [r for r in results if r["status"] == "error"]

print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                    PrivaSee Batch Results                            ║
╠══════════════════════════════════════════════════════════════════════╣
║  Total files processed : {len(results):<45}║
║  Masked successfully   : {len(completed):<45}║
║  No entities found     : {len(no_entities):<45}║
║  Errors                : {len(errors):<45}║
║                                                                      ║
║  Output folder: {OUTPUT_FOLDER:<53}║
╚══════════════════════════════════════════════════════════════════════╝
""")

if errors:
    print("❌ Errors:")
    for r in errors:
        print(f"   • {r['filename']}: {r['error']}")

# Display full results as a Spark DataFrame table
results_df = spark.createDataFrame([
    {
        "filename": r["filename"],
        "status": r["status"] or "",
        "entities_found": r["entities_found"],
        "entities_masked": r["entities_masked"],
        "elapsed_s": r["elapsed_s"],
        "session_id": r["session_id"] or "",
        "output_path": r["output_path"] or "",
        "error": r["error"] or "",
    }
    for r in results
])

display(results_df)

# COMMAND ----------

# Exit with summary for job monitoring
if errors:
    dbutils.notebook.exit(
        f"Completed with {len(errors)} error(s). "
        f"{len(completed)} succeeded, {len(no_entities)} had no entities."
    )
else:
    dbutils.notebook.exit(
        f"All {len(results)} file(s) processed successfully. "
        f"{len(completed)} masked, {len(no_entities)} had no entities."
    )
