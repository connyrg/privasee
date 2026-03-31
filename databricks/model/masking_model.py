"""
Masking Model for Databricks Model Serving

MLflow PyFunc model that applies document masking on the Databricks side.

Pipeline:
  1. Fetch original file from UC volume via Files REST API
  2. Apply masking (PyMuPDF for PDF, PIL for images)
  3. Write masked.pdf to UC volume
  4. Return {session_id, status, entities_masked}

Input (dataframe_records format):
  {
    "dataframe_records": [
      {
        "session_id": "uuid-string",
        "entities_to_mask": "[{...entity dicts...}]"   <- JSON string
      }
    ]
  }

  Each entity dict in ``entities_to_mask`` must include:

    {
      "id": "uuid",
      "entity_type": "Full Name",
      "original_text": "Stephen Parrot",
      "replacement_text": "Jane Doe",
      "strategy": "Fake Data",           // "Black Out" | "Fake Data" | "Entity Label"
      "approved": true,
      "occurrences": [
        {
          "page_number": 1,
          "bounding_boxes": [[0.1, 0.2, 0.3, 0.05]],  // list of [x, y, w, h] (normalised)
          "original_text": "Stephen Parrot"
        },
        {
          "page_number": 2,
          "bounding_boxes": [[0.05, 0.1, 0.2, 0.04]],
          "original_text": "Stephen"                   // partial variant
        }
      ]
    }

  The masking service iterates ``occurrences`` to locate and redact each
  appearance of the entity in the document.

Output:
  {
    "predictions": [
      {"session_id": "...", "status": "complete", "entities_masked": 5}
    ]
  }

  On error: {"session_id": "...", "status": "error", "entities_masked": 0,
             "error_message": "..."}

Environment Variables:
  DATABRICKS_HOST      Workspace URL
  DATABRICKS_TOKEN     PAT with Files API read/write access
  UC_VOLUME_PATH       UC volume base path, e.g. /Volumes/catalog/schema/sessions
"""

import io
import json
import logging
import os
import tempfile
from typing import Any, Dict, List, Tuple

import mlflow.pyfunc
import pandas as pd

logger = logging.getLogger(__name__)

_FILES_API = "/api/2.0/fs/files"
_DIRECTORIES_API = "/api/2.0/fs/directories"


class MaskingModel(mlflow.pyfunc.PythonModel):
    """MLflow model that applies document masking and persists results to UC."""

    def load_context(self, context):
        from .masking_service import MaskingService

        self.masking_service = MaskingService()

        self.uc_volume_path = os.environ.get("UC_VOLUME_PATH", "").rstrip("/")
        self.databricks_host = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
        self.databricks_token = os.environ.get("DATABRICKS_TOKEN", "")

        if not self.uc_volume_path:
            raise ValueError("UC_VOLUME_PATH must be set")
        if not self.databricks_host or not self.databricks_token:
            raise ValueError("DATABRICKS_HOST and DATABRICKS_TOKEN must be set")

        logger.info("MaskingModel ready")

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        """MLflow PyFunc entry point. Masks one session per input row.

        Each row must have `session_id` and `entities_to_mask` (JSON string).
        Returns a DataFrame with columns: session_id, status, entities_masked,
        and error_message (only present on error rows).
        """
        results = []
        for idx, row in model_input.iterrows():
            try:
                results.append(self._process_masking(row))
            except Exception as exc:
                logger.error("Error masking session %s: %s", row.get("session_id"), exc, exc_info=True)
                results.append({
                    "session_id": row.get("session_id", "unknown"),
                    "status": "error",
                    "entities_masked": 0,
                    "error_message": str(exc),
                })
        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Core masking pipeline
    # ------------------------------------------------------------------

    def _process_masking(self, row: pd.Series) -> Dict[str, Any]:
        """Orchestrate fetch → mask → write for a single session row.

        Deserialises `entities_to_mask` from JSON string if needed, fetches the
        original file from UC, applies masking, and writes masked.pdf back to UC.
        Returns a result dict with session_id, status, and entities_masked count.
        """
        session_id = row["session_id"]
        entities_to_mask = row["entities_to_mask"]

        if isinstance(entities_to_mask, str):
            entities_to_mask = json.loads(entities_to_mask)

        logger.info("Masking %d entities for session %s", len(entities_to_mask), session_id)

        file_bytes, filename = self._fetch_original_file(session_id)
        ext = "." + filename.split(".")[-1].lower()

        masked_bytes = self._apply_masking(file_bytes, ext, entities_to_mask)
        self._write_masked_file(session_id, masked_bytes)

        logger.info("Masking complete for session %s", session_id)
        return {
            "session_id": session_id,
            "status": "complete",
            "entities_masked": len(entities_to_mask),
        }

    def _apply_masking(self, file_bytes: bytes, ext: str, entities: List[Dict]) -> bytes:
        """Dispatch to PDF or image masking and return masked bytes as PDF.

        PDFs are masked natively via PyMuPDF. Images (PNG/JPG) are masked via PIL
        then wrapped in a single-page PDF — output is always PDF regardless of input
        format (see TODO #3 for planned format-preserving output).
        Raises ValueError for unsupported file types.
        """
        if ext == ".pdf":
            return self.masking_service.apply_pdf_masks(file_bytes, entities)

        if ext in (".png", ".jpg", ".jpeg"):
            with tempfile.TemporaryDirectory() as tmpdir:
                input_path = os.path.join(tmpdir, f"input{ext}")
                with open(input_path, "wb") as fh:
                    fh.write(file_bytes)
                output_path = os.path.join(tmpdir, "masked.png")
                self.masking_service.apply_masks(input_path, entities, output_path)
                from PIL import Image as PILImage
                img = PILImage.open(output_path).convert("RGB")
                buf = io.BytesIO()
                img.save(buf, format="PDF")
                return buf.getvalue()

        raise ValueError(f"Unsupported file type: {ext}. Supported: .pdf, .png, .jpg, .jpeg")

    # ------------------------------------------------------------------
    # UC volume I/O (Files REST API)
    # ------------------------------------------------------------------

    def _fetch_original_file(self, session_id: str) -> Tuple[bytes, str]:
        """Fetch original.* from UC volume. Returns (bytes, filename)."""
        import requests as _requests

        headers = {"Authorization": f"Bearer {self.databricks_token}"}
        session_path = f"{self.uc_volume_path}/{session_id}"

        list_url = f"{self.databricks_host}{_DIRECTORIES_API}{session_path}/"
        resp = _requests.get(list_url, headers=headers)
        resp.raise_for_status()
        files = resp.json().get("contents", [])

        original = next(
            (f["name"] for f in files if f["name"].startswith("original.")),
            None,
        )
        if not original:
            raise FileNotFoundError(
                f"No original file found in UC volume for session {session_id}"
            )

        file_url = f"{self.databricks_host}{_FILES_API}{session_path}/{original}"
        file_resp = _requests.get(file_url, headers=headers)
        file_resp.raise_for_status()
        return file_resp.content, original

    def _write_masked_file(self, session_id: str, masked_bytes: bytes) -> None:
        """Write masked.pdf to UC volume."""
        import requests as _requests

        headers = {"Authorization": f"Bearer {self.databricks_token}"}
        path = f"{self.uc_volume_path}/{session_id}/masked.pdf"
        url = f"{self.databricks_host}{_FILES_API}{path}"

        resp = _requests.put(url, headers=headers, data=masked_bytes)
        resp.raise_for_status()
        logger.info("Wrote masked.pdf to UC for session %s", session_id)
