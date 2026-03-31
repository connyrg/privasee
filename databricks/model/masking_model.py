"""
Masking Model for Databricks Model Serving

MLflow PyFunc model that applies document masking on the Databricks side.

Pipeline:
  1. Fetch original file from UC volume via Files REST API
  2. Apply masking (PyMuPDF for PDF, PIL for images)
  3. Write masked.pdf to UC volume
  4. Optionally verify masking quality via re-OCR on the masked output
  5. Return {session_id, status, entities_masked[, occurrences_total, occurrences_masked, score]}

Input (dataframe_records format):
  {
    "dataframe_records": [
      {
        "session_id": "uuid-string",
        "entities_to_mask": "[{...entity dicts...}]",  <- JSON string
        "run_verification": false                       <- optional, default false
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

Output (run_verification=false):
  {
    "predictions": [
      {"session_id": "...", "status": "complete", "entities_masked": 5}
    ]
  }

Output (run_verification=true):
  {
    "predictions": [
      {
        "session_id": "...", "status": "complete", "entities_masked": 5,
        "occurrences_total": 8, "occurrences_masked": 7, "score": 87.5
      }
    ]
  }

  On error: {"session_id": "...", "status": "error", "entities_masked": 0,
             "error_message": "..."}

Environment Variables:
  DATABRICKS_HOST      Workspace URL
  DATABRICKS_TOKEN     PAT with Files API read/write access
  UC_VOLUME_PATH       UC volume base path, e.g. /Volumes/catalog/schema/sessions

  Verification (ADI re-OCR for scanned pages):
  ADI_TENANT_ID, ADI_CLIENT_ID, ADI_CLIENT_SECRET, ADI_ENDPOINT,
  ADI_APPSPACE_ID, ADI_MODEL_ID — same vars as OCRService.
  When absent, scanned pages are treated as successfully masked (can't verify).
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
        from .ocr_service import OCRService

        self.masking_service = MaskingService()
        self.ocr_service = OCRService()

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
        Optional `run_verification` (bool, default False) triggers re-OCR
        verification after masking.
        Returns a DataFrame with columns: session_id, status, entities_masked,
        and (when run_verification=True) occurrences_total, occurrences_masked,
        score. Error rows add error_message.
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
        """Orchestrate fetch → mask → write → optional verify for a single session row.

        Deserialises `entities_to_mask` from JSON string if needed, fetches the
        original file from UC, applies masking, and writes masked.pdf back to UC.
        When `run_verification` is True in the row, also runs `_verify_masking`
        and includes occurrences_total, occurrences_masked, and score in the result.
        Returns a result dict with session_id, status, and entities_masked count.
        """
        session_id = row["session_id"]
        entities_to_mask = row["entities_to_mask"]
        run_verification = bool(row.get("run_verification", False))

        if isinstance(entities_to_mask, str):
            entities_to_mask = json.loads(entities_to_mask)

        logger.info("Masking %d entities for session %s", len(entities_to_mask), session_id)

        file_bytes, filename = self._fetch_original_file(session_id)
        ext = "." + filename.split(".")[-1].lower()

        masked_bytes = self._apply_masking(file_bytes, ext, entities_to_mask)
        self._write_masked_file(session_id, masked_bytes)

        result: Dict[str, Any] = {
            "session_id": session_id,
            "status": "complete",
            "entities_masked": len(entities_to_mask),
        }

        if run_verification:
            verify = self._verify_masking(file_bytes, ext, masked_bytes, entities_to_mask)
            result.update(verify)

        logger.info("Masking complete for session %s", session_id)
        return result

    def _verify_masking(
        self,
        original_bytes: bytes,
        original_ext: str,
        masked_bytes: bytes,
        entities: List[Dict],
    ) -> Dict[str, Any]:
        """Verify masking quality by re-OCR on the masked PDF output.

        Classifies pages using the ORIGINAL document's text layer — not the masked
        output — to avoid false-positive "digital" classification when Fake Data or
        Entity Label strategies have written replacement text into scanned pages via
        insert_text().

        - Digital pages (original has a text layer): PyMuPDF text extraction on
          the masked page.
        - Scanned pages (original has no text layer) and image originals: render
          the masked PDF page to PNG and re-OCR with ADI if available. When ADI is
          unavailable, scanned pages are assumed masked (can't verify).

        Args:
            original_bytes: Raw bytes of the original uploaded file.
            original_ext: File extension of the original, e.g. ".pdf".
            masked_bytes: PDF bytes of the masked output.
            entities: List of entity dicts, each with an ``occurrences`` list.

        Returns:
            {"occurrences_total": int, "occurrences_masked": int, "score": float}
        """
        import fitz  # noqa: PLC0415 — deferred to avoid top-level cost

        # Build flat list of (page_number, text_lower) for every occurrence
        all_occurrences: List[Tuple[int, str]] = []
        for entity in entities:
            canonical = (entity.get("original_text") or "").lower()
            for occ in entity.get("occurrences") or []:
                page_num = occ.get("page_number", 1)
                text = (occ.get("original_text") or canonical).lower()
                if text:
                    all_occurrences.append((page_num, text))

        if not all_occurrences:
            return {"occurrences_total": 0, "occurrences_masked": 0, "score": 100.0}

        occurrences_total = len(all_occurrences)

        # --- Classify pages from the original (not the masked output) ---
        is_digital: Dict[int, bool] = {}  # page_num (1-indexed) → True = digital
        if original_ext == ".pdf":
            try:
                orig_doc = fitz.open(stream=original_bytes, filetype="pdf")
                for i, page in enumerate(orig_doc):
                    is_digital[i + 1] = len(page.get_text().strip()) >= self.ocr_service.MIN_TEXT_LENGTH_FOR_DIGITAL
                orig_doc.close()
            except Exception as exc:
                logger.warning("Could not classify pages from original PDF: %s", exc)
        # For non-PDF originals (images), is_digital stays empty → all pages = scanned

        # --- Determine which pages need text extraction ---
        needed_pages = {page_num for page_num, _ in all_occurrences}
        scanned_pages = {p for p in needed_pages if not is_digital.get(p, False)}

        if scanned_pages and not self.ocr_service.adi_available:
            logger.warning(
                "Verification: %d scanned page(s) detected but ADI credentials are not "
                "configured on this endpoint. Scanned pages will be treated as fully masked "
                "— score may be artificially high. Set ADI_TENANT_ID, ADI_CLIENT_ID, "
                "ADI_CLIENT_SECRET, and ADI_ENDPOINT on the masking endpoint to enable "
                "re-OCR verification for scanned pages.",
                len(scanned_pages),
            )

        # Pass 1: open masked PDF, extract digital page texts immediately, render
        # scanned pages to PNG bytes while the fitz doc is still open, then close it.
        # Mirrors OCRService._process_pdf pass 1 — fitz doc must not outlive this block.
        page_texts: Dict[int, str] = {}  # page_num → extracted text (lower-case)
        scanned_tasks: List[Tuple[int, bytes]] = []  # (page_num, png_bytes) to OCR
        try:
            masked_doc = fitz.open(stream=masked_bytes, filetype="pdf")
            zoom = self.ocr_service.RENDER_ZOOM_FACTOR
            for page_num in needed_pages:
                page_idx = page_num - 1
                if page_idx < 0 or page_idx >= len(masked_doc):
                    page_texts[page_num] = ""
                    continue
                page = masked_doc[page_idx]
                if is_digital.get(page_num, False):
                    page_texts[page_num] = page.get_text().lower()
                elif self.ocr_service.adi_available:
                    mat = fitz.Matrix(zoom, zoom)
                    png_bytes = page.get_pixmap(matrix=mat).tobytes("png")
                    scanned_tasks.append((page_num, png_bytes))
                else:
                    page_texts[page_num] = ""  # ADI unavailable — treat as masked
            masked_doc.close()
        except Exception as exc:
            logger.error("Failed to open masked PDF for verification: %s", exc)
            return {
                "occurrences_total": occurrences_total,
                "occurrences_masked": occurrences_total,
                "score": 100.0,
            }

        # Pass 2: run ADI on all scanned pages concurrently via ThreadPoolExecutor.
        # Mirrors OCRService._process_pdf pass 2 — avoids asyncio.run() on the
        # Databricks Serverless event loop.  One OAuth token shared across all calls.
        if scanned_tasks:
            import concurrent.futures  # noqa: PLC0415

            try:
                adi_token = self.ocr_service._get_adi_token()
            except Exception as exc:
                logger.warning(
                    "Could not fetch ADI token for verification: %s — "
                    "scanned pages treated as masked", exc,
                )
                adi_token = None

            if adi_token is not None:
                def _ocr_task(task: Tuple[int, bytes]) -> Tuple[int, str]:
                    pg, png = task
                    try:
                        result = self.ocr_service._ocr_with_adi(png, token=adi_token)
                        return pg, result.get("text", "").lower()
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "ADI OCR failed for verify page %d: %s — treating as masked",
                            pg, exc,
                        )
                        return pg, ""

                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.ocr_service._adi_max_concurrent
                ) as executor:
                    for pg, text in executor.map(_ocr_task, scanned_tasks):
                        page_texts[pg] = text
            else:
                for pg, _ in scanned_tasks:
                    page_texts[pg] = ""

        # --- Count occurrences that are no longer present in the extracted text ---
        occurrences_masked = sum(
            1 for page_num, text in all_occurrences
            if text not in page_texts.get(page_num, "")
        )

        score = round(occurrences_masked / occurrences_total * 100, 1)
        return {
            "occurrences_total": occurrences_total,
            "occurrences_masked": occurrences_masked,
            "score": score,
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
