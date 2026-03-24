"""
Document Intelligence Model for Databricks Model Serving

This MLflow PyFunc model chains:
1. OCR Service (Azure Document Intelligence via Suncorp APIM OAuth) - Extract text and bounding boxes
2. Vision Service (Claude or Databricks or Azure OpenAI) - Detect sensitive entities with word-level bounding boxes

The model implements write-through storage to Unity Catalog volumes.

Environment Variables:
- VISION_SERVICE_PROVIDER: "openai" (default) or "databricks" or "claude"
- ANTHROPIC_API_KEY: Required if provider is "claude"
- DATABRICKS_MODEL_NAME: Required if provider is "databricks"
- AZURE_OPENAI_API_KEY: Required if provider is "openai"
- AZURE_OPENAI_ENDPOINT: Required if provider is "openai"
- AZURE_OPENAI_API_VERSION: Optional, defaults to "2024-02-15-preview"
- AZURE_OPENAI_DEPLOYMENT_NAME: Optional, defaults to "gpt-4o"
- ADI_TENANT_ID: Azure tenant ID for OAuth (defaults to Suncorp tenant)
- ADI_CLIENT_ID: OAuth client ID for ADI
- ADI_CLIENT_SECRET: OAuth client secret for ADI
- ADI_ENDPOINT: APIM endpoint for Document Intelligence
- ADI_APPSPACE_ID: Suncorp AppSpace ID
- ADI_MODEL_ID: Document Intelligence model ID (defaults to "prebuilt-layout")
"""

import concurrent.futures
import copy

import mlflow
import mlflow.pyfunc
import pandas as pd
import json
import base64
import os
import uuid
import logging
import time
from typing import Dict, List, Any
import io
from mimetypes import guess_type

# Our service modules
from .ocr_service import OCRService
from .databricks_service import DatabricksVisionService
from .claude_service import ClaudeVisionService
from .openai_service import OpenAIVisionService
from .fake_data_service import FakeDataService

logger = logging.getLogger(__name__)


# Maximum number of pages whose Vision API calls run concurrently.
# Tune via env var to stay within Azure OpenAI / Anthropic rate limits.
_MAX_CONCURRENT_PAGES: int = int(os.environ.get("VISION_MAX_CONCURRENT_PAGES", "5"))


class DocumentIntelligenceModel(mlflow.pyfunc.PythonModel):
    """
    MLflow model for document de-identification using vision AI and Azure DI.

    Implements the complete pipeline:
    - Document OCR (text + word-level bounding boxes) via ADI OAuth
    - Entity detection with vision AI (Claude or Databricks or Azure OpenAI)
    - Bounding box matching for precise redaction
    - Write-through storage to Unity Catalog volumes

    The vision provider can be toggled via VISION_SERVICE_PROVIDER env var.
    """

    def load_context(self, context):
        """
        Initialize expensive clients once at model startup.
        
        This method is called once when the model is loaded into memory.
        Creating clients here (not in predict) significantly reduces latency.
        
        Args:
            context: MLflow model context (unused but required by interface)
        """
        logger.info("Initializing Document Intelligence Model")

        # Enable auto-tracing for OpenAI
        mlflow.openai.autolog()

        # Initialize OCR service (handles ADI OAuth internally)
        self.ocr_service = OCRService()
        logger.info("OCR service initialized (ADI OAuth)")
        
        # Determine vision service provider
        vision_provider = os.environ.get("VISION_SERVICE_PROVIDER", "openai").lower()
        self.vision_provider = vision_provider
        
        if vision_provider not in ["claude", "databricks", "openai"]:
            raise ValueError(
                f"Invalid VISION_SERVICE_PROVIDER: {vision_provider}. "
                "Must be 'claude' or 'databricks' or 'openai'"
            )
        
        logger.info(f"Vision service provider: {vision_provider}")
        
        # Initialize vision service based on provider
        if vision_provider == "claude":
            anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
            if not anthropic_key:
                raise ValueError("ANTHROPIC_API_KEY must be set when using Claude")
            
            self.vision_service = ClaudeVisionService(api_key=anthropic_key)
            logger.info("Claude vision service initialized")
            
        elif vision_provider == "databricks":
            model_name = os.environ.get("DATABRICKS_MODEL_NAME")
            if not model_name:
                raise ValueError("DATABRICKS_MODEL_NAME must be set when using Databricks")
            
            self.vision_service = DatabricksVisionService(model_name=model_name)
            logger.info("Databricks vision service initialized")
            
        elif vision_provider == "openai":
            # Azure OpenAI configuration
            azure_openai_key = os.environ.get("AZURE_OPENAI_API_KEY")
            azure_openai_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
            azure_openai_api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
            azure_openai_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
            workspace_url: str = os.environ.get("WORKSPACE_URL", "https://suncorp-dev.cloud.databricks.com/")
            workspace_id: str = os.environ.get("WORKSPACE_ID", "1238531023703058")
            proxy_cluster_id: str = os.environ.get("PROXY_CLUSTER_ID", "")
            proxy_port: str = os.environ.get("PROXY_PORT", "8110")
            proxy_route: str = os.environ.get("PROXY_ROUTE", "openai-00010-1")

            if not azure_openai_key:
                raise ValueError("AZURE_OPENAI_API_KEY must be set when using OpenAI")
            if not azure_openai_endpoint:
                raise ValueError("AZURE_OPENAI_ENDPOINT must be set when using OpenAI")
            
            self.vision_service = OpenAIVisionService(
                api_key=azure_openai_key,
                azure_endpoint=azure_openai_endpoint,
                api_version=azure_openai_api_version,
                deployment_name=azure_openai_deployment,
                workspace_url=workspace_url,
                workspace_id=workspace_id,
                proxy_cluster_id=proxy_cluster_id,
                proxy_port=proxy_port,
                proxy_route=proxy_route,
            )
            logger.info(f"Azure OpenAI vision service initialized (deployment: {azure_openai_deployment})")
        
        # Initialize fake data generator
        self.fake_data_service = FakeDataService()
        
        # Get Unity Catalog volume path and Databricks credentials for Files API
        self.uc_volume_path = os.environ.get("UC_VOLUME_PATH")
        if not self.uc_volume_path:
            raise ValueError("UC_VOLUME_PATH must be set")

        self.databricks_host = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
        self.databricks_token = os.environ.get("DATABRICKS_TOKEN", "")
        if not self.databricks_host or not self.databricks_token:
            raise ValueError("DATABRICKS_HOST and DATABRICKS_TOKEN must be set")

        logger.info(f"UC Volume path: {self.uc_volume_path}")
        logger.info("Document Intelligence Model ready")

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        """
        Process document de-identification requests.

        Input (MLflow dataframe_records format):
        ::

            {
              "dataframe_records": [
                {
                  "session_id": "uuid-string",
                  "field_definitions": [
                    {
                      "name": "Full Name",
                      "description": "Person's full name",
                      "strategy": "Fake Data"
                    }
                  ]
                }
              ]
            }

        The model reads ``original{ext}`` from the UC volume using ``session_id``
        — the document bytes are **not** passed in the request payload.

        Output (one prediction per input record):
        ::

            {
              "predictions": [
                {
                  "session_id": "uuid-string",
                  "status": "complete",
                  "entities": [
                    {
                      "id": "uuid-string",
                      "entity_type": "Full Name",
                      "original_text": "Stephen Parrot",
                      "replacement_text": "Jane Doe",
                      "confidence": 0.95,
                      "approved": true,
                      "strategy": "Fake Data",
                      "occurrences": [
                        {
                          "page_number": 1,
                          "original_text": "Stephen Parrot",
                          "bounding_boxes": [[0.1, 0.2, 0.3, 0.05]]
                        },
                        {
                          "page_number": 2,
                          "original_text": "Stephen",
                          "bounding_boxes": [[0.05, 0.1, 0.2, 0.04]]
                        }
                      ]
                    }
                  ],
                  "pages": [
                    {
                      "page_num": 1,
                      "entities": [ ... ]
                    }
                  ]
                }
              ]
            }

        Notes:
          - ``entities`` is the merged flat list (name-variant merging applied).
            The backend prefers this over ``pages`` to avoid overwriting the
            merged result with unmerged per-page data.
          - ``occurrences`` lists every positional appearance after merging,
            including partial name variants (e.g. "Stephen" merged into
            "Stephen Parrot").  Bounding box values are normalised [0, 1].
          - The model also writes ``entities.json`` to the UC volume directly.
          - On error: ``{"session_id": ..., "status": "error", "error_message": ...}``

        Args:
            context: MLflow model context (unused)
            model_input: DataFrame with columns ``session_id`` and
                ``field_definitions`` (list of field definition dicts).

        Returns:
            DataFrame with one result row per input row.
        """
        results = []
        
        for idx, row in model_input.iterrows():
            try:
                result = self._process_document(row)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing row {idx}: {str(e)}", exc_info=True)
                results.append({
                    "session_id": row.get("session_id", "unknown"),
                    "status": "error",
                    "error_message": str(e)
                })
        
        return pd.DataFrame(results)

    def _fetch_original_file(self, session_id: str):
        """
        Fetch the original uploaded file from the UC volume via the Files REST API.

        Returns:
            Tuple of (document_bytes: bytes, document_filename: str)
        """
        import requests as _requests

        _FILES_API = "/api/2.0/fs"
        headers = {"Authorization": f"Bearer {self.databricks_token}"}
        session_path = f"{self.uc_volume_path}/{session_id}"

        # List files in the session directory to find original.*
        list_url = f"{self.databricks_host}{_FILES_API}/directories{session_path}/"
        list_resp = _requests.get(list_url, headers=headers)
        list_resp.raise_for_status()
        files = list_resp.json().get("contents", [])
        original = next(
            (f["name"] for f in files
             if f["name"].startswith("original.")),
            None,
        )
        if not original:
            raise FileNotFoundError(
                f"No original file found in UC volume for session {session_id}"
            )

        file_url = f"{self.databricks_host}{_FILES_API}/files{session_path}/{original}"
        file_resp = _requests.get(file_url, headers=headers)
        file_resp.raise_for_status()
        return file_resp.content, original

    def _process_document(self, row: pd.Series) -> Dict[str, Any]:
        """
        Process a single document through the complete pipeline.

        Args:
            row: DataFrame row with columns:
                - session_id: string
                - field_definitions: list of field definition dicts

        Returns:
            Dictionary with processing results
        """
        session_id = row["session_id"]
        field_definitions = row["field_definitions"]

        logger.info(f"Processing document for session {session_id}")

        # TIMING: Start overall timer
        overall_start = time.time()

        # Fetch document bytes from UC volume via Files REST API
        fetch_start = time.time()
        document_bytes, document_filename = self._fetch_original_file(session_id)
        fetch_time = time.time() - fetch_start
        logger.info(f"⏱️  File fetch: {fetch_time:.2f}s")

        # Extract file extension from stored filename (e.g. "original.pdf" -> "pdf")
        file_extension = document_filename.split('.')[-1].lower()

        # Step 1: OCR - Extract text and word-level bounding boxes
        logger.info(f"Step 1: Running OCR on {document_filename}")
        ocr_start = time.time()
        ocr_result = self.ocr_service.process_document(
            document_bytes=document_bytes,
            file_extension=file_extension
        )
        ocr_time = time.time() - ocr_start
        
        pages = ocr_result
        logger.info(f"OCR completed: {len(pages)} page(s) processed")
        logger.info(f"⏱️  OCR processing: {ocr_time:.2f}s ({ocr_time/max(len(pages), 1):.2f}s per page)")
        
        # Per-document state for replacement pre-generation.
        # Both maps key on normalised original_text so the same value always
        # gets the same replacement across all pages.
        fake_data_consistency: Dict[str, str] = {}
        entity_label_consistency: Dict[str, str] = {}
        entity_label_counters: Dict[str, int] = {}

        result_pages = []

        # Step 2: Vision AI — prepare page inputs then run all pages concurrently.
        # Determine mimetype once (same for every page in a document).
        if file_extension == 'pdf':
            mimetype = 'png'
        else:
            mimetype_full = guess_type(document_filename)[0]
            if mimetype_full and '/' in mimetype_full:
                mimetype = mimetype_full.split('/')[-1]  # 'image/png' -> 'png'
            else:
                mimetype = 'png'

        page_inputs = []
        for page_idx, page_data in enumerate(pages, start=1):
            page_inputs.append({
                "page_number": page_idx,
                "page_image_b64": page_data.get('image_base64', ''),
                "ocr_data": {
                    'text': page_data.get('text', ''),
                    'words': page_data.get('words', []),
                },
                "mimetype": mimetype,
            })

        logger.info(f"Step 2: Running Vision API concurrently for {len(page_inputs)} page(s)")
        vision_start = time.time()

        def _vision_task(page_input: Dict) -> List[Dict]:
            page_number = page_input["page_number"]
            page_image_b64 = page_input["page_image_b64"]
            if not page_image_b64:
                logger.warning(f"No image available for page {page_number}, skipping entity detection")
                return []
            logger.info(f"Vision API starting for page {page_number}")
            t0 = time.time()
            entities = self.vision_service.extract_entities_from_base64(
                image_b64=page_image_b64,
                mimetype=page_input["mimetype"],
                ocr_data=page_input["ocr_data"],
                field_definitions=field_definitions,
                page_number=page_number,
            )
            logger.info(f"⏱️  Vision API (page {page_number}): {time.time() - t0:.2f}s — {len(entities)} entities")
            return entities

        with concurrent.futures.ThreadPoolExecutor(max_workers=_MAX_CONCURRENT_PAGES) as executor:
            all_page_entities = list(executor.map(_vision_task, page_inputs))

        total_vision_time = time.time() - vision_start
        logger.info(f"⏱️  Vision API (all {len(page_inputs)} pages, threaded): {total_vision_time:.2f}s")

        # Step 3: Post-processing (entity IDs, strategies, replacement text)
        for page_input, entities in zip(page_inputs, all_page_entities):
            page_idx = page_input["page_number"]
            logger.info(f"Post-processing page {page_idx}/{len(pages)}: {len(entities)} entities detected")

            enriched_entities = entities

            # Generate entity IDs and set defaults
            for entity in enriched_entities:
                entity['id'] = str(uuid.uuid4())
                entity['approved'] = True  # Default to approved

                # Add strategy from field definitions
                entity_type = entity.get('entity_type')
                matching_field = next(
                    (f for f in field_definitions if f['name'] == entity_type),
                    None
                )
                if matching_field:
                    entity['strategy'] = matching_field.get('strategy', 'Black Out')

                # Pre-generate replacement_text so users can review/edit it in
                # Step 2 before masking is applied.
                if entity.get('strategy') == 'Fake Data' and not entity.get('replacement_text'):
                    # Realistic fake value — same original always maps to same
                    # replacement (e.g. "John Smith" -> "Jane Doe" everywhere).
                    original = entity.get('original_text', '')
                    key = original.lower().strip()
                    if key not in fake_data_consistency:
                        fake_data_consistency[key] = self.fake_data_service.generate(
                            entity.get('entity_type', ''), original
                        )
                    entity['replacement_text'] = fake_data_consistency[key]

                elif entity.get('strategy') == 'Entity Label' and not entity.get('replacement_text'):
                    # Sequential label per entity type: Full_Name_A, Full_Name_B, …
                    # Same original_text always gets the same label.
                    original = entity.get('original_text', '')
                    key = original.lower().strip()
                    if key not in entity_label_consistency:
                        etype = (
                            entity.get('entity_type', 'Unknown')
                            .replace(' ', '_')
                            .replace('-', '_')
                        )
                        entity_label_counters[etype] = entity_label_counters.get(etype, 0) + 1
                        count = entity_label_counters[etype]
                        suffix = chr(64 + count) if count <= 26 else str(count)
                        entity_label_consistency[key] = f"{etype}_{suffix}"
                    entity['replacement_text'] = entity_label_consistency[key]

            result_pages.append({
                'page_num': page_idx,
                'entities': enriched_entities
            })

            logger.info(f"Page {page_idx}: {len(enriched_entities)} entities enriched")
        
        # Flatten and merge entity variants before building the final result so
        # the response carries the same merged entity list that gets written to UC.
        flat_entities: list = []
        for page in result_pages:
            for entity in page.get('entities', []):
                flat_entities.append(entity)

        # Snapshot pre_merge before _merge_entity_variants mutates dicts in-place.
        pre_merge_snapshot = copy.deepcopy(flat_entities)

        merged_entities = self._merge_entity_variants(flat_entities)

        # Prepare final result — include top-level "entities" (merged flat list)
        # so the FastAPI backend receives merged entities from the prediction
        # response and doesn't overwrite the UC-written merged entities.json.
        vision_raw = [
            {"page_number": pi["page_number"], "entities": raw_ents}
            for pi, raw_ents in zip(page_inputs, all_page_entities)
        ]
        result = {
            'session_id': session_id,
            'status': 'complete',
            'pages': result_pages,
            'entities': merged_entities,
            'intermediate_results': {
                'vision_raw': vision_raw,
                'pre_merge': pre_merge_snapshot,
            },
        }

        # Step 4: Write-through to Unity Catalog volume
        write_start = time.time()
        self._write_to_uc_volume(session_id, result, ocr_pages=ocr_result)
        write_time = time.time() - write_start
        logger.info(f"⏱️  UC volume write: {write_time:.2f}s")
        
        # Overall timing summary
        overall_time = time.time() - overall_start
        logger.info(f"\n{'='*80}")
        logger.info(f"⏱️  TIMING SUMMARY for session {session_id}:")
        logger.info(f"  - File fetch:      {fetch_time:6.2f}s ({fetch_time/overall_time*100:5.1f}%)")
        logger.info(f"  - OCR processing:  {ocr_time:6.2f}s ({ocr_time/overall_time*100:5.1f}%)")
        logger.info(f"  - Vision API:      {total_vision_time:6.2f}s ({total_vision_time/overall_time*100:5.1f}%)")
        logger.info(f"  - UC write:        {write_time:6.2f}s ({write_time/overall_time*100:5.1f}%)")
        logger.info(f"  - TOTAL:           {overall_time:6.2f}s")
        logger.info(f"{'='*80}\n")
        
        logger.info(f"Document processing complete for session {session_id}")
        return result

    def _extract_entities_from_page(
        self,
        mimetype: str,
        page_image_b64: str,
        ocr_data: Dict,
        field_definitions: List[Dict],
        page_number: int
    ) -> List[Dict]:
        """
        Extract entities from a page using the configured vision service.
        
        This method delegates to the appropriate vision service (Claude or Databricks or OpenAI)
        which handles prompt building, API calls, and response parsing.
        
        Args:
            mimetype: Image MIME type (e.g., 'png', 'jpeg')
            page_image_b64: Base64-encoded page image
            ocr_data: OCR text and words
            field_definitions: Field definitions for entity detection
            page_number: Page number (1-indexed)
        
        Returns:
            List of detected entities
        """
        try:
            # Delegate to vision service - encapsulates all API interaction
            entities = self.vision_service.extract_entities_from_base64(
                image_b64=page_image_b64,
                mimetype=mimetype,
                ocr_data=ocr_data,
                field_definitions=field_definitions,
                page_number=page_number
            )

            return entities

        except Exception as e:
            logger.error(f"Error extracting entities from page {page_number}: {e}", exc_info=True)
            raise

    @staticmethod
    def _merge_entity_variants(entities: List[Dict]) -> List[Dict]:
        """
        Merge partial name variant entities into their canonical (longest) entity.

        Rules:
        - Only entities with the same entity_type are candidates for merging.
        - A child entity is merged into a parent if the child's original_text
          tokens form a contiguous slice of the parent's tokens, or if the child
          uses a known honorific prefix sharing a suffix with the parent.
        - The child's occurrences are absorbed into the parent; spatially
          duplicate occurrences (child bbox contained in a parent bbox on the
          same page) are dropped.

        Expects entities in the canonical format: each entity has an
        ``occurrences`` list, each occurrence has ``page_number``,
        ``original_text``, and ``bounding_boxes`` ([[x, y, w, h], ...]).

        Returns a new list with child entities removed and parents updated.
        """
        if not entities:
            return entities

        def _is_contained(child_bb: list, parent_bb: list, threshold: float = 0.8) -> bool:
            """Return True if child_bb is >= threshold covered by parent_bb."""
            cx, cy, cw, ch = child_bb
            px, py, pw, ph = parent_bb
            if cw * ch == 0:
                return True
            ix = max(0, min(cx + cw, px + pw) - max(cx, px))
            iy = max(0, min(cy + ch, py + ph) - max(cy, py))
            return (ix * iy) / (cw * ch) >= threshold

        def _occ_is_covered(child_occ: Dict, parent_occs: List[Dict]) -> bool:
            """True if the child occurrence's first bbox is spatially contained
            within any same-page parent occurrence bbox."""
            child_page = child_occ.get("page_number", 1)
            child_bboxes = child_occ.get("bounding_boxes", [])
            if not child_bboxes:
                return False
            child_bb = child_bboxes[0]
            for p_occ in parent_occs:
                if p_occ.get("page_number") != child_page:
                    continue
                for p_bb in p_occ.get("bounding_boxes", []):
                    if _is_contained(child_bb, p_bb):
                        return True
            return False

        # Known honorific/title prefixes used to identify same-person variants
        # e.g. "Mr Doe" and "John Doe" share "Doe" with "Mr" being a title prefix
        _TITLES = frozenset({
            "mr", "mrs", "ms", "miss", "dr", "prof", "professor",
            "sir", "rev", "det", "sgt", "capt", "lt", "col", "cpl",
        })

        sorted_ents = sorted(entities, key=lambda e: len(e.get("original_text", "")), reverse=True)
        merged: set = set()

        for i, parent in enumerate(sorted_ents):
            if i in merged:
                continue
            parent_type = parent.get("entity_type", "")
            parent_tokens = parent.get("original_text", "").lower().strip().split()
            parent_occs = list(parent.get("occurrences", []))

            for j, child in enumerate(sorted_ents):
                if j <= i or j in merged:
                    continue
                if child.get("entity_type", "") != parent_type:
                    continue
                child_tokens = child.get("original_text", "").lower().strip().split()
                n = len(child_tokens)
                if n > len(parent_tokens):
                    continue

                merge_match = False

                if child_tokens == parent_tokens:
                    # Exact same text — merge as additional occurrences (e.g. same name, different page)
                    merge_match = True
                elif n < len(parent_tokens):
                    # Partial match: child tokens form a contiguous slice of parent tokens
                    for k in range(len(parent_tokens) - n + 1):
                        if parent_tokens[k:k + n] == child_tokens:
                            merge_match = True
                            break
                else:
                    # Same token count but not identical: merge if child has a known
                    # title prefix and the remaining suffix tokens all match
                    # e.g. "Mr Doe" → parent "John Doe" because "mr" is a title and "doe" matches
                    shared_k = 0
                    for k in range(1, n + 1):
                        if parent_tokens[-k:] == child_tokens[-k:]:
                            shared_k = k
                        else:
                            break
                    if 0 < shared_k < n:
                        child_diff = child_tokens[:-shared_k]
                        if all(t in _TITLES for t in child_diff):
                            merge_match = True

                if merge_match:
                    for child_occ in child.get("occurrences", []):
                        if not _occ_is_covered(child_occ, parent_occs):
                            parent_occs.append(child_occ)
                    merged.add(j)

            parent["occurrences"] = parent_occs

        result_list = [e for i, e in enumerate(sorted_ents) if i not in merged]
        logger.info(f"Entity variant merge: {len(entities)} → {len(result_list)} entities")
        return result_list

    def _write_to_uc_volume(self, session_id: str, result: Dict, ocr_pages: List[Dict] = None):
        """
        Write processing results to Unity Catalog volume via the Files REST API.

        Flattens entities from all pages into a single list, merges name
        variants (e.g. "Stephen Parrot" absorbs "Stephen"), then writes
        entities.json in the format expected by UCSessionManager.get_entities():
            {"session_id": ..., "status": "awaiting_review", "entities": [...]}

        Args:
            session_id: Session ID for the document
            result: Processing result dictionary with "pages", "entities", and
                "intermediate_results" (vision_raw, pre_merge) keys
            ocr_pages: Raw OCR pages from OCRService. When provided and
                PRIVASEE_DEBUG_INTERMEDIATE=true, written to ocr.json
                (image_base64 stripped to keep file size manageable).
        """
        import requests as _requests
        from datetime import datetime, timezone

        try:
            # Use pre-merged entities from result if available (set by process_document
            # before calling this method). Fall back to flattening+merging from pages
            # for callers that pass a bare pages-only result dict.
            if result.get("entities") is not None:
                flat_entities = result["entities"]
            else:
                flat_entities = []
                for page in result.get("pages", []):
                    for entity in page.get("entities", []):
                        flat_entities.append(entity)
                flat_entities = self._merge_entity_variants(flat_entities)

            payload = {
                "session_id": session_id,
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "status": "awaiting_review",
                "entities": flat_entities,
                "intermediate_results": result.get("intermediate_results", {}),
            }

            _FILES_API = "/api/2.0/fs/files"
            headers = {"Authorization": f"Bearer {self.databricks_token}"}
            base_path = f"{self.uc_volume_path}/{session_id}"

            entities_url = f"{self.databricks_host}{_FILES_API}{base_path}/entities.json"
            resp = _requests.put(entities_url, headers=headers, data=json.dumps(payload))
            resp.raise_for_status()
            logger.info(f"Wrote {len(flat_entities)} entities to UC volume for session {session_id}")

            # Write ocr.json when debug mode is enabled (image_base64 stripped to keep size manageable)
            if ocr_pages and os.environ.get("PRIVASEE_DEBUG_INTERMEDIATE", "false").lower() == "true":
                ocr_payload = {
                    "session_id": session_id,
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                    "pages": [
                        {k: v for k, v in page.items() if k != "image_base64"}
                        for page in ocr_pages
                    ],
                }
                ocr_url = f"{self.databricks_host}{_FILES_API}{base_path}/ocr.json"
                resp = _requests.put(ocr_url, headers=headers, data=json.dumps(ocr_payload))
                resp.raise_for_status()
                logger.info(f"Wrote OCR debug data to UC volume for session {session_id}")

        except Exception as e:
            logger.error(f"Error writing entities to UC volume: {e}")
            raise
