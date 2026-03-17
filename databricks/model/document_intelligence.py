"""
Document Intelligence Model for Databricks Model Serving

This MLflow PyFunc model chains:
1. OCR Service (Azure Document Intelligence via Suncorp APIM OAuth) - Extract text and bounding boxes
2. Vision Service (Claude or Azure OpenAI) - Detect sensitive entities
3. BBox Matcher - Match entities to OCR words for precise redaction coordinates

The model implements write-through storage to Unity Catalog volumes.

Environment Variables:
- VISION_SERVICE_PROVIDER: "openai" (default) or "claude"
- ANTHROPIC_API_KEY: Required if provider is "claude"
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
from .claude_service import ClaudeVisionService
from .openai_service import OpenAIVisionService
from .bbox_matcher import BBoxMatcher
from .fake_data_service import FakeDataService

logger = logging.getLogger(__name__)


class DocumentIntelligenceModel(mlflow.pyfunc.PythonModel):
    """
    MLflow model for document de-identification using vision AI and Azure DI.
    
    Implements the complete pipeline:
    - Document OCR (text + word-level bounding boxes) via ADI OAuth
    - Entity detection with vision AI (Claude or Azure OpenAI)
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
        
        # Initialize OCR service (handles ADI OAuth internally)
        self.ocr_service = OCRService()
        logger.info("OCR service initialized (ADI OAuth)")
        
        # Determine vision service provider
        vision_provider = os.environ.get("VISION_SERVICE_PROVIDER", "openai").lower()
        self.vision_provider = vision_provider
        
        if vision_provider not in ["claude", "openai"]:
            raise ValueError(
                f"Invalid VISION_SERVICE_PROVIDER: {vision_provider}. "
                "Must be 'claude' or 'openai'"
            )
        
        logger.info(f"Vision service provider: {vision_provider}")
        
        # Initialize vision service based on provider
        if vision_provider == "claude":
            anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
            if not anthropic_key:
                raise ValueError("ANTHROPIC_API_KEY must be set when using Claude")
            
            self.vision_service = ClaudeVisionService(api_key=anthropic_key)
            logger.info("Claude vision service initialized")
            
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
        
        # Initialize BBox matcher and fake data generator
        self.bbox_matcher = BBoxMatcher()
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
        
        Args:
            context: MLflow model context (unused)
            model_input: DataFrame with columns:
                - session_id: string
                - field_definitions: list of field definition dicts
        
        Returns:
            DataFrame with processing results for each row
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

        # Process each page
        result_pages = []
        total_vision_time = 0.0
        total_bbox_time = 0.0

        for page_idx, page_data in enumerate(pages, start=1):
            logger.info(f"Processing page {page_idx}/{len(pages)}")
            
            # Prepare OCR data for vision service
            ocr_data = {
                'text': page_data.get('text', ''),
                'words': page_data.get('words', [])
            }
            
            # Step 2: Vision AI - Detect entities
            page_image_b64 = page_data.get('image_base64', '')
            
            if page_image_b64:
                # Call vision service with base64 image
                if file_extension == 'pdf':
                    mimetype = 'png'
                else:
                    # guess_type returns tuple ('image/png', None), extract format
                    mimetype_full = guess_type(document_filename)[0]
                    if mimetype_full and '/' in mimetype_full:
                        mimetype = mimetype_full.split('/')[-1]  # 'image/png' -> 'png'
                    else:
                        mimetype = 'png'  # default fallback
                
                vision_start = time.time()
                entities = self._extract_entities_from_page(
                    mimetype=mimetype,
                    page_image_b64=page_image_b64,
                    ocr_data=ocr_data,
                    field_definitions=field_definitions,
                    page_number=page_idx
                )
                vision_time = time.time() - vision_start
                total_vision_time += vision_time
                logger.info(f"⏱️  Vision API (page {page_idx}): {vision_time:.2f}s - detected {len(entities)} entities")
            else:
                # No image available (e.g., digital PDF page with direct text extraction)
                # Skip entity detection for this page or use text-only mode
                logger.warning(f"No image available for page {page_idx}, skipping entity detection")
                entities = []
            
            # Step 3: BBox Matcher - Enrich entities with bounding boxes
            if entities:
                logger.info(f"Matching {len(entities)} entities to OCR words")
                bbox_start = time.time()
                enriched_entities = self.bbox_matcher.match_entities_to_words(
                    entities=entities,
                    ocr_words=ocr_data['words']
                )
                bbox_time = time.time() - bbox_start
                total_bbox_time += bbox_time
                logger.info(f"⏱️  BBox matching (page {page_idx}): {bbox_time:.2f}s")
            else:
                enriched_entities = []
            
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

                # Add bounding_box (singular flat list) derived from the first
                # occurrence in bounding_boxes.  The backend Entity model requires
                # this field; the masking service uses bounding_boxes (all
                # occurrences) so every appearance in the document is redacted.
                if 'bounding_box' not in entity:
                    first = next(iter(entity.get('bounding_boxes', [])), None)
                    if isinstance(first, dict):
                        entity['bounding_box'] = [first['x'], first['y'], first['width'], first['height']]
                    elif isinstance(first, (list, tuple)) and len(first) == 4:
                        entity['bounding_box'] = list(first)
                    else:
                        entity['bounding_box'] = [0.0, 0.0, 0.0, 0.0]

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
                    # Sequential letter label per entity type: Full_Name_A,
                    # Full_Name_B, … Full_Name_Z, Full_Name_27, …
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
        merged_entities = self._merge_entity_variants(flat_entities)

        # Prepare final result — include top-level "entities" (merged flat list)
        # so the FastAPI backend receives merged entities from the prediction
        # response and doesn't overwrite the UC-written merged entities.json.
        result = {
            'session_id': session_id,
            'status': 'complete',
            'pages': result_pages,
            'entities': merged_entities,
        }

        # Step 4: Write-through to Unity Catalog volume
        write_start = time.time()
        self._write_to_uc_volume(session_id, result)
        write_time = time.time() - write_start
        logger.info(f"⏱️  UC volume write: {write_time:.2f}s")
        
        # Overall timing summary
        overall_time = time.time() - overall_start
        logger.info(f"\n{'='*80}")
        logger.info(f"⏱️  TIMING SUMMARY for session {session_id}:")
        logger.info(f"  - File fetch:      {fetch_time:6.2f}s ({fetch_time/overall_time*100:5.1f}%)")
        logger.info(f"  - OCR processing:  {ocr_time:6.2f}s ({ocr_time/overall_time*100:5.1f}%)")
        logger.info(f"  - Vision API:      {total_vision_time:6.2f}s ({total_vision_time/overall_time*100:5.1f}%)")
        logger.info(f"  - BBox matching:   {total_bbox_time:6.2f}s ({total_bbox_time/overall_time*100:5.1f}%)")
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
        
        This method delegates to the appropriate vision service (Claude or OpenAI)
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
            logger.error(f"Error extracting entities from page {page_number}: {e}")
            return []

    @staticmethod
    def _merge_entity_variants(entities: List[Dict]) -> List[Dict]:
        """
        Merge partial name variant entities into their canonical (longest) entity.

        Rules:
        - Only entities with the same entity_type are candidates for merging.
        - A child entity is merged into a parent if the child's original_text
          tokens form a contiguous slice of the parent's tokens.
        - The child's bbox and page_number are absorbed as an Occurrence on the
          parent; replacement_text is derived dynamically at masking time.
        - Already-merged entities (have occurrences) are left unchanged.

        Returns a new list with child entities removed and parents updated with
        occurrences covering all appearances (own + merged children's).
        """
        if not entities:
            return entities

        # Skip if already merged
        if all(e.get("occurrences") is not None for e in entities):
            return entities

        def _normalise_bb(bb):
            if isinstance(bb, (list, tuple)) and len(bb) == 4:
                return list(bb)
            if isinstance(bb, dict) and all(k in bb for k in ("x", "y", "width", "height")):
                return [bb["x"], bb["y"], bb["width"], bb["height"]]
            return None

        def _to_occurrences(e: Dict) -> List[Dict]:
            if e.get("occurrences") is not None:
                return list(e["occurrences"])
            page = e.get("page_number", 1)
            orig = e.get("original_text", "")
            boxes = e.get("bounding_boxes") or ([e["bounding_box"]] if e.get("bounding_box") else [])
            result = []
            for bb in boxes:
                norm = _normalise_bb(bb)
                if norm:
                    result.append({"page_number": page, "bounding_box": norm, "original_text": orig})
            return result

        def _is_contained(child_bb: list, parent_bb: list, threshold: float = 0.8) -> bool:
            """Return True if child_bb is >= threshold covered by parent_bb."""
            cx, cy, cw, ch = child_bb
            px, py, pw, ph = parent_bb
            if cw * ch == 0:
                return True
            ix = max(0, min(cx + cw, px + pw) - max(cx, px))
            iy = max(0, min(cy + ch, py + ph) - max(cy, py))
            return (ix * iy) / (cw * ch) >= threshold

        sorted_ents = sorted(entities, key=lambda e: len(e.get("original_text", "")), reverse=True)
        merged: set = set()

        for i, parent in enumerate(sorted_ents):
            if i in merged:
                continue
            parent_type = parent.get("entity_type", "")
            parent_tokens = parent.get("original_text", "").lower().strip().split()
            if len(parent_tokens) < 2:
                continue  # single-token entities cannot be parents

            parent_occs = _to_occurrences(parent)

            for j, child in enumerate(sorted_ents):
                if j <= i or j in merged:
                    continue
                if child.get("entity_type", "") != parent_type:
                    continue  # different entity types — different people
                child_tokens = child.get("original_text", "").lower().strip().split()
                n = len(child_tokens)
                if n >= len(parent_tokens):
                    continue  # child must be strictly shorter

                # Check for contiguous token window match
                for k in range(len(parent_tokens) - n + 1):
                    if parent_tokens[k:k + n] == child_tokens:
                        child_page = child.get("page_number", 1)
                        child_boxes = child.get("bounding_boxes") or (
                            [child["bounding_box"]] if child.get("bounding_box") else []
                        )
                        for bb in child_boxes:
                            norm = _normalise_bb(bb)
                            if not norm:
                                continue
                            # Skip if already covered by an existing occurrence on
                            # the same page (e.g. "Stephen" sub-bbox inside "Stephen Parrot").
                            already_covered = any(
                                occ.get("page_number") == child_page
                                and _is_contained(norm, occ["bounding_box"])
                                for occ in parent_occs
                                if occ.get("bounding_box")
                            )
                            if not already_covered:
                                parent_occs.append({
                                    "page_number": child_page,
                                    "bounding_box": norm,
                                    "original_text": child.get("original_text", ""),
                                })
                        merged.add(j)
                        break

            parent["occurrences"] = parent_occs

        # Ensure unmerged entities also get occurrences populated
        for i, entity in enumerate(sorted_ents):
            if i not in merged and entity.get("occurrences") is None:
                entity["occurrences"] = _to_occurrences(entity)

        result_list = [e for i, e in enumerate(sorted_ents) if i not in merged]
        logger.info(f"Entity variant merge: {len(entities)} → {len(result_list)} entities")
        return result_list

    def _write_to_uc_volume(self, session_id: str, result: Dict):
        """
        Write processing results to Unity Catalog volume via the Files REST API.

        Flattens entities from all pages into a single list, merges name
        variants (e.g. "Stephen Parrot" absorbs "Stephen"), then writes
        entities.json in the format expected by UCSessionManager.get_entities():
            {"session_id": ..., "status": "awaiting_review", "entities": [...]}

        Args:
            session_id: Session ID for the document
            result: Processing result dictionary with a "pages" list
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
                    page_num = page.get("page_num", 1)
                    for entity in page.get("entities", []):
                        entity.setdefault("page_number", page_num)
                        flat_entities.append(entity)
                flat_entities = self._merge_entity_variants(flat_entities)

            payload = {
                "session_id": session_id,
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "status": "awaiting_review",
                "entities": flat_entities,
            }

            _FILES_API = "/api/2.0/fs/files"
            headers = {"Authorization": f"Bearer {self.databricks_token}"}
            entities_path = f"{self.uc_volume_path}/{session_id}/entities.json"
            url = f"{self.databricks_host}{_FILES_API}{entities_path}"

            resp = _requests.put(url, headers=headers, data=json.dumps(payload))
            resp.raise_for_status()

            logger.info(f"Wrote {len(flat_entities)} entities to UC volume for session {session_id}")

        except Exception as e:
            logger.error(f"Error writing entities to UC volume: {e}")
            # Don't fail the request if write fails — result is still returned to caller
