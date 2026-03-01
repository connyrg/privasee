"""
Document Intelligence Model for Databricks Model Serving

This MLflow PyFunc model chains:
1. OCR Service (Azure Document Intelligence) - Extract text and bounding boxes
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
"""

import mlflow.pyfunc
import pandas as pd
import json
import base64
import os
import uuid
import logging
from typing import Dict, List, Any
import io
from mimetypes import guess_type

# Azure Document Intelligence
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential

# Our service modules
from .ocr_service import OCRService
from .claude_service import ClaudeVisionService
from .openai_service import OpenAIVisionService
from .bbox_matcher import BBoxMatcher

logger = logging.getLogger(__name__)


class DocumentIntelligenceModel(mlflow.pyfunc.PythonModel):
    """
    MLflow model for document de-identification using vision AI and Azure DI.
    
    Implements the complete pipeline:
    - Document OCR (text + word-level bounding boxes)
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
        
        # Initialize Azure Document Intelligence client
        adi_endpoint = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
        adi_key = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_KEY")
        
        if not adi_endpoint or not adi_key:
            raise ValueError(
                "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT and "
                "AZURE_DOCUMENT_INTELLIGENCE_KEY must be set"
            )
        
        self.adi_client = DocumentIntelligenceClient(
            endpoint=adi_endpoint,
            credential=AzureKeyCredential(adi_key)
        )
        logger.info("Azure Document Intelligence client initialized")
        
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
        
        # Initialize OCR service and BBox matcher
        self.ocr_service = OCRService()
        self.bbox_matcher = BBoxMatcher()
        
        # Get Unity Catalog volume path
        self.uc_volume_path = os.environ.get("UC_VOLUME_PATH")
        if not self.uc_volume_path:
            raise ValueError("UC_VOLUME_PATH must be set")
        
        logger.info(f"UC Volume path: {self.uc_volume_path}")
        logger.info("Document Intelligence Model ready")

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        """
        Process document de-identification requests.
        
        Args:
            context: MLflow model context (unused)
            model_input: DataFrame with columns:
                - session_id: string
                - document_bytes_b64: base64-encoded document bytes
                - document_filename: original filename
                - field_definitions_json: JSON string of field definitions
        
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

    def _process_document(self, row: pd.Series) -> Dict[str, Any]:
        """
        Process a single document through the complete pipeline.
        
        Args:
            row: DataFrame row with document data
        
        Returns:
            Dictionary with processing results
        """
        session_id = row["session_id"]
        document_bytes_b64 = row["document_bytes_b64"]
        document_filename = row["document_filename"]
        field_definitions_json = row["field_definitions_json"]
        
        logger.info(f"Processing document for session {session_id}")
        
        # Parse field definitions
        field_definitions = json.loads(field_definitions_json)
        
        # Decode document bytes
        document_bytes = base64.b64decode(document_bytes_b64)
        
        # Extract file extension from filename
        file_extension = document_filename.split('.')[-1].lower()
        
        # Step 1: OCR - Extract text and word-level bounding boxes
        logger.info(f"Step 1: Running OCR on {document_filename}")
        ocr_result = self.ocr_service.process_document(
            document_bytes=document_bytes,
            file_extension=file_extension
        )
        
        pages = ocr_result
        logger.info(f"OCR completed: {len(pages)} page(s) processed")
        
        # Process each page
        result_pages = []
        
        for page_idx, page_data in enumerate(pages, start=1):
            logger.info(f"Processing page {page_idx}/{len(pages)}")
            
            # Prepare OCR data for vision service
            ocr_data = {
                'text': page_data.get('text', ''),
                'words': page_data.get('words', [])
            }
            
            # Step 2: Vision AI - Detect entities
            # page_image_b64 = page_data.get('image_base64')
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
                entities = self._extract_entities_from_page(
                    mimetype=mimetype,
                    page_image_b64=page_image_b64,
                    ocr_data=ocr_data,
                    field_definitions=field_definitions,
                    page_number=page_idx
                )
            else:
                # No image available (e.g., digital PDF page with direct text extraction)
                # Skip entity detection for this page or use text-only mode
                logger.warning(f"No image available for page {page_idx}, skipping entity detection")
                entities = []
            
            # Step 3: BBox Matcher - Enrich entities with bounding boxes
            if entities:
                logger.info(f"Matching {len(entities)} entities to OCR words")
                enriched_entities = self.bbox_matcher.match_entities_to_words(
                    entities=entities,
                    ocr_words=ocr_data['words']
                )
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
            
            result_pages.append({
                'page_num': page_idx,
                'entities': enriched_entities
            })
            
            logger.info(f"Page {page_idx}: {len(enriched_entities)} entities enriched")
        
        # Prepare final result
        result = {
            'session_id': session_id,
            'status': 'complete',
            'pages': result_pages
        }
        
        # Step 4: Write-through to Unity Catalog volume
        self._write_to_uc_volume(session_id, result)
        
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

    def _write_to_uc_volume(self, session_id: str, result: Dict):
        """
        Write processing results to Unity Catalog volume.
        
        Unity Catalog volumes are mounted as local filesystem paths on Databricks,
        so we can write directly without API calls.
        
        Args:
            session_id: Session ID for the document
            result: Processing result dictionary
        """
        try:
            # Construct filesystem path
            session_dir = os.path.join(self.uc_volume_path, session_id)
            entities_file = os.path.join(session_dir, "entities.json")
            
            # Create session directory if it doesn't exist
            os.makedirs(session_dir, exist_ok=True)
            
            # Write entities.json
            with open(entities_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            logger.info(f"Wrote entities to {entities_file}")
            
        except Exception as e:
            logger.error(f"Error writing to UC volume: {e}")
            # Don't fail the request if write fails - we still return the result
            # The caller can retry the write operation if needed
