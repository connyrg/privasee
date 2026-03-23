"""
Databricks Vision Service
Uses Databricks Foundation Model with vision capabilities to extract entities from documents.
"""

from databricks_openai import DatabricksOpenAI, AsyncDatabricksOpenAI
import base64
import json
import logging
import os
from typing import List, Dict, Optional

from ..utils.nginx_utils import http_client_factory

logger = logging.getLogger(__name__)


class DatabricksVisionService:
    """Service for entity extraction using Databricks Vision API."""

    def __init__(
        self,
        model_name: str = "databricks-claude-haiku-4-5",
    ):
        """
        Initialize Databricks Vision service.

        Args:
            model_name: Name of the Databricks foundation model endpoint (default: databricks-claude-haiku-4-5)
        """
        self.model_name = model_name

        self.client = DatabricksOpenAI()
        self.async_client = AsyncDatabricksOpenAI()

        logger.info(f"Databricks Vision Service initialized (model: {model_name})")

    async def extract_entities(
        self,
        image_path: str,
        ocr_data: Dict,
        field_definitions: List[Dict],
        page_number: int = 1
    ) -> List[Dict]:
        """
        Extract entities from document using Databricks Vision.

        Args:
            image_path: Path to document image
            ocr_data: OCR data from Azure Document Intelligence (text, words, bounding boxes)
            field_definitions: List of field definitions to extract
                [{"name": "Full Name", "description": "...", "strategy": "..."}]
            page_number: Page number this image corresponds to (1-indexed)

        Returns:
            List of extracted entities with bounding boxes:
            [
                {
                    "entity_type": "Full Name",
                    "original_text": "John Doe",
                    "bounding_box": [x, y, width, height],
                    "confidence": 0.95,
                    "page_number": 1
                }
            ]

        Raises:
            FileNotFoundError: If image file doesn't exist
            Exception: For API or processing errors
        """
        try:
            logger.info(f"Extracting entities using Databricks Vision for {len(field_definitions)} field types (page {page_number})")

            # Read and encode image
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            # Build prompt for OpenAI
            prompt = self._build_extraction_prompt(field_definitions, ocr_data)

            # Call Databricks API with vision
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_data}"
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                max_tokens=20000
            )

            # Parse response
            response_text = response.choices[0].message.content
            entities = self._parse_openai_response(response_text, ocr_data, page_number)

            logger.info(f"Successfully extracted {len(entities)} entities")
            return entities

        except FileNotFoundError:
            logger.error(f"Image file not found: {image_path}")
            raise
        except Exception as e:
            logger.error(f"Error extracting entities with Databricks: {str(e)}")
            raise Exception(f"Entity extraction failed: {str(e)}")

    def extract_entities_from_base64(
        self,
        image_b64: str,
        mimetype: str,
        ocr_data: Dict,
        field_definitions: List[Dict],
        page_number: int = 1
    ) -> List[Dict]:
        """
        Extract entities from base64-encoded image using Databricks Vision.
        
        This method is designed for use with OCR services that return base64-encoded
        page images, avoiding the need to write temporary files.

        Args:
            image_b64: Base64-encoded image data
            mimetype: Image MIME type (e.g., 'png', 'jpeg')
            ocr_data: OCR data from Azure Document Intelligence (text, words, bounding boxes)
            field_definitions: List of field definitions to extract
                [{"name": "Full Name", "description": "...", "strategy": "..."}]
            page_number: Page number this image corresponds to (1-indexed)

        Returns:
            List of extracted entities with bounding boxes:
            [
                {
                    "entity_type": "Full Name",
                    "original_text": "John Doe",
                    "bounding_box": [x, y, width, height],
                    "confidence": 0.95,
                    "page_number": 1
                }
            ]

        Raises:
            Exception: For API or processing errors
        """
        try:
            logger.info(f"Extracting entities using Databricks Vision for {len(field_definitions)} field types (page {page_number})")

            # Build prompt for OpenAI
            prompt = self._build_extraction_prompt(field_definitions, ocr_data)

            # Call Databricks API with vision
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/{mimetype};base64,{image_b64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                max_completion_tokens=4096
            )

            # Parse response
            response_text = response.choices[0].message.content
            entities = self._parse_openai_response(response_text, ocr_data, page_number)

            logger.info(f"Successfully extracted {len(entities)} entities")
            return entities

        except Exception as e:
            logger.error(f"Error extracting entities with Databricks: {str(e)}")
            return []

    async def extract_entities_from_base64_async(
        self,
        image_b64: str,
        mimetype: str,
        ocr_data: Dict,
        field_definitions: List[Dict],
        page_number: int = 1
    ) -> List[Dict]:
        """Async version of extract_entities_from_base64 for concurrent page processing."""
        try:
            logger.info(f"Extracting entities using Databricks Vision (async) for {len(field_definitions)} field types (page {page_number})")

            prompt = self._build_extraction_prompt(field_definitions, ocr_data)

            response = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/{mimetype};base64,{image_b64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                max_completion_tokens=4096
            )

            response_text = response.choices[0].message.content
            entities = self._parse_openai_response(response_text, ocr_data, page_number)

            logger.info(f"Successfully extracted {len(entities)} entities (async, page {page_number})")
            return entities

        except Exception as e:
            logger.error(f"Error extracting entities with Databricks (async, page {page_number}): {str(e)}", exc_info=True)
            return []

    def _build_extraction_prompt(
        self,
        field_definitions: List[Dict],
        ocr_data: Dict
    ) -> str:
        """
        Build prompt for Databricks entity extraction.

        Args:
            field_definitions: List of fields to extract
            ocr_data: OCR data with text and bounding boxes

        Returns:
            Formatted prompt string
        """
        # Format field definitions
        fields_text = "\n".join([
            f"- **{field['name']}**: {field['description']}"
            for field in field_definitions
        ])

        # Create OCR context — send all words so Databricks can locate
        # entities anywhere on the page, not just the first ~50.
        all_words = ocr_data.get('words', [])
        ocr_context = json.dumps({
            'text': ocr_data.get('text', '')[:3000],
            'word_count': len(all_words),
            'words': all_words  # Full list with normalized 0-1 bounding boxes
        }, indent=2)

        prompt = f"""You are a document de-identification assistant. Your task is to identify sensitive information in documents that needs to be redacted or replaced.

**Document Context:**
The document has been processed with OCR. Below is the extracted text and structural information:

```json id="9hf3lt"
{ocr_context}
```

**Fields to Identify:**
{fields_text}

---

### **Instructions:**

1. **Comprehensive Analysis**

   * Analyze both textual content and spatial layout from OCR data.
   * Use positional relationships (adjacency, alignment) only when they are strong and unambiguous.

2. **Entity Detection (Names — Deduplication Logic)**

   * Identify ALL instances of the specified field types, including:

     * Full names (e.g., "John Doe")
     * Partial names (e.g., "John", "Doe")
     * Title-based names (e.g., "Mr Doe", "Dr Smith")

   **Critical Rules:**

   * Extract each occurrence independently based on what is actually present in the document.
   * If a full name (e.g., "John Doe") appears as a single contiguous entity:

     * Extract it as **one entity only**
     * Do NOT additionally extract "John" or "Doe" from the same occurrence
   * Only extract partial names ("John", "Doe") if they appear **separately elsewhere** in the document as their own occurrences
   * Avoid duplicate or overlapping entities referring to the same exact text span

   **Examples:**

   * "John Doe" (single occurrence) → ✅ `"John Doe"` only
   * "John Doe" + later "Doe" → ✅ `"John Doe"` and `"Doe"`
   * "Mr Doe" → ✅ `"Mr Doe"` (not `"Doe"` unless it appears separately)

3. **Variation Handling**

   * Handle typos, abbreviations, and semantic equivalents.

4. **Strict Multi-Box Entity Reconstruction (Controlled Adjacency)**

   * Merge tokens only when they form a **continuous reading sequence** with strong spatial evidence.

   **Allowed Merging Scenarios:**

   * Horizontal concatenation (adjacent boxes)
   * Line-break continuation with:

     * Strong horizontal alignment
     * Minimal vertical gap
     * No intervening unrelated tokens

   **Applies to:**

   * Structured fields (dates, IDs, phone numbers)
   * Names (e.g., "John" + "Doe", including across line breaks if clearly continuous)

   **Rules:**

   * Concatenate with natural spacing
   * Compute a **minimum enclosing bounding box**
   * Do NOT merge if ambiguous

5. **Address Extraction Rules (Strict)**

   * Only extract addresses that can pinpoint a specific location.
   * Merge only when forming a clear, continuous address (single line or tight multi-line block)

   **Do NOT extract:**

   * Postcode alone
   * State/region alone
   * Country alone
   * City alone
   * Incomplete fragments

6. **Bounding Box Assignment**

   * Use OCR word-level bounding boxes when possible.
   * For merged entities:

     * Return a single bounding box enclosing all contributing boxes.
   * Ensure normalized coordinates (0.0–1.0).

7. **Confidence Scoring**

   * Assign a confidence score (0.0–1.0).
   * Lower confidence when merging across lines or uncertain grouping.

---

### **Output Format:**

Return a JSON array with this exact structure (no additional text):

```json id="3vu8ye"
[
  {{
    "entity_type": "field name from definitions",
    "original_text": "exact text (merged only if valid continuous entity)",
    "bounding_box": [x, y, width, height],
    "confidence": 0.0-1.0
  }}
]
```

---

### **Important Constraints:**

* Return ONLY the JSON array
* Do NOT include explanations
* Include all valid instances, but avoid duplicates from the same text span
* Do NOT emit partial name entities if they are already part of a single extracted full-name occurrence
* Only include partial names if they appear independently elsewhere
* Merge tokens only when forming a clear continuous entity (including valid line-break continuation)
* Ignore incomplete or non-specific address fragments

---

**Begin analysis:**
"""

        return prompt

    def _parse_openai_response(
        self,
        response_text: str,
        ocr_data: Dict,
        page_number: int = 1
    ) -> List[Dict]:
        """
        Parse Databricks's JSON response into entity list.

        Args:
            response_text: Raw response from Databricks
            ocr_data: Original OCR data for validation
            page_number: Page number for entities

        Returns:
            List of validated entity dictionaries
        """
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_text = response_text.strip()

            if "```json" in json_text:
                # Extract from code block
                start = json_text.find("```json") + 7
                end = json_text.find("```", start)
                json_text = json_text[start:end].strip()
            elif "```" in json_text:
                # Generic code block
                start = json_text.find("```") + 3
                end = json_text.find("```", start)
                json_text = json_text[start:end].strip()

            # Parse JSON
            entities = json.loads(json_text)

            # Validate and normalize
            validated_entities = []
            for entity in entities:
                if self._validate_entity(entity):
                    # Ensure bounding box is in correct format
                    bbox = entity.get('bounding_box', [0, 0, 0, 0])
                    if len(bbox) == 4:
                        validated_entities.append({
                            'entity_type': entity['entity_type'],
                            'original_text': entity['original_text'],
                            'bounding_box': [float(x) for x in bbox],
                            'confidence': float(entity.get('confidence', 0.9)),
                            'page_number': page_number
                        })

            return validated_entities

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Databricks response as JSON: {e}")
            # IMPROVED: Log full response for debugging (up to 2000 chars)
            logger.error(f"Full response text ({len(response_text)} chars): {response_text[:2000]}")
            if len(response_text) > 2000:
                logger.error(f"... (response truncated, total length: {len(response_text)} chars)")
            # Return empty list rather than failing completely
            return []
        except Exception as e:
            logger.error(f"Error parsing Databricks response: {e}")
            logger.error(f"Response text: {response_text[:500]}")
            return []

    def _validate_entity(self, entity: Dict) -> bool:
        """
        Validate entity has required fields.

        Args:
            entity: Entity dictionary to validate

        Returns:
            True if valid, False otherwise
        """
        required_fields = ['entity_type', 'original_text', 'bounding_box']

        for field in required_fields:
            if field not in entity:
                logger.warning(f"Entity missing required field: {field}")
                return False

        bbox = entity.get('bounding_box')
        if not isinstance(bbox, list) or len(bbox) != 4:
            logger.warning(f"Invalid bounding box format: {bbox}")
            return False

        return True

    def test_connection(self) -> bool:
        """
        Test Databricks API connection.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Simple test message
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": "Hello"
                    }
                ],
                max_tokens=10
            )
            return True
        except Exception as e:
            logger.error(f"Databricks API connection test failed: {e}")
            return False
