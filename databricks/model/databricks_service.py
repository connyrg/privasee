"""
Databricks Vision Service
Uses Databricks Foundation Model with vision capabilities to extract entities from documents.
"""

from databricks_openai import DatabricksOpenAI, AsyncDatabricksOpenAI
import base64
import json
import logging
from typing import List, Dict

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
            List of extracted entities in canonical occurrences format:
            [
                {
                    "entity_type": "Full Name",
                    "original_text": "John Doe",
                    "confidence": 0.95,
                    "occurrences": [
                        {
                            "page_number": 1,
                            "original_text": "John Doe",
                            "bounding_boxes": [[x, y, width, height]]
                        }
                    ]
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

            # Build prompt for Databricks
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
                max_completion_tokens=16000
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
            List of extracted entities in canonical occurrences format:
            [
                {
                    "entity_type": "Full Name",
                    "original_text": "John Doe",
                    "confidence": 0.95,
                    "occurrences": [
                        {
                            "page_number": 1,
                            "original_text": "John Doe",
                            "bounding_boxes": [[x, y, width, height]]
                        }
                    ]
                }
            ]

        Raises:
            Exception: For API or processing errors
        """
        try:
            logger.info(f"Extracting entities using Databricks Vision for {len(field_definitions)} field types (page {page_number})")

            # Build prompt for Databricks
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
                max_completion_tokens=16000
            )

            # Parse response
            response_text = response.choices[0].message.content
            entities = self._parse_openai_response(response_text, ocr_data, page_number)

            logger.info(f"Successfully extracted {len(entities)} entities")
            return entities

        except Exception as e:
            logger.error(f"Error extracting entities with Databricks: {str(e)}", exc_info=True)
            raise

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
                max_completion_tokens=16000
            )

            response_text = response.choices[0].message.content
            entities = self._parse_openai_response(response_text, ocr_data, page_number)

            logger.info(f"Successfully extracted {len(entities)} entities (async, page {page_number})")
            return entities

        except Exception as e:
            logger.error(f"Error extracting entities with Databricks (async, page {page_number}): {str(e)}", exc_info=True)
            raise

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

```json
{ocr_context}
```

**Fields to Identify:**
{fields_text}

---

### Instructions

1. **Comprehensive Analysis**
   - Analyze both the document image and the OCR word list together.
   - Use positional relationships (adjacency, alignment) only when they are strong and unambiguous.

2. **Entity Detection — Names (Deduplication Logic)**
   - Identify ALL instances of the specified field types, including full names, partial names, and title-based names.

   **Critical rules:**
   - Extract each occurrence independently based on what actually appears in the document.
   - If a full name ("Stephen Parrot") appears as a single contiguous entity, extract it as **one entity only** — do NOT additionally extract "Stephen" or "Parrot" from that same occurrence.
   - Only extract partial names ("Stephen") if they appear **separately elsewhere** in the document as their own independent occurrence.
   - Avoid duplicate or overlapping entities referring to the same text span.
   - **Do NOT include standalone honorific/title prefixes (Mr, Mrs, Ms, Miss, Dr, Prof, etc.) as separate occurrences of a person's name.** When a name is split across form fields (e.g. Title="Mr", Surname="Potter", Given="Harry James"), only extract the surname and given name fields as occurrences — not the title field.

   **Examples:**
   - "Stephen Parrot" (single occurrence) → ✅ `"Stephen Parrot"` only
   - "Stephen Parrot" + later standalone "Parrot" → ✅ `"Stephen Parrot"` and `"Parrot"`
   - Form fields Title="Mr", Surname="Doe", Given="John" → ✅ occurrences for `"Doe"` and `"John"` only, NOT `"Mr"`

3. **Typo and Variation Grouping**
   - If the same entity appears multiple times with minor OCR typos or spelling variants (e.g. "Kranthi" and "Kranti"), group them under one top-level entity.
   - Use the most likely correct spelling as the top-level `original_text`.
   - Record the exact text as it appears at each location in the occurrence's `original_text`.

4. **What Counts as One Occurrence**
   A set of OCR words forms a single occurrence when they are in a continuous reading sequence with strong spatial evidence:
   - **Horizontal:** tokens are side-by-side with minimal gap on the same line.
   - **Line-break continuation:** tokens appear on consecutive lines, horizontally aligned or near-aligned, with a vertical gap consistent with a normal line break and no unrelated tokens between them.

   Do NOT group words into one occurrence when:
   - There is noticeable vertical separation beyond a typical line break.
   - Alignment is weak or inconsistent.
   - Tokens belong to separate labeled fields or columns.
   - There is ambiguity in grouping.

5. **Address Extraction (Strict)**
   - Only extract addresses that can pinpoint a specific location (must include street/number at minimum).
   - Do NOT extract postcodes, states, countries, or cities in isolation.

6. **Bounding Boxes**
   - For each occurrence, list the bounding box of **each individual OCR word token** that makes up the entity at that location.
   - Copy the exact `x`, `y`, `width`, `height` values from the words list above. Do NOT estimate or interpolate.
   - List every token separately — same-line tokens and line-break tokens alike. They will be merged automatically downstream.

7. **Confidence**
   - Score 0.0–1.0. Lower confidence when grouping across line breaks, merging typo variants, or uncertain grouping.

---

### Output Format

Return a JSON array, no other text:

```json
[
  {{
    "entity_type": "field name from definitions",
    "original_text": "canonical spelling of this entity",
    "confidence": 0.0-1.0,
    "occurrences": [
      {{
        "original_text": "exact text as it appears at this location",
        "bounding_boxes": [
          {{"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0}}
        ]
      }}
    ]
  }}
]
```

---

### Constraints
- Return ONLY the JSON array.
- One top-level entry per unique entity (or typo-variant group). Multiple appearances → multiple items in `occurrences`.
- Do NOT emit partial name entities if they are already part of a full-name occurrence at the same location.
- Copy bounding box values exactly from the words list. All coordinates normalized 0.0–1.0.
- Do NOT merge across columns, sections, or unrelated fields.

Begin analysis:"""

        return prompt

    def _merge_same_line_bboxes(self, bboxes: List[Dict], y_threshold: float = 0.01) -> List[Dict]:
        """Group word-level bboxes by line, merging each line into one rectangle."""
        if not bboxes:
            return []

        sorted_boxes = sorted(bboxes, key=lambda b: (b['y'], b['x']))

        lines = []
        current_line = [sorted_boxes[0]]
        for box in sorted_boxes[1:]:
            if abs(box['y'] - current_line[0]['y']) <= y_threshold:
                current_line.append(box)
            else:
                lines.append(current_line)
                current_line = [box]
        lines.append(current_line)

        merged = []
        for line in lines:
            x0 = min(b['x'] for b in line)
            y0 = min(b['y'] for b in line)
            x1 = max(b['x'] + b['width'] for b in line)
            y1 = max(b['y'] + b['height'] for b in line)
            merged.append({'x': x0, 'y': y0, 'width': x1 - x0, 'height': y1 - y0})

        return merged

    def _parse_openai_response(
        self,
        response_text: str,
        ocr_data: Dict,  # noqa: ARG002 — kept for API compatibility
        page_number: int = 1
    ) -> List[Dict]:
        """Parse Databricks JSON response into canonical occurrences-format entity list."""
        try:
            json_text = response_text.strip()

            if "```json" in json_text:
                start = json_text.find("```json") + 7
                end = json_text.find("```", start)
                json_text = json_text[start:end].strip()
            elif "```" in json_text:
                start = json_text.find("```") + 3
                end = json_text.find("```", start)
                json_text = json_text[start:end].strip()

            raw_entities = json.loads(json_text)
            if not isinstance(raw_entities, list):
                raise ValueError(f"Expected JSON array from Databricks, got {type(raw_entities).__name__}")

            validated_entities = []
            for entity in raw_entities:
                entity_type = entity.get('entity_type')
                canonical_text = entity.get('original_text')
                confidence = float(entity.get('confidence', 0.9))
                raw_occurrences = entity.get('occurrences', [])

                if not entity_type or not canonical_text:
                    logger.warning(f"Entity missing entity_type or original_text: {entity}")
                    continue

                if not raw_occurrences:
                    logger.warning(f"Entity '{canonical_text}' has no occurrences, skipping")
                    continue

                parsed_occurrences = []
                for occurrence in raw_occurrences:
                    occurrence_text = occurrence.get('original_text') or canonical_text
                    raw_bboxes = occurrence.get('bounding_boxes', [])
                    valid_bboxes = [
                        {'x': float(b['x']), 'y': float(b['y']), 'width': float(b['width']), 'height': float(b['height'])}
                        for b in raw_bboxes
                        if all(k in b for k in ('x', 'y', 'width', 'height'))
                    ]
                    if not valid_bboxes:
                        logger.warning(f"Occurrence of '{occurrence_text}' has no valid bounding boxes, skipping")
                        continue

                    merged = self._merge_same_line_bboxes(valid_bboxes)
                    bboxes_flat = [[b['x'], b['y'], b['width'], b['height']] for b in merged]
                    parsed_occurrences.append({
                        'page_number': page_number,
                        'original_text': occurrence_text,
                        'bounding_boxes': bboxes_flat,
                    })

                if parsed_occurrences:
                    validated_entities.append({
                        'entity_type': entity_type,
                        'original_text': canonical_text,
                        'confidence': confidence,
                        'occurrences': parsed_occurrences,
                    })

            return validated_entities

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Databricks response as JSON: {e}")
            logger.error(f"Full response text ({len(response_text)} chars): {response_text[:2000]}")
            if len(response_text) > 2000:
                logger.error(f"... (response truncated, total length: {len(response_text)} chars)")
            raise
        except Exception as e:
            logger.error(f"Error parsing Databricks response: {e}")
            logger.error(f"Response text: {response_text[:500]}")
            raise

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
