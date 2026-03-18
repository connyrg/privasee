"""
Claude Vision Service
Uses Claude AI with vision capabilities to extract entities from documents.
"""

import anthropic
import base64
import json
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class ClaudeVisionService:
    """Service for entity extraction using Claude Vision API."""

    def __init__(self, api_key: str):
        """
        Initialize Claude Vision service.

        Args:
            api_key: Anthropic API key
        """
        if not api_key:
            raise ValueError("Anthropic API key must be provided")

        self.client = anthropic.Anthropic(api_key=api_key)
        self.async_client = anthropic.AsyncAnthropic(api_key=api_key)
        logger.info("Claude Vision Service initialized")

    async def extract_entities(
        self,
        image_path: str,
        ocr_data: Dict,
        field_definitions: List[Dict],
        page_number: int = 1
    ) -> List[Dict]:
        """
        Extract entities from document using Claude Vision.

        Args:
            image_path: Path to document image
            ocr_data: OCR data from Azure (text, words, bounding boxes)
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
            logger.info(f"Extracting entities using Claude Vision for {len(field_definitions)} field types (page {page_number})")

            # Read and encode image
            with open(image_path, "rb") as f:
                image_data = base64.standard_b64encode(f.read()).decode("utf-8")

            # Build prompt for Claude
            prompt = self._build_extraction_prompt(field_definitions, ocr_data)

            # Call Claude API with vision
            message = self.client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=4096,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image_data,
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ],
                    }
                ],
            )

            # Parse response
            response_text = message.content[0].text
            entities = self._parse_claude_response(response_text, ocr_data, page_number)

            logger.info(f"Successfully extracted {len(entities)} entities")
            return entities

        except FileNotFoundError:
            logger.error(f"Image file not found: {image_path}")
            raise
        except Exception as e:
            logger.error(f"Error extracting entities with Claude: {str(e)}")
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
        Extract entities from base64-encoded image using Claude Vision.
        
        This method is designed for use with OCR services that return base64-encoded
        page images, avoiding the need to write temporary files.

        Args:
            image_b64: Base64-encoded image data
            mimetype: Image MIME type (e.g., 'png', 'jpeg')
            ocr_data: OCR data from Azure (text, words, bounding boxes)
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
            logger.info(f"Extracting entities using Claude Vision for {len(field_definitions)} field types (page {page_number})")

            # Build prompt for Claude
            prompt = self._build_extraction_prompt(field_definitions, ocr_data)

            # Call Claude API with vision
            message = self.client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=4096,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": f"image/{mimetype}",
                                    "data": image_b64,
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ],
                    }
                ],
            )

            # Parse response
            response_text = message.content[0].text
            entities = self._parse_claude_response(response_text, ocr_data, page_number)

            logger.info(f"Successfully extracted {len(entities)} entities")
            return entities

        except Exception as e:
            logger.error(f"Error extracting entities with Claude: {str(e)}")
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
            logger.info(f"Extracting entities using Claude Vision (async) for {len(field_definitions)} field types (page {page_number})")

            prompt = self._build_extraction_prompt(field_definitions, ocr_data)

            message = await self.async_client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=4096,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": f"image/{mimetype}",
                                    "data": image_b64,
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ],
                    }
                ],
            )

            response_text = message.content[0].text
            entities = self._parse_claude_response(response_text, ocr_data, page_number)

            logger.info(f"Successfully extracted {len(entities)} entities (async, page {page_number})")
            return entities

        except Exception as e:
            logger.error(f"Error extracting entities with Claude (async, page {page_number}): {str(e)}")
            return []

    def _build_extraction_prompt(
        self,
        field_definitions: List[Dict],
        ocr_data: Dict
    ) -> str:
        """
        Build prompt for Claude entity extraction.

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

        # Create OCR context — send all words so Claude can locate
        # entities anywhere on the page, not just the first ~50.
        all_words = ocr_data.get('words', [])
        ocr_context = json.dumps({
            'text': ocr_data.get('text', '')[:3000],
            'word_count': len(all_words),
            'words': all_words  # Full list with normalized 0-1 bounding boxes
        }, indent=2)

        prompt = f"""You are a document de-identification assistant. Your task is to identify sensitive information in documents that needs to be redacted or replaced.

**Document Context:**
The document has been processed with OCR. Here is the extracted text and structural information:

```json
{ocr_context}
```

**Fields to Identify:**
{fields_text}

**Instructions:**
1. Carefully analyze the document image and OCR data
2. Identify ALL instances of the specified field types
3. Handle variations, typos, and partial matches intelligently
   - Example: "Kranthi" and "Kranti" should be recognized as the same name
   - Example: "SSN" and "Social Security Number" refer to the same field type
4. For each identified entity, determine its bounding box coordinates
5. Match entities to OCR word bounding boxes when possible for accuracy

**Output Format:**
Return a JSON array with this exact structure (no additional text):

```json
[
  {{
    "entity_type": "field name from definitions",
    "original_text": "exact text found in document",
    "bounding_box": [x, y, width, height],
    "confidence": 0.0-1.0
  }}
]
```

**Bounding Box Format:**
- All coordinates are normalized 0.0–1.0 values (fraction of page width/height)
- x: left edge (0 = left margin, 1 = right margin)
- y: top edge (0 = top of page, 1 = bottom of page)
- width: box width as fraction of page width
- height: box height as fraction of page height
- Match the bounding_box values from the words list above as closely as possible

**Important:**
- Return ONLY the JSON array, no explanations
- Include all instances found (even if the same entity appears multiple times)
- Be thorough but precise
- If unsure about an entity, include it with lower confidence (>0.5)

Begin analysis:"""

        return prompt

    def _parse_claude_response(
        self,
        response_text: str,
        ocr_data: Dict,
        page_number: int = 1
    ) -> List[Dict]:
        """
        Parse Claude's JSON response into entity list.

        Args:
            response_text: Raw response from Claude
            ocr_data: Original OCR data for validation

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
            logger.error(f"Failed to parse Claude response as JSON: {e}")
            logger.debug(f"Response text: {response_text[:500]}")
            # Return empty list rather than failing completely
            return []
        except Exception as e:
            logger.error(f"Error parsing Claude response: {e}")
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
        Test Claude API connection.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Simple test message
            message = self.client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=10,
                messages=[
                    {
                        "role": "user",
                        "content": "Hello"
                    }
                ]
            )
            return True
        except Exception as e:
            logger.error(f"Claude API connection test failed: {e}")
            return False
