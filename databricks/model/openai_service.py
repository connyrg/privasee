"""
Azure OpenAI Vision Service
Uses Azure OpenAI GPT-4o with vision capabilities to extract entities from documents.
"""

from openai import AzureOpenAI
import base64
import json
import logging
import os
from typing import List, Dict, Optional

from ..utils.nginx_utils import http_client_factory

logger = logging.getLogger(__name__)


class OpenAIVisionService:
    """Service for entity extraction using Azure OpenAI Vision API."""

    def __init__(
        self,
        api_key: str,
        azure_endpoint: str,
        workspace_url: str = "https://suncorp-dev.cloud.databricks.com/",
        workspace_id: str = "1238531023703058",
        proxy_cluster_id: str = "",
        proxy_port: str = "8110",
        proxy_route: str = "openai-00010-1",
        api_version: str = "2024-02-15-preview",
        deployment_name: str = "gpt-4o"
    ):
        """
        Initialize Azure OpenAI Vision service.

        Args:
            api_key: Azure OpenAI API key
            azure_endpoint: Azure OpenAI endpoint (e.g., https://<resource>.openai.azure.com/)
            api_version: Azure OpenAI API version (default: 2024-02-15-preview)
            deployment_name: Name of the GPT-4o deployment (default: gpt-4o)
        """
        if not api_key:
            raise ValueError("Azure OpenAI API key must be provided")
        if not azure_endpoint:
            raise ValueError("Azure OpenAI endpoint must be provided")

        # Use provided proxy_cluster_id or fall back to environment variable
        if not proxy_cluster_id:
            proxy_cluster_id = os.environ.get("PROXY_CLUSTER_ID", "")
        
        if not proxy_cluster_id:
            raise ValueError("PROXY_CLUSTER_ID must be provided either as parameter or environment variable")

        base_url = f"{workspace_url}/driver-proxy-api/o/{workspace_id}/{proxy_cluster_id}/{proxy_port}/{proxy_route}/openai"

        http_client, http_async_client = http_client_factory(
            workspace_url,
        )

        self.deployment_name = deployment_name
        self.client = AzureOpenAI(
            api_key=api_key,
            # azure_endpoint=azure_endpoint,
            base_url=base_url,
            api_version=api_version,
            http_client=http_client,
        )
        logger.info(f"Azure OpenAI Vision Service initialized (deployment: {deployment_name})")

    async def extract_entities(
        self,
        image_path: str,
        ocr_data: Dict,
        field_definitions: List[Dict],
        page_number: int = 1
    ) -> List[Dict]:
        """
        Extract entities from document using Azure OpenAI Vision.

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
            logger.info(f"Extracting entities using Azure OpenAI Vision for {len(field_definitions)} field types (page {page_number})")

            # Read and encode image
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            # Build prompt for OpenAI
            prompt = self._build_extraction_prompt(field_definitions, ocr_data)

            # Call Azure OpenAI API with vision
            response = self.client.chat.completions.create(
                model=self.deployment_name,
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
                max_tokens=4096
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
            logger.error(f"Error extracting entities with Azure OpenAI: {str(e)}")
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
        Extract entities from base64-encoded image using Azure OpenAI Vision.
        
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
            logger.info(f"Extracting entities using Azure OpenAI Vision for {len(field_definitions)} field types (page {page_number})")

            # Build prompt for OpenAI
            prompt = self._build_extraction_prompt(field_definitions, ocr_data)

            # Call Azure OpenAI API with vision
            response = self.client.chat.completions.create(
                model=self.deployment_name,
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
            logger.error(f"Error extracting entities with Azure OpenAI: {str(e)}")
            return []

    def _build_extraction_prompt(
        self,
        field_definitions: List[Dict],
        ocr_data: Dict
    ) -> str:
        """
        Build prompt for Azure OpenAI entity extraction.

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

        # Create OCR context — send all words so Azure OpenAI can locate
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

    def _parse_openai_response(
        self,
        response_text: str,
        ocr_data: Dict,
        page_number: int = 1
    ) -> List[Dict]:
        """
        Parse Azure OpenAI's JSON response into entity list.

        Args:
            response_text: Raw response from Azure OpenAI
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
            logger.error(f"Failed to parse Azure OpenAI response as JSON: {e}")
            logger.debug(f"Response text: {response_text[:500]}")
            # Return empty list rather than failing completely
            return []
        except Exception as e:
            logger.error(f"Error parsing Azure OpenAI response: {e}")
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
        Test Azure OpenAI API connection.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Simple test message
            response = self.client.chat.completions.create(
                model=self.deployment_name,
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
            logger.error(f"Azure OpenAI API connection test failed: {e}")
            return False
