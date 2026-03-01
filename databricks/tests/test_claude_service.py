"""
Tests for Claude Vision Service

This test suite verifies:
1. Service initialization with API key validation
2. Entity extraction with Claude Vision API
3. Prompt building with OCR context
4. JSON response parsing (with/without markdown code fences)
5. Entity validation (required fields, bounding box format)
6. Error handling for file operations and API failures

All Anthropic API calls are mocked.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, mock_open
import json
import base64

# Mock the anthropic module before importing claude_service
import sys
sys.modules['anthropic'] = MagicMock()
sys.modules['anthropic.types'] = MagicMock()

from databricks.model.claude_service import ClaudeVisionService


class TestClaudeServiceInit(unittest.TestCase):
    """Test Claude Vision service initialization"""

    @patch('databricks.model.claude_service.anthropic.Anthropic')
    def test_init_success(self, mock_anthropic):
        """Test successful initialization with valid API key"""
        service = ClaudeVisionService(api_key="test-api-key-123")
        
        # Verify Anthropic client was created with the key
        mock_anthropic.assert_called_once_with(api_key="test-api-key-123")
        self.assertIsNotNone(service.client)

    def test_init_missing_api_key(self):
        """Test initialization fails with missing API key"""
        with self.assertRaises(ValueError) as context:
            ClaudeVisionService(api_key="")
        self.assertIn("API key must be provided", str(context.exception))

    def test_init_none_api_key(self):
        """Test initialization fails with None API key"""
        with self.assertRaises(ValueError) as context:
            ClaudeVisionService(api_key=None)
        self.assertIn("API key must be provided", str(context.exception))


class TestBuildExtractionPrompt(unittest.TestCase):
    """Test prompt building for Claude Vision"""

    @patch('databricks.model.claude_service.anthropic.Anthropic')
    def setUp(self, mock_anthropic):
        """Set up test service"""
        self.service = ClaudeVisionService(api_key="test-key")

    def test_prompt_includes_field_definitions(self):
        """Test prompt includes all field definitions"""
        field_definitions = [
            {"name": "Full Name", "description": "Person's full name"},
            {"name": "SSN", "description": "Social Security Number"}
        ]
        ocr_data = {"text": "Test document", "words": []}

        prompt = self.service._build_extraction_prompt(field_definitions, ocr_data)

        self.assertIn("Full Name", prompt)
        self.assertIn("Person's full name", prompt)
        self.assertIn("SSN", prompt)
        self.assertIn("Social Security Number", prompt)

    def test_prompt_includes_ocr_text(self):
        """Test prompt includes OCR text content"""
        field_definitions = [{"name": "Name", "description": "Person name"}]
        ocr_data = {
            "text": "John Doe lives at 123 Main St",
            "words": [
                {"text": "John", "bounding_box": {"x": 0.1, "y": 0.1, "width": 0.05, "height": 0.02}},
                {"text": "Doe", "bounding_box": {"x": 0.16, "y": 0.1, "width": 0.04, "height": 0.02}}
            ]
        }

        prompt = self.service._build_extraction_prompt(field_definitions, ocr_data)

        self.assertIn("John Doe lives at 123 Main St", prompt)
        self.assertIn("word_count", prompt)
        self.assertIn("words", prompt)

    def test_prompt_truncates_long_text(self):
        """Test prompt truncates OCR text over 3000 chars"""
        field_definitions = [{"name": "Name", "description": "Person name"}]
        long_text = "A" * 5000
        ocr_data = {"text": long_text, "words": []}

        prompt = self.service._build_extraction_prompt(field_definitions, ocr_data)

        # Should contain truncated text (first 3000 chars)
        self.assertIn("A" * 3000, prompt)
        # Should not contain text beyond 3000 chars
        self.assertNotIn("A" * 3001, prompt)

    def test_prompt_includes_all_words(self):
        """Test prompt includes full words list even if text is truncated"""
        field_definitions = [{"name": "Name", "description": "Person name"}]
        words = [{"text": f"word{i}", "bounding_box": {}} for i in range(100)]
        ocr_data = {"text": "A" * 5000, "words": words}

        prompt = self.service._build_extraction_prompt(field_definitions, ocr_data)

        # Word count should reflect actual count
        self.assertIn('"word_count": 100', prompt)
        # Should include words list
        self.assertIn('"words":', prompt)


class TestParseClaudeResponse(unittest.TestCase):
    """Test JSON response parsing from Claude"""

    @patch('databricks.model.claude_service.anthropic.Anthropic')
    def setUp(self, mock_anthropic):
        """Set up test service"""
        self.service = ClaudeVisionService(api_key="test-key")
        self.ocr_data = {"text": "Test", "words": []}

    def test_parse_clean_json(self):
        """Test parsing clean JSON response"""
        response = json.dumps([
            {
                "entity_type": "Full Name",
                "original_text": "John Doe",
                "bounding_box": [0.1, 0.2, 0.15, 0.03],
                "confidence": 0.95
            }
        ])

        entities = self.service._parse_claude_response(response, self.ocr_data, page_number=1)

        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0]['entity_type'], "Full Name")
        self.assertEqual(entities[0]['original_text'], "John Doe")
        self.assertEqual(entities[0]['bounding_box'], [0.1, 0.2, 0.15, 0.03])
        self.assertEqual(entities[0]['confidence'], 0.95)
        self.assertEqual(entities[0]['page_number'], 1)

    def test_parse_json_with_markdown_json_fence(self):
        """Test parsing JSON wrapped in ```json code fence"""
        response = """Here are the entities:
```json
[
    {
        "entity_type": "SSN",
        "original_text": "123-45-6789",
        "bounding_box": [0.5, 0.5, 0.1, 0.02],
        "confidence": 0.98
    }
]
```
Hope this helps!"""

        entities = self.service._parse_claude_response(response, self.ocr_data, page_number=2)

        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0]['entity_type'], "SSN")
        self.assertEqual(entities[0]['original_text'], "123-45-6789")
        self.assertEqual(entities[0]['page_number'], 2)

    def test_parse_json_with_generic_fence(self):
        """Test parsing JSON wrapped in generic ``` code fence"""
        response = """```
[
    {
        "entity_type": "Address",
        "original_text": "123 Main St",
        "bounding_box": [0.2, 0.3, 0.2, 0.04],
        "confidence": 0.9
    }
]
```"""

        entities = self.service._parse_claude_response(response, self.ocr_data, page_number=1)

        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0]['entity_type'], "Address")

    def test_parse_multiple_entities(self):
        """Test parsing response with multiple entities"""
        response = json.dumps([
            {
                "entity_type": "Name",
                "original_text": "Alice Smith",
                "bounding_box": [0.1, 0.1, 0.1, 0.02],
                "confidence": 0.95
            },
            {
                "entity_type": "Name",
                "original_text": "Bob Jones",
                "bounding_box": [0.1, 0.2, 0.1, 0.02],
                "confidence": 0.93
            }
        ])

        entities = self.service._parse_claude_response(response, self.ocr_data, page_number=1)

        self.assertEqual(len(entities), 2)
        self.assertEqual(entities[0]['original_text'], "Alice Smith")
        self.assertEqual(entities[1]['original_text'], "Bob Jones")

    def test_parse_invalid_json(self):
        """Test parsing invalid JSON returns empty list"""
        response = "This is not valid JSON {[}]"

        entities = self.service._parse_claude_response(response, self.ocr_data, page_number=1)

        self.assertEqual(len(entities), 0)

    def test_parse_empty_array(self):
        """Test parsing empty entity array"""
        response = json.dumps([])

        entities = self.service._parse_claude_response(response, self.ocr_data, page_number=1)

        self.assertEqual(len(entities), 0)

    def test_parse_filters_invalid_entities(self):
        """Test parsing filters out entities that fail validation"""
        response = json.dumps([
            {
                "entity_type": "Valid",
                "original_text": "Good Entity",
                "bounding_box": [0.1, 0.1, 0.1, 0.02],
                "confidence": 0.9
            },
            {
                "entity_type": "Invalid",
                "original_text": "Missing bbox"
                # Missing bounding_box
            },
            {
                "entity_type": "BadBox",
                "original_text": "Bad bbox",
                "bounding_box": [0.1, 0.2]  # Only 2 values instead of 4
            }
        ])

        entities = self.service._parse_claude_response(response, self.ocr_data, page_number=1)

        # Only the valid entity should be returned
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0]['entity_type'], "Valid")

    def test_parse_normalizes_bbox_to_floats(self):
        """Test bounding box values are converted to floats"""
        response = json.dumps([
            {
                "entity_type": "Name",
                "original_text": "Test",
                "bounding_box": ["0.1", "0.2", "0.3", "0.4"],  # Strings
                "confidence": "0.95"  # String
            }
        ])

        entities = self.service._parse_claude_response(response, self.ocr_data, page_number=1)

        self.assertEqual(len(entities), 1)
        # Should be converted to floats
        self.assertEqual(entities[0]['bounding_box'], [0.1, 0.2, 0.3, 0.4])
        self.assertEqual(entities[0]['confidence'], 0.95)

    def test_parse_default_confidence(self):
        """Test missing confidence defaults to 0.9"""
        response = json.dumps([
            {
                "entity_type": "Name",
                "original_text": "Test",
                "bounding_box": [0.1, 0.2, 0.3, 0.4]
                # No confidence
            }
        ])

        entities = self.service._parse_claude_response(response, self.ocr_data, page_number=1)

        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0]['confidence'], 0.9)


class TestValidateEntity(unittest.TestCase):
    """Test entity validation"""

    @patch('databricks.model.claude_service.anthropic.Anthropic')
    def setUp(self, mock_anthropic):
        """Set up test service"""
        self.service = ClaudeVisionService(api_key="test-key")

    def test_validate_complete_entity(self):
        """Test validation passes for complete entity"""
        entity = {
            "entity_type": "Name",
            "original_text": "John Doe",
            "bounding_box": [0.1, 0.2, 0.3, 0.4]
        }

        self.assertTrue(self.service._validate_entity(entity))

    def test_validate_missing_entity_type(self):
        """Test validation fails for missing entity_type"""
        entity = {
            "original_text": "John Doe",
            "bounding_box": [0.1, 0.2, 0.3, 0.4]
        }

        self.assertFalse(self.service._validate_entity(entity))

    def test_validate_missing_original_text(self):
        """Test validation fails for missing original_text"""
        entity = {
            "entity_type": "Name",
            "bounding_box": [0.1, 0.2, 0.3, 0.4]
        }

        self.assertFalse(self.service._validate_entity(entity))

    def test_validate_missing_bounding_box(self):
        """Test validation fails for missing bounding_box"""
        entity = {
            "entity_type": "Name",
            "original_text": "John Doe"
        }

        self.assertFalse(self.service._validate_entity(entity))

    def test_validate_invalid_bbox_format(self):
        """Test validation fails for invalid bounding box format"""
        # Not a list
        entity1 = {
            "entity_type": "Name",
            "original_text": "John",
            "bounding_box": "0.1,0.2,0.3,0.4"
        }
        self.assertFalse(self.service._validate_entity(entity1))

        # Wrong length
        entity2 = {
            "entity_type": "Name",
            "original_text": "John",
            "bounding_box": [0.1, 0.2, 0.3]
        }
        self.assertFalse(self.service._validate_entity(entity2))

        # Too many values
        entity3 = {
            "entity_type": "Name",
            "original_text": "John",
            "bounding_box": [0.1, 0.2, 0.3, 0.4, 0.5]
        }
        self.assertFalse(self.service._validate_entity(entity3))

    def test_validate_empty_bbox(self):
        """Test validation fails for empty bounding box"""
        entity = {
            "entity_type": "Name",
            "original_text": "John",
            "bounding_box": []
        }

        self.assertFalse(self.service._validate_entity(entity))


class TestExtractEntities(unittest.TestCase):
    """Test entity extraction end-to-end"""

    @patch('databricks.model.claude_service.anthropic.Anthropic')
    def setUp(self, mock_anthropic):
        """Set up test service and mocks"""
        self.service = ClaudeVisionService(api_key="test-key")
        self.mock_client = Mock()
        self.service.client = self.mock_client

    @patch('builtins.open', new_callable=mock_open, read_data=b'fake_image_data')
    @patch('databricks.model.claude_service.base64.standard_b64encode')
    def test_extract_entities_success(self, mock_b64encode, mock_file):
        """Test successful entity extraction"""
        # Mock base64 encoding
        mock_b64encode.return_value = b'encoded_image_data'

        # Mock Claude API response
        mock_message = Mock()
        mock_content = Mock()
        mock_content.text = json.dumps([
            {
                "entity_type": "Full Name",
                "original_text": "John Doe",
                "bounding_box": [0.1, 0.2, 0.15, 0.03],
                "confidence": 0.95
            }
        ])
        mock_message.content = [mock_content]
        self.mock_client.messages.create.return_value = mock_message

        # Test data
        image_path = "/fake/path/image.png"
        ocr_data = {"text": "John Doe", "words": []}
        field_definitions = [
            {"name": "Full Name", "description": "Person's full name"}
        ]

        # Call extract_entities (note: it's async but we're testing synchronously)
        # We need to handle the async call
        import asyncio
        entities = asyncio.run(
            self.service.extract_entities(image_path, ocr_data, field_definitions, page_number=1)
        )

        # Verify file was opened
        mock_file.assert_called_once_with(image_path, "rb")

        # Verify Claude API was called
        self.mock_client.messages.create.assert_called_once()
        call_args = self.mock_client.messages.create.call_args

        # Verify model
        self.assertEqual(call_args.kwargs['model'], 'claude-sonnet-4-6')

        # Verify message content includes image and text
        messages = call_args.kwargs['messages']
        self.assertEqual(len(messages), 1)
        content = messages[0]['content']
        self.assertEqual(len(content), 2)
        self.assertEqual(content[0]['type'], 'image')
        self.assertEqual(content[1]['type'], 'text')

        # Verify extracted entities
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0]['entity_type'], "Full Name")
        self.assertEqual(entities[0]['original_text'], "John Doe")
        self.assertEqual(entities[0]['page_number'], 1)

    @patch('builtins.open', side_effect=FileNotFoundError("File not found"))
    def test_extract_entities_file_not_found(self, mock_file):
        """Test extraction fails gracefully when image file not found"""
        image_path = "/nonexistent/image.png"
        ocr_data = {"text": "Test", "words": []}
        field_definitions = [{"name": "Name", "description": "Name"}]

        import asyncio
        with self.assertRaises(FileNotFoundError):
            asyncio.run(
                self.service.extract_entities(image_path, ocr_data, field_definitions)
            )

    @patch('builtins.open', new_callable=mock_open, read_data=b'fake_image_data')
    @patch('databricks.model.claude_service.base64.standard_b64encode')
    def test_extract_entities_api_error(self, mock_b64encode, mock_file):
        """Test extraction handles API errors"""
        mock_b64encode.return_value = b'encoded_image_data'

        # Mock API error
        self.mock_client.messages.create.side_effect = Exception("API Error")

        image_path = "/fake/path/image.png"
        ocr_data = {"text": "Test", "words": []}
        field_definitions = [{"name": "Name", "description": "Name"}]

        import asyncio
        with self.assertRaises(Exception) as context:
            asyncio.run(
                self.service.extract_entities(image_path, ocr_data, field_definitions)
            )

        self.assertIn("Entity extraction failed", str(context.exception))


class TestConnectionTest(unittest.TestCase):
    """Test Claude API connection testing"""

    @patch('databricks.model.claude_service.anthropic.Anthropic')
    def setUp(self, mock_anthropic):
        """Set up test service"""
        self.service = ClaudeVisionService(api_key="test-key")
        self.mock_client = Mock()
        self.service.client = self.mock_client

    def test_connection_success(self):
        """Test successful connection test"""
        mock_message = Mock()
        self.mock_client.messages.create.return_value = mock_message

        result = self.service.test_connection()

        self.assertTrue(result)
        self.mock_client.messages.create.assert_called_once()

    def test_connection_failure(self):
        """Test connection test handles failures"""
        self.mock_client.messages.create.side_effect = Exception("Connection failed")

        result = self.service.test_connection()

        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()
