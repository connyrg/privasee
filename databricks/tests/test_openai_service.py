"""
Tests for Azure OpenAI Vision Service

This test suite verifies:
1. Service initialization with API key and endpoint validation
2. Entity extraction with Azure OpenAI Vision API
3. Prompt building with OCR context
4. JSON response parsing (with/without markdown code fences)
5. Entity validation (required fields, bounding box format)
6. Error handling for file operations and API failures

All Azure OpenAI API calls are mocked.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, mock_open
import json
import base64
import os

# Mock the openai module before importing openai_service
import sys
sys.modules['openai'] = MagicMock()
sys.modules['openai.types'] = MagicMock()
sys.modules['openai.types.chat'] = MagicMock()

from databricks.model.openai_service import OpenAIVisionService


class TestOpenAIServiceInit(unittest.TestCase):
    """Test Azure OpenAI Vision service initialization"""

    @patch('databricks.model.openai_service.http_client_factory')
    @patch('databricks.model.openai_service.AzureOpenAI')
    def test_init_success(self, mock_azure_openai, mock_http_client_factory):
        """Test successful initialization with valid API key and endpoint"""
        # Mock http_client_factory return values
        mock_http_client = Mock()
        mock_http_async_client = Mock()
        mock_http_client_factory.return_value = (mock_http_client, mock_http_async_client)
        
        # Mock environment variable for proxy_cluster_id
        with patch.dict(os.environ, {'PROXY_CLUSTER_ID': 'test-cluster-123'}):
            service = OpenAIVisionService(
                api_key="test-api-key-123",
                azure_endpoint="https://test.openai.azure.com/",
                api_version="2024-02-15-preview",
                deployment_name="test-deployment"
            )
        
        # Verify http_client_factory was called
        mock_http_client_factory.assert_called_once_with(
            "https://suncorp-dev.cloud.databricks.com/"
        )
        
        # Verify AzureOpenAI client was created with correct parameters
        call_kwargs = mock_azure_openai.call_args.kwargs
        self.assertEqual(call_kwargs['api_key'], "test-api-key-123")
        self.assertEqual(call_kwargs['api_version'], "2024-02-15-preview")
        self.assertIn('base_url', call_kwargs)
        # Base URL should end with /openai (not /openai/v1)
        self.assertTrue(call_kwargs['base_url'].endswith('/openai'), 
                       f"Expected base_url to end with '/openai', got: {call_kwargs['base_url']}")
        self.assertIn('test-cluster-123', call_kwargs['base_url'])
        self.assertEqual(call_kwargs['http_client'], mock_http_client)
        # Note: http_async_client is not passed to AzureOpenAI in the current implementation
        
        self.assertIsNotNone(service.client)
        self.assertEqual(service.deployment_name, "test-deployment")

    def test_init_missing_api_key(self):
        """Test initialization fails with missing API key"""
        with self.assertRaises(ValueError) as context:
            OpenAIVisionService(
                api_key="",
                azure_endpoint="https://test.openai.azure.com/"
            )
        self.assertIn("API key must be provided", str(context.exception))

    def test_init_none_api_key(self):
        """Test initialization fails with None API key"""
        with self.assertRaises(ValueError) as context:
            OpenAIVisionService(
                api_key=None,
                azure_endpoint="https://test.openai.azure.com/"
            )
        self.assertIn("API key must be provided", str(context.exception))

    def test_init_missing_endpoint(self):
        """Test initialization fails with missing endpoint"""
        with self.assertRaises(ValueError) as context:
            OpenAIVisionService(
                api_key="test-key",
                azure_endpoint=""
            )
        self.assertIn("Azure OpenAI endpoint must be provided", str(context.exception))

    @patch('databricks.model.openai_service.http_client_factory')
    @patch('databricks.model.openai_service.AzureOpenAI')
    def test_init_with_proxy_cluster_id_parameter(self, mock_azure_openai, mock_http_client_factory):
        """Test initialization works when proxy_cluster_id is passed as parameter"""
        mock_http_client_factory.return_value = (Mock(), Mock())
        
        # Don't set PROXY_CLUSTER_ID in environment, pass it as parameter instead
        service = OpenAIVisionService(
            api_key="test-api-key",
            azure_endpoint="https://test.openai.azure.com/",
            proxy_cluster_id="parameter-cluster-123"
        )
        
        # Verify base_url includes the cluster ID from parameter
        call_kwargs = mock_azure_openai.call_args.kwargs
        self.assertIn('parameter-cluster-123', call_kwargs['base_url'])
    
    @patch('databricks.model.openai_service.http_client_factory')
    @patch('databricks.model.openai_service.AzureOpenAI')
    def test_init_missing_proxy_cluster_id(self, mock_azure_openai, mock_http_client_factory):
        """Test initialization fails when proxy_cluster_id is not provided via parameter or environment"""
        mock_http_client_factory.return_value = (Mock(), Mock())
        
        # Don't provide proxy_cluster_id parameter and clear from environment
        with patch.dict(os.environ, {'AZURE_OPENAI_API_KEY': 'test-key'}, clear=True):
            with self.assertRaises(ValueError) as context:
                OpenAIVisionService(
                    api_key="test-key",
                    azure_endpoint="https://test.openai.azure.com/"
                )
            self.assertIn("PROXY_CLUSTER_ID", str(context.exception))


class TestBuildExtractionPrompt(unittest.TestCase):
    """Test prompt building for Azure OpenAI Vision"""

    @patch('databricks.model.openai_service.http_client_factory')
    @patch('databricks.model.openai_service.AzureOpenAI')
    def setUp(self, mock_azure_openai, mock_http_client_factory):
        """Set up test service"""
        mock_http_client_factory.return_value = (Mock(), Mock())
        
        with patch.dict(os.environ, {'PROXY_CLUSTER_ID': 'test-cluster'}):
            self.service = OpenAIVisionService(
                api_key="test-key",
                azure_endpoint="https://test.openai.azure.com/"
            )

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


class TestParseOpenAIResponse(unittest.TestCase):
    """Test JSON response parsing from Azure OpenAI"""

    @patch('databricks.model.openai_service.http_client_factory')
    @patch('databricks.model.openai_service.AzureOpenAI')
    def setUp(self, mock_azure_openai, mock_http_client_factory):
        """Set up test service"""
        mock_http_client_factory.return_value = (Mock(), Mock())
        
        with patch.dict(os.environ, {'PROXY_CLUSTER_ID': 'test-cluster'}):
            self.service = OpenAIVisionService(
                api_key="test-key",
                azure_endpoint="https://test.openai.azure.com/"
            )
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

        entities = self.service._parse_openai_response(response, self.ocr_data, page_number=1)

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

        entities = self.service._parse_openai_response(response, self.ocr_data, page_number=2)

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

        entities = self.service._parse_openai_response(response, self.ocr_data, page_number=1)

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

        entities = self.service._parse_openai_response(response, self.ocr_data, page_number=1)

        self.assertEqual(len(entities), 2)
        self.assertEqual(entities[0]['original_text'], "Alice Smith")
        self.assertEqual(entities[1]['original_text'], "Bob Jones")

    def test_parse_invalid_json(self):
        """Test parsing invalid JSON returns empty list"""
        response = "This is not valid JSON {[}]"

        entities = self.service._parse_openai_response(response, self.ocr_data, page_number=1)

        self.assertEqual(len(entities), 0)

    def test_parse_empty_array(self):
        """Test parsing empty entity array"""
        response = json.dumps([])

        entities = self.service._parse_openai_response(response, self.ocr_data, page_number=1)

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

        entities = self.service._parse_openai_response(response, self.ocr_data, page_number=1)

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

        entities = self.service._parse_openai_response(response, self.ocr_data, page_number=1)

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

        entities = self.service._parse_openai_response(response, self.ocr_data, page_number=1)

        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0]['confidence'], 0.9)


class TestValidateEntity(unittest.TestCase):
    """Test entity validation"""

    @patch('databricks.model.openai_service.http_client_factory')
    @patch('databricks.model.openai_service.AzureOpenAI')
    def setUp(self, mock_azure_openai, mock_http_client_factory):
        """Set up test service"""
        mock_http_client_factory.return_value = (Mock(), Mock())
        
        with patch.dict(os.environ, {'PROXY_CLUSTER_ID': 'test-cluster'}):
            self.service = OpenAIVisionService(
                api_key="test-key",
                azure_endpoint="https://test.openai.azure.com/"
            )

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
    """Test entity extraction end-to-end with Azure OpenAI"""

    @patch('databricks.model.openai_service.http_client_factory')
    @patch('databricks.model.openai_service.AzureOpenAI')
    def setUp(self, mock_azure_openai, mock_http_client_factory):
        """Set up test service and mocks"""
        mock_http_client_factory.return_value = (Mock(), Mock())
        
        with patch.dict(os.environ, {'PROXY_CLUSTER_ID': 'test-cluster'}):
            self.service = OpenAIVisionService(
                api_key="test-key",
                azure_endpoint="https://test.openai.azure.com/",
                deployment_name="test-deployment"
            )
        self.mock_client = Mock()
        self.service.client = self.mock_client

    @patch('builtins.open', new_callable=mock_open, read_data=b'fake_image_data')
    @patch('databricks.model.openai_service.base64.b64encode')
    def test_extract_entities_success(self, mock_b64encode, mock_file):
        """Test successful entity extraction"""
        # Mock base64 encoding
        mock_b64encode.return_value = b'encoded_image_data'

        # Mock OpenAI API response
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = json.dumps([
            {
                "entity_type": "Full Name",
                "original_text": "John Doe",
                "bounding_box": [0.1, 0.2, 0.15, 0.03],
                "confidence": 0.95
            }
        ])
        mock_choice.message = mock_message
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        self.mock_client.chat.completions.create.return_value = mock_response

        # Test data
        image_path = "/fake/path/image.png"
        ocr_data = {"text": "John Doe", "words": []}
        field_definitions = [
            {"name": "Full Name", "description": "Person's full name"}
        ]

        # Call extract_entities (note: it's async but we're testing synchronously)
        import asyncio
        entities = asyncio.run(
            self.service.extract_entities(image_path, ocr_data, field_definitions, page_number=1)
        )

        # Verify file was opened
        mock_file.assert_called_once_with(image_path, "rb")

        # Verify Azure OpenAI API was called
        self.mock_client.chat.completions.create.assert_called_once()
        call_args = self.mock_client.chat.completions.create.call_args

        # Verify deployment name
        self.assertEqual(call_args.kwargs['model'], 'test-deployment')

        # Verify message content includes image and text
        messages = call_args.kwargs['messages']
        self.assertEqual(len(messages), 1)
        content = messages[0]['content']
        self.assertEqual(len(content), 2)
        self.assertEqual(content[0]['type'], 'image_url')
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
    @patch('databricks.model.openai_service.base64.b64encode')
    def test_extract_entities_api_error(self, mock_b64encode, mock_file):
        """Test extraction handles API errors"""
        mock_b64encode.return_value = b'encoded_image_data'

        # Mock API error
        self.mock_client.chat.completions.create.side_effect = Exception("API Error")

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
    """Test Azure OpenAI API connection testing"""

    @patch('databricks.model.openai_service.http_client_factory')
    @patch('databricks.model.openai_service.AzureOpenAI')
    def setUp(self, mock_azure_openai, mock_http_client_factory):
        """Set up test service"""
        mock_http_client_factory.return_value = (Mock(), Mock())
        
        with patch.dict(os.environ, {'PROXY_CLUSTER_ID': 'test-cluster'}):
            self.service = OpenAIVisionService(
                api_key="test-key",
                azure_endpoint="https://test.openai.azure.com/"
            )
        self.mock_client = Mock()
        self.service.client = self.mock_client

    def test_connection_success(self):
        """Test successful connection test"""
        mock_response = Mock()
        self.mock_client.chat.completions.create.return_value = mock_response

        result = self.service.test_connection()

        self.assertTrue(result)
        self.mock_client.chat.completions.create.assert_called_once()

    def test_connection_failure(self):
        """Test connection test handles failures"""
        self.mock_client.chat.completions.create.side_effect = Exception("Connection failed")

        result = self.service.test_connection()

        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()
