"""
Tests for Document Intelligence MLflow Model

This test suite verifies:
1. Model initialization (load_context) with provider toggle
2. Complete pipeline integration (OCR → Vision AI → BBox Matcher)
3. Unity Catalog volume write-through
4. Error handling (no unhandled exceptions)
5. Multi-page document processing
6. Entity enrichment (IDs, strategy, approved flag)
7. OpenAI and Claude provider toggle

All external services are mocked.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, mock_open
import pandas as pd
import json
import base64
import os
import tempfile

# Mock external dependencies before importing
import sys

# Create a real base class for mlflow.pyfunc.PythonModel
class MockPythonModel:
    """Mock base class that can be inherited from"""
    def load_context(self, context):
        pass
    
    def predict(self, context, model_input):
        pass

# Create a simple namespace for mlflow.pyfunc (not a MagicMock to avoid inheritance issues)
class MockMLflowPyfunc:
    PythonModel = MockPythonModel

# Create a simple namespace for mlflow that has pyfunc as an attribute
class MockMLflow:
    pass

# Setup mock modules with proper attribute chain
mock_mlflow = MockMLflow()
mock_pyfunc = MockMLflowPyfunc()
mock_mlflow.pyfunc = mock_pyfunc  # Set pyfunc as attribute on mlflow

sys.modules['mlflow'] = mock_mlflow
sys.modules['mlflow.pyfunc'] = mock_pyfunc  # Also register in sys.modules
sys.modules['azure.ai.documentintelligence'] = MagicMock()
sys.modules['azure.core.credentials'] = MagicMock()
sys.modules['anthropic'] = MagicMock()
sys.modules['openai'] = MagicMock()

from databricks.model.document_intelligence import DocumentIntelligenceModel


class TestModelInitialization(unittest.TestCase):
    """Test model load_context initialization with provider toggle"""

    @patch('databricks.model.document_intelligence.DocumentIntelligenceClient')
    @patch('databricks.model.document_intelligence.OCRService')
    @patch('databricks.model.document_intelligence.OpenAIVisionService')
    @patch('databricks.model.document_intelligence.BBoxMatcher')
    def test_load_context_default_openai(
        self,
        mock_bbox_matcher,
        mock_openai_service,
        mock_ocr_service,
        mock_adi_client
    ):
        """Test successful model initialization with default Azure OpenAI provider"""
        with patch.dict(os.environ, {
            'AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT': 'https://test.azure.com/',
            'AZURE_DOCUMENT_INTELLIGENCE_KEY': 'test-adi-key',
            'AZURE_OPENAI_API_KEY': 'test-openai-key',
            'AZURE_OPENAI_ENDPOINT': 'https://test-openai.openai.azure.com/',
            'PROXY_CLUSTER_ID': 'test-cluster-123',
            'UC_VOLUME_PATH': '/Volumes/catalog/schema/sessions',
            'DATABRICKS_HOST': 'https://test.databricks.com',
            'DATABRICKS_TOKEN': 'test-token-123'
        }):
            model = DocumentIntelligenceModel()
            model.load_context(context=None)

            # Verify OpenAI is the default
            self.assertEqual(model.vision_provider, 'openai')
            self.assertIsNotNone(model.adi_client)
            self.assertIsNotNone(model.ocr_service)
            self.assertIsNotNone(model.vision_service)
            self.assertIsNotNone(model.bbox_matcher)
            self.assertEqual(model.uc_volume_path, '/Volumes/catalog/schema/sessions')

    @patch('databricks.model.document_intelligence.DocumentIntelligenceClient')
    @patch('databricks.model.document_intelligence.OCRService')
    @patch('databricks.model.document_intelligence.OpenAIVisionService')
    @patch('databricks.model.document_intelligence.BBoxMatcher')
    def test_load_context_explicit_openai(
        self,
        mock_bbox_matcher,
        mock_openai_service,
        mock_ocr_service,
        mock_adi_client
    ):
        """Test model initialization with explicit Azure OpenAI provider"""
        with patch.dict(os.environ, {
            'AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT': 'https://test.azure.com/',
            'AZURE_DOCUMENT_INTELLIGENCE_KEY': 'test-adi-key',
            'AZURE_OPENAI_API_KEY': 'test-openai-key',
            'AZURE_OPENAI_ENDPOINT': 'https://test-openai.openai.azure.com/',
            'VISION_SERVICE_PROVIDER': 'openai',
            'PROXY_CLUSTER_ID': 'test-cluster-123',
            'UC_VOLUME_PATH': '/Volumes/catalog/schema/sessions',
            'DATABRICKS_HOST': 'https://test.databricks.com',
            'DATABRICKS_TOKEN': 'test-token-123'
        }):
            model = DocumentIntelligenceModel()
            model.load_context(context=None)

            self.assertEqual(model.vision_provider, 'openai')

    @patch('databricks.model.document_intelligence.DocumentIntelligenceClient')
    @patch('databricks.model.document_intelligence.OCRService')
    @patch('databricks.model.document_intelligence.ClaudeVisionService')
    @patch('databricks.model.document_intelligence.BBoxMatcher')
    def test_load_context_claude_provider(
        self,
        mock_bbox_matcher,
        mock_claude_service,
        mock_ocr_service,
        mock_adi_client
    ):
        """Test successful model initialization with Claude provider"""
        with patch.dict(os.environ, {
            'AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT': 'https://test.azure.com/',
            'AZURE_DOCUMENT_INTELLIGENCE_KEY': 'test-adi-key',
            'ANTHROPIC_API_KEY': 'test-anthropic-key',
            'VISION_SERVICE_PROVIDER': 'claude',
            'UC_VOLUME_PATH': '/Volumes/catalog/schema/sessions',
            'DATABRICKS_HOST': 'https://test.databricks.com',
            'DATABRICKS_TOKEN': 'test-token-123'
        }):
            model = DocumentIntelligenceModel()
            model.load_context(context=None)

            # Verify Claude provider
            self.assertEqual(model.vision_provider, 'claude')
            self.assertIsNotNone(model.adi_client)
            self.assertIsNotNone(model.ocr_service)
            self.assertIsNotNone(model.vision_service)
            self.assertIsNotNone(model.bbox_matcher)
            self.assertEqual(model.uc_volume_path, '/Volumes/catalog/schema/sessions')

    def test_load_context_missing_adi_credentials(self):
        """Test initialization fails without ADI credentials"""
        with patch.dict(os.environ, {}, clear=True):
            model = DocumentIntelligenceModel()
            with self.assertRaises(ValueError) as context:
                model.load_context(context=None)
            self.assertIn("AZURE_DOCUMENT_INTELLIGENCE", str(context.exception))

    @patch('databricks.model.document_intelligence.DocumentIntelligenceClient')
    def test_load_context_missing_openai_key(self, mock_adi_client):
        """Test initialization fails without Azure OpenAI API key when provider is openai"""
        with patch.dict(os.environ, {
            'AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT': 'https://test.azure.com/',
            'AZURE_DOCUMENT_INTELLIGENCE_KEY': 'test-key',
            'VISION_SERVICE_PROVIDER': 'openai'
        }):
            model = DocumentIntelligenceModel()
            with self.assertRaises(ValueError) as context:
                model.load_context(context=None)
            self.assertIn("AZURE_OPENAI_API_KEY", str(context.exception))

    @patch('databricks.model.document_intelligence.DocumentIntelligenceClient')
    def test_load_context_missing_claude_key(self, mock_adi_client):
        """Test initialization fails without Anthropic API key when provider is claude"""
        with patch.dict(os.environ, {
            'AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT': 'https://test.azure.com/',
            'AZURE_DOCUMENT_INTELLIGENCE_KEY': 'test-key',
            'VISION_SERVICE_PROVIDER': 'claude'
        }):
            model = DocumentIntelligenceModel()
            with self.assertRaises(ValueError) as context:
                model.load_context(context=None)
            self.assertIn("ANTHROPIC_API_KEY", str(context.exception))

    @patch('databricks.model.document_intelligence.DocumentIntelligenceClient')
    def test_load_context_invalid_provider(self, mock_adi_client):
        """Test initialization fails with invalid provider"""
        with patch.dict(os.environ, {
            'AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT': 'https://test.azure.com/',
            'AZURE_DOCUMENT_INTELLIGENCE_KEY': 'test-key',
            'VISION_SERVICE_PROVIDER': 'invalid_provider',
            'AZURE_OPENAI_API_KEY': 'test-key',
            'AZURE_OPENAI_ENDPOINT': 'https://test.openai.azure.com/'
        }):
            model = DocumentIntelligenceModel()
            with self.assertRaises(ValueError) as context:
                model.load_context(context=None)
            self.assertIn("Invalid VISION_SERVICE_PROVIDER", str(context.exception))
            self.assertIn("invalid_provider", str(context.exception))

    @patch('databricks.model.document_intelligence.OpenAIVisionService')
    @patch('databricks.model.document_intelligence.DocumentIntelligenceClient')
    def test_load_context_missing_uc_volume_path(self, mock_adi_client, mock_openai_service):
        """Test initialization fails without UC volume path"""
        with patch.dict(os.environ, {
            'AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT': 'https://test.azure.com/',
            'AZURE_DOCUMENT_INTELLIGENCE_KEY': 'test-key',
            'AZURE_OPENAI_API_KEY': 'test-key',
            'AZURE_OPENAI_ENDPOINT': 'https://test.openai.azure.com/',
            'PROXY_CLUSTER_ID': 'test-cluster-123'
        }):
            model = DocumentIntelligenceModel()
            with self.assertRaises(ValueError) as context:
                model.load_context(context=None)
            self.assertIn("UC_VOLUME_PATH", str(context.exception))


class TestPredictPipelineOpenAI(unittest.TestCase):
    """Test complete document processing pipeline with Azure OpenAI"""

    def setUp(self):
        """Set up test model with mocked Azure OpenAI services"""
        self.model = DocumentIntelligenceModel()
        
        # Mock services for OpenAI provider
        self.model.vision_provider = 'openai'
        self.model.ocr_service = Mock()
        self.model.vision_service = Mock()
        self.model.bbox_matcher = Mock()
        self.model.fake_data_service = Mock()
        self.model.fake_data_service.generate.return_value = "Generated Fake Name"
        self.model.fake_data_service = Mock()
        self.model.fake_data_service.generate.return_value = "Generated Fake Name"
        self.model.fake_data_service = Mock()
        self.model.fake_data_service.generate.return_value = "Generated Fake Name"
        self.model.fake_data_service = Mock()
        self.model.fake_data_service.generate.return_value = "Generated Fake Name"
        self.model.fake_data_service = Mock()
        self.model.fake_data_service.generate.return_value = "Generated Fake Name"
        self.model.uc_volume_path = '/tmp/test_volumes'
        self.model.databricks_host = 'https://test.databricks.com'
        self.model.databricks_token = 'test-token-123'

    def test_predict_single_page_document_openai(self):
        """Test processing single-page document with Azure OpenAI"""
        # Prepare test data
        field_definitions = [
            {"name": "Full Name", "description": "Person's name", "strategy": "Fake Data"}
        ]
        
        input_df = pd.DataFrame([{
            'session_id': 'test-session-123',
            'field_definitions': field_definitions
        }])
        
        # Mock _fetch_original_file() method directly
        self.model._fetch_original_file = Mock(return_value=(b'fake pdf content', 'original.pdf'))
        
        # Mock _write_to_uc_volume() to prevent actual HTTP requests
        self.model._write_to_uc_volume = Mock()

        # Mock OCR service response - returns list directly, not dict with 'pages' key
        self.model.ocr_service.process_document.return_value = [
            {
                'page_num': 1,
                'text': 'John Doe lives here',
                'words': [
                    {'text': 'John', 'bounding_box': {'x': 0.1, 'y': 0.1, 'width': 0.04, 'height': 0.02}},
                    {'text': 'Doe', 'bounding_box': {'x': 0.15, 'y': 0.1, 'width': 0.03, 'height': 0.02}}
                ],
                'image_base64': 'fake_image_base64_data'
            }
        ]

        # Mock vision service extract_entities_from_base64 method
        self.model.vision_service.extract_entities_from_base64.return_value = [
            {
                'entity_type': 'Full Name',
                'original_text': 'John Doe',
                'bounding_box': [0.1, 0.1, 0.08, 0.02],
                'confidence': 0.95,
                'page_number': 1
            }
        ]

        # Mock bbox matcher
        self.model.bbox_matcher.match_entities_to_words.return_value = [
            {
                'entity_type': 'Full Name',
                'original_text': 'John Doe',
                'bounding_box': [0.1, 0.1, 0.08, 0.02],
                'confidence': 0.95,
                'page_number': 1,
                'bounding_boxes': [
                    {'x': 0.1, 'y': 0.1, 'width': 0.08, 'height': 0.02}
                ]
            }
        ]

        # Run prediction
        result_df = self.model.predict(context=None, model_input=input_df)

        # Verify results
        self.assertEqual(len(result_df), 1)
        result = result_df.iloc[0]
        
        self.assertEqual(result['session_id'], 'test-session-123')
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(len(result['pages']), 1)
        self.assertEqual(result['pages'][0]['page_num'], 1)
        self.assertEqual(len(result['pages'][0]['entities']), 1)
        
        entity = result['pages'][0]['entities'][0]
        self.assertEqual(entity['entity_type'], 'Full Name')
        self.assertEqual(entity['original_text'], 'John Doe')
        self.assertEqual(entity['strategy'], 'Fake Data')
        self.assertTrue(entity['approved'])
        self.assertIn('id', entity)

        # Verify UC volume write was attempted via REST API
        self.model._write_to_uc_volume.assert_called_once()


class TestPredictPipelineClaude(unittest.TestCase):
    """Test complete document processing pipeline with Claude"""

    def setUp(self):
        """Set up test model with mocked Claude services"""
        self.model = DocumentIntelligenceModel()
        
        # Mock services for Claude provider
        self.model.vision_provider = 'claude'
        self.model.ocr_service = Mock()
        self.model.vision_service = Mock()
        self.model.bbox_matcher = Mock()
        self.model.fake_data_service = Mock()
        self.model.fake_data_service.generate.return_value = "Generated Fake Name"
        self.model.fake_data_service = Mock()
        self.model.fake_data_service.generate.return_value = "Generated Fake Name"
        self.model.fake_data_service = Mock()
        self.model.fake_data_service.generate.return_value = "Generated Fake Name"
        self.model.fake_data_service = Mock()
        self.model.fake_data_service.generate.return_value = "Generated Fake Name"
        self.model.fake_data_service = Mock()
        self.model.fake_data_service.generate.return_value = "Generated Fake Name"
        self.model.uc_volume_path = '/tmp/test_volumes'
        self.model.databricks_host = 'https://test.databricks.com'
        self.model.databricks_token = 'test-token-123'

    def test_predict_single_page_document_claude(self):
        """Test processing single-page document with Claude"""
        # Prepare test data
        field_definitions = [
            {"name": "Full Name", "description": "Person's name", "strategy": "Fake Data"}
        ]
        
        input_df = pd.DataFrame([{
            'session_id': 'test-session-456',
            'field_definitions': field_definitions
        }])
        
        # Mock _fetch_original_file() method directly
        self.model._fetch_original_file = Mock(return_value=(b'fake pdf content', 'original.pdf'))
        
        # Mock _write_to_uc_volume() to prevent actual HTTP requests
        self.model._write_to_uc_volume = Mock()

        # Mock OCR service response - returns list directly, not dict with 'pages' key
        self.model.ocr_service.process_document.return_value = [
            {
                'page_num': 1,
                'text': 'Jane Smith lives here',
                'words': [
                    {'text': 'Jane', 'bounding_box': {'x': 0.1, 'y': 0.1, 'width': 0.04, 'height': 0.02}},
                    {'text': 'Smith', 'bounding_box': {'x': 0.15, 'y': 0.1, 'width': 0.05, 'height': 0.02}}
                ],
                'image_base64': 'fake_image_base64_data'
            }
        ]

        # Mock vision service extract_entities_from_base64 method
        self.model.vision_service.extract_entities_from_base64.return_value = [
            {
                'entity_type': 'Full Name',
                'original_text': 'Jane Smith',
                'bounding_box': [0.1, 0.1, 0.09, 0.02],
                'confidence': 0.97,
                'page_number': 1
            }
        ]

        # Mock bbox matcher
        self.model.bbox_matcher.match_entities_to_words.return_value = [
            {
                'entity_type': 'Full Name',
                'original_text': 'Jane Smith',
                'bounding_box': [0.1, 0.1, 0.09, 0.02],
                'confidence': 0.97,
                'page_number': 1,
                'bounding_boxes': [
                    {'x': 0.1, 'y': 0.1, 'width': 0.09, 'height': 0.02}
                ]
            }
        ]

        # Run prediction
        result_df = self.model.predict(context=None, model_input=input_df)

        # Verify results
        self.assertEqual(len(result_df), 1)
        result = result_df.iloc[0]
        
        self.assertEqual(result['session_id'], 'test-session-456')
        self.assertEqual(result['status'], 'complete')
        entity = result['pages'][0]['entities'][0]
        self.assertEqual(entity['entity_type'], 'Full Name')
        self.assertEqual(entity['original_text'], 'Jane Smith')


class TestPredictEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""

    def setUp(self):
        """Set up test model"""
        self.model = DocumentIntelligenceModel()
        self.model.vision_provider = 'openai'
        self.model.ocr_service = Mock()
        self.model.vision_service = Mock()
        self.model.bbox_matcher = Mock()
        self.model.fake_data_service = Mock()
        self.model.fake_data_service.generate.return_value = "Generated Fake Name"
        self.model.fake_data_service = Mock()
        self.model.fake_data_service.generate.return_value = "Generated Fake Name"
        self.model.fake_data_service = Mock()
        self.model.fake_data_service.generate.return_value = "Generated Fake Name"
        self.model.fake_data_service = Mock()
        self.model.fake_data_service.generate.return_value = "Generated Fake Name"
        self.model.fake_data_service = Mock()
        self.model.fake_data_service.generate.return_value = "Generated Fake Name"
        self.model.uc_volume_path = '/tmp/test_volumes'
        self.model.databricks_host = 'https://test.databricks.com'
        self.model.databricks_token = 'test-token-123'

    def test_predict_multi_page_document(self):
        """Test processing multi-page document"""
        field_definitions = [
            {"name": "Name", "description": "Person name", "strategy": "Black Out"}
        ]
        
        input_df = pd.DataFrame([{
            'session_id': 'multi-page-session',
            'field_definitions': field_definitions
        }])
        
        # Mock _fetch_original_file() method directly
        self.model._fetch_original_file = Mock(return_value=(b'fake pdf content', 'original.pdf'))
        
        # Mock _write_to_uc_volume() to prevent actual HTTP requests
        self.model._write_to_uc_volume = Mock()

        # Mock OCR with 2 pages - returns list directly, not dict with 'pages' key
        self.model.ocr_service.process_document.return_value = [
            {
                'page_num': 1,
                'text': 'Page 1 with Alice',
                'words': [{'text': 'Alice', 'bounding_box': {'x': 0.1, 'y': 0.1, 'width': 0.05, 'height': 0.02}}],
                'image_base64': 'page1_image'
            },
            {
                'page_num': 2,
                'text': 'Page 2 with Bob',
                'words': [{'text': 'Bob', 'bounding_box': {'x': 0.2, 'y': 0.2, 'width': 0.03, 'height': 0.02}}],
                'image_base64': 'page2_image'
            }
        ]

        # Mock vision service responses for each page
        def extract_entities_side_effect(image_b64, mimetype, ocr_data, field_definitions, page_number):
            if page_number == 1:
                return [{'entity_type': 'Name', 'original_text': 'Alice', 'confidence': 0.9, 'page_number': 1, 'bounding_box': [0.1, 0.1, 0.05, 0.02]}]
            else:
                return [{'entity_type': 'Name', 'original_text': 'Bob', 'confidence': 0.85, 'page_number': 2, 'bounding_box': [0.2, 0.2, 0.03, 0.02]}]

        self.model.vision_service.extract_entities_from_base64.side_effect = extract_entities_side_effect

        # Mock bbox matcher
        self.model.bbox_matcher.match_entities_to_words.side_effect = lambda entities, ocr_words: [
            {**e, 'bounding_boxes': [e['bounding_box']]} for e in entities
        ]

        # Run prediction
        result_df = self.model.predict(context=None, model_input=input_df)

        # Verify results
        result = result_df.iloc[0]
        self.assertEqual(len(result['pages']), 2)
        self.assertEqual(result['pages'][0]['page_num'], 1)
        self.assertEqual(result['pages'][1]['page_num'], 2)

    def test_predict_error_handling(self):
        """Test error handling when processing fails"""
        input_df = pd.DataFrame([{
            'session_id': 'error-session',
            'field_definitions': []
        }])
        
        # Mock _fetch_original_file() to raise exception
        self.model._fetch_original_file = Mock(side_effect=Exception("File fetch failed"))

        # Run prediction
        result_df = self.model.predict(context=None, model_input=input_df)

        # Verify error response
        result = result_df.iloc[0]
        self.assertEqual(result['session_id'], 'error-session')
        self.assertEqual(result['status'], 'error')
        self.assertIn('error_message', result)

    def test_predict_no_image_page(self):
        """Test handling pages without images (digital PDF text extraction)"""
        field_definitions = [{"name": "Name", "description": "Name", "strategy": "Fake Data"}]
        
        input_df = pd.DataFrame([{
            'session_id': 'no-image-session',
            'field_definitions': field_definitions
        }])
        
        # Mock _fetch_original_file() method directly
        self.model._fetch_original_file = Mock(return_value=(b'fake pdf content', 'original.pdf'))
        
        # Mock _write_to_uc_volume() to prevent actual HTTP requests
        self.model._write_to_uc_volume = Mock()

        # Mock OCR with page that has no image_base64 - returns list directly, not dict with 'pages' key
        self.model.ocr_service.process_document.return_value = [
            {
                'page_num': 1,
                'text': 'Digital text',
                'words': [{'text': 'Digital', 'bounding_box': {'x': 0.1, 'y': 0.1, 'width': 0.05, 'height': 0.02}}],
                # No image_base64 key - simulating digital PDF
            }
        ]

        # Run prediction
        result_df = self.model.predict(context=None, model_input=input_df)

        # Verify results - should complete without entities
        result = result_df.iloc[0]
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(len(result['pages'][0]['entities']), 0)


if __name__ == '__main__':
    unittest.main()
