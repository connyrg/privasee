"""Unit tests for adi_utils.py

Tests all Azure Document Intelligence utility functions with mocked HTTP requests.
"""

import unittest
from unittest.mock import patch, mock_open, MagicMock
import base64
import json
from typing import Dict
import sys
import os

# Add parent directory to path to import adi_utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))

# Import the functions to test
from adi_utils import (
    generate_adi_token,
    encode_file_to_base64,
    analyze_document,
    get_analysis_result,
    analyze_document_complete
)


class TestGenerateAdiToken(unittest.TestCase):
    """Test cases for generate_adi_token function"""
    
    @patch('adi_utils.requests.post')
    def test_generate_token_success(self, mock_post):
        """Test successful token generation"""
        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"access_token": "test_token_12345"}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response
        
        # Call function
        token = generate_adi_token(
            tenant_id="test-tenant",
            client_id="test-client",
            client_secret="test-secret"
        )
        
        # Assertions
        self.assertEqual(token, "test_token_12345")
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args[1]
        self.assertIn("https://login.microsoftonline.com/test-tenant/oauth2/token", call_kwargs['url'])
        self.assertEqual(call_kwargs['auth'], ("test-client", "test-secret"))
    
    @patch('adi_utils.requests.post')
    def test_generate_token_with_proxies(self, mock_post):
        """Test token generation with proxy configuration"""
        mock_response = MagicMock()
        mock_response.json.return_value = {"access_token": "test_token"}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response
        
        proxies = {
            'http': 'http://proxy:8989',
            'https': 'http://proxy:8989'
        }
        
        token = generate_adi_token(
            tenant_id="test-tenant",
            client_id="test-client",
            client_secret="test-secret",
            proxies=proxies
        )
        
        self.assertEqual(token, "test_token")
        call_kwargs = mock_post.call_args[1]
        # Proxies parameter not passed in OAuth implementation
        # Proxies not passed in OAuth implementation
        self.assertIn('proxies', call_kwargs)
        self.assertIn('proxies', call_kwargs)
        self.assertEqual(call_kwargs['proxies'], proxies)
    
    @patch('adi_utils.requests.post')
    def test_generate_token_http_error(self, mock_post):
        """Test token generation with HTTP error"""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("401 Unauthorized")
        mock_post.return_value = mock_response
        
        with self.assertRaises(Exception):
            generate_adi_token(
                tenant_id="test-tenant",
                client_id="test-client",
                client_secret="test-secret"
            )


class TestEncodeFileToBase64(unittest.TestCase):
    """Test cases for encode_file_to_base64 function"""
    
    def test_encode_file_success(self):
        """Test successful file encoding"""
        test_content = b"test file content"
        expected_base64 = base64.b64encode(test_content).decode('utf-8')
        
        with patch('builtins.open', mock_open(read_data=test_content)):
            result = encode_file_to_base64("/fake/path/file.pdf")
        
        self.assertEqual(result, expected_base64)
    
    def test_encode_empty_file(self):
        """Test encoding empty file"""
        test_content = b""
        expected_base64 = base64.b64encode(test_content).decode('utf-8')
        
        with patch('builtins.open', mock_open(read_data=test_content)):
            result = encode_file_to_base64("/fake/path/empty.pdf")
        
        self.assertEqual(result, expected_base64)


class TestAnalyzeDocument(unittest.TestCase):
    """Test cases for analyze_document function"""
    
    @patch('adi_utils.encode_file_to_base64')
    @patch('adi_utils.requests.post')
    def test_analyze_document_success(self, mock_post, mock_encode):
        """Test successful document analysis submission"""
        # Mock encode function
        mock_encode.return_value = "base64_encoded_content"
        
        # Mock response
        mock_response = MagicMock()
        mock_response.headers = {
            'Operation-Location': 'https://api.test.com/results/12345'
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response
        
        # Call function
        result = analyze_document(
            file_path="/path/to/doc.pdf",
            token="test_token",
            endpoint_url="https://api.test.com/documentModels/{model}:analyze",
            appspace_id="A-007100"
        )
        
        # Assertions
        self.assertEqual(result, 'https://api.test.com/results/12345')
        mock_encode.assert_called_once_with("/path/to/doc.pdf")
        
        # Verify request parameters
        call_kwargs = mock_post.call_args[1]
        self.assertEqual(call_kwargs['url'], "https://api.test.com/documentModels/prebuilt-layout:analyze")
        self.assertEqual(call_kwargs['headers']['Authorization'], "Bearer test_token")
        self.assertEqual(call_kwargs['headers']['AppspaceId'], "A-007100")
        self.assertEqual(call_kwargs['json']['base64Source'], "base64_encoded_content")
    
    @patch('adi_utils.encode_file_to_base64')
    @patch('adi_utils.requests.post')
    def test_analyze_document_missing_operation_location(self, mock_post, mock_encode):
        """Test error when Operation-Location header is missing"""
        mock_encode.return_value = "base64_content"
        
        mock_response = MagicMock()
        mock_response.headers = {}  # No Operation-Location
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response
        
        with self.assertRaises(ValueError) as context:
            analyze_document(
                file_path="/path/to/doc.pdf",
                token="test_token",
                endpoint_url="https://api.test.com/documentModels/{model}:analyze",
                appspace_id="A-007100"
            )
        
        self.assertIn("Operation-Location", str(context.exception))
    
    @patch('adi_utils.encode_file_to_base64')
    @patch('adi_utils.requests.post')
    def test_analyze_document_custom_parameters(self, mock_post, mock_encode):
        """Test analysis with custom parameters"""
        mock_encode.return_value = "base64_content"
        
        mock_response = MagicMock()
        mock_response.headers = {'Operation-Location': 'https://api.test.com/results/123'}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response
        
        result = analyze_document(
            file_path="/path/to/doc.pdf",
            token="test_token",
            endpoint_url="https://api.test.com/documentModels/{model}:analyze",
            appspace_id="A-007100",
            model_id="custom-model",
            pages="1-5",
            locale="fr-FR",
            output_content_format="text"
        )
        
        # Verify custom parameters in request
        call_kwargs = mock_post.call_args[1]
        self.assertEqual(call_kwargs['url'], "https://api.test.com/documentModels/custom-model:analyze")
        self.assertEqual(call_kwargs['params']['pages'], "1-5")
        self.assertEqual(call_kwargs['params']['locale'], "fr-FR")
        self.assertEqual(call_kwargs['params']['outputContentFormat'], "text")


class TestGetAnalysisResult(unittest.TestCase):
    """Test cases for get_analysis_result function"""
    
    @patch('adi_utils.time.sleep')  # Mock sleep to speed up tests
    @patch('adi_utils.requests.get')
    def test_get_result_immediate_success(self, mock_get, mock_sleep):
        """Test getting results that are immediately ready"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'status': 'succeeded',
            'analyzeResult': {'content': 'Test content'}
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        
        result = get_analysis_result(
            result_location="https://api.test.com/results/123",
            token="test_token",
            appspace_id="A-007100",
            poll_interval=1,
            max_retries=5
        )
        
        self.assertEqual(result['status'], 'succeeded')
        self.assertEqual(result['analyzeResult']['content'], 'Test content')
        mock_get.assert_called()
    
    @patch('adi_utils.time.sleep')
    @patch('adi_utils.requests.get')
    def test_get_result_with_polling(self, mock_get, mock_sleep):
        """Test getting results after multiple polling attempts"""
        # First two calls return 'running', third returns 'succeeded'
        mock_responses = [
            MagicMock(json=lambda: {'status': 'running'}),
            MagicMock(json=lambda: {'status': 'running'}),
            MagicMock(json=lambda: {'status': 'succeeded', 'analyzeResult': {'content': 'Done'}})
        ]
        for resp in mock_responses:
            resp.raise_for_status = MagicMock()
        
        mock_get.side_effect = mock_responses
        
        result = get_analysis_result(
            result_location="https://api.test.com/results/123",
            token="test_token",
            appspace_id="A-007100",
            poll_interval=1,
            max_retries=5
        )
        
        self.assertEqual(result['status'], 'succeeded')
        self.assertEqual(mock_get.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 3)
    
    @patch('adi_utils.time.sleep')
    @patch('adi_utils.requests.get')
    def test_get_result_failed_analysis(self, mock_get, mock_sleep):
        """Test handling failed analysis"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'status': 'failed',
            'error': {'message': 'Analysis failed'}
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        
        with self.assertRaises(RuntimeError) as context:
            get_analysis_result(
                result_location="https://api.test.com/results/123",
                token="test_token",
                appspace_id="A-007100"
            )
        
        self.assertIn("failed", str(context.exception))
    
    @patch('adi_utils.time.sleep')
    @patch('adi_utils.requests.get')
    def test_get_result_timeout(self, mock_get, mock_sleep):
        """Test timeout when polling exceeds max_retries"""
        mock_response = MagicMock()
        mock_response.json.return_value = {'status': 'running'}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        
        with self.assertRaises(TimeoutError) as context:
            get_analysis_result(
                result_location="https://api.test.com/results/123",
                token="test_token",
                appspace_id="A-007100",
                poll_interval=1,
                max_retries=3
            )
        
        self.assertIn("timed out", str(context.exception))
        self.assertEqual(mock_get.call_count, 3)


class TestAnalyzeDocumentComplete(unittest.TestCase):
    """Test cases for analyze_document_complete function"""
    
    @patch('adi_utils.get_analysis_result')
    @patch('adi_utils.analyze_document')
    def test_complete_workflow_success(self, mock_analyze, mock_get_result):
        """Test complete workflow from submission to results"""
        # Mock analyze_document to return operation location
        mock_analyze.return_value = "https://api.test.com/results/123"
        
        # Mock get_analysis_result to return final result
        mock_get_result.return_value = {
            'status': 'succeeded',
            'analyzeResult': {'content': 'Complete result'}
        }
        
        result = analyze_document_complete(
            file_path="/path/to/doc.pdf",
            token="test_token",
            endpoint_url="https://api.test.com/documentModels/{model}:analyze",
            appspace_id="A-007100"
        )
        
        # Assertions
        self.assertEqual(result['status'], 'succeeded')
        self.assertEqual(result['analyzeResult']['content'], 'Complete result')
        
        # Verify both functions were called
        mock_analyze.assert_called_once()
        mock_get_result.assert_called_once_with(
            result_location="https://api.test.com/results/123",
            token="test_token",
            appspace_id="A-007100",
            poll_interval=2,
            max_retries=60,
            proxies=None,
        )
    
    @patch('adi_utils.get_analysis_result')
    @patch('adi_utils.analyze_document')
    def test_complete_workflow_with_custom_params(self, mock_analyze, mock_get_result):
        """Test complete workflow with custom polling parameters"""
        mock_analyze.return_value = "https://api.test.com/results/456"
        mock_get_result.return_value = {'status': 'succeeded', 'analyzeResult': {}}
        
        result = analyze_document_complete(
            file_path="/path/to/doc.pdf",
            token="test_token",
            endpoint_url="https://api.test.com/documentModels/{model}:analyze",
            appspace_id="A-007100",
            poll_interval=5,
            max_retries=30
        )
        
        # Verify custom polling params were passed
        mock_get_result.assert_called_once()
        call_kwargs = mock_get_result.call_args[1]
        self.assertEqual(call_kwargs['poll_interval'], 5)
        self.assertEqual(call_kwargs['max_retries'], 30)
    
    @patch('adi_utils.get_analysis_result')
    @patch('adi_utils.analyze_document')
    def test_complete_workflow_analyze_fails(self, mock_analyze, mock_get_result):
        """Test workflow when initial analysis submission fails"""
        mock_analyze.side_effect = ValueError("Invalid parameters")
        
        with self.assertRaises(ValueError):
            analyze_document_complete(
                file_path="/path/to/doc.pdf",
                token="test_token",
                endpoint_url="https://api.test.com/documentModels/{model}:analyze",
                appspace_id="A-007100"
            )
        
        # get_analysis_result should not be called if analyze fails
        mock_get_result.assert_not_called()


if __name__ == '__main__':
    unittest.main()
