"""
Tests for OCR Service

This test suite verifies:
1. Intelligent digital vs scanned PDF page detection
2. Accurate bounding box conversion from ADI polygon format
3. DOCX text extraction
4. Image OCR processing
5. Error handling for missing credentials

All Azure Document Intelligence API calls are mocked.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import io
import os

# Mock the azure modules before importing ocr_service
import sys
sys.modules['azure.ai.documentintelligence'] = MagicMock()
sys.modules['azure.core.credentials'] = MagicMock()

from databricks.model.ocr_service import OCRService, PageSource


class TestOCRServiceInit(unittest.TestCase):
    """Test OCR service initialization"""
    
    def test_init_success(self):
        """Test successful initialization with valid credentials"""
        with patch.dict(os.environ, {
            'AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT': 'https://test.cognitiveservices.azure.com/',
            'AZURE_DOCUMENT_INTELLIGENCE_KEY': 'test-key-123'
        }):
            service = OCRService()
            self.assertEqual(service.adi_endpoint, 'https://test.cognitiveservices.azure.com/')
            self.assertEqual(service.adi_key, 'test-key-123')
    
    def test_init_missing_credentials(self):
        """Test initialization succeeds without credentials (for digital PDFs/DOCX only)"""
        with patch.dict(os.environ, {}, clear=True):
            service = OCRService()
            self.assertIsNone(service.adi_client)
            self.assertIsNone(service.adi_endpoint)
            self.assertIsNone(service.adi_key)

class TestPolygonToBBox(unittest.TestCase):
    """Test bounding box conversion from ADI polygon format"""
    
    def setUp(self):
        with patch.dict(os.environ, {
            'AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT': 'https://test.cognitiveservices.azure.com/',
            'AZURE_DOCUMENT_INTELLIGENCE_KEY': 'test-key-123'
        }):
            self.service = OCRService()
    
    def test_polygon_to_bbox_standard_rectangle(self):
        """Test conversion of standard rectangular polygon"""
        # ADI polygon: top-left, top-right, bottom-right, bottom-left
        polygon = [100.0, 200.0, 140.0, 200.0, 140.0, 215.0, 100.0, 215.0]
        
        bbox = self.service._polygon_to_bbox(polygon)
        
        self.assertEqual(bbox['x'], 100.0)
        self.assertEqual(bbox['y'], 200.0)
        self.assertEqual(bbox['width'], 40.0)
        self.assertEqual(bbox['height'], 15.0)
    
    def test_polygon_to_bbox_rotated(self):
        """Test conversion of rotated polygon (still computes axis-aligned bbox)"""
        # Slightly rotated word
        polygon = [100.0, 200.0, 141.0, 198.0, 143.0, 213.0, 102.0, 215.0]
        
        bbox = self.service._polygon_to_bbox(polygon)
        
        # Should compute the axis-aligned bounding box
        self.assertEqual(bbox['x'], 100.0)  # min x
        self.assertEqual(bbox['y'], 198.0)  # min y
        self.assertEqual(bbox['width'], 43.0)  # max_x - min_x
        self.assertEqual(bbox['height'], 17.0)  # max_y - min_y
    
    def test_polygon_to_bbox_empty(self):
        """Test handling of empty or invalid polygon"""
        bbox = self.service._polygon_to_bbox([])
        
        self.assertEqual(bbox['x'], 0.0)
        self.assertEqual(bbox['y'], 0.0)
        self.assertEqual(bbox['width'], 0.0)
        self.assertEqual(bbox['height'], 0.0)


class TestDigitalPDFProcessing(unittest.TestCase):
    """Test digital PDF page processing"""
    
    def setUp(self):
        with patch.dict(os.environ, {
            'AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT': 'https://test.cognitiveservices.azure.com/',
            'AZURE_DOCUMENT_INTELLIGENCE_KEY': 'test-key-123'
        }):
            self.service = OCRService()
    
    @patch('fitz.open')
    def test_digital_page_detection(self, mock_fitz_open):
        """Test that pages with sufficient text are detected as digital"""
        # Create mock PDF with digital page (lots of text)
        mock_doc = MagicMock()
        mock_page = MagicMock()
        
        # Simulate digital page with >50 chars of text
        digital_text = "This is a digital PDF page with plenty of text content that exceeds the threshold."
        
        # Mock word extraction with side_effect
        def get_text_side_effect(mode="text"):
            if mode == "words":
                return [
                    (100, 200, 140, 215, "This", 0, 0, 0),
                    (145, 200, 160, 215, "is", 0, 0, 1),
                    (165, 200, 170, 215, "a", 0, 0, 2),
                ]
            else:
                return digital_text
        
        mock_page.get_text.side_effect = get_text_side_effect
        mock_page.rect.width = 595.0
        mock_page.rect.height = 842.0

        # Mock pixmap for rendering page to PNG (needed for vision API)
        mock_pixmap = Mock()
        mock_pixmap.tobytes.return_value = b"fake png bytes"
        mock_page.get_pixmap.return_value = mock_pixmap

        mock_doc.__len__.return_value = 1
        mock_doc.__iter__.return_value = [mock_page]
        mock_doc.__getitem__.return_value = mock_page
        mock_fitz_open.return_value = mock_doc
        
        # Process PDF
        pdf_bytes = b"fake pdf bytes"
        results = self.service._process_pdf(pdf_bytes)
        
        # Verify digital page was detected
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['source'], PageSource.DIGITAL_PDF)
        self.assertEqual(results[0]['page_num'], 1)
        self.assertIn("This", results[0]['text'])
        
        # Verify words have correct bounding boxes
        self.assertGreater(len(results[0]['words']), 0)
        word = results[0]['words'][0]
        self.assertEqual(word['text'], 'This')
        self.assertEqual(word['confidence'], 1.0)
        self.assertAlmostEqual(word['bounding_box']['x'],      100 / 595.0, places=5)
        self.assertAlmostEqual(word['bounding_box']['y'],      200 / 842.0, places=5)
        self.assertAlmostEqual(word['bounding_box']['width'],   40 / 595.0, places=5)
        self.assertAlmostEqual(word['bounding_box']['height'],  15 / 842.0, places=5)


class TestScannedPDFProcessing(unittest.TestCase):
    """Test scanned PDF page processing"""
    
    def setUp(self):
        with patch.dict(os.environ, {
            'AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT': 'https://test.cognitiveservices.azure.com/',
            'AZURE_DOCUMENT_INTELLIGENCE_KEY': 'test-key-123'
        }):
            self.service = OCRService()
    
    @patch('fitz.open')
    def test_scanned_page_detection(self, mock_fitz_open):
        """Test that pages with minimal text are detected as scanned"""
        # Create mock PDF with scanned page (no text)
        mock_doc = MagicMock()
        mock_page = MagicMock()
        
        # Simulate scanned page with <50 chars of text
        mock_page.get_text.return_value = "  "  # Whitespace only
        mock_page.rect.width = 595.0
        mock_page.rect.height = 842.0

        # Mock rendering to PNG
        mock_pixmap = Mock()
        mock_pixmap.tobytes.return_value = b"fake png bytes"
        mock_page.get_pixmap.return_value = mock_pixmap

        mock_doc.__len__.return_value = 1
        mock_doc.__iter__.return_value = [mock_page]
        mock_doc.__getitem__.return_value = mock_page
        mock_fitz_open.return_value = mock_doc
        
        # Mock ADI OCR response
        mock_adi_result = Mock()
        mock_adi_result.content = "Scanned"  # Set the text content
        mock_adi_page = Mock()
        
        mock_word1 = Mock()
        mock_word1.content = "Scanned"
        mock_word1.confidence = 0.98
        mock_word1.polygon = [50.0, 100.0, 120.0, 100.0, 120.0, 115.0, 50.0, 115.0]
        
        mock_adi_page.words = [mock_word1]
        mock_adi_result.pages = [mock_adi_page]
        
        mock_poller = Mock()
        mock_poller.result.return_value = mock_adi_result
        
        self.service.adi_client.begin_analyze_document = Mock(return_value=mock_poller)
        
        # Process PDF
        pdf_bytes = b"fake pdf bytes"
        results = self.service._process_pdf(pdf_bytes)
        
        # Verify scanned page was detected and OCR'd
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['source'], PageSource.SCANNED_PDF)
        self.assertEqual(results[0]['page_num'], 1)
        self.assertIn("Scanned", results[0]['text'])
        
        # Verify OCR words have correct bounding boxes
        self.assertEqual(len(results[0]['words']), 1)
        word = results[0]['words'][0]
        self.assertEqual(word['text'], 'Scanned')
        self.assertEqual(word['confidence'], 0.98)
        # ADI pixel coords normalised by png dimensions (page_pts * zoom = 595*2, 842*2)
        self.assertAlmostEqual(word['bounding_box']['x'],     50.0 / (595.0 * 2), places=5)
        self.assertAlmostEqual(word['bounding_box']['width'], 70.0 / (595.0 * 2), places=5)


class TestDocxProcessing(unittest.TestCase):
    """Test DOCX document processing"""
    
    def setUp(self):
        with patch.dict(os.environ, {
            'AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT': 'https://test.cognitiveservices.azure.com/',
            'AZURE_DOCUMENT_INTELLIGENCE_KEY': 'test-key-123'
        }):
            self.service = OCRService()
    
    @patch('databricks.model.ocr_service.Document')
    def test_docx_text_extraction(self, mock_document):
        """Test text extraction from DOCX file"""
        # Mock document with paragraphs
        mock_doc_instance = Mock()
        
        mock_para1 = Mock()
        mock_para1.text = "First paragraph"
        
        mock_para2 = Mock()
        mock_para2.text = "Second paragraph"
        
        mock_para3 = Mock()
        mock_para3.text = ""  # Empty paragraph should be filtered
        
        mock_doc_instance.paragraphs = [mock_para1, mock_para2, mock_para3]
        mock_document.return_value = mock_doc_instance
        
        # Process DOCX
        docx_bytes = b"fake docx bytes"
        results = self.service._process_docx(docx_bytes)
        
        # Verify results
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['source'], PageSource.DOCX)
        self.assertEqual(results[0]['page_num'], 1)
        self.assertEqual(results[0]['text'], "First paragraph\nSecond paragraph")
        self.assertEqual(results[0]['words'], [])  # DOCX doesn't provide bounding boxes


class TestImageProcessing(unittest.TestCase):
    """Test image file processing"""
    
    def setUp(self):
        with patch.dict(os.environ, {
            'AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT': 'https://test.cognitiveservices.azure.com/',
            'AZURE_DOCUMENT_INTELLIGENCE_KEY': 'test-key-123'
        }):
            self.service = OCRService()
    
    def test_image_ocr(self):
        """Test OCR processing of image files"""
        # Mock ADI OCR response
        mock_adi_result = Mock()
        mock_adi_result.content = "Image Text"  # Set the text content
        mock_adi_page = Mock()
        
        mock_word1 = Mock()
        mock_word1.content = "Image"
        mock_word1.confidence = 0.99
        mock_word1.polygon = [10.0, 20.0, 60.0, 20.0, 60.0, 35.0, 10.0, 35.0]
        
        mock_word2 = Mock()
        mock_word2.content = "Text"
        mock_word2.confidence = 0.97
        mock_word2.polygon = [70.0, 20.0, 110.0, 20.0, 110.0, 35.0, 70.0, 35.0]
        
        mock_adi_page.words = [mock_word1, mock_word2]
        mock_adi_result.pages = [mock_adi_page]
        
        mock_poller = Mock()
        mock_poller.result.return_value = mock_adi_result
        
        self.service.adi_client.begin_analyze_document = Mock(return_value=mock_poller)

        # Process image — mock PIL so fake bytes don't raise
        image_bytes = b"fake image bytes"
        mock_pil_img = Mock()
        mock_pil_img.size = (800, 600)
        with patch('databricks.model.ocr_service.Image.open', return_value=mock_pil_img):
            results = self.service._process_image(image_bytes)
        
        # Verify results
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['source'], PageSource.IMAGE)
        self.assertEqual(results[0]['page_num'], 1)
        self.assertEqual(results[0]['text'], "Image Text")
        
        # Verify words
        self.assertEqual(len(results[0]['words']), 2)
        self.assertEqual(results[0]['words'][0]['text'], 'Image')
        self.assertEqual(results[0]['words'][0]['confidence'], 0.99)
        self.assertEqual(results[0]['words'][1]['text'], 'Text')


class TestProcessDocument(unittest.TestCase):
    """Test main process_document method"""
    
    def setUp(self):
        with patch.dict(os.environ, {
            'AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT': 'https://test.cognitiveservices.azure.com/',
            'AZURE_DOCUMENT_INTELLIGENCE_KEY': 'test-key-123'
        }):
            self.service = OCRService()
    
    def test_unsupported_file_type(self):
        """Test error handling for unsupported file types"""
        with self.assertRaises(ValueError) as context:
            self.service.process_document(b"fake bytes", "unsupported.txt")
        
        self.assertIn("Unsupported", str(context.exception))
    
    @patch.object(OCRService, '_process_pdf')
    def test_pdf_routing(self, mock_process_pdf):
        """Test that PDF files are routed to PDF processor"""
        mock_process_pdf.return_value = []
        
        self.service.process_document(b"fake bytes", "pdf")
        mock_process_pdf.assert_called_once()
    
    @patch.object(OCRService, '_process_docx')
    def test_docx_routing(self, mock_process_docx):
        """Test that DOCX files are routed to DOCX processor"""
        mock_process_docx.return_value = []
        
        self.service.process_document(b"fake bytes", "docx")
        mock_process_docx.assert_called_once()
    
    @patch.object(OCRService, '_process_image')
    def test_image_routing(self, mock_process_image):
        """Test that image files are routed to image processor"""
        mock_process_image.return_value = []
        
        for ext in ['png', 'jpg', 'jpeg']:
            mock_process_image.reset_mock()
            self.service.process_document(b"fake bytes", ext)
            mock_process_image.assert_called_once()


if __name__ == '__main__':
    unittest.main()
