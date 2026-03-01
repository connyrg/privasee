"""
OCR Service for Document Processing

This service runs inside a Databricks Model Serving endpoint and provides
intelligent document text extraction and word-level bounding box detection.

Features:
- Intelligent PDF page detection (digital vs scanned)
- Digital PDF pages: direct text extraction (faster, no OCR)
- Scanned PDF pages: render to PNG and OCR with Azure Document Intelligence
- Word documents: direct text extraction with python-docx
- Images: OCR with Azure Document Intelligence
- Accurate word-level bounding boxes for masking

Dependencies: PyMuPDF (fitz), python-docx, azure-ai-documentintelligence
"""

import os
import io
import base64
from typing import List, Dict, Any
from enum import Enum

import fitz  # PyMuPDF
from docx import Document
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential


class PageSource(str, Enum):
    """Source type for page content"""
    DIGITAL_PDF = "digital_pdf"
    SCANNED_PDF = "scanned_pdf"
    IMAGE = "image"
    DOCX = "docx"


class OCRService:
    """
    Service for extracting text and word-level bounding boxes from documents.
    
    Uses intelligent detection to minimize OCR calls:
    - Digital PDF pages → direct text extraction
    - Scanned PDF pages → render to PNG → OCR
    - Word documents → direct text extraction
    - Images → OCR
    """
    
    # Threshold for determining if a PDF page has enough text to be "digital"
    MIN_TEXT_LENGTH_FOR_DIGITAL = 50
    
    # PyMuPDF zoom factor for rendering scanned pages (2.0 ≈ 144 DPI)
    RENDER_ZOOM_FACTOR = 2.0
    
    def __init__(self):
        """Initialize OCR service with Azure Document Intelligence credentials"""
        self.adi_endpoint = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
        self.adi_key = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_KEY")
        
        if not self.adi_endpoint or not self.adi_key:
            raise ValueError(
                "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT and "
                "AZURE_DOCUMENT_INTELLIGENCE_KEY must be set"
            )
        
        self.adi_client = DocumentIntelligenceClient(
            endpoint=self.adi_endpoint,
            credential=AzureKeyCredential(self.adi_key)
        )
    
    def process_document(
        self, 
        document_bytes: bytes, 
        file_extension: str
    ) -> List[Dict[str, Any]]:
        """
        Process a document and extract text with word-level bounding boxes.
        
        Args:
            document_bytes: Raw document bytes
            file_extension: File extension (pdf, png, jpg, jpeg, docx)
        
        Returns:
            List of page dictionaries with structure:
            {
                "page_num": int,
                "source": "digital_pdf" | "scanned_pdf" | "image" | "docx",
                "text": str,
                "words": [
                    {
                        "text": str,
                        "confidence": float,
                        "bounding_box": {"x": float, "y": float, "width": float, "height": float}
                    }
                ],
                "image_base64": str | None  # Base64-encoded image (None for DOCX)
            }
        """
        ext = file_extension.lower().lstrip('.')
        
        if ext == 'pdf':
            return self._process_pdf(document_bytes)
        elif ext == 'docx':
            return self._process_docx(document_bytes)
        elif ext in ['png', 'jpg', 'jpeg']:
            return self._process_image(document_bytes)
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")
    
    def _process_pdf(self, pdf_bytes: bytes) -> List[Dict[str, Any]]:
        """
        Process PDF with intelligent page detection.
        
        Digital pages (with text layer) are extracted directly.
        Scanned pages (image-only) are rendered and sent to OCR.
        """
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        results = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Try to extract text directly
            text = page.get_text()
            
            # Determine if page is digital or scanned based on text content
            if len(text.strip()) >= self.MIN_TEXT_LENGTH_FOR_DIGITAL:
                # Digital page - extract text and synthesize bounding boxes
                page_result = self._process_digital_pdf_page(page, page_num + 1, text)
            else:
                # Scanned page - render to PNG and OCR
                page_result = self._process_scanned_pdf_page(page, page_num + 1)
            
            results.append(page_result)
        
        doc.close()
        return results
    
    def _process_digital_pdf_page(
        self, 
        page: fitz.Page, 
        page_num: int, 
        text: str
    ) -> Dict[str, Any]:
        """
        Extract text and bounding boxes from digital PDF page.
        
        Uses PyMuPDF's get_text("words") to get word-level bounding boxes.
        Also renders page to PNG for vision API processing.
        """
        # Get word-level information with bounding boxes
        words_data = page.get_text("words")
        
        # Convert PyMuPDF format to our standard format
        # PyMuPDF returns: (x0, y0, x1, y1, word, block_no, line_no, word_no)
        words = []
        for word_tuple in words_data:
            x0, y0, x1, y1, word_text = word_tuple[:5]
            words.append({
                "text": word_text,
                "confidence": 1.0,  # Digital text is 100% confident
                "bounding_box": {
                    "x": float(x0),
                    "y": float(y0),
                    "width": float(x1 - x0),
                    "height": float(y1 - y0)
                }
            })
        
        # Render page to PNG for vision API (OpenAI doesn't support PDF natively)
        mat = fitz.Matrix(self.RENDER_ZOOM_FACTOR, self.RENDER_ZOOM_FACTOR)
        pix = page.get_pixmap(matrix=mat)
        png_bytes = pix.tobytes("png")
        page_image_b64 = base64.b64encode(png_bytes).decode('utf-8')
        
        return {
            "page_num": page_num,
            "source": PageSource.DIGITAL_PDF,
            "text": text,
            "words": words,
            "image_base64": page_image_b64
        }
    
    def _process_scanned_pdf_page(
        self, 
        page: fitz.Page, 
        page_num: int
    ) -> Dict[str, Any]:
        """
        Render scanned PDF page to PNG and OCR with Azure Document Intelligence.
        Also includes the PNG as base64 for vision API processing.
        """
        # Render page to PNG at 2x zoom (≈144 DPI)
        mat = fitz.Matrix(self.RENDER_ZOOM_FACTOR, self.RENDER_ZOOM_FACTOR)
        pix = page.get_pixmap(matrix=mat)
        png_bytes = pix.tobytes("png")
        
        # Encode to base64 for vision API
        page_image_b64 = base64.b64encode(png_bytes).decode('utf-8')
        
        # OCR the rendered image
        ocr_result = self._ocr_with_adi(png_bytes)
        
        return {
            "page_num": page_num,
            "source": PageSource.SCANNED_PDF,
            "text": ocr_result["text"],
            "words": ocr_result["words"],
            "image_base64": page_image_b64
        }
    
    def _process_docx(self, docx_bytes: bytes) -> List[Dict[str, Any]]:
        """
        Extract text from Word document.
        
        Word documents are always digital, so we extract text directly.
        Note: python-docx doesn't provide word-level bounding boxes,
        so we return paragraph-level text with empty words array.
        No visual representation is provided (image_base64 = None).
        """
        doc = Document(io.BytesIO(docx_bytes))
        
        # Extract all paragraphs
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        full_text = "\n".join(paragraphs)
        
        # Word documents don't have page numbers or word-level positions
        # Return as single "page"
        return [{
            "page_num": 1,
            "source": PageSource.DOCX,
            "text": full_text,
            "words": [],  # No spatial information available for DOCX
            "image_base64": None  # No visual representation
        }]
    
    def _process_image(self, image_bytes: bytes) -> List[Dict[str, Any]]:
        """
        OCR an image file with Azure Document Intelligence.
        Also includes the image as base64 for vision API processing.
        """
        # Encode image to base64 for vision API
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # OCR the image
        ocr_result = self._ocr_with_adi(image_bytes)
        
        return [{
            "page_num": 1,
            "source": PageSource.IMAGE,
            "text": ocr_result["text"],
            "words": ocr_result["words"],
            "image_base64": image_b64
        }]
    
    def _ocr_with_adi(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Perform OCR using Azure Document Intelligence prebuilt-read model.
        
        Returns:
            {
                "text": str,
                "words": [{"text": str, "confidence": float, "bounding_box": {...}}]
            }
        """
        # Call Azure Document Intelligence prebuilt-read model
        poller = self.adi_client.begin_analyze_document(
            "prebuilt-read",
            analyze_request=image_bytes,
            content_type="application/octet-stream"
        )
        result = poller.result()
        
        # Extract text and words with bounding boxes
        all_text = []
        all_words = []
        
        for page in result.pages:
            # Extract words with bounding boxes
            for word in page.words:
                # Convert ADI polygon format to our standard bounding box format
                bbox = self._polygon_to_bbox(word.polygon)
                
                all_words.append({
                    "text": word.content,
                    "confidence": word.confidence if hasattr(word, 'confidence') else 1.0,
                    "bounding_box": bbox
                })
                
                all_text.append(word.content)
        
        return {
            "text": " ".join(all_text),
            "words": all_words
        }
    
    def _polygon_to_bbox(self, polygon: List[float]) -> Dict[str, float]:
        """
        Convert ADI polygon format to standard bounding box format.
        
        ADI returns polygons as a flat list of coordinates: [x1, y1, x2, y2, x3, y3, x4, y4]
        representing the four corners of the bounding box (usually a rectangle).
        
        We convert to: {"x": top_left_x, "y": top_left_y, "width": w, "height": h}
        
        Args:
            polygon: List of 8 floats [x1, y1, x2, y2, x3, y3, x4, y4]
        
        Returns:
            Bounding box dict with x, y, width, height
        """
        # Handle empty or invalid polygons
        if not polygon or len(polygon) < 8:
            return {"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0}
        
        # Extract x and y coordinates
        x_coords = [polygon[i] for i in range(0, len(polygon), 2)]
        y_coords = [polygon[i] for i in range(1, len(polygon), 2)]
        
        # Calculate bounding box
        x_min = min(x_coords)
        y_min = min(y_coords)
        x_max = max(x_coords)
        y_max = max(y_coords)
        
        return {
            "x": float(x_min),
            "y": float(y_min),
            "width": float(x_max - x_min),
            "height": float(y_max - y_min)
        }
