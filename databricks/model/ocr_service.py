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

Dependencies: PyMuPDF (fitz), python-docx, requests
"""

import concurrent.futures
import os
import io
import base64
import time
from typing import List, Dict, Any, Optional
from enum import Enum
import logging

import fitz  # PyMuPDF
from docx import Document
from PIL import Image

from ..utils.adi_utils import generate_adi_token, analyze_document_complete

logger = logging.getLogger(__name__)


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
    - Digital PDF pages -> direct text extraction
    - Scanned PDF pages -> render to PNG -> OCR
    - Word documents -> direct text extraction
    - Images -> OCR
    """
    
    # Threshold for determining if a PDF page has enough text to be "digital"
    MIN_TEXT_LENGTH_FOR_DIGITAL = 50
    
    # PyMuPDF zoom factor for rendering scanned pages (2.0 ~= 144 DPI)
    RENDER_ZOOM_FACTOR = 2.0
    
    def __init__(self):
        """
        Initialize OCR service with Azure Document Intelligence OAuth credentials.
        
        ADI credentials are only required for:
        - Scanned PDF pages (image-only, no text layer)
        - Image files (png, jpg, jpeg)
        
        Digital PDFs and DOCX files work without ADI.
        
        Note: HTTP_PROXY and HTTPS_PROXY environment variables are automatically
        used by the requests library, so we don't need to pass them explicitly.
        """
        # OAuth credentials for Suncorp APIM gateway
        self.adi_tenant_id = os.environ.get("ADI_TENANT_ID")
        self.adi_client_id = os.environ.get("ADI_CLIENT_ID")
        self.adi_client_secret = os.environ.get("ADI_CLIENT_SECRET")
        self.adi_api_app_id_uri = os.environ.get(
            "ADI_API_APP_ID_URI",
            "api://aeddc053-d47f-4352-9977-4313e0625905"
        )
        
        # APIM endpoint configuration
        self.adi_endpoint = os.environ.get(
            "ADI_ENDPOINT",
            "https://apim-nonprod-idp.azure-api.net/documentintelligence/documentModels/{model}:analyze"
        )
        self.adi_appspace_id = os.environ.get("ADI_APPSPACE_ID", "A-007100")
        self.adi_model_id = os.environ.get("ADI_MODEL_ID", "prebuilt-layout")
        
        # Check if ADI credentials are available
        self.adi_available = all([
            self.adi_tenant_id,
            self.adi_client_id,
            self.adi_client_secret,
            self.adi_endpoint
        ]) and self.adi_client_id != "dummy_client_id" and self.adi_client_secret != "dummy_secret"

        if self.adi_available:
            logger.info("Azure Document Intelligence OAuth credentials configured")
        else:
            logger.info("No ADI OAuth credentials provided. Service will work with digital PDFs and DOCX only.")

        # Token cache — reused across pages and across requests while the
        # model endpoint is warm.  Tokens are valid for 3600 s; we refresh
        # 60 s before expiry to avoid using a token that expires mid-request.
        self._cached_token: Optional[str] = None
        self._token_expiry: float = 0.0

        # Maximum number of scanned pages whose ADI calls run concurrently.
        # Mirrors the Vision API semaphore pattern in DocumentIntelligenceModel.
        self._adi_max_concurrent: int = int(
            os.environ.get("ADI_MAX_CONCURRENT_PAGES", "5")
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
    
    def _get_adi_token(self) -> str:
        """Return a cached OAuth token, refreshing when within 60 s of expiry."""
        if self._cached_token and time.time() < self._token_expiry - 60:
            logger.debug("Reusing cached ADI OAuth token")
            return self._cached_token
        logger.info("Fetching new ADI OAuth token")
        token = generate_adi_token(
            tenant_id=self.adi_tenant_id,
            client_id=self.adi_client_id,
            client_secret=self.adi_client_secret,
            api_app_id_uri=self.adi_api_app_id_uri,
        )
        self._cached_token = token
        self._token_expiry = time.time() + 3500  # tokens are valid for 3600 s
        return token

    def _process_pdf(self, pdf_bytes: bytes) -> List[Dict[str, Any]]:
        """
        Process PDF with intelligent page detection, running ADI calls concurrently.

        Pass 1 (sync): classify every page via get_text().  Digital pages are
        processed immediately (PyMuPDF only, no network).  Scanned pages are
        rendered to PNG and queued — the fitz document must stay open for this.

        Pass 2 (threaded): all scanned pages are OCR'd via ADI concurrently using
        ThreadPoolExecutor — avoids asyncio.run() which corrupts the Databricks
        Serverless event loop shared with the Vision API's asyncio.gather() tasks.
        A single OAuth token is fetched once and shared across all ADI calls.
        """
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        results: List[Optional[Dict[str, Any]]] = [None] * len(doc)
        scanned_tasks = []

        # Pass 1: classify and render while the doc object is open
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if len(text.strip()) >= self.MIN_TEXT_LENGTH_FOR_DIGITAL:
                results[page_num] = self._process_digital_pdf_page(page, page_num + 1, text)
            else:
                mat = fitz.Matrix(self.RENDER_ZOOM_FACTOR, self.RENDER_ZOOM_FACTOR)
                pix = page.get_pixmap(matrix=mat)
                png_bytes = pix.tobytes("png")
                png_w = page.rect.width * self.RENDER_ZOOM_FACTOR
                png_h = page.rect.height * self.RENDER_ZOOM_FACTOR
                page_image_b64 = base64.b64encode(png_bytes).decode("utf-8")
                scanned_tasks.append(
                    (page_num, page_num + 1, png_bytes, png_w, png_h, page_image_b64)
                )

        doc.close()

        # Pass 2: OCR all scanned pages concurrently via thread pool (no asyncio)
        if scanned_tasks:
            if not self.adi_available:
                raise ValueError(
                    "Scanned PDF page detected but no Azure Document Intelligence "
                    "credentials provided. Please set ADI_TENANT_ID, ADI_CLIENT_ID, "
                    "and ADI_CLIENT_SECRET environment variables with valid credentials."
                )
            token = self._get_adi_token()  # one OAuth call for the whole document

            def _ocr_task(task):
                page_idx, page_num, png_bytes, png_w, png_h, page_image_b64 = task
                logger.info(f"ADI OCR starting for page {page_num}")
                t0 = time.time()
                ocr_result = self._ocr_with_adi(png_bytes, token)
                logger.info(f"⏱️  ADI OCR (page {page_num}): {time.time() - t0:.2f}s")
                for word in ocr_result["words"]:
                    bb = word["bounding_box"]
                    word["bounding_box"] = {
                        "x": bb["x"] / png_w,
                        "y": bb["y"] / png_h,
                        "width": bb["width"] / png_w,
                        "height": bb["height"] / png_h,
                    }
                return page_idx, {
                    "page_num": page_num,
                    "source": PageSource.SCANNED_PDF,
                    "text": ocr_result["text"],
                    "words": ocr_result["words"],
                    "image_base64": page_image_b64,
                }

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self._adi_max_concurrent
            ) as executor:
                for page_idx, page_result in executor.map(_ocr_task, scanned_tasks):
                    results[page_idx] = page_result

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
        
        No Azure Document Intelligence required - uses PyMuPDF only.
        """
        # Get word-level information with bounding boxes
        words_data = page.get_text("words")
        
        # Convert PyMuPDF format to our standard format
        # PyMuPDF returns: (x0, y0, x1, y1, word, block_no, line_no, word_no)
        # Normalise to [0, 1] relative to page dimensions so masking_service
        # can apply rect = fitz.Rect(x*W, y*H, ...) correctly.
        page_w = page.rect.width
        page_h = page.rect.height
        words = []
        for word_tuple in words_data:
            x0, y0, x1, y1, word_text = word_tuple[:5]
            words.append({
                "text": word_text,
                "confidence": 1.0,  # Digital text is 100% confident
                "bounding_box": {
                    "x": float(x0) / page_w,
                    "y": float(y0) / page_h,
                    "width": float(x1 - x0) / page_w,
                    "height": float(y1 - y0) / page_h,
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
    
    def _process_docx(self, docx_bytes: bytes) -> List[Dict[str, Any]]:
        """
        Extract text from Word document.
        
        Word documents are always digital, so we extract text directly.
        Note: python-docx doesn't provide word-level bounding boxes,
        so we return paragraph-level text with empty words array.
        No visual representation is provided (image_base64 = None).
        
        No Azure Document Intelligence required.
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
    
    def _scale_for_vision(self, img: "Image.Image", original_bytes: bytes) -> bytes:
        """
        Scale image up to match the effective resolution of the PDF render path.

        The PDF path renders pages at RENDER_ZOOM_FACTOR=2.0 (≈144 DPI), producing
        ~1190×1684 px for an A4 page.  Images whose longer side is already at or
        above MIN_VISION_PIXELS are returned unchanged; smaller images are upscaled
        proportionally so the vision model receives comparable detail.

        Only used for the vision API call — ADI OCR always receives the original
        bytes to keep OCR coordinate offsets stable.
        """
        MIN_VISION_PIXELS = 1200  # ≈144 DPI for an A4 page (~11.7" tall)

        w, h = img.size
        if max(w, h) >= MIN_VISION_PIXELS:
            return original_bytes  # already high-res enough

        scale = MIN_VISION_PIXELS / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        scaled = img.resize((new_w, new_h), Image.LANCZOS)
        logger.info(
            f"Scaled image for vision API: {w}×{h} → {new_w}×{new_h} "
            f"(×{scale:.2f} to match PDF render quality)"
        )
        buf = io.BytesIO()
        scaled.save(buf, format=img.format or "PNG")
        return buf.getvalue()

    def _process_image(self, image_bytes: bytes) -> List[Dict[str, Any]]:
        """
        OCR an image file with Azure Document Intelligence.
        Also includes the image as base64 for vision API processing.

        Requires Azure Document Intelligence OAuth credentials.
        """
        if not self.adi_available:
            raise ValueError(
                "Image file detected but no Azure Document Intelligence credentials provided. "
                "Please set ADI_TENANT_ID, ADI_CLIENT_ID, and ADI_CLIENT_SECRET "
                "environment variables with valid credentials."
            )

        # Get image dimensions for normalisation (no extra I/O - bytes already in memory)
        img = Image.open(io.BytesIO(image_bytes))
        img_w, img_h = img.size

        # Scale image for vision API to match PDF render quality (~144 DPI equivalent).
        # ADI OCR always receives the original bytes so its coordinate offsets are stable.
        vision_bytes = self._scale_for_vision(img, image_bytes)
        image_b64 = base64.b64encode(vision_bytes).decode('utf-8')

        # OCR the image
        ocr_result = self._ocr_with_adi(image_bytes)

        # Normalise ADI pixel coords to [0, 1] relative to image dimensions
        for word in ocr_result["words"]:
            bb = word["bounding_box"]
            word["bounding_box"] = {
                "x": bb["x"] / img_w,
                "y": bb["y"] / img_h,
                "width": bb["width"] / img_w,
                "height": bb["height"] / img_h,
            }

        return [{
            "page_num": 1,
            "source": PageSource.IMAGE,
            "text": ocr_result["text"],
            "words": ocr_result["words"],
            "image_base64": image_b64
        }]
    
    def _ocr_with_adi(self, image_bytes: bytes, token: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform OCR using Azure Document Intelligence via Suncorp APIM gateway.

        Args:
            image_bytes: Raw image bytes (PNG or JPEG).
            token: Pre-fetched OAuth token.  When None, _get_adi_token() is called
                   (fetches a fresh token or returns the cached one).  Pass a token
                   explicitly when processing multiple pages to avoid redundant
                   OAuth round trips.

        Returns:
            {
                "text": str,  # Full page text
                "words": [...]  # Word-level bounding boxes
            }
        """
        if token is None:
            token = self._get_adi_token()

        # Base64-encode image bytes directly — no temp file needed
        base64_image = base64.b64encode(image_bytes).decode('utf-8')

        # Call ADI via adi_utils (proxies automatically used from env vars)
        result = analyze_document_complete(
            image_b64=base64_image,
            token=token,
            endpoint_url=self.adi_endpoint,
            appspace_id=self.adi_appspace_id,
            model_id=self.adi_model_id
        )

        # Extract text and words from analyzeResult
        analyze_result = result.get("analyzeResult", {})
        full_text = analyze_result.get("content", "")
        words = []

        for page in analyze_result.get("pages", []):
            for word in page.get("words", []):
                bbox = self._polygon_to_bbox(word.get("polygon", []))
                words.append({
                    "text": word.get("content", ""),
                    "confidence": word.get("confidence", 0.0),
                    "bounding_box": bbox
                })

        return {
            "text": full_text,
            "words": words
        }

    def _polygon_to_bbox(self, polygon: List[float]) -> Dict[str, float]:
        """
        Convert ADI polygon (8 points: x1,y1,x2,y2,x3,y3,x4,y4) to bounding box.
        
        Returns: {"x": x_min, "y": y_min, "width": width, "height": height}
        """
        if not polygon or len(polygon) < 8:
            return {"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0}
        
        x_coords = [polygon[i] for i in range(0, 8, 2)]
        y_coords = [polygon[i] for i in range(1, 8, 2)]
        
        x_min = min(x_coords)
        y_min = min(y_coords)
        x_max = max(x_coords)
        y_max = max(y_coords)
        
        return {
            "x": x_min,
            "y": y_min,
            "width": x_max - x_min,
            "height": y_max - y_min
        }
