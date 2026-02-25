"""
OCR Service — Azure Document Intelligence wrapper.

Migrated from the PoC (backend/app/services/ocr_service.py) for use inside
the Databricks model.  The interface is intentionally identical to the PoC
so that tests written against the PoC pass without modification.

Key changes vs. the PoC:
  - Credentials are sourced from Databricks secrets (not environment variables)
    and injected at construction time by DocumentIntelligenceModel.load_context()
  - Input is a base64-encoded PNG (passed over the Model Serving REST API)
    rather than a file path — a decode step will be added to analyse_document()

TODO:
  - Copy implementation from backend/app/services/ocr_service.py in the PoC
  - Add base64 decode + temp-file handling for the Model Serving path
  - Unit-test with a mocked Azure client
"""


class OCRService:
    """Wraps Azure Document Intelligence for OCR."""

    def __init__(self, endpoint: str, api_key: str):
        """
        Args:
            endpoint: Azure Document Intelligence endpoint URL
            api_key:  Azure Document Intelligence API key
        """
        self.endpoint = endpoint
        self.api_key = api_key
        # TODO: initialise azure-ai-documentintelligence client

    def analyze_document(self, image_base64: str) -> dict:
        """
        Run OCR on a base64-encoded PNG.

        Returns:
            dict with keys: words, lines, page_width, page_height

        TODO: implement
        """
        raise NotImplementedError

    def _polygon_to_bbox(self, polygon: list, page_width: float, page_height: float) -> list:
        """
        Convert Azure polygon format to normalised [x, y, w, h].

        TODO: copy from PoC ocr_service.py
        """
        raise NotImplementedError
