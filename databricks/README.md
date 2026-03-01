# Document Intelligence Model for Databricks Model Serving

This directory contains a complete document intelligence pipeline designed to run in Databricks Model Serving endpoints. The system processes documents (PDF, DOCX, PNG, JPG), extracts text with OCR, identifies sensitive entities using Vision AI (Azure OpenAI or Claude), and provides word-level bounding boxes for precise redaction masking.

## Folder Structure

```
databricks/
├── model/
│   ├── __init__.py
│   └── ocr_service.py          # Main OCR service implementation
├── tests/
│   ├── __init__.py
│   └── test_ocr_service.py     # Comprehensive test suite (18 tests)
├── databricks_test_runner      # Notebook to run tests with coverage
├── __init__.py
└── README.md                   # This file
```

## Features

* **Intelligent PDF Page Detection**: Automatically detects whether PDF pages are digital (have text layer) or scanned (image-only)
* **Optimized Processing**: Digital pages bypass OCR for faster, more accurate text extraction
* **Multiple Format Support**: PDF, DOCX, PNG, JPG, JPEG
* **Precise Bounding Boxes**: Word-level coordinates for accurate masking
* **Self-Contained Dependencies**: Uses PyMuPDF (no system dependencies needed)
* **Azure Document Intelligence Integration**: OCR for scanned pages and images

## Architecture

### Processing Flow

```
Input Document
    ↓
File Type Detection
    ↓
├─ PDF ──→ Page Analysis
│           ├─ Digital Page (>50 chars) ──→ PyMuPDF Text Extraction
│           └─ Scanned Page (<50 chars) ──→ Render to PNG (144 DPI) ──→ Azure DI OCR
│
├─ DOCX ──→ python-docx Text Extraction (no OCR needed)
│
└─ Image (PNG/JPG) ──→ Azure DI OCR
```

### Output Format

Each page returns:
```python
{
    "page_num": 1,
    "source": "digital_pdf" | "scanned_pdf" | "image" | "docx",
    "text": "Full page text as a single string",
    "words": [
        {
            "text": "John",
            "confidence": 0.99,
            "bounding_box": {
                "x": 100.0,      # Top-left corner x
                "y": 200.0,      # Top-left corner y
                "width": 40.0,   # Width in points/pixels
                "height": 15.0   # Height in points/pixels
            }
        }
    ]
}
```

## Vision Service Configuration

### Azure OpenAI vs Claude

The Document Intelligence Model supports two vision service providers for entity detection:

* **Azure OpenAI** (Default): Uses GPT-4o with vision capabilities
* **Claude**: Uses Anthropic's Claude Sonnet 4

Toggle between providers using the `VISION_SERVICE_PROVIDER` environment variable.

### Azure OpenAI Configuration

**Environment Variables:**

```bash
# Vision Service Provider (default: "openai")
export VISION_SERVICE_PROVIDER="openai"

# Azure OpenAI Configuration
export AZURE_OPENAI_API_KEY="your-azure-openai-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_API_VERSION="2024-02-15-preview"  # Optional, defaults to 2024-02-15-preview
export AZURE_OPENAI_DEPLOYMENT_NAME="gpt-4o"  # Optional, defaults to gpt-4o
```

**Deployment Setup:**

1. Create an Azure OpenAI resource in Azure Portal
2. Deploy GPT-4o model with vision capabilities
3. Note your deployment name (use this for `AZURE_OPENAI_DEPLOYMENT_NAME`)
4. Get your endpoint and API key from Azure Portal

**Why Azure OpenAI Instead of OpenAI Direct?**

* **Enterprise Compliance**: Data stays within your Azure subscription
* **Private Network**: Use VNet integration and private endpoints
* **Regional Deployment**: Host models in your preferred Azure region
* **Unified Billing**: Consolidated with other Azure services
* **SLA Guarantees**: Enterprise-grade service level agreements

### Claude Configuration (Alternative)

```bash
export VISION_SERVICE_PROVIDER="claude"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

## Installation

### Dependencies

```bash
pip install PyMuPDF python-docx azure-ai-documentintelligence azure-core
```

### Environment Variables

**Required for Azure Document Intelligence (OCR):**

```bash
export AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT="https://your-instance.cognitiveservices.azure.com/"
export AZURE_DOCUMENT_INTELLIGENCE_KEY="your-key-here"
```

**Required for Vision Service (Choose Azure OpenAI or Claude):**

```bash
# For Azure OpenAI (default)
export VISION_SERVICE_PROVIDER="openai"
export AZURE_OPENAI_API_KEY="your-azure-openai-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_API_VERSION="2024-02-15-preview"  # Optional
export AZURE_OPENAI_DEPLOYMENT_NAME="gpt-4o"  # Optional

# OR for Claude
export VISION_SERVICE_PROVIDER="claude"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

**Required for Unity Catalog Storage:**

```bash
export UC_VOLUME_PATH="/Volumes/catalog/schema/volume_name"
```

## Usage

### Basic Usage

```python
from databricks.model.ocr_service import OCRService

# Initialize service (reads credentials from environment)
service = OCRService()

# Process a document
with open("document.pdf", "rb") as f:
    document_bytes = f.read()

# Pass the file extension (without dot)
pages = service.process_document(document_bytes, "pdf")

# Access results
for page in pages:
    print(f"Page {page['page_num']} ({page['source']}):")
    print(f"Text: {page['text'][:100]}...")
    print(f"Word count: {len(page['words'])}")
```

### Digital vs Scanned Detection

The service automatically detects page type based on text content:

```python
# Digital page (has text layer)
# - Extracted directly with PyMuPDF
# - Confidence: 1.0
# - Faster processing, no OCR API call

# Scanned page (image only)
# - Rendered to PNG at 144 DPI
# - Sent to Azure Document Intelligence
# - Confidence: varies (typically 0.95-0.99)
```

### Bounding Box Coordinates

Azure Document Intelligence returns polygons (4 corners), which are converted to our standard format:

```python
# ADI polygon: [x1, y1, x2, y2, x3, y3, x4, y4]
polygon = [100.0, 200.0, 140.0, 200.0, 140.0, 215.0, 100.0, 215.0]

# Converted to bbox:
bbox = {
    "x": 100.0,      # min(x_coords)
    "y": 200.0,      # min(y_coords)
    "width": 40.0,   # max(x) - min(x)
    "height": 15.0   # max(y) - min(y)
}
```

## Testing

### Run Tests

```bash
# Run all tests
python -m pytest databricks/tests/test_ocr_service.py -v

# Run specific test class
python -m pytest databricks/tests/test_ocr_service.py::TestPolygonToBBox -v

# Run with coverage
python -m pytest databricks/tests/test_ocr_service.py --cov=databricks.model.ocr_service --cov-report=term
```

### Test Coverage

The test suite includes:

* **Initialization Tests**: Credential validation
* **Polygon Conversion Tests**: ADI format to standard bbox
* **Digital PDF Tests**: Text extraction and word detection
* **Scanned PDF Tests**: Rendering and OCR
* **DOCX Tests**: Paragraph extraction
* **Image Tests**: OCR processing
* **Routing Tests**: File type detection and dispatch

All Azure Document Intelligence API calls are mocked—no real API credentials needed for testing.

## Performance Considerations

### Digital vs Scanned Detection Threshold

The `MIN_TEXT_LENGTH_FOR_DIGITAL` threshold (default: 50 characters) determines when a page is considered "digital":

* **Too low**: May misclassify scanned pages as digital, missing OCR
* **Too high**: May unnecessarily OCR digital pages, wasting API calls
* **Recommended**: 50 chars (approximately 1-2 short sentences)

### PDF Rendering Resolution

The `RENDER_ZOOM_FACTOR` (default: 2.0) controls PNG rendering quality:

* **1.0**: 72 DPI (low quality, smaller files)
* **2.0**: 144 DPI (recommended for OCR)
* **3.0**: 216 DPI (higher quality, larger files, slower)

## Deployment to Databricks Model Serving

### 1. Package the Service

Create `requirements.txt`:
```
PyMuPDF==1.23.8
python-docx==1.1.0
azure-ai-documentintelligence==1.0.0b1
azure-core==1.29.5
```

### 2. Configure Secrets

Store credentials in Databricks secrets:

```bash
databricks secrets create-scope --scope adi-credentials
databricks secrets put --scope adi-credentials --key endpoint
databricks secrets put --scope adi-credentials --key key
```

### 3. Create Model Serving Endpoint

```python
# In your endpoint initialization
import os
from databricks.model.ocr_service import OCRService

# Load credentials from secrets
os.environ['AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT'] = dbutils.secrets.get('adi-credentials', 'endpoint')
os.environ['AZURE_DOCUMENT_INTELLIGENCE_KEY'] = dbutils.secrets.get('adi-credentials', 'key')

# Initialize service
ocr_service = OCRService()

def score_model(input_data):
    """Model serving endpoint handler"""
    import base64
    
    file_bytes = base64.b64decode(input_data['file_bytes'])
    filename = input_data['filename']
    
    # Extract file extension
    file_extension = filename.split('.')[-1]
    
    pages = ocr_service.process_document(file_bytes, file_extension)
    
    return {"pages": pages}
```

## Troubleshooting

### "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT must be set"

Ensure environment variables are set before initializing `OCRService`:

```python
import os
os.environ['AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT'] = 'https://...'
os.environ['AZURE_DOCUMENT_INTELLIGENCE_KEY'] = 'your-key'
```

### PyMuPDF Import Error

If you see `ModuleNotFoundError: No module named 'fitz'`:

```bash
pip install --upgrade PyMuPDF
```

### Low OCR Confidence

If scanned pages have low confidence scores:

1. Increase `RENDER_ZOOM_FACTOR` to 3.0 for higher resolution
2. Check original document quality (scan at higher DPI)
3. Verify Azure DI endpoint is using "prebuilt-read" model

## Implementation Details

### Why PyMuPDF Instead of Poppler?

* **Self-Contained**: PyMuPDF installs via pip, no system dependencies
* **Model Serving Compatible**: Works in restricted Databricks environments
* **Performance**: Faster text extraction than Poppler utils
* **Bounding Boxes**: Native support for word-level coordinates

### Why 144 DPI for Scanned Pages?

* **OCR Optimal**: Azure Document Intelligence performs best at 150-300 DPI
* **File Size**: Balance between quality and processing speed
* **Claude Vision**: Sufficient resolution for downstream AI processing

### DOCX Limitation

Word documents don't provide spatial information for text, so:

* `words` array is empty
* Only paragraph-level text is returned
* Consider converting DOCX to PDF if masking is required

## License

Copyright © 2024 Suncorp Group. All rights reserved.
