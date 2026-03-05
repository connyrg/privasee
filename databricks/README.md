# PrivaSee — Databricks Model Serving

This directory contains the complete document de-identification pipeline deployed as
a Databricks Model Serving endpoint. The system accepts a `session_id` and
`field_definitions`, fetches the uploaded document from a Unity Catalog volume,
runs OCR and AI-powered entity extraction, and writes results back to UC.

## Folder Structure

```
databricks/
├── model/
│   ├── document_intelligence.py  MLflow PyFunc model — main pipeline entry point
│   ├── ocr_service.py            Azure Document Intelligence OCR
│   ├── openai_service.py         Azure OpenAI vision entity extraction
│   ├── claude_service.py         Claude vision entity extraction (alternative)
│   ├── bbox_matcher.py           Aligns extracted entities to OCR word bounding boxes
│   └── __init__.py
├── notebooks/
│   ├── register_model.py         Log and register the model in Unity Catalog
│   └── deploy_endpoint.py        Create / update the Model Serving endpoint
├── utils/
│   └── databricks_utils.py       Databricks API helpers
└── README.md
```

## Pipeline

```
POST /invocations  (session_id, field_definitions)
    │
    ├─ Fetch original{ext} from UC volume via Files REST API
    │
    ├─ OCRService.process_document()
    │   ├─ PDF: digital pages → PyMuPDF text extraction
    │   │        scanned pages → Azure Document Intelligence
    │   ├─ DOCX: python-docx paragraph extraction
    │   └─ Image (PNG/JPG): Azure Document Intelligence
    │
    ├─ (per page) VisionService.extract_entities_from_base64()
    │   ├─ Azure OpenAI (default): GPT-4o with vision
    │   └─ Claude (alternative): Claude Vision
    │
    ├─ BBoxMatcher.match_entities_to_words()
    │   └─ Aligns entity text spans to OCR word-level bounding boxes
    │
    ├─ Write entities.json to UC volume via Files REST API
    │
    └─ Return {session_id, status, pages: [{page_num, entities}]}
```

## MLflow Interface

### Input (dataframe_records format)

```json
{
  "dataframe_records": [
    {
      "session_id": "uuid-string",
      "field_definitions": [
        {"name": "Full Name", "description": "Person's full name", "strategy": "Fake Data"},
        {"name": "Email", "description": "Email address", "strategy": "Black Out"}
      ]
    }
  ]
}
```

The model fetches the document from:
`{UC_VOLUME_PATH}/{session_id}/original{ext}`

### Output (MLflow predictions format)

```json
{
  "predictions": [
    {
      "session_id": "uuid-string",
      "status": "complete",
      "pages": [
        {
          "page_num": 1,
          "entities": [
            {
              "id": "entity-uuid",
              "entity_type": "Full Name",
              "original_text": "John Doe",
              "replacement_text": "Jane Smith",
              "bounding_box": [0.1, 0.2, 0.3, 0.05],
              "confidence": 0.95,
              "approved": true,
              "page_number": 1,
              "strategy": "Fake Data"
            }
          ]
        }
      ]
    }
  ]
}
```

## Environment Variables

All of the following must be set on the Model Serving endpoint:

### Databricks Files API (for UC volume access)

| Variable | Description |
|---|---|
| `DATABRICKS_HOST` | Workspace URL, e.g. `https://adb-xxxx.azuredatabricks.net` |
| `DATABRICKS_TOKEN` | Service principal token with Files API read/write access |
| `UC_VOLUME_PATH` | UC volume base path, e.g. `/Volumes/catalog/schema/privasee_sessions` |

### Azure Document Intelligence (OCR)

| Variable | Description |
|---|---|
| `AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT` | Azure DI endpoint URL |
| `AZURE_DOCUMENT_INTELLIGENCE_KEY` | Azure DI API key |

### Vision Service (choose one)

**Azure OpenAI (default):**

| Variable | Description |
|---|---|
| `VISION_SERVICE_PROVIDER` | `openai` (default) |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL |
| `AZURE_OPENAI_API_VERSION` | Optional, defaults to `2024-02-15-preview` |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | Optional, defaults to `gpt-4o` |

**Claude (alternative):**

| Variable | Description |
|---|---|
| `VISION_SERVICE_PROVIDER` | `claude` |
| `ANTHROPIC_API_KEY` | Anthropic API key |

## UC Volume Layout

The model reads and writes files under `{UC_VOLUME_PATH}/{session_id}/`:

```
{UC_VOLUME_PATH}/{session_id}/
    original{ext}    — uploaded document (read by model)
    entities.json    — extraction results (written by model, read by backend)
    metadata.json    — session metadata (written/read by backend)
    masked.pdf       — de-identified output (written by backend)
```

`entities.json` format written by the model:

```json
{
  "session_id": "...",
  "saved_at": "2025-01-01T00:00:00+00:00",
  "entities": [ ... ]
}
```

## Deployment

### 1. Register the model

Run `notebooks/register_model.py` in your Databricks workspace. Update the catalog,
schema, and model name variables at the top of the notebook.

### 2. Deploy the endpoint

Run `notebooks/deploy_endpoint.py`. Set `ENDPOINT_NAME` and `MODEL_VERSION` to match
your registered model. The endpoint will have `scale_to_zero` enabled by default.

### 3. Get the invocation URL

After deployment, the invocation URL follows this pattern:
```
https://<databricks-host>/serving-endpoints/<endpoint-name>/invocations
```

Set this as `DATABRICKS_MODEL_ENDPOINT` in the backend environment.

## Testing

```bash
# Run OCR service tests (no real API credentials needed — all mocked)
python -m pytest databricks/tests/ -v

# End-to-end workflow test (requires live credentials)
cd backend
API_BASE_URL=http://localhost:8000 python scripts/e2e_upload_test.py
```

## Troubleshooting

### `No original file found in UC volume for session <id>`

The model could not find `original.*` in `{UC_VOLUME_PATH}/{session_id}/`. Check that:
- The backend successfully uploaded the file (see backend logs for the upload step)
- `UC_VOLUME_PATH` is the same value on both the backend and the model endpoint
- `DATABRICKS_TOKEN` on the endpoint has Files API read access to the UC volume

### `DATABRICKS_HOST and DATABRICKS_TOKEN must be set`

These environment variables were not set on the model serving endpoint config.
Add them in the Databricks UI: Serving → your endpoint → Edit endpoint → Environment variables.

### Low OCR confidence on scanned pages

- Increase `RENDER_ZOOM_FACTOR` in `ocr_service.py` to `3.0` for higher DPI rendering
- Verify the original document was scanned at ≥200 DPI
- Azure Document Intelligence performs best with `prebuilt-read` model (the default)
