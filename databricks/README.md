# PrivaSee — Databricks Model Serving

This directory contains two MLflow PyFunc models deployed as separate Databricks
Model Serving endpoints:

- **Document Intelligence** — OCR + AI entity extraction pipeline
- **Masking** — PDF redaction engine; reads `original{ext}` from UC, writes `masked.pdf`

## Folder Structure

```
databricks/
├── model/
│   ├── document_intelligence.py  Document Intelligence model entry point
│   ├── masking_model.py          Masking model entry point
│   ├── masking_service.py        PyMuPDF redaction engine (used by masking model)
│   ├── fake_data_service.py      Faker-based replacement text generator
│   ├── ocr_service.py            Azure Document Intelligence OCR
│   ├── openai_service.py         Azure OpenAI vision entity extraction
│   ├── claude_service.py         Claude vision entity extraction (alternative)
│   ├── bbox_matcher.py           Aligns extracted entities to OCR word bounding boxes
│   └── __init__.py
├── notebooks/
│   ├── register_model.py             Log and register Document Intelligence in Unity Catalog
│   ├── register_masking_model.ipynb  Log and register Masking model in Unity Catalog
│   ├── deploy_endpoint.py            Create / update a Model Serving endpoint
│   ├── cleanup_sessions.py           Utility to delete old session files from UC volume
│   └── test_endpoint.ipynb           Manual endpoint smoke-test notebook
├── tests/
│   ├── test_document_intelligence.py
│   ├── test_masking_model.py
│   ├── test_masking_service.py
│   ├── test_masking_integration.py
│   ├── test_bbox_matcher.py
│   ├── test_claude_service.py
│   ├── test_ocr_service.py
│   └── test_openai_service.py
├── utils/
│   └── databricks_utils.py       Databricks API helpers
└── README.md
```

## Pipelines

### Document Intelligence

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
    ├─ (per page, async) VisionService.extract_entities_from_base64_async()
    │   ├─ Azure OpenAI (default): GPT-4o with vision
    │   └─ Claude (alternative): Claude Vision
    │   Output: [{entity_type, original_text, confidence, occurrences: [{page_number,
    │            original_text, bounding_boxes: [[x,y,w,h]...]}]}]
    │   Note: each occurrence carries word-level bboxes merged by line
    │
    ├─ Enrich entities
    │   ├─ Assign stable UUID id per entity
    │   └─ Pre-generate replacement_text for user review:
    │       "Fake Data"    → realistic fake value (consistent per original_text)
    │       "Entity Label" → sequential label e.g. Full_Name_A, Full_Name_B
    │                        (letters A–Z, then integers 27, 28, … per type)
    │
    ├─ _merge_entity_variants()
    │   └─ Merge partial-name references (e.g. "John" → occurrence of "John Doe")
    │      and cross-page duplicates into a single entity with combined occurrences
    │
    ├─ Write entities.json to UC volume (includes intermediate_results for debugging)
    │
    └─ Return {session_id, status, entities: [...], pages: [{page_num, entities}]}
```

### Masking

```
POST /invocations  (session_id, entities_to_mask)
    │
    ├─ Fetch original{ext} from UC volume via Files REST API
    │
    ├─ Apply masking
    │   ├─ PDF   → MaskingService.apply_pdf_masks()  (PyMuPDF native)
    │   └─ Image → MaskingService.apply_masks()      (PIL), then wrap in PDF
    │
    ├─ Write masked.pdf to UC volume via Files REST API
    │
    └─ Return {session_id, status: "complete", entities_masked: N}
```

## MLflow Interface

### Document Intelligence

#### Input (dataframe_records format)

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

#### Output (MLflow predictions format)

All positional data lives inside `occurrences`. No entity-level `bounding_box`,
`bounding_boxes`, or `page_number` fields.

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
              "confidence": 0.95,
              "approved": true,
              "strategy": "Fake Data",
              "occurrences": [
                {
                  "page_number": 1,
                  "original_text": "John Doe",
                  "bounding_boxes": [[0.1, 0.2, 0.3, 0.05]]
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}
```

An entity with multiple appearances (e.g. full name on page 1 and first name only on
page 3) carries multiple `occurrences` entries.  Each occurrence's `original_text` may
differ from the entity's canonical `original_text` when the text was merged as a
partial-name variant.  The masking service uses token-alignment to derive the correct
replacement slice for each occurrence.

### Masking

#### Input (dataframe_records format)

```json
{
  "dataframe_records": [
    {
      "session_id": "uuid-string",
      "entities_to_mask": "[{\"id\": \"entity-uuid\", \"entity_type\": \"Full Name\", \"original_text\": \"John Doe\", \"replacement_text\": \"Jane Smith\", \"strategy\": \"Fake Data\", \"approved\": true, \"occurrences\": [{\"page_number\": 1, \"original_text\": \"John Doe\", \"bounding_boxes\": [[0.1, 0.2, 0.3, 0.05]]}]}]"
    }
  ]
}
```

`entities_to_mask` is a **JSON string** (not a nested object) containing the list of
approved entities. Only entities with `approved: true` are redacted.

#### Output (MLflow predictions format)

```json
{
  "predictions": [
    {
      "session_id": "uuid-string",
      "status": "complete",
      "entities_masked": 5
    }
  ]
}
```

## Environment Variables

### Document Intelligence endpoint

#### Databricks Files API (for UC volume access)

| Variable | Description |
|---|---|
| `DATABRICKS_HOST` | Workspace URL, e.g. `https://adb-xxxx.azuredatabricks.net` |
| `DATABRICKS_TOKEN` | Service principal token with Files API read/write access |
| `UC_VOLUME_PATH` | UC volume base path, e.g. `/Volumes/catalog/schema/privasee_sessions` |

#### Azure Document Intelligence (OCR)

| Variable | Description |
|---|---|
| `ADI_TENANT_ID` | Azure tenant ID for ADI OAuth |
| `ADI_CLIENT_ID` | OAuth client ID for Azure Document Intelligence |
| `ADI_CLIENT_SECRET` | OAuth client secret for Azure Document Intelligence |
| `ADI_ENDPOINT` | APIM endpoint URL for Azure Document Intelligence |
| `ADI_APPSPACE_ID` | AppSpace ID (default: `A-007100`) |
| `ADI_MODEL_ID` | Document Intelligence model ID (default: `prebuilt-layout`) |

#### Vision Service (choose one)

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

### Masking endpoint

| Variable | Description |
|---|---|
| `DATABRICKS_HOST` | Workspace URL |
| `DATABRICKS_TOKEN` | Service principal token with Files API read/write access |
| `UC_VOLUME_PATH` | UC volume base path (must match Document Intelligence endpoint) |

## UC Volume Layout

The models read and write files under `{UC_VOLUME_PATH}/{session_id}/`:

```
{UC_VOLUME_PATH}/{session_id}/
    original{ext}    — uploaded document (backend → read by both models)
    entities.json    — extraction results (document intelligence model → backend)
    metadata.json    — session metadata (backend)
    masked.pdf       — de-identified output (masking model)
```

`entities.json` format written by the Document Intelligence model:

```json
{
  "session_id": "...",
  "saved_at": "2025-01-01T00:00:00+00:00",
  "status": "awaiting_review",
  "entities": [
    {
      "id": "uuid",
      "entity_type": "Full Name",
      "original_text": "John Doe",
      "replacement_text": "Jane Smith",
      "confidence": 0.95,
      "approved": true,
      "strategy": "Fake Data",
      "occurrences": [
        {
          "page_number": 1,
          "original_text": "John Doe",
          "bounding_boxes": [[0.1, 0.2, 0.3, 0.05]]
        }
      ]
    }
  ],
  "intermediate_results": {
    "vision_raw": { "1": [ ... ] },
    "pre_merge":  [ ... ]
  }
}
```

`intermediate_results` is stored for debugging only and is not read by the backend
or masking endpoint.

## Deployment

### 1. Register the Document Intelligence model

Run `notebooks/register_model.py` in your Databricks workspace. Update the catalog,
schema, and model name variables at the top of the notebook.

### 2. Register the Masking model

Run `notebooks/register_masking_model.ipynb` in your Databricks workspace. Update
the catalog, schema, and model name variables at the top of the notebook.

### 3. Deploy the endpoints

Run `notebooks/deploy_endpoint.py` for each model. Set `ENDPOINT_NAME` and
`MODEL_VERSION` to match your registered model. The endpoint will have
`scale_to_zero` enabled by default.

### 4. Get the invocation URLs

After deployment, invocation URLs follow this pattern:
```
https://<databricks-host>/serving-endpoints/<endpoint-name>/invocations
```

Set the Document Intelligence endpoint URL as `DATABRICKS_MODEL_ENDPOINT` and the
Masking endpoint URL as `DATABRICKS_MASKING_ENDPOINT` in the backend environment.

## Testing

```bash
# Run unit tests (no real API credentials needed — all mocked)
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

- Verify the original document was scanned at ≥200 DPI
- Azure Document Intelligence performs best with `prebuilt-read` model (the default)
- The vision service downscales images to 1500px on the longer side before calling the LLM (downscale-only; smaller images are left at original size to avoid payload issues)
