# PrivaSee — Architecture

## Overview

PrivaSee is a document de-identification tool that masks sensitive information
in PDFs and images.  Three independently deployable components communicate over HTTPS.

```
┌───────────────────────────────────────────────────────────────────┐
│                    Posit Connect                                   │
│                                                                   │
│  ┌──────────────────────┐      ┌──────────────────────────────┐  │
│  │  Dash Frontend        │─────▶│  FastAPI Backend             │  │
│  │  (frontend_dash/)     │ HTTP │  (uvicorn / ASGI)            │  │
│  └──────────────────────┘      └──────────┬─────────────────-─┘  │
│                                           │                       │
└───────────────────────────────────────────┼───────────────────────┘
                                            │
                         ┌──────────────────▼──────────────────┐
                         │          Databricks                  │
                         │                                      │
                         │  ┌──────────────────────────────┐   │
                         │  │  Document Intelligence        │   │
                         │  │  Model Serving endpoint       │   │
                         │  │  • Azure Document Intelligence│   │
                         │  │  • Azure OpenAI / Claude      │   │
                         │  └──────────────────────────────┘   │
                         │                                      │
                         │  ┌──────────────────────────────┐   │
                         │  │  Masking Model Serving        │   │
                         │  │  endpoint                     │   │
                         │  │  • PyMuPDF redaction engine   │   │
                         │  └──────────────────────────────┘   │
                         │                                      │
                         │  ┌──────────────────────────────┐   │
                         │  │  Unity Catalog Volume         │   │
                         │  │  (shared session state)       │   │
                         │  └──────────────────────────────┘   │
                         └─────────────────────────────────────┘
```

## Components

### Frontend (`frontend_dash/`)
- Dash (Python) application deployed to Posit Connect
- Communicates with the backend via direct HTTP calls (`requests` library)
- Fetches PDF content server-side and returns `data:` URIs for iframe display (avoids cross-origin routing issues on Posit Connect)
- Supports two processing modes:
  - **Single document** — upload → review entities → generate masked PDF → compare
  - **Batch** — upload multiple PDFs → automatic upload/process/mask/verify per file → results table with masking scores

### Backend (`backend/`)
- FastAPI application deployed to Posit Connect
- Handles file upload, orchestrates the workflow, and serves output files
- Delegates entity extraction to the Document Intelligence Model Serving endpoint
- Delegates masking to the Masking Model Serving endpoint
- Persists session state (metadata + binary artefacts) to a UC volume via the
  Databricks Files REST API (`/api/2.0/fs/files`)

Key API endpoints:

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/upload` | Upload a document, create a session |
| `POST` | `/api/process` | Extract entities via Databricks |
| `POST` | `/api/approve-and-mask` | Apply redactions via Databricks |
| `POST` | `/api/sessions/{id}/verify` | Verify masking by extracting text from masked PDF |
| `GET` | `/api/sessions/{id}` | Get session metadata |
| `DELETE` | `/api/sessions/{id}` | Delete session and all UC artefacts |
| `GET` | `/api/files/{folder}/{filename}` | Serve original or masked PDF |
| `GET` | `/api/templates` | List built-in system templates |
| `GET` | `/api/configs` | List saved field configurations |

### Databricks Models (`databricks/model/`)

Two MLflow PyFunc models, each deployed to its own Model Serving endpoint:

**DocumentIntelligenceModel** (`document_intelligence.py`)
- Fetches the uploaded document from the UC volume via the Files REST API
- Runs Azure Document Intelligence (OCR) and Azure OpenAI / Claude Vision (entity extraction)
- Pre-generates replacement text for Fake Data and Entity Label strategies
- Writes `entities.json` to the session directory in the UC volume

**MaskingModel** (`masking_model.py`)
- Receives `session_id` + approved entities from the backend
- Fetches `original{ext}` from the UC volume
- Applies redactions using PyMuPDF (black out, fake data replacement, entity label)
- Writes `masked.pdf` back to the UC volume

### Shared Storage — Unity Catalog Volume
- Path configured via `UC_VOLUME_PATH` environment variable
- Per-session directory layout:
  ```
  {UC_VOLUME_PATH}/{session_id}/
      metadata.json          — session status and original filename (backend)
      original{ext}          — uploaded document, e.g. original.pdf (backend)
      entities.json          — extracted entities (document intelligence model → backend)
      masking_decisions.json — audit record: every entity with approved flag and replacement_text (backend, written before masking call)
      masked.pdf             — de-identified output (masking model)
  ```
- Accessed by the backend and both model serving endpoints via the
  Databricks Files REST API (`/api/2.0/fs/files`)

## Data Flow

### Single document mode

1. User uploads document → backend creates session in UC volume (`metadata.json` + `original{ext}`)
2. User selects field definitions → backend calls the Document Intelligence endpoint with `session_id` + `field_definitions`
3. Document Intelligence model fetches the document from UC via Files REST API
4. Document Intelligence model runs OCR + entity extraction; pre-generates replacement text for Fake Data / Entity Label strategies
5. Document Intelligence model writes `entities.json` to UC; status updated to `awaiting_review`
6. User reviews, edits replacement text, and approves entities in the frontend
7. Backend calls the Masking endpoint with `session_id` + approved entities
8. Masking model fetches `original{ext}` from UC, applies redactions, writes `masked.pdf` to UC
9. Backend updates session status to `completed`; frontend provides download link

### Batch mode (frontend-orchestrated)

The Dash frontend iterates the single-document flow automatically per file:

1. For each uploaded PDF, the frontend calls upload → process → approve-and-mask in sequence
2. After masking, the frontend calls `POST /api/sessions/{id}/verify`:
   - Backend fetches `masked.pdf` from UC and extracts its text layer (PyMuPDF)
   - Checks case-insensitively whether each entity's original text still appears
   - Returns a masking score (0–100) and per-entity verification result
3. Results are collected into a summary table with colour-coded verdicts:
   - ≥ 90%: Excellent
   - 70–89%: Review recommended
   - < 70%: Masking incomplete
4. "Process Another Batch" resets the batch state and deletes all session artefacts from UC

> Note: `verify` uses text extraction, so it returns score = 100 for scanned (image-only) PDFs regardless of actual redaction quality.

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Session storage | UC volume via Files REST API | Shared between backend and model without a database |
| Entity extraction | Databricks Model Serving | Scales independently; keeps heavy AI/OCR deps off backend |
| Masking | Databricks Model Serving | Keeps PyMuPDF off backend; model reads/writes UC directly |
| Document transfer to models | Models fetch from UC | Avoids sending large file bytes over HTTP; consistent with session-based architecture |
| Frontend | Dash (Python) | Deployable to Posit Connect without a separate Node build step |
| Deployment | Posit Connect | Existing enterprise deployment platform |
