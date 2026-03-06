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

### Backend (`backend/`)
- FastAPI application deployed to Posit Connect
- Handles file upload, orchestrates the workflow, and serves output files
- Delegates entity extraction to the Document Intelligence Model Serving endpoint
- Delegates masking to the Masking Model Serving endpoint
- Persists session state (metadata + binary artefacts) to a UC volume via the
  Databricks Files REST API (`/api/2.0/fs/files`)

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
      metadata.json     — session status and original filename (backend)
      original{ext}     — uploaded document, e.g. original.pdf (backend)
      entities.json     — extracted entities (document intelligence model → backend)
      masked.pdf        — de-identified output (masking model)
  ```
- Accessed by the backend and both model serving endpoints via the
  Databricks Files REST API (`/api/2.0/fs/files`)

## Data Flow

1. User uploads document → backend creates session in UC volume (`metadata.json` + `original{ext}`)
2. User selects field definitions → backend calls the Document Intelligence endpoint with `session_id` + `field_definitions`
3. Document Intelligence model fetches the document from UC via Files REST API
4. Document Intelligence model runs OCR + entity extraction; pre-generates replacement text for Fake Data / Entity Label strategies
5. Document Intelligence model writes `entities.json` to UC; status updated to `awaiting_review`
6. User reviews, edits replacement text, and approves entities in the frontend
7. Backend calls the Masking endpoint with `session_id` + approved entities
8. Masking model fetches `original{ext}` from UC, applies redactions, writes `masked.pdf` to UC
9. Backend updates session status to `completed`; frontend provides download link

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Session storage | UC volume via Files REST API | Shared between backend and model without a database |
| Entity extraction | Databricks Model Serving | Scales independently; keeps heavy AI/OCR deps off backend |
| Masking | Databricks Model Serving | Keeps PyMuPDF off backend; model reads/writes UC directly |
| Document transfer to models | Models fetch from UC | Avoids sending large file bytes over HTTP; consistent with session-based architecture |
| Frontend | Dash (Python) | Deployable to Posit Connect without a separate Node build step |
| Deployment | Posit Connect | Existing enterprise deployment platform |
