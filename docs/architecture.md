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
                         │  │  Model Serving endpoint       │   │
                         │  │  (MLflow PyFunc)              │   │
                         │  │                               │   │
                         │  │  • Azure Document Intelligence│   │
                         │  │  • Azure OpenAI / Claude      │   │
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
- Proxies PDF file serving through its own Flask routes to avoid cross-origin iframe issues

### Backend (`backend/`)
- FastAPI application deployed to Posit Connect
- Handles file upload, orchestrates masking, serves output files
- Delegates entity extraction to the Databricks Model Serving endpoint
- Persists session state (metadata + binary artefacts) to a UC volume via the
  Databricks Files REST API (`/api/2.0/fs/files`)

### Databricks Model (`databricks/model/`)
- MLflow PyFunc model (`DocumentIntelligenceModel`) registered in Unity Catalog
- Deployed to a Databricks Model Serving endpoint
- On each request, fetches the uploaded document from the UC volume via the Files REST API
- Runs Azure Document Intelligence (OCR) and Azure OpenAI / Claude Vision (entity extraction)
- Writes a flat `entities.json` to the session directory in the UC volume
- Returns entity list with normalised bounding boxes

### Shared Storage — Unity Catalog Volume
- Path configured via `UC_VOLUME_PATH` environment variable
- Per-session directory layout:
  ```
  {UC_VOLUME_PATH}/{session_id}/
      metadata.json     — session status and original filename
      original{ext}     — uploaded document (e.g. original.pdf)
      entities.json     — extracted entities (written by model, read by backend)
      masked.pdf        — de-identified output (written by backend)
  ```
- Accessed by both the backend and the model serving endpoint via the
  Databricks Files REST API (`/api/2.0/fs/files`)

## Data Flow

1. User uploads document → backend creates session in UC volume (`metadata.json` + `original{ext}`)
2. User selects field definitions → backend calls Model Serving endpoint with `session_id` + `field_definitions`
3. Model fetches the document from the UC volume via Files REST API
4. Model runs OCR (Azure Document Intelligence) + entity extraction (Azure OpenAI or Claude Vision)
5. Model writes `entities.json` to the session directory in UC
6. Backend parses the model response and saves entities; status updated to `awaiting_review`
7. User reviews and approves entities in the frontend
8. Backend applies visual masks using `MaskingService` (in-process, PyMuPDF)
9. Masked PDF written to UC volume; download URL returned to frontend

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Session storage | UC volume via Files REST API | Shared between backend and model without a database |
| Entity extraction | Databricks Model Serving | Scales independently; keeps heavy AI/OCR deps off backend |
| Document transfer to model | Model fetches from UC | Avoids sending large file bytes over HTTP; consistent with session-based architecture |
| Masking | In-process (FastAPI) | Low latency; no model serving overhead needed |
| Frontend | Dash (Python) | Deployable to Posit Connect without a separate Node build step |
| Deployment | Posit Connect | Existing enterprise deployment platform |
