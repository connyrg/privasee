# PrivaSee — Architecture

> TODO: expand this document once the implementation is complete.

## Overview

PrivaSee is a document de-identification tool that masks sensitive information
in single-page PDFs.  The target architecture has three independently
deployable components that communicate over HTTPS.

```
┌───────────────────────────────────────────────────────────────────┐
│                    Posit Connect                                   │
│                                                                   │
│  ┌──────────────────────┐      ┌──────────────────────────────┐  │
│  │  React Frontend      │─────▶│  FastAPI Backend             │  │
│  │  (static bundle)     │ HTTP │  (uvicorn / ASGI)            │  │
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
                         │  │  • Claude Vision              │   │
                         │  └──────────────────────────────┘   │
                         │                                      │
                         │  ┌──────────────────────────────┐   │
                         │  │  Unity Catalog Volume         │   │
                         │  │  (shared session state)       │   │
                         │  └──────────────────────────────┘   │
                         └─────────────────────────────────────┘
```

## Components

### Frontend (`frontend/`)
- React 18 + Vite, styled with Tailwind CSS
- Copied from the PoC with minimal changes
- Deployed as a static bundle to Posit Connect
- Communicates with the backend via a `/api` proxy

### Backend (`backend/`)
- FastAPI application deployed to Posit Connect
- Handles file upload, orchestrates masking, serves output PDFs
- Delegates entity extraction to the Databricks Model Serving endpoint
- Persists session state (metadata + binary artefacts) to a UC volume

### Databricks Model (`databricks/model/`)
- MLflow PyFunc model registered in Unity Catalog
- Deployed to a Databricks Model Serving endpoint
- Encapsulates Azure Document Intelligence (OCR) and Claude Vision (NER)
- Returns entity list with normalised bounding boxes

### Shared Storage — Unity Catalog Volume
- Path configured via `UC_VOLUME_PATH` environment variable
- Stores per-session JSON state and binary artefacts
- Accessed by the backend via the Databricks SDK Files API

## Data Flow

1. User uploads PDF → backend creates session in UC volume
2. Backend calls Model Serving endpoint with base64 page image
3. Model runs OCR + Claude entity extraction, returns entity list
4. Backend persists entities to UC volume session
5. User approves entities in frontend
6. Backend applies visual masks using `MaskingService` (in-process)
7. Masked PDF written to UC volume; download URL returned to frontend

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Session storage | UC volume | Shared between frontend and backend without a DB |
| Entity extraction | Databricks Model Serving | Scales independently; keeps large deps off backend |
| Masking | In-process (FastAPI) | Low latency; no model serving overhead needed |
| Deployment | Posit Connect | Existing enterprise deployment platform |
