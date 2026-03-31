# PrivaSee

Document de-identification tool — intelligently masks sensitive information
in PDFs and images using Azure Document Intelligence, Claude Vision, and Azure OpenAI.

Supports both **single document** and **batch** processing modes.

## Architecture

Three independently deployable components:

| Component | Location | Deployed to |
|---|---|---|
| Dash frontend | `frontend_dash/` | Posit Connect |
| FastAPI backend | `backend/` | Posit Connect |
| MLflow model | `databricks/` | Databricks Model Serving |

Session state and binary artefacts are shared via a Unity Catalog volume.

See [docs/architecture.md](docs/architecture.md) for the full architecture diagram.

## Quick start (local development)

```bash
cp backend/.env.template backend/.env
# fill in backend/.env with your credentials

# Terminal 1 — backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Terminal 2 — Dash frontend
cd frontend_dash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
API_BASE_URL=http://localhost:8000 python app.py
# Frontend: http://localhost:8050
# Backend:  http://localhost:8000
```

See [docs/setup.md](docs/setup.md) for detailed prerequisites and setup.

## Deployment

See [docs/deployment.md](docs/deployment.md) for CI/CD pipeline details.

## Repository structure

```
privasee/
├── .claude/commands/        Shared Claude Code skills (/update-docs, /update-readme, /update-docstrings)
├── .github/workflows/       CI/CD pipelines
├── frontend_dash/           Dash frontend (primary UI)
│   ├── app.py               Entry point — deployable via rsconnect deploy dash
│   ├── assets/              Static assets (custom.css)
│   ├── README.md
│   └── requirements.txt
├── frontend/                Legacy React + Vite app (not actively deployed)
├── backend/
│   ├── app/
│   │   ├── main.py              FastAPI application — all endpoints
│   │   ├── models.py            Pydantic request/response models
│   │   ├── config_manager.py    Named config persistence on UC volume
│   │   └── session_manager.py   UC volume session persistence (Files REST API)
│   ├── tests/
│   ├── scripts/
│   │   └── e2e_upload_test.py   End-to-end workflow validation script
│   ├── requirements.txt
│   └── .env.template
├── databricks/
│   ├── model/               MLflow PyFunc model source
│   │   ├── document_intelligence.py  Document intelligence model — OCR + entity extraction
│   │   ├── masking_model.py          Masking model — applies redactions to PDF in UC
│   │   ├── masking_service.py        PyMuPDF redaction engine (used by masking model)
│   │   ├── fake_data_service.py      Faker-based replacement text generator
│   │   ├── ocr_service.py            Azure Document Intelligence OCR
│   │   ├── openai_service.py         Azure OpenAI vision entity extraction
│   │   ├── claude_service.py         Claude vision entity extraction (alternative)
│   │   └── bbox_matcher.py           Entity-to-word bounding box alignment
│   ├── notebooks/
│   │   ├── register_model.py    MLflow model registration in Unity Catalog
│   │   └── deploy_endpoint.py   Model Serving endpoint deployment
│   └── utils/
├── docs/
│   ├── architecture.md
│   ├── setup.md
│   └── deployment.md
└── start.sh                 Legacy local dev launcher (React frontend)
```

