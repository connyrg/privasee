# PrivaSee

Document de-identification tool — intelligently masks sensitive information
in single-page PDFs using Azure Document Intelligence and Claude Vision.

## Architecture

Three independently deployable components:

| Component | Location | Deployed to |
|---|---|---|
| React frontend | `frontend/` | Posit Connect |
| FastAPI backend | `backend/` | Posit Connect |
| MLflow model | `databricks/` | Databricks Model Serving |

Session state and binary artefacts are shared via a Unity Catalog volume.

See [docs/architecture.md](docs/architecture.md) for the full architecture diagram.

## Quick start (local development)

```bash
cp backend/.env.template backend/.env
# fill in backend/.env with your credentials

./start.sh
# Frontend: http://localhost:5173
# Backend:  http://localhost:8000
```

See [docs/setup.md](docs/setup.md) for detailed prerequisites and setup.

## Deployment

See [docs/deployment.md](docs/deployment.md) for CI/CD pipeline details.

## Repository structure

```
privasee/
├── .github/workflows/       CI/CD pipelines
├── frontend/                React + Vite app (copied from PoC)
├── backend/
│   ├── app/
│   │   ├── main.py          FastAPI application entry point
│   │   ├── models.py        Pydantic data models
│   │   ├── session_manager.py  UC volume session persistence
│   │   └── services/
│   │       ├── masking_service.py   Visual PDF masking (from PoC)
│   │       └── mapping_manager.py  Consistent entity replacement (from PoC)
│   ├── tests/
│   ├── requirements.txt
│   └── .env.template
├── databricks/
│   ├── model/               MLflow PyFunc model source
│   ├── notebooks/           Registration and deployment notebooks
│   └── tests/
├── docs/
└── start.sh                 Local dev launcher
```

## Status

This repository is a migration of the
[PoC](https://github.com/nkranthiram/privasee) to the target three-component
architecture.  The structure and placeholder files are in place; implementation
will be added in subsequent steps.
