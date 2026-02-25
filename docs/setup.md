# PrivaSee — Local Development Setup

> TODO: verify and expand these instructions once the implementation is complete.

## Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Python | 3.11+ | |
| Node.js | 20+ | |
| npm | 10+ | |
| Poppler | any | Required by pdf2image — `brew install poppler` on macOS |
| Databricks workspace | — | For session storage and model serving |

## 1. Clone the repository

```bash
git clone https://github.com/your-org/privasee.git
cd privasee
```

## 2. Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.template .env
# Edit .env and fill in all required values (see comments in the file)
```

## 3. Frontend

```bash
cd frontend
npm install
```

## 4. Run locally

From the repository root:

```bash
./start.sh
```

This starts:
- FastAPI backend on http://localhost:8000
- React dev server on http://localhost:5173 (with `/api` proxied to the backend)

## 5. Environment variables

See `backend/.env.template` for the full list of required variables.

Key variables:

| Variable | Description |
|---|---|
| `DATABRICKS_HOST` | Databricks workspace URL |
| `DATABRICKS_TOKEN` | Personal access token |
| `DATABRICKS_MODEL_ENDPOINT` | Model Serving invocation URL |
| `UC_VOLUME_PATH` | UC volume base path for sessions |
| `ALLOWED_ORIGINS` | Comma-separated CORS allowed origins |

## 6. Databricks setup

TODO: document:
- Creating the UC volume
- Configuring Databricks secrets (Azure + Anthropic keys)
- Running register_model.py and deploy_endpoint.py notebooks
