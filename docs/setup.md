# PrivaSee — Local Development Setup

## Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Python | 3.11+ | Required by both backend and frontend_dash |
| Databricks workspace | — | For session storage and model serving |
| Azure subscription | — | For Azure Document Intelligence and Azure OpenAI |

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

Start the backend:

```bash
uvicorn app.main:app --reload --port 8000
# API docs: http://localhost:8000/docs
```

## 3. Dash frontend

```bash
cd frontend_dash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Start the frontend (pointing at the local backend):

```bash
API_BASE_URL=http://localhost:8000 python app.py
# UI: http://localhost:8050
```

## 4. Environment variables

### Backend (`backend/.env`)

| Variable | Required | Description |
|---|---|---|
| `DATABRICKS_HOST` | Yes | Workspace URL, e.g. `https://adb-xxxx.azuredatabricks.net` |
| `DATABRICKS_TOKEN` | Yes | PAT with UC Files API permissions |
| `DATABRICKS_MODEL_ENDPOINT` | Yes | Full Model Serving invocation URL |
| `UC_VOLUME_PATH` | Yes | UC volume base path, e.g. `/Volumes/catalog/schema/privasee_sessions` |
| `ALLOWED_ORIGINS` | Yes (prod) | Comma-separated CORS origins, e.g. `https://connect.example.com` |
| `MOCK_DATABRICKS` | No | Set `true` to skip Databricks and return mock entities (local dev only) |
| `MAX_FILE_SIZE_MB` | No | Upload size cap in MB (default: 10) |

### Dash frontend (`frontend_dash/.env` or shell env)

| Variable | Required | Description |
|---|---|---|
| `API_BASE_URL` | Yes | Backend URL, e.g. `http://localhost:8000` or the Posit Connect backend URL |

## 5. Databricks setup

### UC Volume

Create a Unity Catalog volume to store session artefacts:

```sql
CREATE VOLUME <catalog>.<schema>.privasee_sessions;
```

Set `UC_VOLUME_PATH=/Volumes/<catalog>/<schema>/privasee_sessions` in the backend env.

### Model Serving endpoint

1. Configure Databricks secrets for Azure and AI credentials (see `databricks/notebooks/register_model.py`)
2. Run `databricks/notebooks/register_model.py` to register the MLflow model in Unity Catalog
3. Run `databricks/notebooks/deploy_endpoint.py` to create the Model Serving endpoint
4. Note the endpoint invocation URL and set it as `DATABRICKS_MODEL_ENDPOINT` in the backend env
5. Add `DATABRICKS_HOST` and `DATABRICKS_TOKEN` as environment variables on the model serving endpoint config (required for the model to fetch documents from the UC volume)

### Required secrets in the Databricks endpoint environment

| Variable | Description |
|---|---|
| `DATABRICKS_HOST` | Workspace URL (same as backend) |
| `DATABRICKS_TOKEN` | Service principal token with Files API read access to the UC volume |
| `UC_VOLUME_PATH` | Same value as backend |
| `AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT` | Azure DI endpoint URL |
| `AZURE_DOCUMENT_INTELLIGENCE_KEY` | Azure DI API key |
| `VISION_SERVICE_PROVIDER` | `openai` (default) or `claude` |
| `AZURE_OPENAI_API_KEY` | Required when `VISION_SERVICE_PROVIDER=openai` |
| `AZURE_OPENAI_ENDPOINT` | Required when `VISION_SERVICE_PROVIDER=openai` |
| `ANTHROPIC_API_KEY` | Required when `VISION_SERVICE_PROVIDER=claude` |

## 6. Run the end-to-end test

Once both services are running and credentials are configured:

```bash
cd backend
API_BASE_URL=http://localhost:8000 python scripts/e2e_upload_test.py
```

This validates the full upload → process → approve-and-mask workflow against real Databricks infrastructure.
