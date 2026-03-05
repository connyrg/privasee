# PrivaSee — Deployment Guide

## Overview

Each component is deployed independently via GitHub Actions:

| Component | Workflow | Target |
|---|---|---|
| Dash frontend | `.github/workflows/deploy-frontend.yml` | Posit Connect |
| Backend | `.github/workflows/deploy-backend.yml` | Posit Connect (FastAPI) |
| Databricks model | `.github/workflows/deploy-databricks.yml` | Databricks Model Serving |

---

## Posit Connect — Dash Frontend

Deploy from the `frontend_dash/` directory:

```bash
pip install rsconnect-python
rsconnect deploy dash \
  --server $POSIT_CONNECT_URL \
  --api-key $POSIT_CONNECT_TOKEN \
  --title "PrivaSee" \
  frontend_dash/
```

After deployment, set the following environment variables on the content item in the
Posit Connect UI (Content → Settings → Environment Variables):

| Variable | Value |
|---|---|
| `API_BASE_URL` | Deployed backend URL (e.g. `https://connect.example.com/privasee-api`) |
| `SSL_VERIFY` | `false` if the backend is behind a Posit Connect proxy with an internal/self-signed certificate; omit otherwise |

---

## Posit Connect — Backend

Deploy from the `backend/` directory:

```bash
pip install rsconnect-python
rsconnect deploy fastapi \
  --server $POSIT_CONNECT_URL \
  --api-key $POSIT_CONNECT_TOKEN \
  --title "PrivaSee API" \
  --entrypoint app.main:app \
  backend/
```

After deployment, set the following environment variables on the content item in the
Posit Connect UI (Content → Settings → Environment Variables):

| Variable | Value |
|---|---|
| `DATABRICKS_HOST` | Workspace URL |
| `DATABRICKS_TOKEN` | Personal access token |
| `DATABRICKS_MODEL_ENDPOINT` | Full serving endpoint invocation URL |
| `UC_VOLUME_PATH` | `/Volumes/<catalog>/<schema>/privasee_sessions` |
| `ALLOWED_ORIGINS` | Deployed Dash frontend URL (e.g. `https://connect.example.com/privasee`) |

---

## Databricks — Model Serving

### 1. Register the model

Run `databricks/notebooks/register_model.py` in your Databricks workspace. This logs
the `DocumentIntelligenceModel` as an MLflow PyFunc model and registers it in Unity Catalog.

### 2. Deploy the endpoint

Run `databricks/notebooks/deploy_endpoint.py` to create or update the Model Serving
endpoint. Adjust `ENDPOINT_NAME` and `MODEL_VERSION` at the top of the notebook.

### 3. Configure endpoint environment variables

In the Databricks UI (Serving → your endpoint → Edit endpoint → Environment variables),
set:

| Variable | Description |
|---|---|
| `DATABRICKS_HOST` | Workspace URL — used by the model to fetch documents from UC |
| `DATABRICKS_TOKEN` | Service principal token with Files API read/write access to UC volume |
| `UC_VOLUME_PATH` | Same value as the backend |
| `AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT` | Azure DI endpoint URL |
| `AZURE_DOCUMENT_INTELLIGENCE_KEY` | Azure DI API key |
| `VISION_SERVICE_PROVIDER` | `openai` (default) or `claude` |
| `AZURE_OPENAI_API_KEY` | Required when provider is `openai` |
| `AZURE_OPENAI_ENDPOINT` | Required when provider is `openai` |
| `ANTHROPIC_API_KEY` | Required when provider is `claude` |

### 4. Update after a model version bump

Re-run `deploy_endpoint.py` with the updated model version number. The endpoint
will perform a rolling update with zero downtime.

---

## GitHub Secrets

The following secrets must be configured in the repository
(Settings → Secrets and variables → Actions):

| Secret | Used by |
|---|---|
| `POSIT_CONNECT_URL` | deploy-frontend, deploy-backend |
| `POSIT_CONNECT_TOKEN` | deploy-frontend, deploy-backend |
| `DATABRICKS_HOST` | deploy-databricks, deploy-backend |
| `DATABRICKS_TOKEN` | deploy-databricks, deploy-backend |
| `DATABRICKS_MODEL_ENDPOINT` | deploy-backend |
| `UC_VOLUME_PATH` | deploy-backend |
| `ALLOWED_ORIGINS` | deploy-backend |

---

## Rollback

- **Frontend / Backend (Posit Connect):** redeploy a previous bundle via the
  Posit Connect web UI or by re-running the workflow on an earlier commit.
- **Databricks model:** change the endpoint's served model version in the
  Databricks UI or re-run `deploy_endpoint.py` with the desired version number.
