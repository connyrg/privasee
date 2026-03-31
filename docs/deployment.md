# PrivaSee — Deployment Guide

## Overview

Each component is deployed independently via manual `rsconnect` or Databricks notebook commands.
GitHub Actions workflows run CI checks (tests, lint, build) on pull requests but do not deploy automatically.

| Component | Deployed to | How |
|---|---|---|
| Dash frontend | Posit Connect | `rsconnect deploy dash` (manual) |
| FastAPI backend | Posit Connect | `rsconnect deploy fastapi` (manual) |
| Databricks models | Databricks Model Serving | Run notebooks manually (see below) |

CI workflows (`.github/workflows/`) run on PRs to catch failures early:

| Workflow | Trigger | What it does |
|---|---|---|
| `deploy-backend.yml` | PRs touching `backend/` | Runs the full backend test suite |
| `deploy-frontend.yml` | PRs/pushes touching `frontend/` | Lints and builds the legacy React frontend (not actively deployed) |
| `deploy-databricks.yml` | PRs/pushes touching `databricks/` | Placeholder — Databricks CI not yet configured |

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
| `DATABRICKS_MODEL_ENDPOINT` | Document Intelligence Model Serving invocation URL |
| `DATABRICKS_MASKING_ENDPOINT` | Masking Model Serving invocation URL |
| `UC_VOLUME_PATH` | `/Volumes/<catalog>/<schema>/privasee_sessions` |
| `ALLOWED_ORIGINS` | Deployed Dash frontend URL (e.g. `https://connect.example.com/privasee`) |

---

## Databricks — Model Serving

There are two independent Model Serving endpoints: one for document intelligence (OCR + entity extraction) and one for masking (PDF redaction).

### Document Intelligence endpoint

#### 1. Register the model

Run `databricks/notebooks/register_model.py` in your Databricks workspace. This logs
`DocumentIntelligenceModel` as an MLflow PyFunc model and registers it in Unity Catalog.

#### 2. Deploy the endpoint

Run `databricks/notebooks/deploy_endpoint.py`. Adjust `ENDPOINT_NAME` and `MODEL_VERSION`
at the top of the notebook.

#### 3. Configure endpoint environment variables

In the Databricks UI (Serving → your endpoint → Edit endpoint → Environment variables), set:

| Variable | Description |
|---|---|
| `DATABRICKS_HOST` | Workspace URL |
| `DATABRICKS_TOKEN` | Service principal token with Files API read/write access to UC volume |
| `UC_VOLUME_PATH` | Same value as the backend |
| `ADI_TENANT_ID` | Azure tenant ID for ADI OAuth |
| `ADI_CLIENT_ID` | OAuth client ID for Azure Document Intelligence |
| `ADI_CLIENT_SECRET` | OAuth client secret for Azure Document Intelligence |
| `ADI_ENDPOINT` | APIM endpoint URL for Azure Document Intelligence |
| `ADI_APPSPACE_ID` | AppSpace ID (default: `A-007100`) |
| `ADI_MODEL_ID` | Document Intelligence model ID (default: `prebuilt-layout`) |
| `VISION_SERVICE_PROVIDER` | `openai` (default) or `claude` |
| `AZURE_OPENAI_API_KEY` | Required when provider is `openai` |
| `AZURE_OPENAI_ENDPOINT` | Required when provider is `openai` |
| `ANTHROPIC_API_KEY` | Required when provider is `claude` |

### Masking endpoint

#### 1. Register the model

Run `databricks/notebooks/register_masking_model.ipynb` in your Databricks workspace.
Register it under a separate name (e.g. `privasee_masking`) in Unity Catalog.

#### 2. Deploy the endpoint

Run `databricks/notebooks/deploy_masking_endpoint.py` with the masking model name and version.

#### 3. Configure endpoint environment variables

| Variable | Required | Description |
|---|---|---|
| `DATABRICKS_HOST` | Yes | Workspace URL |
| `DATABRICKS_TOKEN` | Yes | Service principal token with Files API read/write access to UC volume |
| `UC_VOLUME_PATH` | Yes | Same value as the backend |
| `ADI_TENANT_ID` | For scanned PDFs | Azure tenant ID for ADI OAuth |
| `ADI_CLIENT_ID` | For scanned PDFs | OAuth client ID for Azure Document Intelligence |
| `ADI_CLIENT_SECRET` | For scanned PDFs | OAuth client secret for Azure Document Intelligence |
| `ADI_ENDPOINT` | For scanned PDFs | APIM endpoint URL for Azure Document Intelligence |
| `ADI_APPSPACE_ID` | For scanned PDFs | AppSpace ID (default: `A-007100`) |
| `ADI_MODEL_ID` | For scanned PDFs | Document Intelligence model ID (default: `prebuilt-layout`) |

The ADI variables are only needed when batch mode verification is used on scanned PDFs.
Without them, `run_verification=True` requests will treat scanned pages as fully masked
(score reported as 100% for those pages) rather than re-OCR'ing them.

#### 4. Update after a model version bump

Re-run `deploy_endpoint.py` with the updated model version number. The endpoint
will perform a rolling update with zero downtime.

---

## GitHub Secrets

The following secrets are used by CI workflows
(Settings → Secrets and variables → Actions).
Deployment credentials are set directly on the Posit Connect content item or Databricks endpoint — not stored as GitHub secrets.

| Secret | Used by |
|---|---|
| `DATABRICKS_HOST` | `deploy-databricks` (when Databricks CI is configured) |
| `DATABRICKS_TOKEN` | `deploy-databricks` (when Databricks CI is configured) |

---

## Rollback

- **Frontend / Backend (Posit Connect):** redeploy a previous bundle via the
  Posit Connect web UI or by re-running the workflow on an earlier commit.
- **Databricks model:** change the endpoint's served model version in the
  Databricks UI or re-run `deploy_endpoint.py` with the desired version number.
