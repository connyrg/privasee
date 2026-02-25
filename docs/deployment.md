# PrivaSee — Deployment Guide

> TODO: expand once CI/CD pipelines are implemented.

## Overview

Each component is deployed independently via GitHub Actions:

| Component | Workflow | Target |
|---|---|---|
| Frontend | `.github/workflows/deploy-frontend.yml` | Posit Connect (static) |
| Backend | `.github/workflows/deploy-backend.yml` | Posit Connect (FastAPI) |
| Databricks model | `.github/workflows/deploy-databricks.yml` | Databricks Model Serving |

## Posit Connect — Frontend

TODO: document:
- rsconnect manifest format for static React apps
- Required Posit Connect content settings (vanity URL, access control)
- Environment variable injection

## Posit Connect — Backend

TODO: document:
- rsconnect FastAPI deployment command
- Setting environment variables on the content item
- CORS configuration for the frontend origin

## Databricks — Model Serving

TODO: document:
- Unity Catalog model registration steps
- Endpoint configuration (compute size, scale-to-zero)
- Secret scope setup for Azure and Anthropic credentials
- Updating the endpoint after a model version bump

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

## Rollback

- **Frontend / Backend (Posit Connect):** redeploy a previous bundle via the
  Posit Connect web UI or by re-running the workflow on an earlier commit.
- **Databricks model:** change the endpoint's served model version in the
  Databricks UI or re-run deploy_endpoint.py with the desired version number.
