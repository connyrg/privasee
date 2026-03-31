# PrivaSee — Dash Frontend

Single-file Dash application deployed to Posit Connect. All UI logic lives in `app.py`.

## Structure

```
frontend_dash/
├── app.py              Entry point — all layout, callbacks, and Flask proxy routes
├── assets/
│   └── custom.css      Custom styles
└── requirements.txt
```

## Running locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

API_BASE_URL=http://localhost:8000 python app.py
# UI: http://localhost:8050
```

Set `MOCK_DATABRICKS=true` on the backend to skip real Databricks calls during local development.

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `API_BASE_URL` | Yes | Backend URL, e.g. `http://localhost:8000` or the Posit Connect backend URL |
| `SSL_VERIFY` | No | Set `false` when the backend uses an internal/self-signed certificate. Defaults to `true`. |

## Processing modes

**Single mode** — one document at a time:
1. Upload a PDF or image
2. Configure field definitions (or load a saved config / system template)
3. Process → review extracted entities → approve and mask
4. Compare original vs masked; download masked PDF

**Batch mode** — multiple documents automatically:
1. Upload multiple files
2. Configure field definitions
3. Process All — frontend iterates upload → process → mask → verify per file
4. Results table with per-file entity counts and masking score (≥90% Excellent, 70–89% Review recommended, <70% Masking incomplete)

## Key architecture notes

- **Single file** — all layout helpers, callbacks, and Flask proxy routes are in `app.py` (~2400 lines)
- **No Flask proxy routes for PDF display** — PDFs are fetched server-side by Dash callbacks and returned as `data:application/pdf;base64,...` URIs; do not use Flask routes or iframe src paths for this (Posit Connect routing breaks them)
- **Do not set `app.config.requests_pathname_prefix`** — Posit Connect controls this value
- **Batch state machine** — `store-batch-cursor` + `store-batch-phase` drive a `dcc.Interval`-based state machine; `batch_tick` and `batch_do_blocking` callbacks advance through phases per file
- **Config status output** — multiple callbacks write to `config-status`; `refresh_config_list` is the primary owner (no `allow_duplicate`); all others use `allow_duplicate=True` with `prevent_initial_call=True`

## Deployment

```bash
pip install rsconnect-python
rsconnect deploy dash \
  --server $POSIT_CONNECT_URL \
  --api-key $POSIT_CONNECT_TOKEN \
  --title "PrivaSee" \
  frontend_dash/
```

After deployment, set `API_BASE_URL` and (if needed) `SSL_VERIFY` in the Posit Connect content item's environment variables.

See [../docs/deployment.md](../docs/deployment.md) for full deployment instructions.
