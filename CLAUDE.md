# PrivaSee — Claude Code Guide

## Project Overview

PrivaSee is a document de-identification tool. Users upload documents, an AI extracts sensitive entities (names, dates, addresses, etc.), the user reviews them, and a masking model redacts the approved entities.

**Three independently deployable components:**
- `backend/` — FastAPI API server (Posit Connect)
- `frontend_dash/` — Dash UI (Posit Connect)
- `databricks/` — MLflow PyFunc models on Databricks Model Serving

The legacy `frontend/` (React) is no longer used.

---

## Architecture

```
Dash Frontend  ──HTTP──▶  FastAPI Backend  ──HTTP──▶  Databricks Model Serving
                                │                      (Document Intelligence)
                                │                      (Masking)
                                │
                                └──────────────────▶  Unity Catalog Volume
                                                       (session state store)
```

**No database.** All session state (metadata, files, entities, masked PDFs) lives in a UC volume:
```
{UC_VOLUME_PATH}/{session_id}/
  metadata.json         — session status, filename
  original{ext}         — uploaded document
  entities.json         — extracted entities with bounding boxes
  masked.pdf            — final redacted output
```

Accessed via Databricks Files REST API (`/api/2.0/fs/files`).

### Request Flow
1. `POST /api/upload` → backend creates session in UC
2. `POST /api/process` → backend calls Document Intelligence endpoint with `session_id` + `field_definitions`; model fetches document from UC, runs OCR + vision AI, writes `entities.json`
3. User reviews entities in UI
4. `POST /api/approve-and-mask` → backend calls Masking endpoint; model reads `original`, applies redactions, writes `masked.pdf`; in batch mode (`run_verification=True`) the model also re-OCRs the masked output and returns `occurrences_total`, `occurrences_masked`, `score`

---

## Running Tests

### Backend
```bash
cd backend
make test-fast        # unit + contract (fast feedback)
make test-unit        # unit only
make test-integration # full API with mocked Databricks
make test-all         # full suite + 70% coverage requirement
```

### Databricks models
```bash
cd databricks
python -m pytest tests/ -v
```

**Backend test structure:**
- `tests/unit/` — pure functions, no I/O
- `tests/integration/` — real FastAPI via `httpx.AsyncClient` + `ASGITransport`, mocked Databricks
- `tests/contracts/` — verify mock response shapes match real service schemas
- `pytest.ini` — `asyncio_mode = auto`

---

## Key Conventions

### Backend
- All endpoints in `app/main.py`; models in `app/models.py`
- Custom HTTP exception handler: `{"error": "...", "status_code": ...}` for 4xx/5xx; Pydantic 422 → `{"detail": [...]}`
- `UCSessionManager` uses GET-then-PUT for updates (preserves existing fields)
- `VALID_STATUSES = ["uploaded", "processing", "awaiting_review", "completed"]`

### Databricks models

**Model files (`databricks/model/`):**
- `document_intelligence.py` — main MLflow PyFunc model
- `openai_service.py` — GPT-5 vision calls (primary provider)
- `claude_service.py` — Databricks-hosted Claude (secondary provider; lower rate limits)
- `databricks_service.py` — provider routing / Databricks Foundation Model API
- `masking_model.py`, `masking_service.py` — masking MLflow model
- `fake_data_service.py`, `ocr_service.py`, `bbox_matcher.py` — supporting services

**Entity format:** Occurrences-based. Each `Entity` has `occurrences: List[Occurrence]`; each `Occurrence` carries `bounding_boxes: List[BoundingBox]` for one positional appearance in the document. LLM outputs compact array `[x, y, w, h]`; parser converts to `BoundingBox` dicts before merging.

**Performance conventions:**
- Compact bbox array format `[x, y, w, h]` in LLM prompt output (not dicts) — reduces token usage
- All call sites in `openai_service.py` use `reasoning_effort="low"` (GPT-5 default is medium — reduces latency with acceptable quality)
- OCR word list uses compact keys `{"t": text, "b": [x,y,w,h]}` (confidence field dropped)
- Per-page processing via `ThreadPoolExecutor` — one Vision API call per page concurrently

### MaskingService — form widget handling
- Form widgets are updated **in place** (`widget.field_value = new_val; widget.update()`) rather than painted over — painting does not clear the widget render layer
- **Multi-widget span** (one entity occurrence covering several widgets, e.g. DD/MM/YYYY): `_resolve_component_replacement()` aligns each widget to its slice of the replacement text
  - Separator split: tries `/`, `-`, `.`, whitespace — works when `FakeDataService` preserves the separator (dates, grouped IDs)
  - Character-level fallback: strips separators, ranks widgets by (y, x) reading order — works for 10 single-char Medicare widgets
  - Address spans: part count varies by replacement, falls back to assigning full replacement (acceptable)
- Component splitting only applies to **Fake Data** strategy; Redact (`[MASKED]`) and Entity Label apply the full value to every widget unchanged
- `FakeDataService` guarantees format preservation: `_reconstruct_date` mirrors separator + component order; `_preserve_structure` replaces char-by-char keeping separators intact — this is what makes separator/character splitting reliable

### Frontend (Dash)
- Single file: `frontend_dash/app.py`
- **Do NOT** use Flask proxy routes or `app.config.requests_pathname_prefix` — Posit Connect routing breaks them
- PDF display uses `data:application/pdf;base64,...` URIs fetched server-side
- Download links use `download="filename.pdf"` attribute, no `target="_blank"`
- Multiple callbacks write to `config-status` — `refresh_config_list` is the primary owner (no `allow_duplicate`); all others use `allow_duplicate=True` with `prevent_initial_call=True`
- **Batch mode** uses an interval-driven state machine: `store-batch-cursor` + `store-batch-phase` advance through upload → process (polled) → masking_run per file; `batch_tick` callback drives transitions on each `dcc.Interval` tick. Verification (`run_verification=True`) is bundled into the masking_run phase — no separate verify step

---

## Known Constraints

- **Nginx 60s hard timeout** on the Databricks driver proxy — managed by platform team, cannot be changed. Mitigate by keeping LLM output tokens low (`reasoning_effort="low"`, compact bbox format, `max_completion_tokens=10000`)
- **Databricks model serving timeout** is 120s (separate from nginx timeout above)
- **Posit Connect** controls `requests_pathname_prefix` — don't set it in app code
- **`asyncio.run()` corrupts the Databricks Serverless event loop** — use `ThreadPoolExecutor` for blocking I/O inside async model serving code instead
- **MLflow model signature enforced by Databricks Model Serving** — columns not declared in `input_schema` are silently dropped before `predict()` is called. The masking model signature (`register_masking_model.ipynb`) must include `run_verification` (boolean, optional) in `input_schema` and `occurrences_total`, `occurrences_masked`, `score` in `output_schema`. Re-register a new model version after any signature change.

---

## Testing Patterns

### Patching Databricks in backend integration tests
```python
monkeypatch.setattr(main_module, "MOCK_DATABRICKS", False)
monkeypatch.setattr(main_module, "DATABRICKS_MODEL_ENDPOINT", "https://fake/ep")
mock_http = AsyncMock()
mock_http.post.side_effect = httpx.TimeoutException("timed out")
with patch("app.main.httpx.AsyncClient") as MockClient:
    MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_http)
    MockClient.return_value.__aexit__ = AsyncMock(return_value=False)
    response = await client.post("/api/process", json=payload)
```

### Patching masking in backend tests
```python
with patch("app.main._apply_masking_sync", return_value=b"%PDF-1.4 fake %%EOF"):
    response = await client.post("/api/approve-and-mask", json=payload)
```

### Databricks model test fixtures
- `make_bbox(x, y, w, h)` → returns `[x, y, w, h]` array (compact format)
- `make_bbox_dict(x, y, w, h)` → returns `{"x":..., "y":..., "width":..., "height":...}` — only for `_merge_same_line_bboxes` tests (internal method takes dicts)
- `_create_pdf_with_widgets(spec)` → creates PDF with AcroForm text widgets; returns `(pdf_bytes, {field_name: fitz.Rect})`
- `_widget_values(result_bytes)` → returns `{field_name: field_value}` from first page — use to assert widget masking
- `_mock_widget(field_value, x0, ...)` → duck-type widget with `.field_value` and `.rect` — use for `_resolve_component_replacement` unit tests
- `_make_verify_model()` → `MaskingModel` with mocked `MaskingService`/`OCRService` pre-configured for `_verify_masking` tests; sets `adi_available=False`, `MIN_TEXT_LENGTH_FOR_DIGITAL=50`, `RENDER_ZOOM_FACTOR=2.0`, `_adi_max_concurrent=5`
- `_mock_fitz_ctx(...)` → context manager using `patch.dict(sys.modules, {"fitz": mock_fitz})` — required because `fitz` is imported locally inside `_verify_masking` (deferred import), so `patch("databricks.model.masking_model.fitz")` fails

---

## Documentation Maintenance

Three slash commands are available to help keep docs in sync (run them before committing):
- `/update-docs` — updates CLAUDE.md, TODO.md, and MEMORY.md
- `/update-readme` — updates README files and `docs/`
- `/update-docstrings` — updates inline docstrings in changed Python files

Before every commit, update these three files to keep them in sync:

| File | Audience | Contains | Update when |
|---|---|---|---|
| **CLAUDE.md** | Team + AI | Architecture, conventions, constraints, test/deploy instructions | Conventions or architecture change |
| **TODO.md** | Team + AI | Open bugs and planned features only | Work starts, completes, or new issues found |
| **MEMORY.md** | AI only (`.claude/`) | Current branch state, test count, what's implemented — things not derivable from the other two | End of every session |

---

## Deployment

### Posit Connect
```bash
# Frontend
rsconnect deploy dash --server <URL> --api-key <TOKEN> --title "PrivaSee" frontend_dash/

# Backend
rsconnect deploy fastapi --server <URL> --api-key <TOKEN> --title "PrivaSee API" --entrypoint app.main:app backend/
```

**Backend env vars:** `DATABRICKS_HOST`, `DATABRICKS_TOKEN`, `DATABRICKS_MODEL_ENDPOINT`, `DATABRICKS_MASKING_ENDPOINT`, `UC_VOLUME_PATH`, `ALLOWED_ORIGINS`

**Frontend env vars:** `API_BASE_URL`, `SSL_VERIFY`

### Databricks Model Serving
- Register and deploy via notebooks in `databricks/notebooks/`
- Two endpoints: Document Intelligence + Masking (independent)
- CI via GitHub Actions runs tests only — deployment is manual
