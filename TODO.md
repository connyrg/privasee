# PrivaSee — Known Issues / TODO

## Bugs

### ~~1. Split form fields masked with full replacement text~~ ✅ Fixed

**Area:** Databricks model — `masking_service.py`
**Fixed in:** `feat/llm-bbox-extraction`

**What was wrong:** When an entity spanned multiple separate form field widgets (e.g. DD / MM / YYYY
in three separate input boxes), every widget was replaced with the full replacement text
(e.g. `"02/03/2025"`) instead of its aligned component (`"02"`, `"03"`, `"2025"`).
Only affected interactive PDF form widgets — regular text bboxes were unaffected.

**Fix:** Added `MaskingService._resolve_component_replacement()` which:
1. **Separator split** — tries `/`, `-`, `.`, then whitespace; if original and replacement split
   into the same number of parts and the widget value matches a part, returns the aligned
   replacement part. Uses widget x-position to break ties when the same value appears in
   multiple parts (e.g. DD == MM == `"01"`).
2. **Character-level fallback** — strips separators, ranks widgets by reading order (y, x),
   maps to corresponding character(s) in the replacement. Handles 10-widget single-char
   Medicare numbers.

**Scope / limitations:**
- ✅ Dates (DD/MM/YYYY, D-M-Y, etc.) — separator preserved by `FakeDataService._reconstruct_date`
- ✅ Medicare grouped widgets (5+4+1 or similar) — separator preserved by `_preserve_structure`
- ✅ Medicare 10 individual single-char widgets — character-level alignment
- ❌ Address multi-widget spans — component count varies, falls back to previous behaviour (no regression)
- Only applied for **Fake Data** strategy; Redact and Entity Label assign the full value unchanged (correct)

---

### ~~2. Masking text misaligned on image documents~~ ✅ Fixed

**Area:** Databricks model — `masking_service.py`
**Fixed in:** `feat/llm-bbox-extraction`

**Root cause:** `insert_text` origin was always `(rect.x0 + 2, rect.y1 - 2)` in raw PDF
coords.  For pages with `rotation=180` or `rotation=270` (common in scanned PDFs captured
upside-down or sideways), `rect.x0` maps to the *display-right* edge of the box — so text
started at the right edge and extended further right, landing outside the box.

**Fix:** Rotation-aware origin selection in `apply_pdf_masks`:
- `rotation=0/90`:  `(rect.x0 + 2, rect.y1 - 2)` — raw left = display left ✓
- `rotation=180`:   `(rect.x1 - 2, rect.y1 - 2)` — raw right = display left ✓
- `rotation=270`:   `(rect.x0 + 2, rect.y0 + 2)` — raw y₀ = display left ✓

---

### 3. Masked output always PDF regardless of original format
**Area:** Databricks model — `masking_model.py` / `masking_service.py`
**Description:** When the original document is a PNG or JPG image, the masked output is
always returned as a PDF (PIL wraps the masked image in a single-page PDF). The output
should match the original format — a PNG input should produce a masked PNG, a JPG input
should produce a masked JPG at the same quality/resolution.
**Expected:** Output format and image quality mirror the original upload.
**Open questions:**
- Where is the PDF wrapping done? (`masking_model.py` likely wraps PIL output in PDF before writing to UC)
- Is the masked PNG/JPG written to UC directly, or does it always go through the PDF wrapper?
- Does the backend/frontend assume the masked file is always `masked.pdf`, or is the filename/extension flexible?
- For JPG: what quality level should be used (original EXIF quality, a fixed value, lossless)?

---

### 4. Claude on Databricks hitting 429 rate limits
**Area:** Databricks model — `databricks_service.py`
**Description:** Databricks-hosted Claude has much lower rate limits than Azure OpenAI.
Concurrent per-page calls frequently hit 429 errors.
**Expected:** Graceful handling — backoff, retry, and/or reduced concurrency when using Claude.
**Approach options:**
- Exponential backoff with jitter on 429 (beyond built-in `max_retries`)
- Reduce `ThreadPoolExecutor` `max_workers` when provider is Claude
- Token-bucket rate limiter to cap requests/sec
**Open questions:**
- What is the exact rate limit (RPM or TPM)? For Claude Haiku 4.6, the TPM limit is 200,000.

---

### ~~5. Improvement to Batch mode result table~~ ✅ Fixed

**Area:** Databricks model — `masking_model.py`; Backend — `app/main.py`, `app/models.py`; Frontend — `frontend_dash/app.py`
**Fixed in:** `feat/llm-bbox-extraction`

**What was done:** Moved verification from a separate `/api/sessions/{id}/verify` backend endpoint into the Masking Databricks model itself. The masking model now accepts `run_verification` (bool, default `false`); in batch mode the frontend sends `true` and the model runs a two-pass hybrid re-OCR on the masked output (PyMuPDF text extraction for digital PDF pages, Azure Document Intelligence for scanned pages). Returns `occurrences_total`, `occurrences_masked`, and `score`. The batch results table now shows File | Entities | Occurrences Found | Occurrences Masked | Score | Status | Download.

**Key design choices:**
- Page classification uses the ORIGINAL file's text layer (not masked output) to avoid false "digital" classification from `insert_text()` replacement text on scanned pages
- Two-pass `ThreadPoolExecutor` pattern mirrors `OCRService._process_pdf` — fitz doc closed before concurrent ADI calls
- Scanned pages without ADI credentials are treated as masked (safe fallback) with a warning log
- Single-file mode leaves `run_verification=False` to avoid ADI latency
