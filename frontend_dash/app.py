"""
PrivaSee — Dash frontend
Deployable on Posit Connect via: rsconnect deploy dash --title "PrivaSee" .

Set API_BASE_URL env var to point at the deployed FastAPI backend.
"""
# export http_proxy="" && export https_proxy="" && rsconnect deploy dash  --server  https://sds-posit-connect-prod.int.corp.sun/ --api-key $POSIT_CONNECT_API_KEY --entrypoint app.py [--new] -t PrivaSee . --insecure  --exclude venv/

from __future__ import annotations

import base64
import os
import time
import uuid

import dash
import dash_bootstrap_components as dbc
import requests as req
from dash import ALL, Input, Output, State, callback, ctx, dcc, html, no_update
from dash.exceptions import PreventUpdate
from dotenv import load_dotenv
from flask import Response, request as flask_request

load_dotenv()
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
# Set SSL_VERIFY=false in Posit Connect env vars when the backend uses an
# internal/self-signed certificate that the Python runtime doesn't trust.
_ssl_verify_env = os.getenv("SSL_VERIFY", "true").lower()
SSL_VERIFY: bool | str = False if _ssl_verify_env == "false" else True

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP],
    suppress_callback_exceptions=True,  # required for pattern-matching callbacks
    title="PrivaSee",
)
server = app.server  # expose Flask server for rsconnect

# ---------------------------------------------------------------------------
# Flask PDF proxy routes
# Proxy FastAPI file-serving through the Dash server so iframes stay same-origin.
# ---------------------------------------------------------------------------


@server.route("/pdf/original/<session_id>")
def proxy_original_pdf(session_id: str) -> Response:
    url = f"{API_BASE_URL}/api/files/uploads/{session_id}.pdf"
    headers = {
        "accept": "application/json",
        "Authorization": f"Key {os.environ.get('POSIT_CONNECT_API_KEY', '')}"
    }
    try:
        r = req.get(url, headers=headers, timeout=30, verify=SSL_VERIFY)
        r.raise_for_status()
        as_download = flask_request.args.get("dl") == "1"
        headers = {"Content-Type": "application/pdf"}
        if as_download:
            headers["Content-Disposition"] = f"attachment; filename={session_id}.pdf"
        return Response(r.content, headers=headers)
    except Exception:
        return Response("PDF not available", status=404)


@server.route("/pdf/masked/<session_id>")
def proxy_masked_pdf(session_id: str) -> Response:
    url = f"{API_BASE_URL}/api/files/output/{session_id}_masked.pdf"
    headers = {
        "accept": "application/json",
        "Authorization": f"Key {os.environ.get('POSIT_CONNECT_API_KEY', '')}"
    }
    try:
        r = req.get(url, headers=headers, timeout=30, verify=SSL_VERIFY)
        r.raise_for_status()
        as_download = flask_request.args.get("dl") == "1"
        headers = {"Content-Type": "application/pdf"}
        if as_download:
            headers["Content-Disposition"] = f"attachment; filename={session_id}_masked.pdf"
        return Response(r.content, headers=headers)
    except Exception:
        return Response("PDF not available", status=404)


# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

STRATEGIES = ["Fake Data", "Black Out", "Entity Label"]


def _new_field(name: str = "", description: str = "", strategy: str = "Fake Data") -> dict:
    return {"id": str(uuid.uuid4()), "name": name, "description": description, "strategy": strategy}


DEFAULT_FIELDS = [_new_field()]

STRATEGY_GUIDE = (
    "Fake Data — replace with realistic synthetic values. "
    "Black Out — redact with a solid black rectangle. "
    "Entity Label — replace with a labelled placeholder (e.g. Full_Name_1)."
)

# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------


def _navbar() -> dbc.Navbar:
    return dbc.Navbar(
        dbc.Container(
            [
                html.Span(
                    [html.I(className="bi bi-shield-lock-fill me-2 text-primary"), "PrivaSee"],
                    className="navbar-brand mb-0",
                ),
                html.Small("Document De-identification", className="text-muted"),
            ],
            fluid=True,
        ),
        color="white",
        className="shadow-sm mb-4",
    )


def _step_indicator_content(step: int) -> list:
    labels = ["Configure", "Review", "Compare"]
    items = []
    for i, label in enumerate(labels, start=1):
        if i < step:
            badge_cls = "step-badge done"
            label_cls = "text-success fw-semibold ms-2"
            badge_text = html.I(className="bi bi-check-lg")
        elif i == step:
            badge_cls = "step-badge active"
            label_cls = "text-primary fw-semibold ms-2"
            badge_text = str(i)
        else:
            badge_cls = "step-badge inactive"
            label_cls = "text-muted ms-2"
            badge_text = str(i)
        items.append(
            html.Div(
                [html.Span(badge_text, className=badge_cls), html.Span(label, className=label_cls)],
                className="d-flex align-items-center",
            )
        )
        if i < 3:
            items.append(html.Div(className="flex-grow-1 mx-3", style={"height": "1px", "background": "#dee2e6"}))
    return items


def _field_row(field: dict, idx: int, total: int) -> html.Div:
    """Render one field-definition row."""
    return html.Div(
        dbc.Row(
            [
                dbc.Col(
                    dbc.Input(
                        id={"type": "field-name", "index": field["id"]},
                        value=field["name"],
                        placeholder="e.g. Full Name",
                        debounce=True,
                        size="sm",
                    ),
                    width=3,
                ),
                dbc.Col(
                    dbc.Input(
                        id={"type": "field-desc", "index": field["id"]},
                        value=field["description"],
                        placeholder="e.g. Patient's full name",
                        debounce=True,
                        size="sm",
                    ),
                    width=5,
                ),
                dbc.Col(
                    dcc.Dropdown(
                        id={"type": "field-strategy", "index": field["id"]},
                        options=[{"label": s, "value": s} for s in STRATEGIES],
                        value=field["strategy"],
                        clearable=False,
                        style={"fontSize": "0.875rem"},
                    ),
                    width=3,
                ),
                dbc.Col(
                    dbc.Button(
                        html.I(className="bi bi-trash"),
                        id={"type": "delete-btn", "index": field["id"]},
                        color="outline-danger",
                        size="sm",
                        disabled=total <= 1,
                    ),
                    width=1,
                    className="d-flex align-items-center",
                ),
            ],
            className="g-2",
        ),
        className="field-row",
    )


# ---------------------------------------------------------------------------
# Step 1 — Configure (static shell; dynamic parts populated by callbacks)
# ---------------------------------------------------------------------------


def _step1_layout() -> html.Div:
    return html.Div(
        [
            # Upload card
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H5("Upload Document", className="card-title mb-3"),
                        dcc.Upload(
                            id="pdf-upload",
                            children=html.Div(
                                [
                                    html.I(className="bi bi-cloud-upload fs-2 text-muted"),
                                    html.P("Drag & drop a PDF, or click to browse", className="mt-2 mb-1 text-muted"),
                                    html.Small("PDF only · max 10 MB", className="text-muted"),
                                ]
                            ),
                            accept=".pdf",
                            className="upload-zone",
                        ),
                        # File card / error shown here after upload attempt
                        html.Div(id="upload-status", className="mt-2"),
                        # Session info banner (page count etc.)
                        html.Div(id="session-banner", className="mt-2"),
                    ]
                ),
                className="mb-4",
            ),
            # Field definitions card
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H5("De-identification Rules", className="card-title mb-1"),
                        html.Small(STRATEGY_GUIDE, className="text-muted d-block mb-3"),
                        # Column headers
                        dbc.Row(
                            [
                                dbc.Col(html.Small("Field Name", className="text-muted fw-semibold"), width=3),
                                dbc.Col(html.Small("Description", className="text-muted fw-semibold"), width=5),
                                dbc.Col(html.Small("Strategy", className="text-muted fw-semibold"), width=3),
                                dbc.Col(width=1),
                            ],
                            className="g-2 mb-2 px-2",
                        ),
                        # Dynamic field rows injected by render_fields callback
                        html.Div(id="fields-container"),
                        dbc.Button(
                            [html.I(className="bi bi-plus-lg me-1"), "Add Field"],
                            id="add-field-btn",
                            color="outline-secondary",
                            size="sm",
                            className="mt-2",
                        ),
                        # Validation warning
                        html.Div(id="fields-warning", className="mt-2"),
                    ]
                ),
                className="mb-4",
            ),
            # Process button
            dbc.Button(
                [html.I(className="bi bi-play-fill me-2"), "Process Document"],
                id="process-btn",
                color="primary",
                size="lg",
                className="w-100",
                disabled=True,
            ),
            # Loading indicator shown while processing
            dcc.Loading(html.Div(id="process-loading"), type="circle", color="#0284c7"),
        ],
        id="step-1-content",
    )


# ---------------------------------------------------------------------------
# Step 2 — Review (shown when store-step == 2)
# ---------------------------------------------------------------------------


def _step2_layout() -> html.Div:
    return html.Div(
        [
            dbc.Card(
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                dbc.Col(html.H5("Review Extracted Entities", className="card-title mb-0"), width="auto"),
                                dbc.Col(
                                    html.Div(id="entity-count-label", className="text-muted small"),
                                    className="d-flex align-items-center",
                                ),
                                dbc.Col(
                                    dbc.Button(
                                        "Select All / Deselect All",
                                        id="select-all-btn",
                                        color="outline-secondary",
                                        size="sm",
                                    ),
                                    width="auto",
                                    className="ms-auto",
                                ),
                            ],
                            className="mb-3 align-items-center",
                        ),
                        html.Small(
                            "Tip: check the rows you want to mask. Edit the Replacement column to customise values.",
                            className="text-muted d-block mb-2",
                        ),
                        dash.dash_table.DataTable(
                            id="entity-table",
                            columns=[
                                {"name": "Page", "id": "page_number", "editable": False},
                                {"name": "Type", "id": "entity_type", "editable": False},
                                {"name": "Original Text", "id": "original_text", "editable": False},
                                {"name": "Replacement", "id": "replacement_text", "editable": True},
                                {"name": "Confidence", "id": "confidence_pct", "editable": False},
                            ],
                            data=[],
                            row_selectable="multi",
                            selected_rows=[],
                            page_size=20,
                            style_table={"overflowX": "auto"},
                            style_header={
                                "backgroundColor": "#f8fafc",
                                "fontWeight": "600",
                                "fontSize": "0.8125rem",
                                "color": "#475569",
                                "borderBottom": "2px solid #e2e8f0",
                            },
                            style_cell={
                                "fontSize": "0.875rem",
                                "padding": "0.5rem 0.75rem",
                                "textAlign": "left",
                                "border": "1px solid #e2e8f0",
                            },
                            style_data_conditional=[
                                # Highlight selected rows green
                                {
                                    "if": {"state": "selected"},
                                    "backgroundColor": "#f0fdf4",
                                    "border": "1px solid #86efac",
                                },
                                # Confidence colour coding
                                {
                                    "if": {"filter_query": "{confidence} >= 0.9", "column_id": "confidence_pct"},
                                    "color": "#16a34a",
                                    "fontWeight": "600",
                                },
                                {
                                    "if": {
                                        "filter_query": "{confidence} >= 0.7 && {confidence} < 0.9",
                                        "column_id": "confidence_pct",
                                    },
                                    "color": "#d97706",
                                    "fontWeight": "600",
                                },
                                {
                                    "if": {"filter_query": "{confidence} < 0.7", "column_id": "confidence_pct"},
                                    "color": "#dc2626",
                                    "fontWeight": "600",
                                },
                            ],
                            style_data={"border": "1px solid #e2e8f0"},
                        ),
                    ]
                ),
                className="mb-4",
            ),
            dbc.Button(
                [html.I(className="bi bi-shield-check me-2"), "Generate Masked PDF"],
                id="generate-btn",
                color="success",
                size="lg",
                className="w-100",
                disabled=True,
            ),
            dcc.Loading(html.Div(id="generate-loading"), type="circle", color="#16a34a"),
        ],
        id="step-2-content",
        style={"display": "none"},
    )


# ---------------------------------------------------------------------------
# Step 3 — Compare (shown when store-step == 3)
# ---------------------------------------------------------------------------


def _step3_layout() -> html.Div:
    return html.Div(
        [
            # Success banner
            dbc.Alert(
                [html.I(className="bi bi-shield-fill-check me-2"), html.Span(id="entities-masked-count")],
                color="success",
                className="mb-4",
                id="mask-success-banner",
            ),
            dbc.Row(
                [
                    # Original PDF pane
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                html.H6(
                                                    [html.I(className="bi bi-file-text me-2"), "Original Document"],
                                                    className="mb-0",
                                                )
                                            ),
                                            dbc.Col(
                                                dbc.Switch(id="show-original-switch", value=True, label="Show"),
                                                width="auto",
                                            ),
                                            dbc.Col(
                                                html.A(
                                                    [html.I(className="bi bi-download me-1"), "Download"],
                                                    id="original-download-link",
                                                    className="btn btn-sm btn-outline-secondary",
                                                    target="_blank",
                                                ),
                                                width="auto",
                                            ),
                                        ],
                                        align="center",
                                        className="mb-2",
                                    ),
                                    html.Div(
                                        html.Iframe(id="original-iframe", className="pdf-frame"),
                                        id="original-iframe-div",
                                    ),
                                ]
                            )
                        ),
                        lg=6,
                        className="mb-4",
                    ),
                    # Masked PDF pane
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                html.H6(
                                                    [html.I(className="bi bi-shield me-2 text-success"), "Masked Document"],
                                                    className="mb-0",
                                                )
                                            ),
                                            dbc.Col(
                                                dbc.Switch(id="show-masked-switch", value=True, label="Show"),
                                                width="auto",
                                            ),
                                            dbc.Col(
                                                html.A(
                                                    [html.I(className="bi bi-download me-1"), "Download"],
                                                    id="masked-download-link",
                                                    className="btn btn-sm btn-outline-primary",
                                                    target="_blank",
                                                ),
                                                width="auto",
                                            ),
                                        ],
                                        align="center",
                                        className="mb-2",
                                    ),
                                    html.Div(
                                        html.Iframe(id="masked-iframe", className="pdf-frame"),
                                        id="masked-iframe-div",
                                    ),
                                ]
                            ),
                            style={"borderColor": "#86efac"},
                        ),
                        lg=6,
                        className="mb-4",
                    ),
                ]
            ),
            # Info cards
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Small("Original", className="text-muted d-block"),
                                    html.Span("Unprotected", className="fw-semibold text-danger"),
                                ]
                            )
                        ),
                        md=4,
                        className="mb-3",
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Small("Entities Masked", className="text-muted d-block"),
                                    html.Span(id="entities-masked-count-card", className="fw-semibold text-success"),
                                ]
                            )
                        ),
                        md=4,
                        className="mb-3",
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Small("Processing", className="text-muted d-block"),
                                    html.Span("Server-side", className="fw-semibold"),
                                ]
                            )
                        ),
                        md=4,
                        className="mb-3",
                    ),
                ]
            ),
            dbc.Button(
                [html.I(className="bi bi-arrow-counterclockwise me-2"), "Process New Document"],
                id="reset-btn",
                color="outline-secondary",
                className="w-100",
            ),
        ],
        id="step-3-content",
        style={"display": "none"},
    )


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

app.layout = dbc.Container(
    [
        # --- Stores ---
        dcc.Store(id="store-step", data=1),
        dcc.Store(id="store-session", data=None),
        dcc.Store(id="store-fields", data=DEFAULT_FIELDS),
        dcc.Store(id="store-entities", data=None),
        dcc.Store(id="store-mask-result", data=None),
        dcc.Store(id="error-msg", data=None),
        # --- UI ---
        _navbar(),
        # Step indicator
        html.Div(id="step-indicator", className="d-flex align-items-center mb-4"),
        # Error alert (hidden by default)
        dbc.Alert(
            id="error-alert",
            is_open=False,
            dismissable=True,
            color="danger",
            className="mb-4",
        ),
        # Step content
        _step1_layout(),
        _step2_layout(),
        _step3_layout(),
        # Footer
        html.Footer(
            html.Small("PrivaSee — Document De-identification · All processing is server-side", className="text-muted"),
            className="text-center py-4 mt-4 border-top",
        ),
    ],
    fluid=True,
    style={"maxWidth": "1100px"},
)

# ===========================================================================
# Callbacks
# ===========================================================================

# ---------------------------------------------------------------------------
# Step indicator
# ---------------------------------------------------------------------------


@callback(Output("step-indicator", "children"), Input("store-step", "data"))
def update_step_indicator(step: int) -> list:
    return _step_indicator_content(step or 1)


# ---------------------------------------------------------------------------
# Error alert
# ---------------------------------------------------------------------------


@callback(
    Output("error-alert", "children"),
    Output("error-alert", "is_open"),
    Input("error-msg", "data"),
)
def show_error(msg: str | None):
    if msg:
        return msg, True
    return no_update, False


# ---------------------------------------------------------------------------
# Step visibility
# ---------------------------------------------------------------------------


@callback(
    Output("step-1-content", "style"),
    Output("step-2-content", "style"),
    Output("step-3-content", "style"),
    Input("store-step", "data"),
)
def toggle_steps(step: int):
    show = {}
    hide = {"display": "none"}
    step = step or 1
    return (
        show if step == 1 else hide,
        show if step == 2 else hide,
        show if step == 3 else hide,
    )


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


@callback(
    Output("store-session", "data"),
    Output("upload-status", "children"),
    Output("session-banner", "children"),
    Input("pdf-upload", "contents"),
    State("pdf-upload", "filename"),
    prevent_initial_call=True,
)
def handle_upload(contents: str | None, filename: str | None):
    if not contents or not filename:
        raise PreventUpdate

    if not filename.lower().endswith(".pdf"):
        err = dbc.Alert("Only PDF files are accepted.", color="danger", dismissable=True)
        return no_update, err, no_update

    _, content_string = contents.split(",", 1)
    file_bytes = base64.b64decode(content_string)

    if len(file_bytes) > 10 * 1024 * 1024:
        err = dbc.Alert("File must be under 10 MB.", color="danger", dismissable=True)
        return no_update, err, no_update

    headers = {
        "accept": "application/json",
        "Authorization": f"Key {os.environ.get('POSIT_CONNECT_API_KEY', '')}"
    }

    try:
        resp = req.post(
            f"{API_BASE_URL}/api/upload",
            headers=headers,
            files={"file": (filename, file_bytes, "application/pdf")},
            timeout=120,
            verify=SSL_VERIFY,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        err = dbc.Alert(f"Upload failed: {exc}", color="danger", dismissable=True)
        return no_update, err, no_update

    session = {
        "session_id": data["session_id"],
        "filename": data["filename"],
        "page_count": data.get("page_count", 1),
    }

    size_kb = len(file_bytes) // 1024
    file_card = dbc.Card(
        dbc.CardBody(
            dbc.Row(
                [
                    dbc.Col(html.I(className="bi bi-file-earmark-pdf text-danger fs-4"), width="auto"),
                    dbc.Col(
                        [
                            html.Div(filename, className="fw-semibold"),
                            html.Small(f"{size_kb} KB", className="text-muted"),
                        ]
                    ),
                    dbc.Col(
                        dbc.Badge("Uploaded", color="success", className="ms-auto"),
                        width="auto",
                        className="d-flex align-items-center",
                    ),
                ],
                align="center",
            )
        ),
        className="border-success",
    )

    page_count = session["page_count"]
    banner = None
    if page_count and page_count > 1:
        banner = dbc.Alert(
            [html.I(className="bi bi-info-circle me-2"), f"Multi-page PDF detected: {page_count} pages will all be processed."],
            color="info",
            className="mb-0 py-2",
        )

    return session, file_card, banner


# ---------------------------------------------------------------------------
# Enable "Process Document" button when a session exists
# ---------------------------------------------------------------------------


@callback(
    Output("process-btn", "disabled"),
    Input("store-session", "data"),
)
def toggle_process_btn(session):
    return not (session and session.get("session_id"))


# ---------------------------------------------------------------------------
# Dynamic field rows
# ---------------------------------------------------------------------------


@callback(
    Output("fields-container", "children"),
    Input("store-fields", "data"),
)
def render_fields(fields: list) -> list:
    if not fields:
        return []
    total = len(fields)
    return [_field_row(f, i, total) for i, f in enumerate(fields)]


@callback(
    Output("store-fields", "data", allow_duplicate=True),
    Input("add-field-btn", "n_clicks"),
    State("store-fields", "data"),
    prevent_initial_call=True,
)
def add_field(n_clicks: int, fields: list) -> list:
    if not n_clicks:
        raise PreventUpdate
    return (fields or []) + [_new_field()]


@callback(
    Output("store-fields", "data", allow_duplicate=True),
    Input({"type": "delete-btn", "index": ALL}, "n_clicks"),
    State("store-fields", "data"),
    prevent_initial_call=True,
)
def remove_field(n_clicks_list: list, fields: list) -> list:
    if not any(n for n in (n_clicks_list or []) if n):
        raise PreventUpdate
    triggered = ctx.triggered_id
    if triggered is None or not isinstance(triggered, dict):
        raise PreventUpdate
    field_id = triggered["index"]
    new_fields = [f for f in fields if f["id"] != field_id]
    return new_fields if new_fields else fields  # always keep at least one


# ---------------------------------------------------------------------------
# Sync field text / strategy changes back to store-fields
# We read all current values when Process is clicked (see process_document),
# but we also sync changes to the store so add/remove preserves typed values.
# ---------------------------------------------------------------------------


@callback(
    Output("store-fields", "data", allow_duplicate=True),
    Input({"type": "field-name", "index": ALL}, "value"),
    Input({"type": "field-desc", "index": ALL}, "value"),
    Input({"type": "field-strategy", "index": ALL}, "value"),
    State("store-fields", "data"),
    prevent_initial_call=True,
)
def sync_fields(names: list, descs: list, strategies: list, fields: list) -> list:
    if not fields:
        raise PreventUpdate
    updated = []
    for i, f in enumerate(fields):
        updated.append(
            {
                "id": f["id"],
                "name": names[i] if i < len(names) else f["name"],
                "description": descs[i] if i < len(descs) else f["description"],
                "strategy": strategies[i] if i < len(strategies) else f["strategy"],
            }
        )
    return updated


# ---------------------------------------------------------------------------
# Process document
# ---------------------------------------------------------------------------


@callback(
    Output("store-entities", "data"),
    Output("store-step", "data", allow_duplicate=True),
    Output("error-msg", "data", allow_duplicate=True),
    Output("process-loading", "children"),
    Input("process-btn", "n_clicks"),
    State("store-session", "data"),
    State("store-fields", "data"),
    prevent_initial_call=True,
)
def process_document(n_clicks: int, session: dict | None, fields: list | None):
    if not n_clicks or not session:
        raise PreventUpdate

    session_id = session.get("session_id")
    if not session_id:
        raise PreventUpdate

    # Build field definitions from store (already synced by sync_fields)
    field_definitions = [
        {"name": f["name"], "description": f["description"], "strategy": f["strategy"]}
        for f in (fields or [])
        if f.get("name") and f.get("description")
    ]

    if not field_definitions:
        return no_update, no_update, "Please fill in at least one field name and description.", no_update

    headers = {
        "accept": "application/json",
        "Authorization": f"Key {os.environ.get('POSIT_CONNECT_API_KEY', '')}"
    }
    try:
        resp = req.post(
            f"{API_BASE_URL}/api/process",
            json={"session_id": session_id, "field_definitions": field_definitions},
            timeout=900,  # 15 minutes — AI model call can be slow
            headers=headers,
            verify=SSL_VERIFY,
        )
        resp.raise_for_status()
        entities = resp.json().get("entities", [])
    except Exception as exc:
        return no_update, no_update, f"Processing failed: {exc}", no_update

    return entities, 2, None, no_update


# ---------------------------------------------------------------------------
# Step 2 — populate DataTable when entities arrive
# ---------------------------------------------------------------------------


@callback(
    Output("entity-table", "data"),
    Output("entity-table", "selected_rows"),
    Output("entity-count-label", "children"),
    Input("store-entities", "data"),
    Input("store-step", "data"),
)
def populate_entity_table(entities: list | None, step: int):
    if step != 2 or not entities:
        return [], [], ""

    rows = sorted(entities, key=lambda e: (e.get("page_number", 0), e.get("entity_type", "")))
    table_data = [
        {
            "id": e.get("id", ""),
            "page_number": e.get("page_number", ""),
            "entity_type": e.get("entity_type", ""),
            "original_text": e.get("original_text", ""),
            "replacement_text": e.get("replacement_text", ""),
            "confidence_pct": f"{e.get('confidence', 0) * 100:.0f}%",
            "confidence": e.get("confidence", 0),  # raw float for style_data_conditional
        }
        for e in rows
    ]
    # Select all rows by default
    selected = list(range(len(table_data)))
    count_label = f"{len(selected)} of {len(table_data)} selected for masking"
    return table_data, selected, count_label


# ---------------------------------------------------------------------------
# Update count label when selection changes
# ---------------------------------------------------------------------------


@callback(
    Output("entity-count-label", "children", allow_duplicate=True),
    Input("entity-table", "selected_rows"),
    State("entity-table", "data"),
    prevent_initial_call=True,
)
def update_count_label(selected_rows: list, data: list):
    total = len(data) if data else 0
    selected = len(selected_rows) if selected_rows else 0
    return f"{selected} of {total} selected for masking"


# ---------------------------------------------------------------------------
# Select all / deselect all
# ---------------------------------------------------------------------------


@callback(
    Output("entity-table", "selected_rows", allow_duplicate=True),
    Input("select-all-btn", "n_clicks"),
    State("entity-table", "data"),
    State("entity-table", "selected_rows"),
    prevent_initial_call=True,
)
def toggle_select_all(n_clicks: int, data: list, selected: list):
    if not n_clicks or data is None:
        raise PreventUpdate
    if len(selected) == len(data):
        return []  # deselect all
    return list(range(len(data)))  # select all


# ---------------------------------------------------------------------------
# Enable "Generate Masked PDF" button when rows are selected
# ---------------------------------------------------------------------------


@callback(
    Output("generate-btn", "disabled"),
    Input("entity-table", "selected_rows"),
)
def toggle_generate_btn(selected_rows: list):
    return not selected_rows


# ---------------------------------------------------------------------------
# Approve and mask
# ---------------------------------------------------------------------------


@callback(
    Output("store-mask-result", "data"),
    Output("store-step", "data", allow_duplicate=True),
    Output("error-msg", "data", allow_duplicate=True),
    Output("generate-loading", "children"),
    Input("generate-btn", "n_clicks"),
    State("store-session", "data"),
    State("entity-table", "data"),
    State("entity-table", "selected_rows"),
    prevent_initial_call=True,
)
def approve_and_mask(n_clicks: int, session: dict | None, table_data: list | None, selected_rows: list | None):
    if not n_clicks or not session or not table_data or not selected_rows:
        raise PreventUpdate

    session_id = session.get("session_id")
    if not session_id:
        raise PreventUpdate

    approved_ids = [table_data[i]["id"] for i in selected_rows if i < len(table_data)]
    updated_entities = [
        {"id": row["id"], "replacement_text": row["replacement_text"]}
        for row in table_data
    ]

    headers = {
        "accept": "application/json",
        "Authorization": f"Key {os.environ.get('POSIT_CONNECT_API_KEY', '')}"
    }
    try:
        resp = req.post(
            f"{API_BASE_URL}/api/approve-and-mask",
            headers=headers,
            json={
                "session_id": session_id,
                "approved_entity_ids": approved_ids,
                "updated_entities": updated_entities,
            },
            timeout=300,  # 5 minutes
            verify=SSL_VERIFY,
        )
        resp.raise_for_status()
        result = resp.json()
    except Exception as exc:
        return no_update, no_update, f"Masking failed: {exc}", no_update

    return result, 3, None, no_update


# ---------------------------------------------------------------------------
# Step 3 — populate compare view
# ---------------------------------------------------------------------------


@callback(
    Output("original-iframe", "src"),
    Output("masked-iframe", "src"),
    Output("original-download-link", "href"),
    Output("masked-download-link", "href"),
    Output("entities-masked-count", "children"),
    Output("entities-masked-count-card", "children"),
    Input("store-mask-result", "data"),
    State("store-session", "data"),
)
def populate_compare(mask_result: dict | None, session: dict | None):
    if not mask_result or not session:
        return "", "", "", "", "", ""

    session_id = session["session_id"]
    count = mask_result.get("entities_masked", 0)
    label = f"{count} {'entity' if count == 1 else 'entities'} masked"

    return (
        f"/pdf/original/{session_id}",
        f"/pdf/masked/{session_id}",
        f"/pdf/original/{session_id}?dl=1",
        f"/pdf/masked/{session_id}?dl=1",
        label,
        label,
    )


# ---------------------------------------------------------------------------
# Show / hide PDF panes
# ---------------------------------------------------------------------------


@callback(
    Output("original-iframe-div", "style"),
    Input("show-original-switch", "value"),
)
def toggle_original(show: bool):
    return {} if show else {"display": "none"}


@callback(
    Output("masked-iframe-div", "style"),
    Input("show-masked-switch", "value"),
)
def toggle_masked(show: bool):
    return {} if show else {"display": "none"}


# ---------------------------------------------------------------------------
# Reset workflow
# ---------------------------------------------------------------------------


@callback(
    Output("store-step", "data", allow_duplicate=True),
    Output("store-session", "data", allow_duplicate=True),
    Output("store-fields", "data", allow_duplicate=True),
    Output("store-entities", "data", allow_duplicate=True),
    Output("store-mask-result", "data", allow_duplicate=True),
    Output("error-msg", "data", allow_duplicate=True),
    Input("reset-btn", "n_clicks"),
    prevent_initial_call=True,
)
def reset_workflow(n_clicks: int):
    if not n_clicks:
        raise PreventUpdate
    return 1, None, DEFAULT_FIELDS, None, None, None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, port=8050)
