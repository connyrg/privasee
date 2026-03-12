"""
PrivaSee — Dash frontend
Deployable on Posit Connect via: rsconnect deploy dash --title "PrivaSee" .

Set API_BASE_URL env var to point at the deployed FastAPI backend.
"""
# export http_proxy="" && export https_proxy="" && rsconnect deploy dash  --server  https://sds-posit-connect-prod.int.corp.sun/ --api-key $POSIT_CONNECT_API_KEY --entrypoint app.py [--new] -t PrivaSee . --insecure  --exclude venv/

from __future__ import annotations

import base64
import json
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


def _new_field(name: str = "", description: str = "", strategy: str = "Fake Data", source: str = "custom") -> dict:
    return {"id": str(uuid.uuid4()), "name": name, "description": description, "strategy": strategy, "source": source}


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


def _step_indicator_content(step: int, mode: str = "single") -> list:
    if mode == "batch":
        labels = ["Configure", "Processing", "Results"]
    else:
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
    source = field.get("source", "custom")
    badge = dbc.Badge(
        "System" if source == "system" else "Custom",
        color="primary" if source == "system" else "secondary",
        pill=True,
        style={"fontSize": "0.7rem"},
    )
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
                    width=4,
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
                dbc.Col(badge, width="auto", className="d-flex align-items-center"),
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
            # Mode toggle
            dbc.Card(
                dbc.CardBody(
                    dbc.Row(
                        [
                            dbc.Col(html.Small("Mode:", className="text-muted fw-semibold"), width="auto", className="d-flex align-items-center"),
                            dbc.Col(
                                dbc.RadioItems(
                                    id="mode-toggle",
                                    options=[
                                        {"label": "Single Document", "value": "single"},
                                        {"label": "Batch (Multiple Documents)", "value": "batch"},
                                    ],
                                    value="single",
                                    inline=True,
                                    className="mb-0",
                                ),
                            ),
                        ],
                        align="center",
                    )
                ),
                className="mb-3",
            ),
            # Upload card — single mode
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
                id="single-upload-card",
            ),
            # Upload card — batch mode (hidden by default)
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H5("Upload Documents", className="card-title mb-3"),
                        dcc.Upload(
                            id="batch-upload",
                            children=html.Div(
                                [
                                    html.I(className="bi bi-cloud-upload fs-2 text-muted"),
                                    html.P("Drag & drop PDFs, or click to browse", className="mt-2 mb-1 text-muted"),
                                    html.Small("PDF only · max 10 MB each · up to 20 files", className="text-muted"),
                                ]
                            ),
                            accept=".pdf",
                            multiple=True,
                            className="upload-zone",
                        ),
                        html.Div(id="batch-file-list", className="mt-2"),
                    ]
                ),
                className="mb-4",
                id="batch-upload-card",
                style={"display": "none"},
            ),
            # Field definitions card
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H5("De-identification Rules", className="card-title mb-1"),
                        html.Small(STRATEGY_GUIDE, className="text-muted d-block mb-3"),
                        # System template row
                        dbc.Row(
                            [
                                dbc.Col(html.Small("Templates:", className="text-muted fw-semibold"), width="auto", className="d-flex align-items-center"),
                                dbc.Col(
                                    dcc.Dropdown(
                                        id="template-dropdown",
                                        placeholder="Load a system template...",
                                        options=[],
                                        clearable=True,
                                        style={"fontSize": "0.875rem"},
                                    ),
                                    width=5,
                                ),
                                dbc.Col(
                                    dbc.Button("Load", id="template-load-btn", color="outline-secondary", size="sm", disabled=True),
                                    width="auto",
                                ),
                            ],
                            className="g-2 mb-2 align-items-center",
                        ),
                        # Config save / load row
                        dbc.Row(
                            [
                                dbc.Col(
                                    dcc.Dropdown(
                                        id="config-load-dropdown",
                                        placeholder="Load saved config...",
                                        options=[],
                                        clearable=True,
                                        style={"fontSize": "0.875rem"},
                                    ),
                                    width=5,
                                ),
                                dbc.Col(
                                    dbc.Button("Load", id="config-load-btn", color="outline-primary", size="sm", disabled=True),
                                    width="auto",
                                ),
                                dbc.Col(
                                    dbc.Input(id="config-save-name", placeholder="Save as...", size="sm", debounce=True),
                                    width=3,
                                ),
                                dbc.Col(
                                    dbc.Button("Save", id="config-save-btn", color="outline-success", size="sm", disabled=True),
                                    width="auto",
                                ),
                            ],
                            className="g-2 mb-2 align-items-center",
                        ),
                        # JSON import row
                        dbc.Row(
                            [
                                dbc.Col(html.Small("Import JSON:", className="text-muted fw-semibold"), width="auto", className="d-flex align-items-center"),
                                dbc.Col(
                                    dcc.Upload(
                                        id="config-json-upload",
                                        children=dbc.Button("Browse...", color="outline-secondary", size="sm"),
                                        accept=".json",
                                        multiple=False,
                                    ),
                                    width="auto",
                                ),
                                dbc.Col(
                                    html.Small("Upload a saved config .json file to load its fields.", className="text-muted"),
                                    className="d-flex align-items-center",
                                ),
                            ],
                            className="g-2 mb-2 align-items-center",
                        ),
                        html.Div(id="config-status", className="mb-2"),
                        # Column headers
                        dbc.Row(
                            [
                                dbc.Col(html.Small("Field Name", className="text-muted fw-semibold"), width=3),
                                dbc.Col(html.Small("Description", className="text-muted fw-semibold"), width=4),
                                dbc.Col(
                                    html.Small(
                                        [
                                            "Strategy",
                                            html.I(
                                                className="bi bi-info-circle ms-1",
                                                id="strategy-header-info",
                                                style={"cursor": "pointer"},
                                            ),
                                            dbc.Tooltip(STRATEGY_GUIDE, target="strategy-header-info", placement="top"),
                                        ],
                                        className="text-muted fw-semibold",
                                    ),
                                    width=3,
                                ),
                                dbc.Col(width="auto"),
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
            # Process button — single mode
            dbc.Button(
                [html.I(className="bi bi-play-fill me-2"), "Process Document"],
                id="process-btn",
                color="primary",
                size="lg",
                className="w-100",
                disabled=True,
            ),
            # Process button — batch mode (hidden by default)
            dbc.Button(
                [html.I(className="bi bi-play-fill me-2"), "Process Documents"],
                id="batch-process-btn",
                color="primary",
                size="lg",
                className="w-100",
                disabled=True,
                style={"display": "none"},
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
                [html.I(className="bi bi-arrow-left me-2"), "Back to Configure"],
                id="back-to-configure-btn",
                color="outline-secondary",
                size="sm",
                className="mb-3",
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
                                                    download="original.pdf",
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
                                                    download="masked.pdf",
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
            dbc.Alert(
                [
                    html.I(className="bi bi-shield-lock me-2"),
                    "Your documents are processed server-side only. Files are stored temporarily in your "
                    "organisation's cloud storage and are never sent to external services.",
                ],
                color="primary",
                className="mb-4",
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
# Batch Step 2 — Processing (shown in batch mode when store-step == 2)
# ---------------------------------------------------------------------------


def _batch_step2_layout() -> html.Div:
    return html.Div(
        [
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H5("Processing Documents", className="card-title mb-3"),
                        dcc.Loading(
                            html.Div(id="batch-progress-label", className="text-muted"),
                            type="circle",
                            color="#0284c7",
                        ),
                    ]
                ),
                className="mb-4",
            ),
        ],
        id="batch-step-2-content",
        style={"display": "none"},
    )


# ---------------------------------------------------------------------------
# Batch Step 3 — Results (shown in batch mode when store-step == 3)
# ---------------------------------------------------------------------------


def _batch_step3_layout() -> html.Div:
    return html.Div(
        [
            html.Div(id="batch-summary-banner", className="mb-4"),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H5("Batch Results", className="card-title mb-3"),
                        html.Div(id="batch-results-table"),
                    ]
                ),
                className="mb-4",
            ),
            dbc.Alert(
                [
                    html.I(className="bi bi-shield-lock me-2"),
                    "Your documents are processed server-side only. Files are stored temporarily in your "
                    "organisation's cloud storage and are never sent to external services.",
                ],
                color="primary",
                className="mb-4",
            ),
            dbc.Button(
                [html.I(className="bi bi-arrow-counterclockwise me-2"), "Process Another Batch"],
                id="batch-reset-btn",
                color="outline-secondary",
                className="w-100",
            ),
        ],
        id="batch-step-3-content",
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
        dcc.Store(id="store-configs", data=[]),
        dcc.Store(id="store-entities", data=None),
        dcc.Store(id="store-mask-result", data=None),
        dcc.Store(id="error-msg", data=None),
        dcc.Store(id="store-mode", data="single"),
        dcc.Store(id="store-batch-files", data=[]),
        dcc.Store(id="store-batch-results", data=[]),
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
        _batch_step2_layout(),
        _batch_step3_layout(),
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


@callback(
    Output("step-indicator", "children"),
    Input("store-step", "data"),
    Input("store-mode", "data"),
)
def update_step_indicator(step: int, mode: str) -> list:
    return _step_indicator_content(step or 1, mode or "single")


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
    Output("batch-step-2-content", "style"),
    Output("batch-step-3-content", "style"),
    Input("store-step", "data"),
    Input("store-mode", "data"),
)
def toggle_steps(step: int, mode: str):
    show = {}
    hide = {"display": "none"}
    step = step or 1
    batch = (mode == "batch")
    return (
        show if step == 1 else hide,
        show if (step == 2 and not batch) else hide,
        show if (step == 3 and not batch) else hide,
        show if (step == 2 and batch) else hide,
        show if (step == 3 and batch) else hide,
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

    headers = {"Authorization": f"Key {os.environ.get('POSIT_CONNECT_API_KEY', '')}"}

    def _fetch_b64(url: str) -> str:
        try:
            r = req.get(url, headers=headers, timeout=30, verify=SSL_VERIFY)
            r.raise_for_status()
            return "data:application/pdf;base64," + base64.b64encode(r.content).decode()
        except Exception:
            return ""

    original_src = _fetch_b64(f"{API_BASE_URL}/api/files/uploads/{session_id}.pdf")
    masked_src = _fetch_b64(f"{API_BASE_URL}/api/files/output/{session_id}_masked.pdf")

    return (
        original_src,
        masked_src,
        original_src,
        masked_src,
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
# Config management
# ---------------------------------------------------------------------------


@callback(
    Output("store-configs", "data"),
    Output("config-load-dropdown", "options"),
    Input("store-step", "data"),
)
def refresh_config_list(step: int):
    """Fetch saved configs from the API whenever the user is on step 1."""
    if step != 1:
        raise PreventUpdate
    headers = {"Authorization": f"Key {os.environ.get('POSIT_CONNECT_API_KEY', '')}"}
    try:
        r = req.get(f"{API_BASE_URL}/api/configs", headers=headers, verify=SSL_VERIFY, timeout=10)
        if r.ok:
            configs = r.json()
            options = [{"label": c["config_name"], "value": c["key"]} for c in configs]
            return configs, options
    except Exception:
        pass
    return [], []


@callback(
    Output("config-load-btn", "disabled"),
    Input("config-load-dropdown", "value"),
)
def toggle_load_btn(value):
    return not bool(value)


@callback(
    Output("store-fields", "data", allow_duplicate=True),
    Output("config-status", "children"),
    Input("config-load-btn", "n_clicks"),
    State("config-load-dropdown", "value"),
    prevent_initial_call=True,
)
def load_config(n_clicks, key):
    if not n_clicks or not key:
        raise PreventUpdate
    headers = {"Authorization": f"Key {os.environ.get('POSIT_CONNECT_API_KEY', '')}"}
    try:
        r = req.get(f"{API_BASE_URL}/api/configs/{key}", headers=headers, verify=SSL_VERIFY, timeout=10)
        r.raise_for_status()
        field_defs = r.json().get("field_definitions", [])
        fields = [_new_field(f["name"], f["description"], f["strategy"]) for f in field_defs]
        alert = dbc.Alert("Config loaded.", color="success", dismissable=True, duration=3000)
        return fields, alert
    except Exception as exc:
        return no_update, dbc.Alert(f"Failed to load config: {exc}", color="danger", dismissable=True)


@callback(
    Output("config-save-btn", "disabled"),
    Input("config-save-name", "value"),
)
def toggle_save_btn(name):
    return not bool(name and name.strip())


@callback(
    Output("config-status", "children", allow_duplicate=True),
    Output("store-configs", "data", allow_duplicate=True),
    Output("config-load-dropdown", "options", allow_duplicate=True),
    Input("config-save-btn", "n_clicks"),
    State("config-save-name", "value"),
    State("store-fields", "data"),
    prevent_initial_call=True,
)
def save_config(n_clicks, name, fields):
    if not n_clicks or not name:
        raise PreventUpdate
    # Filter out rows with empty name or description (same as process_document)
    field_defs = [
        {"name": f["name"], "description": f["description"], "strategy": f["strategy"]}
        for f in (fields or [])
        if f.get("name") and f.get("description")
    ]
    if not field_defs:
        return dbc.Alert("Please fill in at least one field name and description before saving.", color="warning", dismissable=True), no_update, no_update
    headers = {"Authorization": f"Key {os.environ.get('POSIT_CONNECT_API_KEY', '')}"}
    try:
        r = req.post(
            f"{API_BASE_URL}/api/configs",
            headers=headers,
            json={"config_name": name, "field_definitions": field_defs},
            verify=SSL_VERIFY,
            timeout=10,
        )
        r.raise_for_status()
        # Refresh the dropdown list
        r2 = req.get(f"{API_BASE_URL}/api/configs", headers=headers, verify=SSL_VERIFY, timeout=10)
        configs = r2.json() if r2.ok else []
        options = [{"label": c["config_name"], "value": c["key"]} for c in configs]
        alert = dbc.Alert(f"Config \"{name}\" saved.", color="success", dismissable=True, duration=3000)
        return alert, configs, options
    except Exception as exc:
        return dbc.Alert(f"Failed to save config: {exc}", color="danger", dismissable=True), no_update, no_update


@callback(
    Output("template-dropdown", "options"),
    Input("store-step", "data"),
)
def refresh_template_list(step: int):
    """Populate the system template dropdown when the user is on step 1."""
    if step != 1:
        raise PreventUpdate
    headers = {"Authorization": f"Key {os.environ.get('POSIT_CONNECT_API_KEY', '')}"}
    try:
        r = req.get(f"{API_BASE_URL}/api/templates", headers=headers, verify=SSL_VERIFY, timeout=10)
        if r.ok:
            return [{"label": t["template_name"], "value": t["key"]} for t in r.json()]
    except Exception:
        pass
    return []


@callback(
    Output("template-load-btn", "disabled"),
    Input("template-dropdown", "value"),
)
def toggle_template_load_btn(value):
    return not bool(value)


@callback(
    Output("store-fields", "data", allow_duplicate=True),
    Output("config-status", "children", allow_duplicate=True),
    Input("template-load-btn", "n_clicks"),
    State("template-dropdown", "value"),
    prevent_initial_call=True,
)
def load_template(n_clicks, key):
    if not n_clicks or not key:
        raise PreventUpdate
    headers = {"Authorization": f"Key {os.environ.get('POSIT_CONNECT_API_KEY', '')}"}
    try:
        r = req.get(f"{API_BASE_URL}/api/templates/{key}", headers=headers, verify=SSL_VERIFY, timeout=10)
        r.raise_for_status()
        data = r.json()
        fields = [_new_field(f["name"], f["description"], f["strategy"], "system") for f in data.get("field_definitions", [])]
        alert = dbc.Alert(f"Template \"{data['template_name']}\" loaded.", color="success", dismissable=True, duration=3000)
        return fields, alert
    except Exception as exc:
        return no_update, dbc.Alert(f"Failed to load template: {exc}", color="danger", dismissable=True)


# ---------------------------------------------------------------------------
# Back to Configure (Step 2 → Step 1)
# ---------------------------------------------------------------------------


@callback(
    Output("store-step", "data", allow_duplicate=True),
    Input("back-to-configure-btn", "n_clicks"),
    prevent_initial_call=True,
)
def back_to_configure(n_clicks: int):
    if not n_clicks:
        raise PreventUpdate
    return 1


# ---------------------------------------------------------------------------
# Import config from local JSON file
# ---------------------------------------------------------------------------


@callback(
    Output("store-fields", "data", allow_duplicate=True),
    Output("config-status", "children", allow_duplicate=True),
    Input("config-json-upload", "contents"),
    State("config-json-upload", "filename"),
    prevent_initial_call=True,
)
def import_config_json(contents, filename):
    if not contents:
        raise PreventUpdate
    try:
        _, encoded = contents.split(",", 1)
        raw = json.loads(base64.b64decode(encoded).decode("utf-8"))
        # Accept both {"field_definitions": [...]} and bare [...]
        field_defs = raw if isinstance(raw, list) else raw.get("field_definitions", [])
        if not field_defs:
            raise ValueError("No field_definitions found in JSON.")
        fields = [_new_field(f.get("name", ""), f.get("description", ""), f.get("strategy", "Fake Data")) for f in field_defs]
        alert = dbc.Alert(f"Imported {len(fields)} field(s) from {filename}.", color="success", dismissable=True, duration=3000)
        return fields, alert
    except Exception as exc:
        return no_update, dbc.Alert(f"Failed to import config: {exc}", color="danger", dismissable=True)


# ---------------------------------------------------------------------------
# Batch mode callbacks
# ---------------------------------------------------------------------------


@callback(
    Output("store-mode", "data"),
    Output("store-batch-files", "data", allow_duplicate=True),
    Input("mode-toggle", "value"),
    prevent_initial_call=True,
)
def toggle_mode(mode: str):
    return mode, []


@callback(
    Output("single-upload-card", "style"),
    Output("batch-upload-card", "style"),
    Output("process-btn", "style"),
    Output("batch-process-btn", "style"),
    Input("store-mode", "data"),
)
def show_upload_zones(mode: str):
    show = {}
    hide = {"display": "none"}
    batch = (mode == "batch")
    return (
        hide if batch else show,
        show if batch else hide,
        hide if batch else show,
        show if batch else hide,
    )


@callback(
    Output("store-batch-files", "data", allow_duplicate=True),
    Input("batch-upload", "contents"),
    State("batch-upload", "filename"),
    State("store-batch-files", "data"),
    prevent_initial_call=True,
)
def handle_batch_upload(contents_list, filenames, existing_files):
    if not contents_list:
        raise PreventUpdate
    files = list(existing_files or [])
    errors = []
    for contents, filename in zip(contents_list, filenames):
        if not filename.lower().endswith(".pdf"):
            errors.append(f"{filename}: only PDF files are accepted.")
            continue
        _, content_string = contents.split(",", 1)
        file_bytes = base64.b64decode(content_string)
        if len(file_bytes) > 10 * 1024 * 1024:
            errors.append(f"{filename}: file exceeds 10 MB limit.")
            continue
        # Avoid duplicates by filename
        if any(f["filename"] == filename for f in files):
            continue
        files.append({"filename": filename, "content": content_string})
    return files


@callback(
    Output("batch-file-list", "children"),
    Input("store-batch-files", "data"),
)
def render_batch_file_list(files: list):
    if not files:
        return html.Small("No files selected.", className="text-muted")
    rows = []
    for i, f in enumerate(files):
        size_kb = len(base64.b64decode(f["content"])) // 1024
        rows.append(
            dbc.Row(
                [
                    dbc.Col(html.I(className="bi bi-file-earmark-pdf text-danger"), width="auto", className="d-flex align-items-center"),
                    dbc.Col(html.Small(f["filename"], className="fw-semibold"), className="d-flex align-items-center"),
                    dbc.Col(html.Small(f"{size_kb} KB", className="text-muted"), width="auto", className="d-flex align-items-center"),
                    dbc.Col(
                        dbc.Button(
                            html.I(className="bi bi-x"),
                            id={"type": "batch-remove-btn", "index": i},
                            color="outline-danger",
                            size="sm",
                            n_clicks=0,
                        ),
                        width="auto",
                    ),
                ],
                className="g-2 mb-1 align-items-center",
                key=str(i),
            )
        )
    return rows


@callback(
    Output("store-batch-files", "data", allow_duplicate=True),
    Input({"type": "batch-remove-btn", "index": ALL}, "n_clicks"),
    State("store-batch-files", "data"),
    prevent_initial_call=True,
)
def remove_batch_file(n_clicks_list, files):
    if not any(n for n in n_clicks_list):
        raise PreventUpdate
    triggered = ctx.triggered_id
    if triggered is None:
        raise PreventUpdate
    idx = triggered["index"]
    return [f for i, f in enumerate(files) if i != idx]


@callback(
    Output("batch-process-btn", "disabled"),
    Input("store-batch-files", "data"),
)
def toggle_batch_process_btn(files: list):
    return not bool(files)


@callback(
    Output("store-batch-results", "data"),
    Output("store-step", "data", allow_duplicate=True),
    Output("batch-progress-label", "children"),
    Output("process-loading", "children", allow_duplicate=True),
    Input("batch-process-btn", "n_clicks"),
    State("store-batch-files", "data"),
    State("store-fields", "data"),
    prevent_initial_call=True,
)
def run_batch(n_clicks: int, files: list, fields: list):
    if not n_clicks or not files:
        raise PreventUpdate

    headers = {
        "accept": "application/json",
        "Authorization": f"Key {os.environ.get('POSIT_CONNECT_API_KEY', '')}",
    }
    field_defs = [
        {"name": f["name"], "description": f["description"], "strategy": f.get("strategy", "Fake Data")}
        for f in (fields or [])
        if f.get("name") and f.get("description")
    ]
    if not field_defs:
        raise PreventUpdate  # Should not happen — process-btn is disabled without valid fields, but guard anyway

    results = []

    for file_info in files:
        filename = file_info["filename"]
        file_bytes = base64.b64decode(file_info["content"])
        session_id = None
        error = None
        entities_found = 0
        entities_masked = 0
        score = None
        verdict = None

        # Step 1: Upload
        try:
            resp = req.post(
                f"{API_BASE_URL}/api/upload",
                headers=headers,
                files={"file": (filename, file_bytes, "application/pdf")},
                timeout=120,
                verify=SSL_VERIFY,
            )
            resp.raise_for_status()
            session_id = resp.json()["session_id"]
        except Exception as exc:
            results.append({"filename": filename, "session_id": None, "entities_found": 0,
                            "entities_masked": 0, "score": None, "verdict": "Upload failed", "error": str(exc)})
            continue

        # Step 2: Process
        try:
            resp = req.post(
                f"{API_BASE_URL}/api/process",
                headers=headers,
                json={"session_id": session_id, "field_definitions": field_defs},
                timeout=300,
                verify=SSL_VERIFY,
            )
            resp.raise_for_status()
            entities = resp.json().get("entities", [])
            entities_found = len(entities)
        except Exception as exc:
            # Clean up session
            try:
                req.delete(f"{API_BASE_URL}/api/sessions/{session_id}", headers=headers, verify=SSL_VERIFY, timeout=10)
            except Exception:
                pass
            results.append({"filename": filename, "session_id": session_id, "entities_found": 0,
                            "entities_masked": 0, "score": None, "verdict": "Processing failed", "error": str(exc)})
            continue

        if not entities:
            results.append({"filename": filename, "session_id": session_id, "entities_found": 0,
                            "entities_masked": 0, "score": 100.0, "verdict": "No entities found", "error": None})
            continue

        # Step 3: Mask (all entities approved)
        try:
            approved_ids = [e["id"] for e in entities]
            resp = req.post(
                f"{API_BASE_URL}/api/approve-and-mask",
                headers=headers,
                json={"session_id": session_id, "approved_entity_ids": approved_ids},
                timeout=300,
                verify=SSL_VERIFY,
            )
            resp.raise_for_status()
            entities_masked = resp.json().get("entities_masked", 0)
        except Exception as exc:
            results.append({"filename": filename, "session_id": session_id, "entities_found": entities_found,
                            "entities_masked": 0, "score": None, "verdict": "Masking failed", "error": str(exc)})
            continue

        # Step 4: Verify
        try:
            resp = req.post(
                f"{API_BASE_URL}/api/sessions/{session_id}/verify",
                headers=headers,
                json={"entities": entities},
                timeout=60,
                verify=SSL_VERIFY,
            )
            resp.raise_for_status()
            verify_data = resp.json()
            score = verify_data["score"]
            if score >= 90:
                verdict = "Excellent"
            elif score >= 70:
                verdict = "Review recommended"
            else:
                verdict = "Masking incomplete"
        except Exception as exc:
            score = None
            verdict = "Verify failed"
            error = str(exc)

        results.append({
            "filename": filename,
            "session_id": session_id,
            "entities_found": entities_found,
            "entities_masked": entities_masked,
            "score": score,
            "verdict": verdict,
            "error": error,
        })

    return results, 3, f"Processed {len(results)} of {len(files)} files.", None


@callback(
    Output("batch-summary-banner", "children"),
    Output("batch-results-table", "children"),
    Input("store-batch-results", "data"),
    Input("store-step", "data"),
    Input("store-mode", "data"),
    prevent_initial_call=True,
)
def render_batch_results(results: list, step: int, mode: str):
    if mode != "batch" or step != 3 or not results:
        raise PreventUpdate

    total = len(results)
    excellent = sum(1 for r in results if r.get("verdict") == "Excellent")
    errors = sum(1 for r in results if r.get("error"))

    if errors == 0 and excellent == total:
        banner_color = "success"
        banner_icon = "bi-shield-fill-check"
        banner_msg = f"All {total} document(s) masked successfully."
    elif errors > 0:
        banner_color = "warning"
        banner_icon = "bi-exclamation-triangle"
        banner_msg = f"{total - errors}/{total} document(s) processed. {errors} error(s) occurred."
    else:
        banner_color = "info"
        banner_icon = "bi-info-circle"
        banner_msg = f"{total} document(s) processed."

    banner = dbc.Alert(
        [html.I(className=f"bi {banner_icon} me-2"), banner_msg],
        color=banner_color,
        className="mb-0",
    )

    VERDICT_COLORS = {
        "Excellent": "success",
        "Review recommended": "warning",
        "Masking incomplete": "danger",
        "No entities found": "secondary",
    }

    rows = []
    for r in results:
        verdict = r.get("verdict") or "—"
        score = r.get("score")
        score_str = f"{score:.0f}%" if score is not None else "—"
        badge_color = VERDICT_COLORS.get(verdict, "secondary")
        if r.get("error") and verdict not in ("No entities found",):
            verdict_badge = dbc.Badge(verdict, color="danger", className="ms-1")
        else:
            verdict_badge = dbc.Badge(verdict, color=badge_color, className="ms-1")
        rows.append(
            html.Tr([
                html.Td(r["filename"]),
                html.Td(str(r.get("entities_found", 0))),
                html.Td(str(r.get("entities_masked", 0))),
                html.Td(score_str),
                html.Td(verdict_badge),
            ])
        )

    table = dbc.Table(
        [
            html.Thead(html.Tr([
                html.Th("File"), html.Th("Entities Found"),
                html.Th("Entities Masked"), html.Th("Score"), html.Th("Status"),
            ])),
            html.Tbody(rows),
        ],
        bordered=True,
        hover=True,
        responsive=True,
        size="sm",
        className="mb-0",
    )
    return banner, table


@callback(
    Output("store-step", "data", allow_duplicate=True),
    Output("store-batch-files", "data", allow_duplicate=True),
    Output("store-batch-results", "data", allow_duplicate=True),
    Output("error-msg", "data", allow_duplicate=True),
    Input("batch-reset-btn", "n_clicks"),
    State("store-batch-results", "data"),
    prevent_initial_call=True,
)
def batch_reset_workflow(n_clicks: int, results: list):
    if not n_clicks:
        raise PreventUpdate
    headers = {"Authorization": f"Key {os.environ.get('POSIT_CONNECT_API_KEY', '')}"}
    for r in (results or []):
        sid = r.get("session_id")
        if sid:
            try:
                req.delete(f"{API_BASE_URL}/api/sessions/{sid}", headers=headers, verify=SSL_VERIFY, timeout=10)
            except Exception:
                pass
    return 1, [], [], None


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
    Output("upload-status", "children", allow_duplicate=True),
    Output("session-banner", "children", allow_duplicate=True),
    Output("pdf-upload", "contents", allow_duplicate=True),
    Input("reset-btn", "n_clicks"),
    State("store-session", "data"),
    prevent_initial_call=True,
)
def reset_workflow(n_clicks: int, session_id: str):
    if not n_clicks:
        raise PreventUpdate
    if session_id:
        headers = {"Authorization": f"Key {os.environ.get('POSIT_CONNECT_API_KEY', '')}"}
        try:
            req.delete(
                f"{API_BASE_URL}/api/sessions/{session_id}",
                headers=headers,
                verify=SSL_VERIFY,
                timeout=10,
            )
        except Exception:
            pass  # Non-fatal — local stores are cleared regardless
    return 1, None, DEFAULT_FIELDS, None, None, None, None, None, None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, port=8050)
