"""
Integration tests for:
  GET  /api/templates              — list all system templates
  GET  /api/templates/{key}        — get a single template by key
  POST /api/configs                — save a named field config
  GET  /api/configs                — list saved configs
  GET  /api/configs/{key}          — get a config by key

Templates are hardcoded in main.py (_SYSTEM_TEMPLATES) and require no storage.
Configs use _config_manager which is patched via monkeypatch.
"""

from unittest.mock import MagicMock

import pytest

import app.main as main_module
from app.config_manager import ConfigManager


# ---------------------------------------------------------------------------
# Config manager fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_config_manager(monkeypatch) -> MagicMock:
    """Patch app.main._config_manager with a MagicMock."""
    mock = MagicMock(spec=ConfigManager)
    monkeypatch.setattr(main_module, "_config_manager", mock)
    return mock


# ===========================================================================
# GET /api/templates
# ===========================================================================


@pytest.mark.integration
async def test_list_templates_returns_200(client):
    """GET /api/templates returns 200 with a non-empty list."""
    response = await client.get("/api/templates")

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, list)
    assert len(body) >= 1


@pytest.mark.integration
async def test_list_templates_includes_required_fields(client):
    """Each template summary must have key, template_name, description, field_count."""
    response = await client.get("/api/templates")

    body = response.json()
    for tmpl in body:
        assert "key" in tmpl
        assert "template_name" in tmpl
        assert "description" in tmpl
        assert "field_count" in tmpl
        assert isinstance(tmpl["field_count"], int)


@pytest.mark.integration
async def test_list_templates_includes_common_pii(client):
    """The 'common_pii' template must always be present."""
    response = await client.get("/api/templates")

    keys = [t["key"] for t in response.json()]
    assert "common_pii" in keys


# ===========================================================================
# GET /api/templates/{key}
# ===========================================================================


@pytest.mark.integration
async def test_get_template_returns_field_definitions(client):
    """GET /api/templates/common_pii returns the template with field_definitions."""
    response = await client.get("/api/templates/common_pii")

    assert response.status_code == 200
    body = response.json()
    assert body["key"] == "common_pii"
    assert isinstance(body["field_definitions"], list)
    assert len(body["field_definitions"]) >= 1

    for fd in body["field_definitions"]:
        assert "name" in fd
        assert "description" in fd
        assert "strategy" in fd


@pytest.mark.integration
async def test_get_template_returns_404_for_unknown_key(client):
    """404 for a template key that does not exist."""
    response = await client.get("/api/templates/does-not-exist")

    assert response.status_code == 404
    assert "error" in response.json()


# ===========================================================================
# POST /api/configs
# ===========================================================================


@pytest.mark.integration
async def test_save_config_returns_201_with_summary(
    client, mock_config_manager
):
    """POST /api/configs returns 201 with config_name and key."""
    mock_config_manager.save_config.return_value = "my_config"

    response = await client.post(
        "/api/configs",
        json={
            "config_name": "My Config",
            "field_definitions": [
                {"name": "Full Name", "description": "Person's full name", "strategy": "Fake Data"}
            ],
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["config_name"] == "My Config"
    assert body["key"] == "my_config"
    assert "saved_at" in body


@pytest.mark.integration
async def test_save_config_calls_config_manager_with_correct_args(
    client, mock_config_manager
):
    """save_config is called with config_name and the field_definitions list."""
    mock_config_manager.save_config.return_value = "full_name_only"

    await client.post(
        "/api/configs",
        json={
            "config_name": "Full Name Only",
            "field_definitions": [
                {"name": "Full Name", "description": "A person's name", "strategy": "Black Out"}
            ],
        },
    )

    mock_config_manager.save_config.assert_called_once()
    call_kwargs = mock_config_manager.save_config.call_args
    assert call_kwargs.kwargs["config_name"] == "Full Name Only"
    field_defs = call_kwargs.kwargs["field_definitions"]
    assert field_defs[0]["name"] == "Full Name"


@pytest.mark.integration
async def test_save_config_returns_503_on_storage_failure(
    client, mock_config_manager
):
    """503 when ConfigManager.save_config raises an exception."""
    mock_config_manager.save_config.side_effect = Exception("Storage down")

    response = await client.post(
        "/api/configs",
        json={
            "config_name": "My Config",
            "field_definitions": [
                {"name": "Email", "description": "Email address", "strategy": "Black Out"}
            ],
        },
    )

    assert response.status_code == 503


@pytest.mark.integration
async def test_save_config_returns_503_when_config_manager_not_configured(
    client, monkeypatch
):
    """503 when _config_manager is None (Databricks not configured)."""
    monkeypatch.setattr(main_module, "_config_manager", None)

    response = await client.post(
        "/api/configs",
        json={
            "config_name": "Test",
            "field_definitions": [
                {"name": "Email", "description": "Email address", "strategy": "Black Out"}
            ],
        },
    )

    assert response.status_code == 503


# ===========================================================================
# GET /api/configs
# ===========================================================================


@pytest.mark.integration
async def test_list_configs_returns_summaries(client, mock_config_manager):
    """GET /api/configs returns a list of config summaries."""
    mock_config_manager.list_configs.return_value = [
        {"config_name": "My Config", "key": "my_config", "saved_at": "2024-01-01T00:00:00+00:00"},
    ]

    response = await client.get("/api/configs")

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, list)
    assert body[0]["config_name"] == "My Config"
    assert body[0]["key"] == "my_config"


@pytest.mark.integration
async def test_list_configs_returns_empty_list_when_none_saved(
    client, mock_config_manager
):
    """GET /api/configs returns [] when no configs have been saved."""
    mock_config_manager.list_configs.return_value = []

    response = await client.get("/api/configs")

    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.integration
async def test_list_configs_returns_503_on_storage_failure(
    client, mock_config_manager
):
    """503 when ConfigManager.list_configs raises."""
    mock_config_manager.list_configs.side_effect = Exception("Storage error")

    response = await client.get("/api/configs")

    assert response.status_code == 503


# ===========================================================================
# GET /api/configs/{key}
# ===========================================================================


@pytest.mark.integration
async def test_get_config_returns_field_definitions(client, mock_config_manager):
    """GET /api/configs/{key} returns the full config including field_definitions."""
    mock_config_manager.get_config.return_value = {
        "config_name": "My Config",
        "key": "my_config",
        "saved_at": "2024-01-01T00:00:00+00:00",
        "field_definitions": [
            {"name": "Full Name", "description": "A person's name", "strategy": "Fake Data"}
        ],
    }

    response = await client.get("/api/configs/my_config")

    assert response.status_code == 200
    body = response.json()
    assert body["key"] == "my_config"
    assert len(body["field_definitions"]) == 1
    assert body["field_definitions"][0]["name"] == "Full Name"


@pytest.mark.integration
async def test_get_config_returns_404_when_not_found(client, mock_config_manager):
    """404 when get_config returns None."""
    mock_config_manager.get_config.return_value = None

    response = await client.get("/api/configs/does-not-exist")

    assert response.status_code == 404
    assert "error" in response.json()


@pytest.mark.integration
async def test_get_config_returns_503_on_storage_failure(client, mock_config_manager):
    """503 when ConfigManager.get_config raises."""
    mock_config_manager.get_config.side_effect = Exception("Storage error")

    response = await client.get("/api/configs/some_key")

    assert response.status_code == 503
