"""
Integration tests for POST /api/sessions/{session_id}/verify.

The endpoint retrieves masked.pdf from UC storage, extracts its text layer
with PyMuPDF, and checks whether each entity's original_text still appears.

Tests use create_pdf_with_text (from conftest) to produce PDFs whose text
content is precisely known, so we can assert masking scores deterministically.
"""

import pytest

from tests.conftest import create_pdf_with_text

_SESSION_ID = "test-session-id"

# An entity that we'll check against the PDF text
_ENTITY_JOHN = {
    "id": "entity-1",
    "entity_type": "Full Name",
    "original_text": "John Smith",
    "replacement_text": "Jane Doe",
    "confidence": 0.95,
    "approved": True,
    "occurrences": [
        {"page_number": 1, "original_text": "John Smith", "bounding_boxes": [[0.05, 0.08, 0.45, 0.025]]},
    ],
}

_ENTITY_EMAIL = {
    "id": "entity-2",
    "entity_type": "Email",
    "original_text": "john@example.com",
    "replacement_text": "[REDACTED]",
    "confidence": 0.9,
    "approved": True,
    "occurrences": [
        {"page_number": 1, "original_text": "john@example.com", "bounding_boxes": [[0.05, 0.15, 0.45, 0.025]]},
    ],
}


# ===========================================================================
# Happy path
# ===========================================================================


@pytest.mark.integration
async def test_verify_returns_score_100_when_all_entities_masked(
    client, override_databricks_dependency
):
    """Score is 100 when none of the original entity texts appear in the PDF."""
    sm = override_databricks_dependency
    # PDF contains "Hello World" but NOT the entity texts
    pdf_bytes = create_pdf_with_text([("Hello World", 50, 100)])
    sm.get_file.return_value = pdf_bytes

    response = await client.post(
        f"/api/sessions/{_SESSION_ID}/verify",
        json={"entities": [_ENTITY_JOHN]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["score"] == 100.0
    assert body["masked_count"] == 1
    assert body["total"] == 1
    assert body["entities"][0]["masked"] is True


@pytest.mark.integration
async def test_verify_returns_score_0_when_no_entities_masked(
    client, override_databricks_dependency
):
    """Score is 0 when the original entity text is still present in the PDF."""
    sm = override_databricks_dependency
    pdf_bytes = create_pdf_with_text([("John Smith", 50, 100)])
    sm.get_file.return_value = pdf_bytes

    response = await client.post(
        f"/api/sessions/{_SESSION_ID}/verify",
        json={"entities": [_ENTITY_JOHN]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["score"] == 0.0
    assert body["masked_count"] == 0
    assert body["entities"][0]["masked"] is False


@pytest.mark.integration
async def test_verify_partial_score_for_mixed_results(
    client, override_databricks_dependency
):
    """Score is 50 when exactly half of the entities are masked."""
    sm = override_databricks_dependency
    # PDF contains email but NOT John Smith
    pdf_bytes = create_pdf_with_text([("john@example.com", 50, 100)])
    sm.get_file.return_value = pdf_bytes

    response = await client.post(
        f"/api/sessions/{_SESSION_ID}/verify",
        json={"entities": [_ENTITY_JOHN, _ENTITY_EMAIL]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["score"] == 50.0
    assert body["masked_count"] == 1
    assert body["total"] == 2


@pytest.mark.integration
async def test_verify_is_case_insensitive(
    client, override_databricks_dependency
):
    """Matching is case-insensitive: 'JOHN SMITH' counts as not masked for 'John Smith'."""
    sm = override_databricks_dependency
    pdf_bytes = create_pdf_with_text([("JOHN SMITH", 50, 100)])
    sm.get_file.return_value = pdf_bytes

    response = await client.post(
        f"/api/sessions/{_SESSION_ID}/verify",
        json={"entities": [_ENTITY_JOHN]},
    )

    assert response.status_code == 200
    assert response.json()["entities"][0]["masked"] is False


@pytest.mark.integration
async def test_verify_checks_occurrence_original_text_too(
    client, override_databricks_dependency
):
    """If an occurrence has a different original_text (partial name), that is also checked."""
    sm = override_databricks_dependency
    # PDF still has "John" (a partial occurrence) but not "John Smith"
    pdf_bytes = create_pdf_with_text([("John still here", 50, 100)])
    sm.get_file.return_value = pdf_bytes

    entity_with_partial_occ = {
        **_ENTITY_JOHN,
        "occurrences": [
            {"page_number": 1, "original_text": "John Smith", "bounding_boxes": [[0.1, 0.1, 0.3, 0.04]]},
            {"page_number": 2, "original_text": "John", "bounding_boxes": [[0.1, 0.1, 0.2, 0.04]]},
        ],
    }

    response = await client.post(
        f"/api/sessions/{_SESSION_ID}/verify",
        json={"entities": [entity_with_partial_occ]},
    )

    assert response.status_code == 200
    # "john" still appears in the PDF text → not masked
    assert response.json()["entities"][0]["masked"] is False


@pytest.mark.integration
async def test_verify_empty_entities_returns_score_100(
    client, override_databricks_dependency
):
    """Score is 100 (fully masked) when the entities list is empty."""
    sm = override_databricks_dependency
    pdf_bytes = create_pdf_with_text([("Anything here", 50, 100)])
    sm.get_file.return_value = pdf_bytes

    response = await client.post(
        f"/api/sessions/{_SESSION_ID}/verify",
        json={"entities": []},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["score"] == 100.0
    assert body["total"] == 0
    assert body["masked_count"] == 0


# ===========================================================================
# Error responses
# ===========================================================================


@pytest.mark.integration
async def test_verify_returns_404_when_masked_pdf_not_found(
    client, override_databricks_dependency
):
    """404 when masked.pdf does not exist in UC storage."""
    sm = override_databricks_dependency
    sm.get_file.side_effect = FileNotFoundError("masked.pdf not found")

    response = await client.post(
        f"/api/sessions/{_SESSION_ID}/verify",
        json={"entities": [_ENTITY_JOHN]},
    )

    assert response.status_code == 404
    assert "error" in response.json()


@pytest.mark.integration
async def test_verify_returns_503_on_storage_failure(
    client, override_databricks_dependency
):
    """503 when get_file raises an unexpected exception."""
    sm = override_databricks_dependency
    sm.get_file.side_effect = Exception("Storage down")

    response = await client.post(
        f"/api/sessions/{_SESSION_ID}/verify",
        json={"entities": [_ENTITY_JOHN]},
    )

    assert response.status_code == 503
