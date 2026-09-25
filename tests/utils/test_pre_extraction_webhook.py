"""Tests for the optional pre-extraction webhook in document_loader.

The webhook is enabled only when PRE_EXTRACTION_WEBHOOK_URL is set. These tests
exercise the helper directly so we can cover the branching logic without
standing up FastAPI.
"""

import importlib

import pytest
import responses
from langchain_core.documents import Document


WEBHOOK_URL = "http://ocr-sidecar.test/extract"


@pytest.fixture
def enabled_webhook(monkeypatch):
    """Reload config + document_loader so the module-level constants pick up
    the new environment variables."""
    monkeypatch.setenv("PRE_EXTRACTION_WEBHOOK_URL", WEBHOOK_URL)
    monkeypatch.setenv("PRE_EXTRACTION_WEBHOOK_MIN_CHARS", "100")
    monkeypatch.setenv("PRE_EXTRACTION_WEBHOOK_TIMEOUT", "5")

    import app.config as config_module
    import app.utils.document_loader as loader_module

    importlib.reload(config_module)
    importlib.reload(loader_module)
    yield loader_module

    # Reset after the test so other tests see the default (disabled) state.
    monkeypatch.delenv("PRE_EXTRACTION_WEBHOOK_URL", raising=False)
    importlib.reload(config_module)
    importlib.reload(loader_module)


@pytest.fixture
def disabled_webhook(monkeypatch):
    monkeypatch.delenv("PRE_EXTRACTION_WEBHOOK_URL", raising=False)
    import app.config as config_module
    import app.utils.document_loader as loader_module

    importlib.reload(config_module)
    importlib.reload(loader_module)
    yield loader_module


def _make_pdf(tmp_path, name="input.pdf"):
    p = tmp_path / name
    p.write_bytes(b"%PDF-1.4\n%fake\n")
    return str(p)


def test_webhook_disabled_is_noop(disabled_webhook, tmp_path):
    """When the feature flag is empty, the helper must return the input
    unchanged without issuing any HTTP requests."""
    docs = [Document(page_content="", metadata={"source": "a.pdf"})]
    file_path = _make_pdf(tmp_path)
    out = disabled_webhook.maybe_enrich_with_webhook(file_path, docs)
    assert out is docs


def test_above_threshold_skips_webhook(enabled_webhook, tmp_path):
    """If average chars per page already exceeds the threshold we trust the
    existing extraction and do not call the webhook."""
    docs = [
        Document(page_content="a" * 150, metadata={"source": "a.pdf"}),
        Document(page_content="b" * 150, metadata={"source": "a.pdf"}),
    ]
    file_path = _make_pdf(tmp_path)

    # responses active but no registered endpoints — an unexpected call would
    # raise ConnectionError, which is exactly what we want to assert.
    with responses.RequestsMock() as rsps:
        out = enabled_webhook.maybe_enrich_with_webhook(file_path, docs)
        assert out == docs
        assert len(rsps.calls) == 0


def test_below_threshold_triggers_webhook_and_replaces_documents(
    enabled_webhook, tmp_path
):
    """Pages with effectively no text should be sent to the webhook and the
    result should replace the documents, annotated with ocr metadata."""
    docs = [
        Document(page_content="", metadata={"source": "scanned.pdf"}),
        Document(page_content="  ", metadata={"source": "scanned.pdf"}),
    ]
    file_path = _make_pdf(tmp_path, name="scanned.pdf")

    with responses.RequestsMock() as rsps:
        rsps.add(
            responses.POST,
            WEBHOOK_URL,
            json={
                "text": "OCRed contract content",
                "provider": "azure-di",
                "pages_processed": 2,
            },
            status=200,
        )
        out = enabled_webhook.maybe_enrich_with_webhook(file_path, docs)

    assert len(out) == 1
    assert out[0].page_content == "OCRed contract content"
    assert out[0].metadata["ocr_used"] is True
    assert out[0].metadata["ocr_provider"] == "azure-di"
    assert out[0].metadata["source"] == "scanned.pdf"


def test_webhook_failure_falls_back_to_original(enabled_webhook, tmp_path):
    """HTTP errors must never break ingest; we keep the original documents."""
    docs = [Document(page_content="", metadata={"source": "scanned.pdf"})]
    file_path = _make_pdf(tmp_path, name="scanned.pdf")

    with responses.RequestsMock() as rsps:
        rsps.add(responses.POST, WEBHOOK_URL, status=500)
        out = enabled_webhook.maybe_enrich_with_webhook(file_path, docs)

    assert out == docs


def test_webhook_empty_text_falls_back(enabled_webhook, tmp_path):
    """A 200 response with empty text means OCR had nothing to offer; keep
    whatever PyPDF returned rather than destroy the extraction."""
    docs = [Document(page_content="", metadata={"source": "scanned.pdf"})]
    file_path = _make_pdf(tmp_path, name="scanned.pdf")

    with responses.RequestsMock() as rsps:
        rsps.add(
            responses.POST,
            WEBHOOK_URL,
            json={"text": "", "provider": "azure-di"},
            status=200,
        )
        out = enabled_webhook.maybe_enrich_with_webhook(file_path, docs)

    assert out == docs


def test_webhook_handles_empty_document_list(enabled_webhook, tmp_path):
    """Empty list: avg_chars is 0 → webhook is called."""
    file_path = _make_pdf(tmp_path, name="scanned.pdf")

    with responses.RequestsMock() as rsps:
        rsps.add(
            responses.POST,
            WEBHOOK_URL,
            json={"text": "rescued text", "provider": "azure-di"},
            status=200,
        )
        out = enabled_webhook.maybe_enrich_with_webhook(file_path, [])

    assert len(out) == 1
    assert out[0].page_content == "rescued text"
    assert out[0].metadata["ocr_used"] is True
