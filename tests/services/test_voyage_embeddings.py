import os
import subprocess
import sys

import pytest
from langchain_voyageai import VoyageAIEmbeddings

from app.services.voyage_embeddings import (
    VOYAGE_EMBEDDING_MODELS,
    VoyageEmbeddings,
    init_voyage_embeddings,
)

KEY = "test-key-not-real"


def test_model_is_required():
    with pytest.raises(ValueError, match="EMBEDDINGS_MODEL is required"):
        init_voyage_embeddings(model=None, api_key=KEY)
    with pytest.raises(ValueError, match="EMBEDDINGS_MODEL is required"):
        init_voyage_embeddings(model="", api_key=KEY)


@pytest.mark.parametrize(
    "model", ["text-embedding-3-small", "nomic-embed-text", "voyage-nope"]
)
def test_unknown_or_other_provider_model_is_rejected(model):
    with pytest.raises(ValueError, match="Unsupported Voyage embeddings model"):
        init_voyage_embeddings(model=model, api_key=KEY)


def test_api_key_is_required():
    with pytest.raises(ValueError, match="VOYAGE_API_KEY"):
        init_voyage_embeddings(model="voyage-4-lite", api_key="")


@pytest.mark.parametrize("model", ["voyage-finance-2", "voyage-law-2"])
def test_fixed_dimension_models_reject_other_dimensions(model):
    with pytest.raises(ValueError, match="Allowed values: 1024"):
        init_voyage_embeddings(model=model, api_key=KEY, dimensions=512)


def test_matryoshka_models_reject_unsupported_dimension():
    with pytest.raises(ValueError, match="256, 512, 1024, 2048"):
        init_voyage_embeddings(model="voyage-4", api_key=KEY, dimensions=768)


@pytest.mark.parametrize("model", list(VOYAGE_EMBEDDING_MODELS))
def test_every_listed_model_builds_without_truncation(model):
    emb = init_voyage_embeddings(model=model, api_key=KEY, dimensions=1024)
    assert emb.model == model
    assert emb.output_dimension == 1024
    assert emb.truncation is False


def test_output_dimension_is_omitted_unless_requested():
    emb = init_voyage_embeddings(model="voyage-4", api_key=KEY)
    assert emb.output_dimension is None


def test_vector_count_mismatch_is_rejected(monkeypatch):
    # voyage-context-* auto-chunking can return more vectors than inputs.
    monkeypatch.setattr(
        VoyageAIEmbeddings, "embed_documents", lambda self, texts: [[0.1], [0.2]]
    )
    emb = init_voyage_embeddings(model="voyage-context-4", api_key=KEY)
    with pytest.raises(ValueError, match="returned 2 vectors for 1 texts"):
        emb.embed_documents(["one long document"])
    assert emb.embed_documents(["a", "b"]) == [[0.1], [0.2]]


async def test_vector_count_mismatch_is_rejected_async(monkeypatch):
    async def fake(self, texts):
        return []

    monkeypatch.setattr(VoyageAIEmbeddings, "aembed_documents", fake)
    emb = init_voyage_embeddings(model="voyage-context-3", api_key=KEY)
    with pytest.raises(ValueError, match="returned 0 vectors for 1 texts"):
        await emb.aembed_documents(["x"])


def test_app_refuses_to_start_without_model(tmp_path):
    env = {
        **os.environ,
        "EMBEDDINGS_PROVIDER": "voyage",
        "VOYAGE_API_KEY": KEY,
        "RAG_UPLOAD_DIR": str(tmp_path),
    }
    env.pop("EMBEDDINGS_MODEL", None)
    result = subprocess.run(
        [sys.executable, "-c", "import app.config"],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode != 0
    assert "EMBEDDINGS_MODEL is required" in result.stderr
