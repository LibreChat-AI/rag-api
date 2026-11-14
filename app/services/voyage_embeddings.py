# app/services/voyage_embeddings.py
"""Voyage AI embeddings.

Thin, strict layer over ``langchain_voyageai.VoyageAIEmbeddings`` (which calls
the official ``voyageai`` SDK). Differences from the bare LangChain class:

* There is no default model. ``EMBEDDINGS_MODEL`` must name one of the models
  listed in ``VOYAGE_EMBEDDING_MODELS``; anything else fails at startup.
* ``EMBEDDINGS_DIMENSIONS`` is checked against the dimensions the chosen model
  accepts instead of waiting for the API to answer with a 400.
* ``truncation`` is off: an input longer than the model context is an error,
  not a silently clipped vector.
* ``embed_documents`` must return exactly one vector per text. The
  ``voyage-context-*`` models run with auto-chunking, so the API may split a
  long input into several chunks; the vector store pairs vectors with texts by
  position, so a count mismatch is rejected rather than stored misaligned.
"""

from typing import List, Mapping, Optional, Tuple

from langchain_voyageai import VoyageAIEmbeddings

# Output dimensions per model (all models default to 1024).
_MATRYOSHKA_DIMENSIONS: Tuple[int, ...] = (256, 512, 1024, 2048)
_FIXED_DIMENSIONS: Tuple[int, ...] = (1024,)

VOYAGE_EMBEDDING_MODELS: Mapping[str, Tuple[int, ...]] = {
    # General purpose and code, 32k context.
    "voyage-4-large": _MATRYOSHKA_DIMENSIONS,
    "voyage-4": _MATRYOSHKA_DIMENSIONS,
    "voyage-4-lite": _MATRYOSHKA_DIMENSIONS,
    "voyage-code-4": _MATRYOSHKA_DIMENSIONS,
    "voyage-3-large": _MATRYOSHKA_DIMENSIONS,
    "voyage-3.5": _MATRYOSHKA_DIMENSIONS,
    "voyage-3.5-lite": _MATRYOSHKA_DIMENSIONS,
    "voyage-code-3": _MATRYOSHKA_DIMENSIONS,
    # Domain specific, fixed 1024 dimensions (finance-2: 32k, law-2: 16k context).
    "voyage-finance-2": _FIXED_DIMENSIONS,
    "voyage-law-2": _FIXED_DIMENSIONS,
    # Contextualized chunk embeddings (auto-chunking, 32k tokens per chunk).
    "voyage-context-4": _MATRYOSHKA_DIMENSIONS,
    "voyage-context-3": _MATRYOSHKA_DIMENSIONS,
}


def _check_one_vector_per_text(
    texts: List[str], vectors: List[List[float]], model: str
) -> List[List[float]]:
    if len(vectors) != len(texts):
        raise ValueError(
            f"Voyage model '{model}' returned {len(vectors)} vectors for "
            f"{len(texts)} texts. Auto-chunking split at least one input into "
            "several chunks; lower CHUNK_SIZE so every text fits in one chunk."
        )
    return vectors


class VoyageEmbeddings(VoyageAIEmbeddings):
    """``VoyageAIEmbeddings`` that rejects a vector/text count mismatch."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors = super().embed_documents(texts)
        return _check_one_vector_per_text(texts, vectors, self.model)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors = await super().aembed_documents(texts)
        return _check_one_vector_per_text(texts, vectors, self.model)


def init_voyage_embeddings(
    model: Optional[str],
    api_key: Optional[str],
    dimensions: Optional[int] = None,
) -> VoyageEmbeddings:
    supported = ", ".join(VOYAGE_EMBEDDING_MODELS)
    if not model:
        raise ValueError(
            "EMBEDDINGS_MODEL is required when EMBEDDINGS_PROVIDER=voyage "
            f"(there is no default). Supported models: {supported}"
        )
    if model not in VOYAGE_EMBEDDING_MODELS:
        raise ValueError(
            f"Unsupported Voyage embeddings model '{model}'. "
            f"Supported models: {supported}"
        )
    if not api_key:
        raise ValueError(
            "VOYAGE_API_KEY (or RAG_VOYAGE_API_KEY) is required when "
            "EMBEDDINGS_PROVIDER=voyage"
        )
    allowed = VOYAGE_EMBEDDING_MODELS[model]
    if dimensions is not None and dimensions not in allowed:
        raise ValueError(
            f"EMBEDDINGS_DIMENSIONS={dimensions} is not supported by Voyage "
            f"model '{model}'. Allowed values: {', '.join(map(str, allowed))}"
        )
    return VoyageEmbeddings(
        model=model,
        api_key=api_key,
        output_dimension=dimensions,
        truncation=False,
    )
