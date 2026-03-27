"""
embedder.py — embedding support for RAG.

Two backends:
  1. Sentence-transformers (legacy, app-level RAG):
       all-mpnet-base-v2  768D  PDF/TXT
       all-MiniLM-L6-v2   384D  CSV/JSON

  2. Gemini (Vertex AI) — text-embedding-004, 768D
       Used for user-uploaded document ingestion + retrieval via messages endpoint.
       Single unified model for both app data queries and user attachment queries.
"""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import numpy as np
import structlog
from rag.document_type_detector import MINILM_MODEL, MPNET_MODEL

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = structlog.get_logger()

GEMINI_EMBED_MODEL = "text-embedding-004"
GEMINI_EMBED_DIM = 768

# Model cache: model_name → loaded SentenceTransformer instance
_model_cache: dict[str, SentenceTransformer] = {}

# Map doc_type → sentence-transformer model name (legacy app-level RAG)
_DOC_TYPE_TO_MODEL: dict[str, str] = {
    "pdf":     MPNET_MODEL,
    "txt":     MPNET_MODEL,
    "csv":     MINILM_MODEL,
    "json":    MINILM_MODEL,
    "db_rows": MINILM_MODEL,
}


def model_for_doc_type(doc_type: str) -> str:
    """Return sentence-transformer model ID for the given doc_type."""
    return _DOC_TYPE_TO_MODEL.get(doc_type, MPNET_MODEL)


def _load_model_sync(model_name: str) -> SentenceTransformer:
    """Load (or retrieve from cache) a SentenceTransformer model. Sync."""
    if model_name not in _model_cache:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            ) from exc
        logger.info("loading_embedding_model", model=model_name)
        _model_cache[model_name] = SentenceTransformer(model_name)
        logger.info("embedding_model_loaded", model=model_name)
    return _model_cache[model_name]


async def embed_texts(texts: list[str], model_name: str) -> np.ndarray:
    """
    Encode a list of texts with a sentence-transformer model (app-level RAG).
    Returns float32 ndarray of shape (len(texts), embedding_dim), L2-normalised.
    """
    if not texts:
        return np.array([], dtype=np.float32)

    def _encode() -> np.ndarray:
        model = _load_model_sync(model_name)
        return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    vectors: np.ndarray = await asyncio.to_thread(_encode)
    return vectors.astype(np.float32)


async def embed_query(text: str, model_name: str) -> np.ndarray:
    """Encode a single query string with sentence-transformer. Returns 1-D float32 ndarray."""
    result = await embed_texts([text], model_name)
    return result[0] if len(result) > 0 else np.array([], dtype=np.float32)


# ── Gemini embeddings (Vertex AI text-embedding-004, 768D) ────────────────────


async def embed_texts_gemini(texts: list[str]) -> np.ndarray:
    """
    Encode a list of texts using Vertex AI text-embedding-004 (768D).
    L2-normalised so cosine similarity == inner product.
    Used for user-uploaded document ingestion and unified retrieval.
    """
    if not texts:
        return np.array([], dtype=np.float32)

    from rag.llm_clients import _get_client

    client = _get_client()
    vectors: list[list[float]] = []
    for text in texts:
        result = await client.aio.models.embed_content(
            model=GEMINI_EMBED_MODEL,
            content=text,
        )
        vectors.append(list(result.embedding.values))

    arr = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return arr / norms


async def embed_query_gemini(text: str) -> np.ndarray:
    """Encode a single query string using Gemini text-embedding-004. Returns 1-D float32."""
    result = await embed_texts_gemini([text])
    return result[0] if len(result) > 0 else np.array([], dtype=np.float32)
