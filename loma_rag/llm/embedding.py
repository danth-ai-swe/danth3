"""Embedding helpers: dense (Azure), sparse (BM42), late-interaction (ColBERT).

Sparse and ColBERT models are loaded lazily on first call (~0.5s + 1.5s
cold start, then cached process-wide).
"""
from __future__ import annotations

import sys
import time

from openai import AzureOpenAI

from loma_rag.config.settings import azure
from loma_rag.constant.models import BM42_MODEL, COLBERT_MODEL
from loma_rag.constant.thresholds import DENSE_EMBED_CACHE_SIZE
from loma_rag.util.cache import LRUCache

_dense_cache = LRUCache(DENSE_EMBED_CACHE_SIZE)
_sparse_model = None
_colbert_model = None


def embed_dense_batch(client: AzureOpenAI, texts: list[str], retries: int = 3) -> list[list[float]]:
    """Batched dense embed with retry. Used by the indexer.

    Mirrors the legacy index.py:embed_dense — prints retry attempts to stderr
    for operator visibility into transient embed failures.
    """
    last: Exception | None = None
    for attempt in range(retries):
        try:
            resp = client.embeddings.create(model=azure.embedding_model, input=texts)
            return [d.embedding for d in resp.data]
        except Exception as e:  # noqa: BLE001
            last = e
            wait = 2 ** attempt
            print(f"  ! embed retry {attempt+1}/{retries} after {wait}s: {e}", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"embed failed after {retries} retries: {last}")


def embed_dense_single(client: AzureOpenAI, text: str) -> list[float]:
    """Cached single-text dense embed (used by the retriever hot path)."""
    cached = _dense_cache.get(text)
    if cached is not None:
        return cached
    vec = embed_dense_batch(client, [text])[0]
    _dense_cache.put(text, vec)
    return vec


def embed_sparse(text: str):
    """Sparse BM42 embed; lazy-load model on first call."""
    global _sparse_model
    if _sparse_model is None:
        from fastembed import SparseTextEmbedding
        _sparse_model = SparseTextEmbedding(model_name=BM42_MODEL)
    return next(iter(_sparse_model.embed([text])))


def embed_colbert(text: str):
    """ColBERT late-interaction embed; lazy-load model on first call."""
    global _colbert_model
    if _colbert_model is None:
        from fastembed import LateInteractionTextEmbedding
        _colbert_model = LateInteractionTextEmbedding(model_name=COLBERT_MODEL)
    return next(iter(_colbert_model.embed([text])))
