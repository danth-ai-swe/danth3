"""Custom exceptions for the LOMA RAG package.

Catch-all base: RagError. Subclasses are domain-specific so callers
that care about a particular failure mode can target it.
"""
from __future__ import annotations


class RagError(Exception):
    """Base for all RAG-package errors."""


class LLMError(RagError):
    """Failure calling Azure OpenAI / chat / embedding endpoints."""


class VectorDBError(RagError):
    """Qdrant connection, schema, upsert, or query failure."""


class RetrievalError(RagError):
    """Retrieval pipeline failure (search, fuse, rerank, KG expand)."""


class IngestError(RagError):
    """Ingestion pipeline failure (parsing, chunking, JSONL write)."""


class WebFallbackError(RagError):
    """SearXNG fetch, page-fetch, or FAISS-rerank failure."""
