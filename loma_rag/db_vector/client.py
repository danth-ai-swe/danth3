"""Qdrant client factory."""
from __future__ import annotations

from qdrant_client import QdrantClient

from loma_rag.config.settings import qdrant


def make_qdrant_client() -> QdrantClient:
    return QdrantClient(url=qdrant.url)
