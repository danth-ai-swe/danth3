"""Stable point-ID generation and retried upsert."""
from __future__ import annotations

import sys
import time
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from loma_rag.constant.thresholds import EMBED_BATCH, UPSERT_BATCH
from loma_rag.util.retry import with_retries

NAMESPACE = uuid.UUID("9c3a3a4e-2e2c-4d8a-bcd7-1f59f3a1c800")  # for stable point IDs


def stable_id(key: str) -> str:
    return str(uuid.uuid5(NAMESPACE, key))


def upsert_with_retry(qc: QdrantClient, coll: str, points: list[PointStruct], retries: int = 3) -> None:
    """Upsert with retry — ColBERT multivector payloads can fail transiently
    on connection abort under big request loads."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            qc.upsert(collection_name=coll, points=points, wait=True)
            return
        except Exception as e:  # noqa: BLE001
            last = e
            wait = 2 ** attempt
            print(f"    ! upsert retry {attempt+1}/{retries} after {wait}s: {e}",
                  file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"upsert failed after {retries} retries: {last}")
