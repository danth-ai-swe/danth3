"""Shared background thread pool for fire-and-forget work in the chat pipeline."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

BG_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="rag-bg")
