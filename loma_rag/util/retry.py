"""Generic retry helper for flaky external calls (HTTP, LLM, vector DB)."""
from __future__ import annotations

import time
from typing import Callable, TypeVar

T = TypeVar("T")


def with_retries(
    fn: Callable[[], T],
    retries: int = 3,
    base_delay: float = 1.0,
    label: str = "operation",
) -> T:
    """Call fn() with exponential backoff on exception.

    Matches the legacy behavior of embed_dense / _upsert_with_retry:
        - First retry waits base_delay seconds (default 1.0).
        - On final failure, raises RuntimeError("<label> failed after N retries: <last>")
          with the original exception chained via `from`.

    Used as:
        result = with_retries(lambda: client.embed(texts), retries=3, label="embed")
    """
    last_exc: Exception | None = None
    for i in range(retries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last_exc = e
            if i == retries - 1:
                break
            time.sleep(base_delay * (2 ** i))
    raise RuntimeError(f"{label} failed after {retries} retries: {last_exc}") from last_exc
