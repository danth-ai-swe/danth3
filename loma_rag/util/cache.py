"""Bounded LRU cache — not thread-safe, but safe enough for single-process FastAPI use."""
from __future__ import annotations

from collections import OrderedDict

__all__ = ["LRUCache"]


class LRUCache:
    """Tiny bounded LRU. Threadsafe-ish for our use (single-process FastAPI)."""

    def __init__(self, maxsize: int) -> None:
        self._d: "OrderedDict[str, object]" = OrderedDict()
        self._max = maxsize

    def get(self, key: str):
        if key in self._d:
            self._d.move_to_end(key)
            return self._d[key]
        return None

    def put(self, key: str, value) -> None:
        if key in self._d:
            self._d.move_to_end(key)
        elif len(self._d) >= self._max:
            self._d.popitem(last=False)
        self._d[key] = value
