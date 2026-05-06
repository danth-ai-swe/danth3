"""Resolve node references into edges within a lesson scope."""
from __future__ import annotations

import re

from loma_rag.model.domain import Edge, IngestNode


def build_edges(nodes: list[IngestNode]) -> tuple[list[Edge], list[str]]:
    """Resolve Related Nodes within each lesson (by raw id OR name)."""
    by_lesson: dict[str, dict[str, IngestNode]] = {}
    for n in nodes:
        d = by_lesson.setdefault(n.lesson_id, {})
        d[n.raw_id.lower()] = n
        d[n.name.lower()] = n

    edges: list[Edge] = []
    seen: set[tuple[str, str]] = set()
    unresolved: list[str] = []

    for n in nodes:
        if not n.related_raw:
            continue
        parts = re.split(r"[,;|/\n]+", n.related_raw)
        for p in parts:
            p = p.strip().rstrip(".").rstrip(",").strip()
            if not p or p.lower() in {"etc", "etc.", "..."}:
                continue
            target = by_lesson.get(n.lesson_id, {}).get(p.lower())
            if not target:
                unresolved.append(f"{n.lesson_id}::{n.raw_id} -> {p!r}")
                continue
            if target.node_id == n.node_id:
                continue
            key = tuple(sorted([n.node_id, target.node_id]))
            if key in seen:
                continue
            seen.add(key)
            edges.append(Edge(src=n.node_id, dst=target.node_id))
    return edges, unresolved
