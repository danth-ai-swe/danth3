"""CLI display helpers — used by REPL and ad-hoc scripts."""
from __future__ import annotations

from loma_rag.model.domain import RetrievalResult, WebDoc


def print_loma_context(result: RetrievalResult) -> None:
    print("\n--- retrieved LOMA context ---")
    for i, c in enumerate(result.chunks, 1):
        print(
            f"  #{i}  rerank={c.rerank_score:+.2f}  rrf={c.fused_score:.2f}  "
            f"{c.chunk_id}  | {c.section} > {c.subsection!r}"
        )
    if result.related_nodes:
        names = ", ".join(n.name for n in result.related_nodes[:8])
        more = "" if len(result.related_nodes) <= 8 else f", +{len(result.related_nodes)-8} more"
        print(f"  related nodes ({len(result.related_nodes)}): {names}{more}")
    parts = [f"{k}={v*1000:.0f}ms" for k, v in result.timings.items()]
    print(f"  timings: {{ {', '.join(parts)} }}")


def print_loma_sources(result: RetrievalResult) -> None:
    print("\n--- sources ---")
    for c in result.chunks:
        sub = f" > {c.subsection}" if c.subsection else ""
        print(f"  [{c.chunk_id}]  {c.lesson_id}  |  {c.section}{sub}")


def print_web_sources(docs: list[WebDoc]) -> None:
    print("\n--- web sources ---")
    for i, d in enumerate(docs, 1):
        print(f"  [WEB_{i}]  ({d.score:.3f})  {d.title}")
        print(f"            {d.url}")
