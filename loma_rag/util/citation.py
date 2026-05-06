"""Citation builders for LOMA chunks and web docs."""
from __future__ import annotations

from loma_rag.model.api_models import Citation


def build_citations(raw: list[dict]) -> list[Citation]:
    return [Citation(**c) for c in raw]


def format_loma_citations(answer: str, chunks) -> list[dict]:
    """Return only chunks whose chunk_id appears in `answer`."""
    cited: list[dict] = []
    for c in chunks:
        if c.chunk_id and c.chunk_id in answer:
            sub = c.subsection or ""
            cited.append({
                "label": c.chunk_id,
                "lesson_id": c.lesson_id,
                "course": c.course,
                "section": c.section,
                "subsection": sub,
                "rerank_score": float(c.rerank_score),
            })
    return cited


def format_web_citations(docs) -> list[dict]:
    return [
        {
            "label": f"WEB_{i}",
            "title": d.title,
            "url": d.url,
            "score": float(d.score),
        }
        for i, d in enumerate(docs, 1)
    ]
