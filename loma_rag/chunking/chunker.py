"""Heading-aware chunking + tiny-chunk merging + chunk<->node linking."""
from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.text.paragraph import Paragraph
from docx.table import Table

from loma_rag.chunking.docx_blocks import iter_block_items, table_to_md
from loma_rag.chunking.text import norm_ws, SKIP_PARAS
from loma_rag.constant.thresholds import MAX_CHUNK_CHARS, MIN_CHUNK_CHARS
from loma_rag.model.domain import Chunk, IngestNode


def chunk_docx(path: Path) -> list[Chunk]:
    from loma_rag.ingest.filename import parse_filename
    meta = parse_filename(path.name)
    if not meta:
        return []
    doc = Document(path)
    chunks: list[Chunk] = []
    section = ""
    subsection = ""
    buf: list[str] = []
    seq = 0

    def flush() -> None:
        nonlocal buf, seq
        text = "\n\n".join(s for s in buf if s).strip()
        buf = []
        if not text:
            return
        cid = f"{meta['lesson_id']}_C{seq:03d}"
        seq += 1
        chunks.append(Chunk(
            chunk_id=cid,
            course=meta["course"],
            module=meta["module"],
            lesson=meta["lesson"],
            lesson_id=meta["lesson_id"],
            section=section,
            subsection=subsection,
            text=text,
            char_count=len(text),
            token_estimate=len(text) // 4,
            source=path.name,
        ))

    for block in iter_block_items(doc):
        if isinstance(block, Paragraph):
            style = block.style.name if block.style else ""
            text = block.text.strip()
            if not text:
                continue
            if style in ("Heading 1", "Heading 2"):
                # Module/Lesson headers — flush, but don't open new section yet
                flush()
                continue
            if style == "Heading 3":
                flush()
                section = text
                subsection = ""
                continue
            if style == "Heading 4":
                flush()
                subsection = text
                continue
            if text.lower() in SKIP_PARAS:
                continue
            buf.append(text)
        else:  # Table
            md = table_to_md(block)
            if md:
                buf.append(md)

        # split if buffer too big within a single subsection
        if sum(len(s) for s in buf) > MAX_CHUNK_CHARS:
            flush()

    flush()
    return _merge_tiny(chunks)


def _merge_tiny(chunks: list[Chunk]) -> list[Chunk]:
    """Merge tiny chunks (< MIN_CHUNK_CHARS) into a same-section neighbor.
    Prefer backward merge; fall back to forward merge for section-leading tinies.
    """
    if not chunks:
        return chunks

    # Backward merge.
    out: list[Chunk] = [chunks[0]]
    for c in chunks[1:]:
        prev = out[-1]
        if (
            len(c.text) < MIN_CHUNK_CHARS
            and prev.section == c.section
            and len(prev.text) + len(c.text) < MAX_CHUNK_CHARS
        ):
            head = f"### {c.subsection}\n" if c.subsection else ""
            prev.text = f"{prev.text}\n\n{head}{c.text}"
            prev.char_count = len(prev.text)
            prev.token_estimate = prev.char_count // 4
        else:
            out.append(c)

    # Forward merge: section-leading tiny chunk absorbs into next.
    i = 0
    final: list[Chunk] = []
    while i < len(out):
        c = out[i]
        nxt = out[i + 1] if i + 1 < len(out) else None
        if (
            len(c.text) < MIN_CHUNK_CHARS
            and nxt is not None
            and nxt.section == c.section
            and len(c.text) + len(nxt.text) < MAX_CHUNK_CHARS
        ):
            head = f"### {nxt.subsection}\n" if nxt.subsection else ""
            nxt.text = f"{c.text}\n\n{head}{nxt.text}"
            nxt.char_count = len(nxt.text)
            nxt.token_estimate = nxt.char_count // 4
            final.append(nxt)
            i += 2
        else:
            final.append(c)
            i += 1
    return final


def link_chunks_to_nodes(chunks: list[Chunk], nodes: list[IngestNode]) -> None:
    """For each chunk, add node_ids whose Name appears as a whole word in chunk text."""
    by_lesson: dict[str, list[IngestNode]] = {}
    for n in nodes:
        if n.name:
            by_lesson.setdefault(n.lesson_id, []).append(n)

    # precompile patterns per lesson, longest names first to bias toward
    # specific terms when names overlap
    patterns: dict[str, list[tuple[re.Pattern, str]]] = {}
    for lid, ns in by_lesson.items():
        ns_sorted = sorted(ns, key=lambda x: -len(x.name))
        patterns[lid] = [
            (re.compile(r"\b" + re.escape(n.name.lower()) + r"\b"), n.node_id)
            for n in ns_sorted
        ]

    for c in chunks:
        text_low = c.text.lower()
        refs: list[str] = []
        for pat, nid in patterns.get(c.lesson_id, []):
            if pat.search(text_low):
                refs.append(nid)
        c.node_refs = refs
