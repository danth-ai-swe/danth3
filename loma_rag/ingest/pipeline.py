"""Ingest orchestrator: scan files -> parse -> chunk -> link -> write JSONL.

Inputs:  D:/t/LOMA/**  (docx Knowledge Files + xlsx Knowledge Nodes + xlsx Quizzes)
Outputs: out/{chunks,nodes,edges,quizzes}.jsonl

Run: python -m loma_rag.ingest.pipeline
"""
from __future__ import annotations

import sys
from pathlib import Path

from loma_rag.chunking.chunker import chunk_docx, link_chunks_to_nodes
from loma_rag.config.settings import OUT_DIR, PROJECT_ROOT
from loma_rag.ingest.graph_builder import build_edges
from loma_rag.ingest.tree_scanner import build_syllabus_chunks, build_toc_chunks
from loma_rag.ingest.xlsx_parser import parse_nodes, parse_quiz
from loma_rag.model.domain import Chunk, IngestNode, Quiz
from loma_rag.util.io import write_jsonl

# D:/t/LOMA — the directory containing LOMA 281 / LOMA 291 sub-folders.
# PROJECT_ROOT is D:/t/LOMA/rag, so its parent is D:/t/LOMA.
ROOT = PROJECT_ROOT.parent
OUT = OUT_DIR


def find_files(root: Path) -> tuple[list[Path], list[Path], list[Path]]:
    from loma_rag.chunking.text import norm_ws

    docx_files: list[Path] = []
    node_files: list[Path] = []
    quiz_files: list[Path] = []
    for p in root.rglob("*.docx"):
        if "Knowledge File" in norm_ws(p.name):
            docx_files.append(p)
    for p in root.rglob("*.xlsx"):
        nm = norm_ws(p.name)
        if "Knowledge Node" in nm:
            node_files.append(p)
        elif "03_Quiz" in p.parts:
            quiz_files.append(p)
    return sorted(docx_files), sorted(node_files), sorted(quiz_files)


def main() -> int:
    print(f"Scanning {ROOT}")
    docx_files, node_files, quiz_files = find_files(ROOT)
    print(f"  docx Knowledge Files : {len(docx_files)}")
    print(f"  xlsx Knowledge Nodes : {len(node_files)}")
    print(f"  xlsx Quizzes         : {len(quiz_files)}")

    print("\nParsing knowledge nodes")
    all_nodes: list[IngestNode] = []
    for p in node_files:
        try:
            all_nodes.extend(parse_nodes(p))
        except Exception as e:
            print(f"  ! {p.name}: {e}", file=sys.stderr)
    print(f"  -> {len(all_nodes)} nodes")

    print("\nBuilding edges")
    edges, unresolved = build_edges(all_nodes)
    print(f"  -> {len(edges)} edges  ({len(unresolved)} unresolved references)")
    if unresolved[:5]:
        for u in unresolved[:5]:
            print(f"     unresolved: {u}")
        if len(unresolved) > 5:
            print(f"     (+ {len(unresolved)-5} more)")

    print("\nChunking docx")
    all_chunks: list[Chunk] = []
    for p in docx_files:
        try:
            all_chunks.extend(chunk_docx(p))
        except Exception as e:
            print(f"  ! {p.name}: {e}", file=sys.stderr)
    print(f"  -> {len(all_chunks)} chunks")

    print("\nBuilding TOC chunks (course + module outline)")
    toc_chunks = build_toc_chunks(ROOT)
    all_chunks.extend(toc_chunks)
    print(f"  -> {len(toc_chunks)} TOC chunks (total now {len(all_chunks)})")

    print("\nBuilding Syllabus chunks (metadata + schedule)")
    syllabus_chunks = build_syllabus_chunks(ROOT)
    all_chunks.extend(syllabus_chunks)
    print(f"  -> {len(syllabus_chunks)} syllabus chunks (total now {len(all_chunks)})")

    print("\nLinking chunks -> nodes")
    link_chunks_to_nodes(all_chunks, all_nodes)
    linked = sum(1 for c in all_chunks if c.node_refs)
    total_refs = sum(len(c.node_refs) for c in all_chunks)
    print(f"  -> {linked}/{len(all_chunks)} chunks have >=1 node ref ({total_refs} refs total)")

    print("\nParsing quizzes")
    all_quizzes: list[Quiz] = []
    for p in quiz_files:
        try:
            all_quizzes.extend(parse_quiz(p))
        except Exception as e:
            print(f"  ! {p.name}: {e}", file=sys.stderr)
    print(f"  -> {len(all_quizzes)} quiz items")

    print(f"\nWriting JSONL to {OUT}")
    write_jsonl(all_chunks, OUT / "chunks.jsonl")
    write_jsonl(all_nodes, OUT / "nodes.jsonl")
    write_jsonl(edges, OUT / "edges.jsonl")
    write_jsonl(all_quizzes, OUT / "quizzes.jsonl")

    # quick stats
    if all_chunks:
        sizes = [c.char_count for c in all_chunks]
        sizes.sort()
        med = sizes[len(sizes) // 2]
        p95 = sizes[int(len(sizes) * 0.95)]
        print(f"\nChunk size (chars): min={sizes[0]} med={med} p95={p95} max={sizes[-1]}")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
