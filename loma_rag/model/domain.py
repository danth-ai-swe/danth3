"""Domain dataclasses shared across pipelines."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class AnswerResult:
    """Programmatic result of a single Q&A turn — used by tests."""
    query: str
    used_path: str   # "loma" | "web" | "refused" | "no_web_results" | "error"
    answer_text: str = ""
    en_search_query: str = ""               # populated when path=="web"
    loma_chunks: list = field(default_factory=list)
    related_nodes: list = field(default_factory=list)
    direct_node_ids: list = field(default_factory=list)  # nodes from direct KG semantic search
    web_docs: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    error: str = ""


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    section: str
    subsection: str
    course: str
    module: str
    lesson: str
    lesson_id: str
    source: str
    node_refs: list[str] = field(default_factory=list)
    fused_score: float = 0.0   # RRF score from Qdrant
    rerank_score: float = 0.0  # cross-encoder score (0 if rerank disabled)


@dataclass
class RetrievalResult:
    query: str                          # original user query
    chunks: list[RetrievedChunk]
    related_nodes: list                 # list[graph.Node] (chunk-derived + direct KG hits)
    timings: dict[str, float]
    search_text: str = ""               # English text actually used for vector search
    direct_node_ids: list[str] = field(default_factory=list)  # nodes from direct KG semantic search


@dataclass
class Chunk:
    chunk_id: str
    course: str
    module: str
    lesson: str
    lesson_id: str
    section: str         # H3
    subsection: str      # H4 ("" if none)
    text: str
    char_count: int
    token_estimate: int  # rough: chars/4
    source: str
    node_refs: list[str] = field(default_factory=list)


@dataclass
class IngestNode:
    node_id: str         # globally unique: f"{lesson_id}::{raw_id}"
    raw_id: str
    name: str
    definition: str
    category: str
    domain_tags: str
    related_raw: str
    course: str
    module: str
    lesson: str
    lesson_id: str
    source: str


@dataclass
class Edge:
    src: str
    dst: str
    relation: str = "related_to"


@dataclass
class Quiz:
    quiz_id: str
    course: str
    module: str
    lesson: str
    lesson_id: str
    no: int
    question: str
    options: list[str]
    correct_idx: int        # 1-based
    correct_text: str
    difficulty: str
    source: str


@dataclass
class WebDoc:
    url: str
    title: str
    snippet: str
    content: str = ""
    engine: str = ""
    rank_in_search: int = 0
    score: float = 0.0  # set after FAISS rank
