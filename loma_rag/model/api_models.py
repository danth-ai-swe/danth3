"""Pydantic models exposed by the FastAPI surface."""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question")
    top_k: int = Field(7, ge=1, le=20, description="LOMA chunks to feed the LLM")
    web_k: int = Field(5, ge=1, le=10, description="Web docs to feed the LLM in fallback")


class Citation(BaseModel):
    label: str
    source: Optional[str] = None  # original file name containing the chunk
    lesson_id: Optional[str] = None
    course: Optional[str] = None
    section: Optional[str] = None
    subsection: Optional[str] = None
    rerank_score: Optional[float] = None
    title: Optional[str] = None
    url: Optional[str] = None
    score: Optional[float] = None


class GraphNode(BaseModel):
    node_id: str
    name: str
    category: str
    definition: str
    lesson_id: str
    direct_hit: bool = False  # True if pulled by direct KG semantic search


class ChatData(BaseModel):
    path: str
    answer: str
    citations: list[Citation] = []
    related_nodes: list[GraphNode] = []
    en_search_query: str = ""
    intent: str = ""
    web_search_used: bool = False


class ChatResponse(BaseModel):
    success: bool
    data: Optional[ChatData] = None
    error: str = ""
