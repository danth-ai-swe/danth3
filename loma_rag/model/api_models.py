"""Pydantic models exposed by the FastAPI surface."""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, field_validator

_ALLOWED_OPTION_IDS: frozenset[str] = frozenset("ABCD")


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


class QuizOption(BaseModel):
    id: str = Field(..., pattern=r"^[A-D]$")
    content: str = Field(..., min_length=1, max_length=1000)


class QuizChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000,
                          description="Quiz question text")
    options: list[QuizOption] = Field(..., min_length=2, max_length=4,
                                      description="Multiple-choice options")
    query: str = Field(..., min_length=1, max_length=2000,
                       description="Learner's prompt")
    top_k: int = Field(7, ge=1, le=20)
    web_k: int = Field(5, ge=1, le=10)

    @field_validator("options", mode="after")
    @classmethod
    def _check_option_ids(cls, options: list[QuizOption]) -> list[QuizOption]:
        bad: list[str] = [o.id for o in options if o.id not in _ALLOWED_OPTION_IDS]
        if bad:
            raise ValueError(f"option.id must be one of A/B/C/D, got {bad}")
        seen: set[str] = set()
        dup: list[str] = []
        for o in options:
            if o.id in seen:
                dup.append(o.id)
            seen.add(o.id)
        if dup:
            raise ValueError(f"duplicate option.id values: {dup}")
        return options


class QuizChatData(BaseModel):
    path: str
    intent: str
    answer: Optional[str] = None       # "A"/"B"/"C"/"D" only when intent=="answer"
    message: str = ""
    citations: list[Citation] = []
    related_nodes: list[GraphNode] = []
    en_search_query: str = ""
    web_search_used: bool = False


class QuizChatResponse(BaseModel):
    success: bool
    data: Optional[QuizChatData] = None
    error: str = ""
