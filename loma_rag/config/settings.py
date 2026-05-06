"""Centralised configuration loaded from .env once at import time.

All env reads previously scattered across chat.py / retrieval.py /
index.py / web_fallback.py / api.py funnel through here.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Project root (parent of this loma_rag/ package directory).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "out"

# Load .env exactly once.
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class AzureConfig:
    api_base: str = os.environ.get("OPENAI_API_BASE", "")
    api_key: str = os.environ.get("OPENAI_API_KEY", "")
    api_version: str = os.environ.get("OPENAI_API_VERSION", "2025-01-01-preview")
    chat_model: str = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o")
    detect_model: str = os.environ.get("OPENAI_DETECT_MODEL", "gpt-4o-mini")
    short_answer_model: str = os.environ.get("OPENAI_SHORT_ANSWER_MODEL", "gpt-4o-mini")
    embedding_model: str = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


@dataclass(frozen=True)
class QdrantConfig:
    url: str = os.environ.get("QDRANT_URL", "http://localhost:6333")


@dataclass(frozen=True)
class WebConfig:
    searxng_url: str = os.environ.get("SEARXNG_URL", "http://localhost:8080")


@dataclass(frozen=True)
class RagConfig:
    rerank_model: str = os.environ.get("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
    prefetch_limit: int = int(os.environ.get("PREFETCH_LIMIT", "100"))
    node_rrf_threshold: float = float(os.environ.get("NODE_RRF_THRESHOLD", "0.4"))
    direct_node_top_k: int = int(os.environ.get("DIRECT_NODE_TOP_K", "5"))
    sniff_chars: int = int(os.environ.get("SNIFF_CHARS", "25"))
    early_exit_rerank: float = float(os.environ.get("EARLY_EXIT_RERANK", "-2.0"))
    early_exit_rrf: float = float(os.environ.get("EARLY_EXIT_RRF", "0.10"))
    high_confidence_rerank: float = float(os.environ.get("HIGH_CONFIDENCE_RERANK", "2.0"))
    high_confidence_rrf: float = float(os.environ.get("HIGH_CONFIDENCE_RRF", "0.50"))


@dataclass(frozen=True)
class APIConfig:
    host: str = os.environ.get("API_HOST", "127.0.0.1")
    port: int = int(os.environ.get("API_PORT", "8000"))


# Eagerly-built singletons (frozen dataclasses, safe to share).
azure = AzureConfig()
qdrant = QdrantConfig()
web = WebConfig()
rag = RagConfig()
api = APIConfig()
