# LOMA RAG Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize 11 flat-root Python files into a layered package `loma_rag/` with single-responsibility modules, while preserving runtime behavior exactly.

**Architecture:** New package `loma_rag/` with sub-packages `config/`, `constant/`, `model/`, `prompt/`, `llm/`, `db_vector/`, `chunking/`, `ingest/`, `rag/`, `api/`, `util/`. Single root file `loma_rag/exceptions.py`. Top-level `main.py` dispatcher; tools relocated to `scripts/` and `tests/`. Migration is leaf-first so every intermediate state is importable. Original files moved to `_legacy/` after import-check passes; deleted only after full E2E validation.

**Tech Stack:** Python 3.10+, openai (AzureOpenAI), qdrant-client, fastembed (BM42 + ColBERT), networkx, faiss, fastapi, python-docx, openpyxl, requests, beautifulsoup4, transformers (cross-encoder rerank).

**Spec:** `docs/superpowers/specs/2026-05-06-rag-refactor-design.md`

**Repo note:** Not a git repository. Replace any `git commit` step with a "checkpoint" — verify the import check passes, then proceed. Do not delete originals until Phase 6.

**Imports convention in this plan:**
- "Extract `<source>:<line-range>`" means copy that block of lines verbatim from the source file. Adjust only `import` statements at the top.
- "Public exports" means symbols importable from outside the new module; keep underscore-prefixed names module-private unless noted.
- After each task touches the source, leave the original file untouched until Phase 6 (we copy, not move, until then).

---

## Phase 1 — Foundations

No internal deps. After Phase 1, the new package compiles but consumers still use legacy.

### Task 1: Create package skeleton

**Files:**
- Create: `loma_rag/__init__.py`
- Create: `loma_rag/config/__init__.py`
- Create: `loma_rag/constant/__init__.py`
- Create: `loma_rag/model/__init__.py`
- Create: `loma_rag/prompt/__init__.py`
- Create: `loma_rag/llm/__init__.py`
- Create: `loma_rag/db_vector/__init__.py`
- Create: `loma_rag/chunking/__init__.py`
- Create: `loma_rag/ingest/__init__.py`
- Create: `loma_rag/rag/__init__.py`
- Create: `loma_rag/api/__init__.py`
- Create: `loma_rag/api/routes/__init__.py`
- Create: `loma_rag/util/__init__.py`
- Create: `scripts/` (no __init__ — not a package)
- Create: `tests/__init__.py`

- [ ] **Step 1: Create empty `__init__.py` for every package directory listed above.**

Each `__init__.py` contains a single line:
```python
"""LOMA RAG package."""
```

Sub-packages may use a more specific docstring, e.g. `loma_rag/rag/__init__.py`:
```python
"""RAG orchestration: retriever, pipeline, graph, web fallback."""
```

- [ ] **Step 2: Verify importability.**

Run:
```powershell
python -c "import loma_rag; import loma_rag.config; import loma_rag.constant; import loma_rag.model; import loma_rag.prompt; import loma_rag.llm; import loma_rag.db_vector; import loma_rag.chunking; import loma_rag.ingest; import loma_rag.rag; import loma_rag.api; import loma_rag.api.routes; import loma_rag.util; print('OK')"
```
Expected: `OK`

---

### Task 2: Create `loma_rag/exceptions.py`

**Files:**
- Create: `loma_rag/exceptions.py`

- [ ] **Step 1: Write the file.**

```python
"""Custom exceptions for the LOMA RAG package.

Catch-all base: RagError. Subclasses are domain-specific so callers
that care about a particular failure mode can target it.
"""
from __future__ import annotations


class RagError(Exception):
    """Base for all RAG-package errors."""


class LLMError(RagError):
    """Failure calling Azure OpenAI / chat / embedding endpoints."""


class VectorDBError(RagError):
    """Qdrant connection, schema, upsert, or query failure."""


class RetrievalError(RagError):
    """Retrieval pipeline failure (search, fuse, rerank, KG expand)."""


class IngestError(RagError):
    """Ingestion pipeline failure (parsing, chunking, JSONL write)."""


class WebFallbackError(RagError):
    """SearXNG fetch, page-fetch, or FAISS-rerank failure."""
```

- [ ] **Step 2: Verify.**

```powershell
python -c "from loma_rag.exceptions import RagError, RetrievalError, IngestError, LLMError, WebFallbackError, VectorDBError; print('OK')"
```
Expected: `OK`

---

### Task 3: Create `loma_rag/config/settings.py`

**Files:**
- Create: `loma_rag/config/settings.py`
- Reference: `chat.py:51-67`, `retrieval.py:64-77`, `index.py:46-65`, `web_fallback.py:31-40`, `api.py:34-35`

- [ ] **Step 1: Write the file.**

```python
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
```

- [ ] **Step 2: Verify.**

```powershell
python -c "from loma_rag.config.settings import azure, qdrant, web, rag, api; print(azure.chat_model, qdrant.url)"
```
Expected: prints chat model name and qdrant URL.

---

### Task 4: Create `constant/` modules

**Files:**
- Create: `loma_rag/constant/collections.py`
- Create: `loma_rag/constant/models.py`
- Create: `loma_rag/constant/tokens.py`
- Create: `loma_rag/constant/thresholds.py`

- [ ] **Step 1: `loma_rag/constant/collections.py`**

```python
"""Qdrant collection names and vector dimensions."""
CHUNKS_COLL = "loma_chunks"
NODES_COLL = "loma_nodes"

# Vector dimensions
DENSE_DIM = 1536      # text-embedding-3-small
COLBERT_DIM = 128     # colbert-v2 token-vector dimension
```

- [ ] **Step 2: `loma_rag/constant/models.py`**

```python
"""Model identifiers (defaults; runtime values come from config.settings)."""
DEFAULT_DENSE_MODEL = "text-embedding-3-small"
DEFAULT_CHAT_MODEL = "gpt-4o"
DEFAULT_DETECT_MODEL = "gpt-4o-mini"
DEFAULT_SHORT_ANSWER_MODEL = "gpt-4o-mini"
COLBERT_MODEL = "colbert-ir/colbertv2.0"
BM42_MODEL = "Qdrant/bm42-all-minilm-l6-v2-attentions"
DEFAULT_RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
```

- [ ] **Step 3: `loma_rag/constant/tokens.py`**

```python
"""Sentinel tokens used in LLM prompting / response parsing."""
INSUFFICIENT_TOKEN = "[INSUFFICIENT_CONTEXT]"
```

- [ ] **Step 4: `loma_rag/constant/thresholds.py`**

```python
"""Numeric thresholds and limits."""

# Chunking
MIN_CHUNK_CHARS = 200
MAX_CHUNK_CHARS = 6000  # ~1500 tokens

# Embed / upsert batch
EMBED_BATCH = 32
UPSERT_BATCH = 16

# LRU cache sizes
TRANSLATE_CACHE_MAX = 256
ANALYZE_CACHE_MAX = 512
LANG_CACHE_MAX = 512
DENSE_EMBED_CACHE_SIZE = 1024
TRANSLATE_FOR_SEARCH_CACHE_SIZE = 512
```

- [ ] **Step 5: Verify.**

```powershell
python -c "from loma_rag.constant.collections import CHUNKS_COLL, NODES_COLL, DENSE_DIM, COLBERT_DIM; from loma_rag.constant.tokens import INSUFFICIENT_TOKEN; from loma_rag.constant.thresholds import MIN_CHUNK_CHARS, MAX_CHUNK_CHARS; from loma_rag.constant.models import COLBERT_MODEL, BM42_MODEL; print('OK')"
```
Expected: `OK`

---

### Task 5: Create `model/api_models.py` and `model/domain.py`

**Files:**
- Create: `loma_rag/model/api_models.py` — extract from `api.py:44-79`
- Create: `loma_rag/model/domain.py` — extract from `chat.py:141-153` (AnswerResult), `retrieval.py:102-126` (RetrievedChunk, RetrievalResult), `ingest.py:96-109` (Chunk), `ingest.py:229-249` (Node→IngestNode, Edge), `ingest.py:368-381` (Quiz), `web_fallback.py:44-54` (WebDoc)

- [ ] **Step 1: Write `loma_rag/model/api_models.py` — paste verbatim from `api.py:44-79`.**

Header:
```python
"""Pydantic models exposed by the FastAPI surface."""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field
```

Then paste the four classes: `ChatRequest`, `Citation`, `GraphNode`, `ChatResponse` exactly as defined in `api.py:44-79`.

- [ ] **Step 2: Write `loma_rag/model/domain.py` — header.**

```python
"""Domain dataclasses shared across pipelines."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
```

- [ ] **Step 3: Append `AnswerResult` from `chat.py:141-153`** (paste verbatim).

- [ ] **Step 4: Append `RetrievedChunk` and `RetrievalResult` from `retrieval.py:102-126`** (paste verbatim).

- [ ] **Step 5: Append `Chunk` from `ingest.py:96-109`** (paste verbatim).

- [ ] **Step 6: Append `IngestNode` (rename of `Node`) and `Edge` from `ingest.py:229-249`.**

Read `ingest.py` lines 229-249, paste, then in the new file rename `class Node` → `class IngestNode`. Update any internal references (within those lines) that say `Node(` to `IngestNode(`. Do NOT touch `graph.py` — its `Node` is a different class.

- [ ] **Step 7: Append `Quiz` from `ingest.py:368-381`** (paste verbatim).

- [ ] **Step 8: Append `WebDoc` from `web_fallback.py:44-54`** (paste verbatim).

- [ ] **Step 9: Verify.**

```powershell
python -c "from loma_rag.model.api_models import ChatRequest, Citation, GraphNode, ChatResponse; from loma_rag.model.domain import AnswerResult, RetrievedChunk, RetrievalResult, Chunk, IngestNode, Edge, Quiz, WebDoc; print('OK')"
```
Expected: `OK`

---

### Task 6: Create Phase 1 `util/` modules

**Files:**
- Create: `loma_rag/util/cache.py` — extract `_LRUCache` from `retrieval.py:36-58`
- Create: `loma_rag/util/io.py` — extract `load_jsonl` from `index.py:242-246` and `write_jsonl` from `ingest.py:903-907`
- Create: `loma_rag/util/retry.py` — new helper, generalising the retry pattern in `index.py:78-92` (embed) and `index.py:162-176` (upsert)
- Create: `loma_rag/util/concurrency.py` — extract `_BG_POOL` from `chat.py:118`
- Create: `loma_rag/util/language.py` — extract `_LANG_NAMES` from `chat.py:316-332`, `language_name` from `chat.py:380-382`, `_detect_user_language` from `chat.py:384-394`, `_LANG_CODE_RE` from `chat.py:334`
- Create: `loma_rag/util/sse.py` — extract `_sse_event` from `api.py:192-194`

- [ ] **Step 1: `loma_rag/util/cache.py`**

```python
"""Thread-unsafe LRU cache used as a module-level memo across the package."""
from __future__ import annotations

from collections import OrderedDict
```

Then paste `_LRUCache` from `retrieval.py:36-58` verbatim. Rename to `LRUCache` (drop underscore — it is now public to the package). Add `__all__ = ["LRUCache"]` at top.

- [ ] **Step 2: `loma_rag/util/io.py`**

```python
"""JSONL read/write helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def load_jsonl(p: Path) -> list[dict]:
    """Read a JSONL file into a list of dicts."""
    out = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def write_jsonl(items: Iterable, path: Path) -> None:
    """Write an iterable of dataclass-or-dict items to JSONL.

    Dataclass instances are serialised via dataclasses.asdict; dicts pass through.
    """
    from dataclasses import asdict, is_dataclass
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            if is_dataclass(it):
                obj = asdict(it)
            else:
                obj = it
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
```

Compare against `index.py:242-246` and `ingest.py:903-907` — if either uses `dataclasses.asdict` differently, match that exact serialisation. The reference behaviour is verbatim; do not re-implement.

- [ ] **Step 3: `loma_rag/util/retry.py`**

```python
"""Generic retry helper for flaky external calls (HTTP, LLM, vector DB)."""
from __future__ import annotations

import time
from typing import Callable, TypeVar

T = TypeVar("T")


def with_retries(fn: Callable[[], T], retries: int = 3, base_delay: float = 0.5) -> T:
    """Call fn() with exponential backoff on exception.

    Used as: result = with_retries(lambda: client.embed(texts), retries=3)
    """
    last_exc: Exception | None = None
    for i in range(retries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last_exc = e
            if i == retries - 1:
                break
            time.sleep(base_delay * (2 ** i))
    raise last_exc  # type: ignore[misc]
```

- [ ] **Step 4: `loma_rag/util/concurrency.py`**

```python
"""Shared background thread pool for fire-and-forget work in the chat pipeline."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

BG_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="rag-bg")
```

- [ ] **Step 5: `loma_rag/util/language.py`**

Header:
```python
"""Language code/name helpers and heuristic-based user-language detection."""
from __future__ import annotations

import re
```

Paste `_LANG_NAMES` from `chat.py:316-332` verbatim, but rename to `LANG_NAMES` (drop underscore).
Paste `_LANG_CODE_RE` from `chat.py:334` verbatim, rename to `LANG_CODE_RE`.
Paste `language_name` from `chat.py:380-382` verbatim — internal references `_LANG_NAMES` → `LANG_NAMES`.
Paste `_detect_user_language` from `chat.py:384-394` verbatim — keep underscore; this is a private helper used by `chat`/`pipeline`. (Or rename to `detect_user_language` for export. Choose one and keep all callers consistent.) **Decision: rename to `detect_user_language` (public).**

- [ ] **Step 6: `loma_rag/util/sse.py`**

```python
"""Server-Sent Events frame encoder."""
from __future__ import annotations

import json


def sse_event(payload: dict) -> bytes:
    """Format a payload as a single SSE frame."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")
```

- [ ] **Step 7: Verify.**

```powershell
python -c "from loma_rag.util.cache import LRUCache; from loma_rag.util.io import load_jsonl, write_jsonl; from loma_rag.util.retry import with_retries; from loma_rag.util.concurrency import BG_POOL; from loma_rag.util.language import language_name, detect_user_language, LANG_NAMES; from loma_rag.util.sse import sse_event; print('OK')"
```
Expected: `OK`

---

### Task 7: Phase 1 import gate

- [ ] **Step 1: Run a single-pass import of every Phase 1 module.**

```powershell
python -c "import loma_rag.exceptions, loma_rag.config.settings, loma_rag.constant.collections, loma_rag.constant.models, loma_rag.constant.tokens, loma_rag.constant.thresholds, loma_rag.model.api_models, loma_rag.model.domain, loma_rag.util.cache, loma_rag.util.io, loma_rag.util.retry, loma_rag.util.concurrency, loma_rag.util.language, loma_rag.util.sse; print('PHASE 1 OK')"
```
Expected: `PHASE 1 OK`

- [ ] **Step 2: Checkpoint.** Phase 1 stable; legacy code untouched.

---

## Phase 2 — Providers

### Task 8: Create `prompt/` modules

**Files:**
- Create: `loma_rag/prompt/system.py` — extract `LOMA_SYSTEM` (`chat.py:156-167`), `WEB_SYSTEM` (`chat.py:170-179`)
- Create: `loma_rag/prompt/analyzer.py` — extract `_LANG_DETECT_SYSTEM` (`chat.py:181-191`), `_ANALYZE_QUERY_SYSTEM` (`chat.py:193-230`), `_ANALYZE_RE` (`chat.py:232`), `_LANG_CODE_RE` (`chat.py:334`)
- Create: `loma_rag/prompt/translate.py` — extract `TRANSLATE_SYSTEM` (`chat.py:397-405`), `TRANSLATE_FOR_SEARCH_SYSTEM` (`retrieval.py:82-90`)
- Create: `loma_rag/prompt/builder.py` — extract `build_loma_user_prompt` (`chat.py:406-436`), `build_web_user_prompt` (`chat.py:438-459`)
- Create: `loma_rag/prompt/judge.py` — extract judge system + user-prompt template from `eval.py` (search for `build_judge_user`)

- [ ] **Step 1: `loma_rag/prompt/system.py`**

Header:
```python
"""Top-level system prompts for the LOMA tutor and the web-fallback flow."""
```

Paste `LOMA_SYSTEM` and `WEB_SYSTEM` verbatim from the source ranges. **Do not reformat string contents.**

- [ ] **Step 2: `loma_rag/prompt/analyzer.py`**

Header:
```python
"""Query-analysis and language-detect prompts + parsing regexes."""
import re
```

Paste `_LANG_DETECT_SYSTEM`, `_ANALYZE_QUERY_SYSTEM` verbatim.
Paste `_ANALYZE_RE` (already a `re.compile(...)`) — note source uses `_re.compile`; replace `_re` with `re` since we import the canonical name.
Paste `_LANG_CODE_RE` similarly.

Public names: drop leading underscore — `LANG_DETECT_SYSTEM`, `ANALYZE_QUERY_SYSTEM`, `ANALYZE_RE`, `LANG_CODE_RE`.

(Note: `LANG_CODE_RE` also appears in `util/language.py` from Task 6 — keep it canonical there. Delete the duplicate from this file; the analyzer imports it from `loma_rag.util.language` instead. Update header to `from loma_rag.util.language import LANG_CODE_RE`.)

- [ ] **Step 3: `loma_rag/prompt/translate.py`**

Header:
```python
"""Translation prompts: user-language → English query, and search-translation."""
```

Paste both strings verbatim.

- [ ] **Step 4: `loma_rag/prompt/builder.py`**

Header:
```python
"""Build user-message bodies: LOMA chunks payload, web docs payload."""
from __future__ import annotations

from loma_rag.model.domain import RetrievalResult, WebDoc
from loma_rag.util.language import language_name
```

Paste `build_loma_user_prompt` and `build_web_user_prompt` verbatim from `chat.py:406-459`. The functions reference `language_name` and types — the new imports provide them.

- [ ] **Step 5: `loma_rag/prompt/judge.py`**

Read `eval.py` and locate the judge system prompt + `build_judge_user` function (search for "judge" or "Pick one"). Extract to:

```python
"""Eval judge prompts (LLM-as-judge, closed/open book)."""
from __future__ import annotations
```

Paste system prompt strings + the `build_judge_user` function verbatim. If `build_judge_user` references local helpers, copy those too (or move them into `util/`).

- [ ] **Step 6: Verify.**

```powershell
python -c "from loma_rag.prompt.system import LOMA_SYSTEM, WEB_SYSTEM; from loma_rag.prompt.analyzer import LANG_DETECT_SYSTEM, ANALYZE_QUERY_SYSTEM, ANALYZE_RE; from loma_rag.prompt.translate import TRANSLATE_SYSTEM, TRANSLATE_FOR_SEARCH_SYSTEM; from loma_rag.prompt.builder import build_loma_user_prompt, build_web_user_prompt; from loma_rag.prompt.judge import build_judge_user; print('OK')"
```
Expected: `OK`

---

### Task 9: Create `llm/openai_client.py`

**Files:**
- Create: `loma_rag/llm/openai_client.py` — extract `make_chat_client` from `chat.py:461-470`, `make_async_chat_client` from `chat.py:471-481`, `make_dense_client` from `index.py:70-76`

- [ ] **Step 1: Write the file.**

```python
"""Azure OpenAI client factories.

Single source of truth — replaces the duplicated factories in chat.py,
eval.py, and index.py.
"""
from __future__ import annotations

from openai import AsyncAzureOpenAI, AzureOpenAI

from loma_rag.config.settings import azure


def make_chat_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=azure.api_key,
        api_version=azure.api_version,
        azure_endpoint=azure.api_base,
    )


def make_async_chat_client() -> AsyncAzureOpenAI:
    return AsyncAzureOpenAI(
        api_key=azure.api_key,
        api_version=azure.api_version,
        azure_endpoint=azure.api_base,
    )


# Dense embeddings use the same Azure deployment under a different model name;
# reusing the same client class is fine.
def make_dense_client() -> AzureOpenAI:
    return make_chat_client()
```

If the original `make_chat_client` does anything else (custom headers, timeouts), copy those settings here verbatim.

- [ ] **Step 2: Verify.**

```powershell
python -c "from loma_rag.llm.openai_client import make_chat_client, make_async_chat_client, make_dense_client; print('OK')"
```
Expected: `OK`

---

### Task 10: Create `llm/embedding.py`

**Files:**
- Create: `loma_rag/llm/embedding.py`
- Reference: `index.py:78-92` (`embed_dense` batch), `retrieval.py:220-242` (`Retriever._embed_dense`, `_embed_sparse`, `_embed_colbert`)

- [ ] **Step 1: Write header.**

```python
"""Embedding helpers: dense (Azure), sparse (BM42), late-interaction (ColBERT).

Sparse and ColBERT models are loaded lazily on first call (~0.5s + 1.5s
cold start, then cached process-wide).
"""
from __future__ import annotations

from openai import AzureOpenAI

from loma_rag.config.settings import azure
from loma_rag.constant.models import BM42_MODEL, COLBERT_MODEL
from loma_rag.constant.thresholds import DENSE_EMBED_CACHE_SIZE
from loma_rag.util.cache import LRUCache
from loma_rag.util.retry import with_retries

_dense_cache = LRUCache(DENSE_EMBED_CACHE_SIZE)
_sparse_model = None
_colbert_model = None
```

- [ ] **Step 2: Define `embed_dense_batch`** — port `embed_dense` from `index.py:78-92`. Use `with_retries` instead of inline retry loop.

```python
def embed_dense_batch(client: AzureOpenAI, texts: list[str]) -> list[list[float]]:
    """Batched dense embed with retry. Used by the indexer."""
    def _call() -> list[list[float]]:
        resp = client.embeddings.create(model=azure.embedding_model, input=texts)
        return [d.embedding for d in resp.data]
    return with_retries(_call, retries=3)
```

If `index.py:78-92` does additional work (chunking the batch, logging), preserve that.

- [ ] **Step 3: Define `embed_dense_single`** — port `Retriever._embed_dense` from `retrieval.py:220-232`.

```python
def embed_dense_single(client: AzureOpenAI, text: str) -> list[float]:
    """Cached single-text dense embed (used by the retriever hot path)."""
    cached = _dense_cache.get(text)
    if cached is not None:
        return cached
    vec = embed_dense_batch(client, [text])[0]
    _dense_cache.put(text, vec)
    return vec
```

- [ ] **Step 4: Define `embed_sparse` and `embed_colbert`** — port `_embed_sparse`, `_embed_colbert` from `retrieval.py:234-242`.

```python
def embed_sparse(text: str):
    """Sparse BM42 embed; lazy-load model on first call."""
    global _sparse_model
    if _sparse_model is None:
        from fastembed import SparseTextEmbedding
        _sparse_model = SparseTextEmbedding(model_name=BM42_MODEL)
    return next(iter(_sparse_model.embed([text])))


def embed_colbert(text: str):
    """ColBERT late-interaction embed; lazy-load model on first call."""
    global _colbert_model
    if _colbert_model is None:
        from fastembed import LateInteractionTextEmbedding
        _colbert_model = LateInteractionTextEmbedding(model_name=COLBERT_MODEL)
    return next(iter(_colbert_model.embed([text])))
```

If `retrieval.py:234-242` returns/uses different shapes, match exactly.

- [ ] **Step 5: Verify.**

```powershell
python -c "from loma_rag.llm.embedding import embed_dense_batch, embed_dense_single, embed_sparse, embed_colbert; print('OK')"
```
Expected: `OK`

---

### Task 11: Create `db_vector/{client,collection,upsert}.py`

**Files:**
- Create: `loma_rag/db_vector/client.py`
- Create: `loma_rag/db_vector/collection.py` — extract `ensure_collection` from `index.py:94-156`
- Create: `loma_rag/db_vector/upsert.py` — extract `stable_id` (`index.py:158-160`), `_upsert_with_retry` (`index.py:162-176`), `NAMESPACE` (`index.py:65`), batch sizes

- [ ] **Step 1: `loma_rag/db_vector/client.py`**

```python
"""Qdrant client factory."""
from __future__ import annotations

from qdrant_client import QdrantClient

from loma_rag.config.settings import qdrant


def make_qdrant_client() -> QdrantClient:
    return QdrantClient(url=qdrant.url)
```

- [ ] **Step 2: `loma_rag/db_vector/collection.py`**

Header:
```python
"""Qdrant collection schema setup (3 named vectors: dense + sparse + ColBERT)."""
from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    Modifier,
    MultiVectorComparator,
    MultiVectorConfig,
    OptimizersConfigDiff,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    SparseIndexParams,
    SparseVectorParams,
    StrictModeConfig,
    VectorParams,
)

from loma_rag.constant.collections import COLBERT_DIM, DENSE_DIM
```

Paste `ensure_collection` body from `index.py:94-156` verbatim.

- [ ] **Step 3: `loma_rag/db_vector/upsert.py`**

```python
"""Stable point-ID generation and retried upsert."""
from __future__ import annotations

import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from loma_rag.constant.thresholds import EMBED_BATCH, UPSERT_BATCH
from loma_rag.util.retry import with_retries

NAMESPACE = uuid.UUID("9c3a3a4e-2e2c-4d8a-bcd7-1f59f3a1c800")


def stable_id(key: str) -> str:
    return str(uuid.uuid5(NAMESPACE, key))


def upsert_with_retry(qc: QdrantClient, coll: str, points: list[PointStruct], retries: int = 3) -> None:
    with_retries(lambda: qc.upsert(collection_name=coll, points=points, wait=True), retries=retries)
```

If the original `_upsert_with_retry` (`index.py:162-176`) does logging or partial-batch handling, preserve that exactly.

- [ ] **Step 4: Verify.**

```powershell
python -c "from loma_rag.db_vector.client import make_qdrant_client; from loma_rag.db_vector.collection import ensure_collection; from loma_rag.db_vector.upsert import stable_id, upsert_with_retry, NAMESPACE; print('OK')"
```
Expected: `OK`

---

### Task 12: Phase 2 import gate

- [ ] **Step 1: Single-pass import.**

```powershell
python -c "import loma_rag.prompt.system, loma_rag.prompt.analyzer, loma_rag.prompt.translate, loma_rag.prompt.builder, loma_rag.prompt.judge, loma_rag.llm.openai_client, loma_rag.llm.embedding, loma_rag.db_vector.client, loma_rag.db_vector.collection, loma_rag.db_vector.upsert; print('PHASE 2 OK')"
```
Expected: `PHASE 2 OK`

- [ ] **Step 2: Checkpoint.**

---

## Phase 3 — Domain logic

### Task 13: Create `chunking/` modules

**Files:**
- Create: `loma_rag/chunking/text.py` — extract `_norm_ws` (`ingest.py:42-49`), `_SKIP_PARAS` (`ingest.py:46`)
- Create: `loma_rag/chunking/docx_blocks.py` — extract `iter_block_items` (`ingest.py:63-76`), `table_to_md` (`ingest.py:78-94`)
- Create: `loma_rag/chunking/chunker.py` — extract `chunk_docx` (`ingest.py:111-177`), `_merge_tiny` (`ingest.py:179-227`), `link_chunks_to_nodes` (`ingest.py:339-366`)

- [ ] **Step 1: `loma_rag/chunking/text.py`**

```python
"""Plain-text normalisation helpers used during chunking."""
from __future__ import annotations
```

Paste `_norm_ws` and `_SKIP_PARAS` verbatim. Public name: `norm_ws` (drop underscore); keep `SKIP_PARAS`.

- [ ] **Step 2: `loma_rag/chunking/docx_blocks.py`**

```python
"""Iterate paragraphs+tables from a docx Document, render tables as Markdown."""
from __future__ import annotations

from typing import Iterator

from docx.document import Document as _Doc
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table, _Cell
from docx.text.paragraph import Paragraph
```

Paste `iter_block_items` and `table_to_md` verbatim.

- [ ] **Step 3: `loma_rag/chunking/chunker.py`**

```python
"""Heading-aware chunking + tiny-chunk merging + chunk↔node linking."""
from __future__ import annotations

from pathlib import Path

from docx import Document

from loma_rag.chunking.docx_blocks import iter_block_items, table_to_md
from loma_rag.chunking.text import norm_ws, SKIP_PARAS
from loma_rag.constant.thresholds import MAX_CHUNK_CHARS, MIN_CHUNK_CHARS
from loma_rag.model.domain import Chunk, IngestNode
```

Paste `chunk_docx`, `_merge_tiny`, `link_chunks_to_nodes` verbatim. Update internal references:
- `_norm_ws` → `norm_ws`
- `_SKIP_PARAS` → `SKIP_PARAS`
- `Node` → `IngestNode` (in `link_chunks_to_nodes` only)

- [ ] **Step 4: Verify.**

```powershell
python -c "from loma_rag.chunking.text import norm_ws, SKIP_PARAS; from loma_rag.chunking.docx_blocks import iter_block_items, table_to_md; from loma_rag.chunking.chunker import chunk_docx, link_chunks_to_nodes; print('OK')"
```
Expected: `OK`

---

### Task 14: Create `ingest/` parser modules

**Files:**
- Create: `loma_rag/ingest/filename.py` — extract `FILENAME_RE` (`ingest.py:39`), `parse_filename` (`ingest.py:51-61`)
- Create: `loma_rag/ingest/xlsx_parser.py` — extract `parse_nodes` (`ingest.py:251-303`), `parse_quiz` (`ingest.py:383-467`), `_parse_syllabus_meta` (`ingest.py:638-650`), `_parse_schedule` (`ingest.py:652-708`)
- Create: `loma_rag/ingest/tree_scanner.py` — extract `_COURSE_DIR_RE`, `_MODULE_DIR_RE`, `_LESSON_DIR_RE` (`ingest.py:469-471`), `_M_RE`, `_L_RE` (`ingest.py:634-635`), `_extract_learning_objectives` (`ingest.py:474-506`), `_scan_course_tree` (`ingest.py:508-543`), `build_toc_chunks` (`ingest.py:545-636`), `_find_syllabus` (`ingest.py:710-717`), `build_syllabus_chunks` (`ingest.py:719-885`)
- Create: `loma_rag/ingest/graph_builder.py` — extract `build_edges` (`ingest.py:305-337`)
- Create: `loma_rag/ingest/docx_parser.py` — thin orchestrator that re-exports `chunk_docx` from chunking

- [ ] **Step 1: `loma_rag/ingest/filename.py`**

```python
"""Parse LOMA-style course filenames (e.g. LOMA281_M2L3_Foo.docx)."""
from __future__ import annotations

import re
```

Paste `FILENAME_RE` and `parse_filename` verbatim.

- [ ] **Step 2: `loma_rag/ingest/xlsx_parser.py`**

```python
"""Parse Knowledge-Node and Quiz xlsx files, plus syllabus sheets."""
from __future__ import annotations

from pathlib import Path

from openpyxl import load_workbook

from loma_rag.chunking.text import norm_ws
from loma_rag.ingest.filename import parse_filename
from loma_rag.model.domain import IngestNode, Quiz
```

Paste `parse_nodes`, `parse_quiz`, `_parse_syllabus_meta`, `_parse_schedule` verbatim. Internal references: `_norm_ws` → `norm_ws`, `Node` → `IngestNode`.

- [ ] **Step 3: `loma_rag/ingest/tree_scanner.py`**

```python
"""Scan course/module/lesson directory trees to build TOC + syllabus chunks."""
from __future__ import annotations

import re
from pathlib import Path

from loma_rag.model.domain import Chunk
```

Paste the regex constants and the five functions verbatim. If they reference helpers from `xlsx_parser` (e.g. `_parse_syllabus_meta`), import them.

- [ ] **Step 4: `loma_rag/ingest/graph_builder.py`**

```python
"""Resolve node references into edges within a lesson scope."""
from __future__ import annotations

from loma_rag.model.domain import Edge, IngestNode
```

Paste `build_edges` verbatim. `Node` → `IngestNode` if referenced.

- [ ] **Step 5: `loma_rag/ingest/docx_parser.py`**

```python
"""Docx ingestion entrypoint — re-exports chunking primitives."""
from loma_rag.chunking.chunker import chunk_docx

__all__ = ["chunk_docx"]
```

- [ ] **Step 6: Verify.**

```powershell
python -c "from loma_rag.ingest.filename import parse_filename, FILENAME_RE; from loma_rag.ingest.xlsx_parser import parse_nodes, parse_quiz; from loma_rag.ingest.tree_scanner import build_toc_chunks, build_syllabus_chunks; from loma_rag.ingest.graph_builder import build_edges; from loma_rag.ingest.docx_parser import chunk_docx; print('OK')"
```
Expected: `OK`

---

### Task 15: Create `db_vector/search.py`

**Files:**
- Create: `loma_rag/db_vector/search.py`
- Reference: `retrieval.py:243-414` (the body of `Retriever.retrieve`) for the Prefetch+RRF call shape, `retrieval.py:415-466` for `_search_nodes_direct`

- [ ] **Step 1: Read `retrieval.py:243-466` to identify the exact Qdrant query call.**

The hybrid search portion uses `qc.query_points(...)` with `Prefetch` over the three named vectors and RRF. Extract just the I/O — given prepared embeddings, return Qdrant scored points.

- [ ] **Step 2: Write the file.**

```python
"""Hybrid search abstraction over Qdrant.

Inputs are already-computed embeddings. Outputs Qdrant scored-point lists.
"""
from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import Fusion, FusionQuery, NamedVector, Prefetch, SparseVector

from loma_rag.constant.collections import CHUNKS_COLL, NODES_COLL


def hybrid_search(
    qc: QdrantClient,
    dense_emb: list[float],
    sparse_emb,
    colbert_vecs,
    prefetch_limit: int,
    final_limit: int,
):
    """Run dense+sparse+ColBERT prefetch with RRF fusion against the chunks collection."""
    # Body: extract from retrieval.py:243-414 — the prefetch list + qc.query_points call.
    raise NotImplementedError("Paste the qc.query_points(...) call from retrieval.py:243-414 here, verbatim.")


def search_nodes_direct(qc: QdrantClient, dense_emb: list[float], sparse_emb, colbert_vecs, top_k: int):
    """Run direct semantic search against the nodes collection."""
    # Body: extract from retrieval.py:415-466 (`Retriever._search_nodes_direct`).
    raise NotImplementedError("Paste the body from retrieval.py:415-466 here.")
```

The two `raise NotImplementedError` lines are placeholders for the engineer to replace with the actual calls extracted from `retrieval.py`. Read those exact line ranges and paste the function bodies, adjusting `self._qc` → `qc`, etc.

- [ ] **Step 3: Verify.**

```powershell
python -c "from loma_rag.db_vector.search import hybrid_search, search_nodes_direct; print('OK')"
```
Expected: `OK`

---

### Task 16: Create `llm/{completion,streaming}.py`

**Files:**
- Create: `loma_rag/llm/completion.py` — extract `select_answer_model` (`chat.py:86-107`), `analyze_query` (`chat.py:255-285`), `analyze_query_async` (`chat.py:287-315`), `detect_language_llm` (`chat.py:352-378`), `translate_to_english_query` (`chat.py:483-499`)
- Create: `loma_rag/llm/streaming.py` — extract `stream_with_sentinel_detect` (`chat.py:509-568`), `complete_with_sentinel_detect` (`chat.py:570-587`)

- [ ] **Step 1: `loma_rag/llm/completion.py`**

```python
"""Synchronous + async helpers wrapping Azure OpenAI chat completions."""
from __future__ import annotations

from loma_rag.config.settings import azure
from loma_rag.constant.thresholds import ANALYZE_CACHE_MAX, LANG_CACHE_MAX, TRANSLATE_CACHE_MAX
from loma_rag.prompt.analyzer import ANALYZE_QUERY_SYSTEM, ANALYZE_RE, LANG_DETECT_SYSTEM
from loma_rag.prompt.translate import TRANSLATE_SYSTEM
from loma_rag.util.cache import LRUCache
from loma_rag.util.language import LANG_CODE_RE

_translate_cache = LRUCache(TRANSLATE_CACHE_MAX)
_analyze_cache = LRUCache(ANALYZE_CACHE_MAX)
_lang_cache = LRUCache(LANG_CACHE_MAX)
```

Paste `select_answer_model` from `chat.py:86-107` verbatim. The `_BIG_MODEL_KEYWORDS`, `_PREFER_BIG_MODEL` module-level constants from `chat.py:73-83` go here too (above the function).

Paste `analyze_query`, `analyze_query_async`, `detect_language_llm`, `translate_to_english_query` verbatim. Internal references:
- `_translate_cache_get/put`, `_analyze_cache_get/put`, `_lang_cache_get/put` → use `_translate_cache.get/put`, etc.
- `_ANALYZE_QUERY_SYSTEM` → `ANALYZE_QUERY_SYSTEM`
- `_LANG_DETECT_SYSTEM` → `LANG_DETECT_SYSTEM`
- `TRANSLATE_SYSTEM` → keep
- `_ANALYZE_RE` → `ANALYZE_RE`
- `_LANG_CODE_RE` → `LANG_CODE_RE`
- `CHAT_MODEL`, `DETECT_MODEL` → `azure.chat_model`, `azure.detect_model`

- [ ] **Step 2: `loma_rag/llm/streaming.py`**

```python
"""Sentinel-detect streaming + non-streaming helpers used by the answer pipeline."""
from __future__ import annotations

from loma_rag.config.settings import azure
from loma_rag.constant.tokens import INSUFFICIENT_TOKEN
```

Paste `stream_with_sentinel_detect` and `complete_with_sentinel_detect` verbatim. Internal references:
- `INSUFFICIENT_TOKEN` → keep
- `CHAT_MODEL` → `azure.chat_model`
- `SNIFF_CHARS` → `from loma_rag.config.settings import rag` then `rag.sniff_chars`

- [ ] **Step 3: Verify.**

```powershell
python -c "from loma_rag.llm.completion import select_answer_model, analyze_query, analyze_query_async, detect_language_llm, translate_to_english_query; from loma_rag.llm.streaming import stream_with_sentinel_detect, complete_with_sentinel_detect; print('OK')"
```
Expected: `OK`

---

### Task 17: Create `util/{citation,printer}.py`

**Files:**
- Create: `loma_rag/util/citation.py` — extract `_build_citations` (`api.py:119-120`), `_format_loma_citations` (`chat.py:728-743`), `_format_web_citations` (`chat.py:745-755`)
- Create: `loma_rag/util/printer.py` — extract `print_loma_context` (`chat.py:589-602`), `print_loma_sources` (`chat.py:604-609`), `print_web_sources` (`chat.py:611-618`)

- [ ] **Step 1: `loma_rag/util/citation.py`**

```python
"""Citation builders for LOMA chunks and web docs."""
from __future__ import annotations

from loma_rag.model.api_models import Citation


def build_citations(raw: list[dict]) -> list[Citation]:
    return [Citation(**c) for c in raw]
```

Paste `format_loma_citations` and `format_web_citations` (drop leading underscore on rename) verbatim from `chat.py:728-755`.

- [ ] **Step 2: `loma_rag/util/printer.py`**

```python
"""CLI display helpers — used by REPL and ad-hoc scripts."""
from __future__ import annotations

from loma_rag.model.domain import RetrievalResult, WebDoc
```

Paste the three print functions verbatim.

- [ ] **Step 3: Verify.**

```powershell
python -c "from loma_rag.util.citation import build_citations, format_loma_citations, format_web_citations; from loma_rag.util.printer import print_loma_context, print_loma_sources, print_web_sources; print('OK')"
```
Expected: `OK`

---

### Task 18: Phase 3 import gate

- [ ] **Step 1: Single-pass import.**

```powershell
python -c "import loma_rag.chunking.chunker, loma_rag.ingest.docx_parser, loma_rag.ingest.xlsx_parser, loma_rag.ingest.tree_scanner, loma_rag.ingest.graph_builder, loma_rag.db_vector.search, loma_rag.llm.completion, loma_rag.llm.streaming, loma_rag.util.citation, loma_rag.util.printer; print('PHASE 3 OK')"
```
Expected: `PHASE 3 OK`

- [ ] **Step 2: Checkpoint.**

---

## Phase 4 — Orchestration

### Task 19: Create `rag/graph.py`

**Files:**
- Create: `loma_rag/rag/graph.py`
- Reference: `graph.py` (entire file)

- [ ] **Step 1: Copy `graph.py` to `loma_rag/rag/graph.py`.**

Replace the top-of-file env loading (`load_dotenv(...)`, `HERE = ...`, `OUT = ...`) with:
```python
from loma_rag.config.settings import OUT_DIR as OUT
```

The `Node` class stays as `Node` here (it's the KG node, distinct from `IngestNode`).

- [ ] **Step 2: Verify.**

```powershell
python -c "from loma_rag.rag.graph import KG, Node, load_graph; print('OK')"
```
Expected: `OK`

---

### Task 20: Create `rag/web_fallback.py`

**Files:**
- Create: `loma_rag/rag/web_fallback.py`
- Reference: `web_fallback.py` (entire file)

- [ ] **Step 1: Copy `web_fallback.py` to `loma_rag/rag/web_fallback.py`.**

Replace top-of-file env loading with:
```python
from loma_rag.config.settings import azure, web
```

Replace internal:
- `DEFAULT_SEARXNG` → `web.searxng_url`
- `DENSE_MODEL` → `azure.embedding_model`

The `WebDoc` dataclass already moved to `model/domain.py` in Task 5 — replace local definition with `from loma_rag.model.domain import WebDoc`.

- [ ] **Step 2: Verify.**

```powershell
python -c "from loma_rag.rag.web_fallback import WebFallback, search_searxng, fetch_page_text, enrich_pages, faiss_rank; print('OK')"
```
Expected: `OK`

---

### Task 21: Create `rag/{reranker,fusion,analyzer}.py`

**Files:**
- Create: `loma_rag/rag/reranker.py`
- Create: `loma_rag/rag/fusion.py`
- Create: `loma_rag/rag/analyzer.py`

These three are smaller than `retriever.py` — they hold logic currently inlined inside `Retriever.retrieve` (`retrieval.py:127-414`). The plan: read that range and split out.

- [ ] **Step 1: `loma_rag/rag/fusion.py`**

```python
"""RRF helpers (currently inline in Retriever.retrieve)."""
from __future__ import annotations
```

If RRF math is done by Qdrant's `Fusion.RRF` query, this module may end up tiny — just a constant or wrapper. If there is also Python-side reciprocal-rank fusion (e.g. for combining graph scores with chunk scores), extract that here as `rrf_combine(rank_lists, k=60) -> dict`.

Read `retrieval.py:243-414`. Identify any non-Qdrant RRF math. Extract into `rrf_combine(...)`.

- [ ] **Step 2: `loma_rag/rag/reranker.py`**

```python
"""Cross-encoder reranker (Xenova/ms-marco-MiniLM-L-6-v2)."""
from __future__ import annotations

from loma_rag.config.settings import rag


class Reranker:
    """Lazy-loaded cross-encoder.

    Mirror of the `_reranker` field formerly on Retriever; pulled out
    so retriever.py is small and reranking can be swapped.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or rag.rerank_model
        self._model = None

    def _ensure(self) -> None:
        if self._model is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            # Replicate the loader used in retrieval.py:127-194.
            # Paste verbatim from retrieval.py.
            raise NotImplementedError("Paste reranker loader from retrieval.py:160-194.")

    def rerank(self, query: str, passages: list[str]) -> list[float]:
        """Return reranker scores for each passage; higher = better."""
        self._ensure()
        # Paste rerank scoring loop from retrieval.py (search for `.rerank` or scoring call).
        raise NotImplementedError("Paste rerank scoring from retrieval.py.")
```

Read `retrieval.py:160-194` and the rerank usage in `Retriever.retrieve` to fill the two NotImplementedErrors.

- [ ] **Step 3: `loma_rag/rag/analyzer.py`**

```python
"""Query analysis: language detect + intent classify, with caching."""
from __future__ import annotations

from loma_rag.llm.completion import analyze_query as _llm_analyze
from loma_rag.llm.completion import analyze_query_async as _llm_analyze_async


def analyze_query(client, text: str) -> tuple[str, str]:
    """Return (lang_code, normalised_query)."""
    return _llm_analyze(client, text)


async def analyze_query_async(async_client, text: str) -> tuple[str, str]:
    return await _llm_analyze_async(async_client, text)
```

Thin wrapper — present so `rag/` exports a single namespace for query analysis without callers reaching into `llm/`.

- [ ] **Step 4: Verify.**

```powershell
python -c "from loma_rag.rag.fusion import rrf_combine; from loma_rag.rag.reranker import Reranker; from loma_rag.rag.analyzer import analyze_query, analyze_query_async; print('OK')"
```
Expected: `OK`

(If `rrf_combine` is not actually defined because all RRF is server-side, skip the import in the verify step.)

---

### Task 22: Create `rag/retriever.py`

**Files:**
- Create: `loma_rag/rag/retriever.py`
- Reference: `retrieval.py:127-466` (the `Retriever` class)

This is the largest single carry-over. The class body retains its public interface (`__init__`, `retrieve`, `translate_for_search`) but delegates to the new modules.

- [ ] **Step 1: Read `retrieval.py:127-466`.**

Identify the public surface: `Retriever(rerank: bool, expand_graph: bool, ...)`, `retriever.retrieve(query, top_k=...)`. The `_ensure_clients`, `_embed_*`, `_search_nodes_direct` helpers all become calls into the new modules.

- [ ] **Step 2: Write the file.**

```python
"""Retriever orchestrator — composes embedding, hybrid search, rerank, KG expand."""
from __future__ import annotations

from typing import Optional

from loma_rag.config.settings import azure, rag as rag_cfg
from loma_rag.constant.collections import CHUNKS_COLL, NODES_COLL
from loma_rag.db_vector.client import make_qdrant_client
from loma_rag.db_vector.search import hybrid_search, search_nodes_direct
from loma_rag.exceptions import RetrievalError
from loma_rag.llm.embedding import embed_colbert, embed_dense_single, embed_sparse
from loma_rag.llm.openai_client import make_dense_client
from loma_rag.model.domain import RetrievalResult, RetrievedChunk
from loma_rag.prompt.translate import TRANSLATE_FOR_SEARCH_SYSTEM
from loma_rag.rag.reranker import Reranker
from loma_rag.util.cache import LRUCache


class Retriever:
    def __init__(self, rerank: bool = True, expand_graph: bool = True) -> None:
        # Mirror the original constructor from retrieval.py:130-159.
        # Replace direct env reads / client construction with the new module calls.
        self.rerank = rerank
        self.expand_graph = expand_graph
        self._qc = None
        self._dense_client = None
        self._reranker: Optional[Reranker] = None
        # Lazy KG (loaded only when expand_graph is true).
        self._kg = None
        self._sparse_model = None
        self._translate_cache = LRUCache(rag_cfg.prefetch_limit)  # carry-over

    def _ensure_clients(self) -> None:
        if self._qc is None:
            self._qc = make_qdrant_client()
        if self._dense_client is None:
            self._dense_client = make_dense_client()
        if self.rerank and self._reranker is None:
            self._reranker = Reranker()
        # Replicate any other lazy loads from retrieval.py:160-194 here.

    def translate_for_search(self, query: str) -> str:
        # Paste body from retrieval.py:195-218 verbatim.
        # Replace internal `_TRANSLATE_FOR_SEARCH_SYSTEM` → `TRANSLATE_FOR_SEARCH_SYSTEM`.
        # Replace internal `CHAT_MODEL` → `azure.chat_model`.
        raise NotImplementedError("Paste body from retrieval.py:195-218.")

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        """Orchestrate: embed → hybrid_search → rerank → KG expand."""
        self._ensure_clients()
        # Paste body from retrieval.py:243-414, replacing:
        #   self._embed_dense(...)  → embed_dense_single(self._dense_client, ...)
        #   self._embed_sparse(...) → embed_sparse(...)
        #   self._embed_colbert(...) → embed_colbert(...)
        #   self._qc.query_points(...) → hybrid_search(self._qc, ..., prefetch_limit=rag_cfg.prefetch_limit, final_limit=top_k)
        #   self._search_nodes_direct(...) → search_nodes_direct(self._qc, ...)
        #   reranker call           → self._reranker.rerank(query, passages)
        #   KG expand               → uses self._kg (load lazily via load_graph() from rag.graph)
        raise NotImplementedError("Paste retrieve body from retrieval.py:243-414.")
```

Read `retrieval.py:127-466` end-to-end and fill the three `NotImplementedError` stubs by pasting the original bodies, with the substitutions noted above.

- [ ] **Step 3: Verify.**

```powershell
python -c "from loma_rag.rag.retriever import Retriever; r = Retriever(rerank=False, expand_graph=False); print('OK')"
```
Expected: `OK` (instantiation only — does not call any external service).

---

### Task 23: Create `rag/pipeline.py`

**Files:**
- Create: `loma_rag/rag/pipeline.py`
- Reference: `chat.py:501-507` (`_prep_web_docs`), `chat.py:620-726` (`run_query`), `chat.py:757-968` (`stream_query`), `chat.py:970-1050` (`answer_query`)

- [ ] **Step 1: Write header.**

```python
"""Top-level RAG pipelines: run_query, stream_query, answer_query."""
from __future__ import annotations

from loma_rag.config.settings import rag as rag_cfg
from loma_rag.constant.tokens import INSUFFICIENT_TOKEN
from loma_rag.llm.completion import select_answer_model
from loma_rag.llm.streaming import complete_with_sentinel_detect, stream_with_sentinel_detect
from loma_rag.model.domain import AnswerResult, RetrievalResult
from loma_rag.prompt.builder import build_loma_user_prompt, build_web_user_prompt
from loma_rag.prompt.system import LOMA_SYSTEM, WEB_SYSTEM
from loma_rag.rag.retriever import Retriever
from loma_rag.rag.web_fallback import WebFallback
from loma_rag.util.citation import format_loma_citations, format_web_citations
from loma_rag.util.concurrency import BG_POOL
from loma_rag.util.language import detect_user_language
```

- [ ] **Step 2: Paste `_prep_web_docs` verbatim.**

(Public name `prep_web_docs` — drop underscore.)

- [ ] **Step 3: Paste `run_query` verbatim** from `chat.py:620-726`.

Substitutions:
- `INSUFFICIENT_TOKEN` → keep
- `LOMA_SYSTEM`, `WEB_SYSTEM` → keep
- `_format_loma_citations` → `format_loma_citations`
- `_format_web_citations` → `format_web_citations`
- `select_answer_model` → keep
- `_prep_web_docs` → `prep_web_docs`
- `_BG_POOL` → `BG_POOL`
- `_detect_user_language` → `detect_user_language`
- `complete_with_sentinel_detect` → keep (now imported)

- [ ] **Step 4: Paste `stream_query` verbatim** from `chat.py:757-968` with the same substitutions.

- [ ] **Step 5: Paste `answer_query` verbatim** from `chat.py:970-1050`.

- [ ] **Step 6: Verify.**

```powershell
python -c "from loma_rag.rag.pipeline import run_query, stream_query, answer_query, prep_web_docs; print('OK')"
```
Expected: `OK`

---

### Task 24: Create `ingest/{pipeline,indexer}.py`

**Files:**
- Create: `loma_rag/ingest/pipeline.py` — extract `find_files` (`ingest.py:887-901`) and `main` (`ingest.py:909-end`)
- Create: `loma_rag/ingest/indexer.py` — extract `index_docs` (`index.py:179-241`) and `main` (`index.py:248-end`)

- [ ] **Step 1: `loma_rag/ingest/pipeline.py`**

```python
"""Ingestion orchestrator: scan inputs → produce chunks/nodes/edges/quizzes JSONL."""
from __future__ import annotations

import sys
from pathlib import Path

from loma_rag.config.settings import OUT_DIR, PROJECT_ROOT
from loma_rag.ingest.docx_parser import chunk_docx
from loma_rag.ingest.graph_builder import build_edges
from loma_rag.ingest.tree_scanner import build_syllabus_chunks, build_toc_chunks
from loma_rag.ingest.xlsx_parser import parse_nodes, parse_quiz
from loma_rag.chunking.chunker import link_chunks_to_nodes
from loma_rag.util.io import write_jsonl
```

Paste `find_files` and `main` verbatim. Update:
- `ROOT` → `PROJECT_ROOT.parent`  (note: `ingest.py:32` uses `... .parent.parent`; `PROJECT_ROOT` is already `parent.parent` of `loma_rag/config/settings.py` → equals `D:/t/LOMA/rag`. The original `ROOT = ... .parent.parent` from `ingest.py` was `D:/t/LOMA`. So replace `ROOT` with `PROJECT_ROOT.parent`.)
- `OUT` → `OUT_DIR`
- `write_jsonl` → keep (now imported)

- [ ] **Step 2: `loma_rag/ingest/indexer.py`**

```python
"""Read JSONL outputs from ingest/pipeline and upsert into Qdrant."""
from __future__ import annotations

import sys

from loma_rag.config.settings import OUT_DIR
from loma_rag.constant.collections import CHUNKS_COLL, DENSE_DIM, NODES_COLL
from loma_rag.constant.thresholds import EMBED_BATCH, UPSERT_BATCH
from loma_rag.db_vector.client import make_qdrant_client
from loma_rag.db_vector.collection import ensure_collection
from loma_rag.db_vector.upsert import stable_id, upsert_with_retry
from loma_rag.llm.embedding import embed_dense_batch, embed_sparse, embed_colbert
from loma_rag.llm.openai_client import make_dense_client
from loma_rag.util.io import load_jsonl
```

Paste `index_docs` and `main` verbatim. Update:
- `make_dense_client` / `make_qdrant_client` → use new factories
- `_upsert_with_retry` → `upsert_with_retry`
- `OUT` → `OUT_DIR`
- model loading for sparse/colbert: replace inline `SparseTextEmbedding(...)` and `LateInteractionTextEmbedding(...)` with calls to `embed_sparse`/`embed_colbert` if signatures match. Otherwise keep inline construction.

- [ ] **Step 3: Verify.**

```powershell
python -c "from loma_rag.ingest.pipeline import find_files, main; from loma_rag.ingest.indexer import index_docs; print('OK')"
```
Expected: `OK`

---

### Task 25: Phase 4 import gate

- [ ] **Step 1: Single-pass import.**

```powershell
python -c "import loma_rag.rag.graph, loma_rag.rag.web_fallback, loma_rag.rag.fusion, loma_rag.rag.reranker, loma_rag.rag.analyzer, loma_rag.rag.retriever, loma_rag.rag.pipeline, loma_rag.ingest.pipeline, loma_rag.ingest.indexer; print('PHASE 4 OK')"
```
Expected: `PHASE 4 OK`

- [ ] **Step 2: Checkpoint.**

---

## Phase 5 — Interface

### Task 26: Create `api/app.py` and `api/routes/{health,chat}.py`

**Files:**
- Create: `loma_rag/api/app.py`
- Create: `loma_rag/api/routes/health.py`
- Create: `loma_rag/api/routes/chat.py`
- Reference: `api.py:82-229`

- [ ] **Step 1: `loma_rag/api/routes/health.py`**

```python
"""GET /health — liveness probe."""
from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}
```

- [ ] **Step 2: `loma_rag/api/routes/chat.py`**

```python
"""POST /chat (non-streaming) and /chat/stream (SSE)."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from loma_rag.model.api_models import ChatRequest, ChatResponse, Citation, GraphNode
from loma_rag.rag.pipeline import run_query, stream_query
from loma_rag.util.citation import build_citations
from loma_rag.util.sse import sse_event

router = APIRouter()
```

Paste the body of the `chat()` handler from `api.py:130-189` verbatim. Decorate with `@router.post("/chat", response_model=ChatResponse)` instead of `@app.post(...)`.
Paste `chat_stream()` from `api.py:197-221` verbatim. Decorate with `@router.post("/chat/stream")`.
Replace `_sse_event` → `sse_event`, `_build_citations` → `build_citations`.

- [ ] **Step 3: `loma_rag/api/app.py`**

```python
"""FastAPI app factory + lifespan (eager retriever/client warm-up)."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from loma_rag.api.routes.chat import router as chat_router
from loma_rag.api.routes.health import router as health_router
from loma_rag.llm.openai_client import make_chat_client
from loma_rag.rag.retriever import Retriever
from loma_rag.rag.web_fallback import WebFallback


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[startup] initialising retriever and clients…")
    retriever = Retriever(rerank=True, expand_graph=True)
    chat_client = make_chat_client()
    web_fb = WebFallback(dense_client=chat_client)

    print("[startup] warming up sparse model + reranker + KG…")
    retriever._ensure_clients()
    list(retriever._sparse_model.embed(["warmup"]))
    if retriever._reranker is not None:
        list(retriever._reranker.rerank("warmup", ["warmup passage"]))

    app.state.retriever = retriever
    app.state.chat_client = chat_client
    app.state.web_fallback = web_fb
    print("[startup] ready.")
    yield
    print("[shutdown] bye.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="LOMA RAG Chatbot",
        description="Hybrid RAG (Qdrant) + GraphRAG (KG) + HyDE + SearXNG/FAISS fallback",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.include_router(health_router)
    app.include_router(chat_router)
    return app


app = create_app()  # uvicorn-style: `uvicorn loma_rag.api.app:app`
```

- [ ] **Step 4: Verify.**

```powershell
python -c "from loma_rag.api.app import create_app, app; from loma_rag.api.routes.health import router as h; from loma_rag.api.routes.chat import router as c; print('OK', len(app.routes))"
```
Expected: `OK <N>` where N >= 3 (health + 2 chat).

---

### Task 27: Create `main.py` dispatcher

**Files:**
- Create: `D:/t/LOMA/rag/main.py`

- [ ] **Step 1: Write the file.**

```python
"""LOMA RAG CLI dispatcher.

Subcommands:
  main.py serve     — run the FastAPI server (uvicorn)
  main.py repl      — interactive chat REPL (was chat.py)
  main.py ingest    — run the ingestion pipeline (docx/xlsx → JSONL)
  main.py index     — build/rebuild Qdrant collections from JSONL
"""
from __future__ import annotations

import sys


def cmd_serve(argv: list[str]) -> int:
    import uvicorn
    from loma_rag.config.settings import api as api_cfg
    uvicorn.run("loma_rag.api.app:app", host=api_cfg.host, port=api_cfg.port)
    return 0


def cmd_repl(argv: list[str]) -> int:
    # Replicate chat.py:1052-end main(). Import locally to keep startup fast for other subcommands.
    from loma_rag.rag.pipeline import answer_query  # noqa: F401  (used by REPL)
    # Paste argparse + REPL loop body from chat.py main(), substituting:
    #   run_query / stream_query → loma_rag.rag.pipeline.{run_query, stream_query}
    #   make_chat_client          → loma_rag.llm.openai_client.make_chat_client
    #   Retriever                 → loma_rag.rag.retriever.Retriever
    #   WebFallback               → loma_rag.rag.web_fallback.WebFallback
    raise NotImplementedError("Paste REPL body from chat.py main().")


def cmd_ingest(argv: list[str]) -> int:
    from loma_rag.ingest.pipeline import main as ingest_main
    return ingest_main()


def cmd_index(argv: list[str]) -> int:
    from loma_rag.ingest.indexer import main as index_main
    return index_main()


COMMANDS = {
    "serve": cmd_serve,
    "repl": cmd_repl,
    "ingest": cmd_ingest,
    "index": cmd_index,
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print("usage: main.py {serve|repl|ingest|index} [args...]")
        return 2
    return COMMANDS[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Fill in `cmd_repl`** by pasting `chat.py:1052-end main()` body with substitutions.

- [ ] **Step 3: Verify.**

```powershell
python main.py
```
Expected: `usage: main.py {serve|repl|ingest|index} [args...]` (exit code 2).

```powershell
python main.py ingest --help
```
Expected: ingest pipeline's argparse usage (or no error, depending on whether `ingest.main()` accepts `--help`).

---

### Task 28: Move scripts and tests

**Files:**
- Create: `scripts/eval.py` — copied from `eval.py` with imports updated
- Create: `scripts/bench.py` — copied from `bench.py` with imports updated
- Create: `tests/qa_cases.py` — copied from `qa_cases.py` (no changes; data-only)
- Create: `tests/test_qa.py` — copied from `test_qa.py` with imports updated

- [ ] **Step 1: `scripts/eval.py`** — copy `eval.py`, update header imports:

```python
from loma_rag.llm.openai_client import make_chat_client
from loma_rag.prompt.judge import build_judge_user
from loma_rag.rag.retriever import Retriever
from loma_rag.rag.pipeline import answer_query
```

Replace any local `make_chat_client`, judge prompt strings, etc. with imports.

- [ ] **Step 2: `scripts/bench.py`** — copy `bench.py`, update:

```python
from tests.qa_cases import TESTS
```

(No other changes — bench uses httpx against a running server.)

- [ ] **Step 3: `tests/qa_cases.py`** — copy `qa_cases.py` byte-for-byte.

- [ ] **Step 4: `tests/test_qa.py`** — copy `test_qa.py`, update:

```python
from loma_rag.llm.openai_client import make_chat_client
from loma_rag.rag.pipeline import run_query
from loma_rag.rag.retriever import Retriever
from loma_rag.rag.web_fallback import WebFallback
from loma_rag.util.language import detect_user_language as detect_language
from tests.qa_cases import TESTS
```

- [ ] **Step 5: Verify imports compile.**

```powershell
python -c "import scripts.eval, scripts.bench, tests.qa_cases, tests.test_qa" 
```

Note: `scripts/` has no `__init__.py`. Run instead:
```powershell
python -c "import importlib.util; [importlib.util.spec_from_file_location(n, p) for n, p in [('eval','scripts/eval.py'),('bench','scripts/bench.py')]]; print('OK')"
```
Or simpler — just compile-check:
```powershell
python -m py_compile scripts/eval.py scripts/bench.py tests/qa_cases.py tests/test_qa.py
```
Expected: no output, exit 0.

---

### Task 29: Phase 5 import gate

- [ ] **Step 1: Single-pass import.**

```powershell
python -c "import loma_rag.api.app, loma_rag.api.routes.health, loma_rag.api.routes.chat; from loma_rag.api.app import create_app; create_app(); print('PHASE 5 OK')"
```
Expected: `PHASE 5 OK` (lifespan does NOT run on `create_app()` — only on uvicorn startup, so this is safe).

- [ ] **Step 2: Checkpoint.**

---

## Phase 6 — Validation & Cleanup

### Task 30: Move legacy files to `_legacy/`

**Files:**
- Create: `_legacy/`
- Move (don't copy): `api.py`, `bench.py`, `chat.py`, `eval.py`, `graph.py`, `index.py`, `ingest.py`, `qa_cases.py`, `retrieval.py`, `test_qa.py`, `web_fallback.py` → `_legacy/`

- [ ] **Step 1: Create `_legacy/` and move files.**

```powershell
New-Item -ItemType Directory -Path "_legacy" -Force
Move-Item api.py, bench.py, chat.py, eval.py, graph.py, index.py, ingest.py, qa_cases.py, retrieval.py, test_qa.py, web_fallback.py _legacy/
```

- [ ] **Step 2: Verify root is now clean.**

```powershell
Get-ChildItem -Filter *.py
```
Expected: only `main.py` at root.

---

### Task 31: Static validation

- [ ] **Step 1: Full package import.**

```powershell
python -c "import loma_rag.api.app, loma_rag.rag.pipeline, loma_rag.rag.retriever, loma_rag.ingest.pipeline, loma_rag.ingest.indexer, loma_rag.api.routes.chat, loma_rag.api.routes.health; print('STATIC OK')"
```
Expected: `STATIC OK`

- [ ] **Step 2: App instantiation.**

```powershell
python -c "from loma_rag.api.app import create_app; app = create_app(); print('routes:', [r.path for r in app.routes])"
```
Expected: list including `/health`, `/chat`, `/chat/stream`.

- [ ] **Step 3: py_compile sweep.**

```powershell
python -m compileall loma_rag scripts tests main.py
```
Expected: exits 0, all files compile.

---

### Task 32: Smoke validation (requires `.env`)

- [ ] **Step 1: REPL CLI parses without error.**

```powershell
python main.py repl --help
```
Expected: argparse usage prints, exit 0 (or 2 if `--help` returns exit 2 — depending on argparse).

- [ ] **Step 2: Retriever instantiates.**

```powershell
python -c "from loma_rag.rag.retriever import Retriever; r = Retriever(rerank=False, expand_graph=False); print('OK')"
```
Expected: `OK`

- [ ] **Step 3: API factory works.**

```powershell
python -c "from loma_rag.api.app import create_app; create_app(); print('OK')"
```

---

### Task 33: End-to-end validation (requires Qdrant + Azure + indexed data)

- [ ] **Step 1: Pre-flight checklist.**

Confirm with user (or check directly):
- Qdrant is running: `curl http://localhost:6333/healthz` returns 200.
- `.env` has `OPENAI_API_KEY` and `OPENAI_API_BASE`.
- `out/` contains `chunks.jsonl`, `nodes.jsonl`, `edges.jsonl`.
- The chunks/nodes have already been indexed into Qdrant (i.e. `loma_chunks` and `loma_nodes` collections exist).

If anything is missing, run the relevant prerequisite (`python main.py ingest` and/or `python main.py index`) before continuing.

- [ ] **Step 2: Run the full QA test.**

```powershell
python -m tests.test_qa
```

Expected: pass-rate ≥ 94% (the pre-refactor baseline from `SESSION_NOTES.md`).

- [ ] **Step 3: Compare with baseline.**

If pass-rate is lower than baseline:
1. Capture the failures (which case IDs failed, which questions).
2. Cross-check against the pre-refactor `out/qa_run.log` if available.
3. Identify the diverging module — likely a missed substitution. Fix and re-run.

If pass-rate matches or exceeds baseline: proceed to Task 34.

- [ ] **Step 4: API smoke (optional but recommended).**

```powershell
Start-Process -NoNewWindow python -ArgumentList "main.py serve"
# in another shell:
curl http://127.0.0.1:8000/health
curl -Method POST http://127.0.0.1:8000/chat -ContentType "application/json" -Body '{"query":"What is antiselection?"}'
```
Expected: `/health` returns `{"status":"ok"}`; `/chat` returns a JSON `ChatResponse` body.

---

### Task 34: Final cleanup

- [ ] **Step 1: Delete `_legacy/` only after Task 33 passes.**

```powershell
Remove-Item -Recurse -Force _legacy
```

- [ ] **Step 2: Remove `__pycache__` directories at root.**

```powershell
Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force
```

- [ ] **Step 3: Final tree check.**

```powershell
Get-ChildItem -Filter *.py
```
Expected: only `main.py`.

```powershell
Get-ChildItem loma_rag -Directory
```
Expected: 11 directories (`api`, `chunking`, `config`, `constant`, `db_vector`, `ingest`, `llm`, `model`, `prompt`, `rag`, `util`).

- [ ] **Step 4: Summary message.**

State to user: which tasks completed, final QA pass-rate, any deviations from the plan, and whether Qdrant connection / API smoke worked.

---

## Plan self-review

**Spec coverage:** Every section/folder in the spec has a corresponding Task: §3 folder tree → Tasks 1, 5, 6, 8, 11, 13, 14, 15, 17, 19–24, 26; §4 symbol mapping → embedded in each Task's "Reference" line and "Substitutions" lists; §5 public interface → Tasks 22, 23, 26, 27 expose those symbols; §6 migration order → Phases 1–6; §7 validation → Tasks 31–33; §8 risks → addressed in Task 22 (NotImplementedError stubs force engineer to read source carefully), Task 30 (move not delete), Task 33 (rollback gate); §9 decisions → enforced (Approach 3, Option A, IngestNode rename, judge prompt in `prompt/`).

**Placeholder scan:** Three `NotImplementedError` stubs in Tasks 15, 21, 22 — these are intentional placeholders the engineer must replace by pasting source-line ranges. They are NOT acceptable to leave in the final code; each is paired with an exact line range to extract from the legacy file. The engineer must read the legacy file, paste the body, and replace the `raise NotImplementedError(...)`.

**Type consistency:** `Retriever(rerank: bool, expand_graph: bool)` constructor signature matches Task 22 → consumed in Task 26 (`api/app.py`) and Task 27 (`main.py repl`). `embed_dense_single(client, text)` and `embed_dense_batch(client, texts)` signatures defined in Task 10 → consumed in Tasks 22, 24. `LRUCache(maxsize)` signature defined in Task 6 → consumed in Tasks 10, 16, 22. `make_chat_client()`, `make_dense_client()`, `make_qdrant_client()` factory shapes defined in Tasks 9, 11 → consumed throughout. `WebFallback(dense_client=...)` constructor preserved from legacy (Task 20) → consumed in Tasks 26, 27, 28.

No fixes needed.
