# LOMA RAG Refactor — Design Spec

**Date:** 2026-05-06
**Author:** danth3@fpt.com (with Claude)
**Status:** Approved (pending implementation plan)

---

## 1. Goal

Refactor the LOMA RAG codebase from 11 flat-root Python files (~196 KB total) into a layered package `loma_rag/` with clear single-responsibility modules. Behavior must be identical post-refactor; the only intentional code-level changes are:

- Deduplication of repeated implementations (`make_chat_client`, `_LRUCache`, model-name constants).
- Extraction of magic numbers/strings into `constant/` (values unchanged).
- Introduction of a small custom-exception hierarchy where bare `Exception` is currently raised.

Out of scope: any pipeline logic change, prompt-text changes, model swaps, dependency upgrades, or new features.

## 2. Current state

11 root-level Python files:

| File | LOC class | Responsibility |
|------|-----------|----------------|
| `chat.py` | 41 KB | Largest file. Mixes prompts, LLM clients, sentinel-detection streaming, query analysis, language detection, citation formatting, REPL CLI. |
| `ingest.py` | 34 KB | docx/xlsx parsing, chunking, node/edge/quiz extraction, TOC/syllabus building, ingest pipeline. |
| `retrieval.py` | 20 KB | `Retriever` class with hybrid Qdrant search, RRF fusion, ColBERT/BM42/dense embedding, cross-encoder reranking, KG expansion. |
| `qa_cases.py` | 20 KB | Test data (`TESTS` global). |
| `eval.py` | 13 KB | LLM-judge evaluation harness. Duplicates `make_chat_client`. |
| `test_qa.py` | 11 KB | E2E QA test runner. |
| `index.py` | 10 KB | Qdrant collection setup + upsert. Duplicates `make_dense_client`, embedding logic. |
| `graph.py` | 10 KB | KG `Node`/`KG` classes, `load_graph`. |
| `bench.py` | 9 KB | API latency benchmark using httpx. |
| `api.py` | 7 KB | FastAPI app, `/chat`, `/chat/stream` endpoints, lifespan. |
| `web_fallback.py` | 7 KB | SearXNG search + FAISS rerank fallback. |

### Cross-file imports (current)

```
api.py        → chat, retrieval, web_fallback
chat.py       → retrieval, web_fallback
retrieval.py  → graph (lazy)
test_qa.py    → chat, qa_cases, retrieval, web_fallback
bench.py      → qa_cases
eval.py       → retrieval
```

### Identified duplication

- `make_chat_client` in both `chat.py` (line 461) and `eval.py`.
- `make_dense_client` in `index.py` (line 70); equivalent embedding-client setup logic appears inline in `retrieval.py`.
- `_LRUCache` defined in `retrieval.py:36`; `chat.py` reimplements the same pattern as ad-hoc dicts (`_translate_cache_*`, `_analyze_cache_*`, `_lang_cache_*`).
- Collection-name constants split across files: `CHUNKS_COLLECTION`/`CHUNKS_COLL`, `NODES_COLLECTION`/`NODES_COLL`.
- Model-name env reads (`OPENAI_EMBEDDING_MODEL`, `OPENAI_CHAT_MODEL`) repeated in 4+ files.

## 3. Target structure

Approach 3 (chosen): folder root `D:/t/LOMA/rag/` keeps its name; sub-package is `loma_rag/`. Imports look like `from loma_rag.rag.retriever import Retriever`.

Option A consolidation (chosen): `clean/` merged into `chunking/`; `helper/` merged into `util/`; `router/` merged into `api/routes/`; `exception/` collapsed to a single file `loma_rag/exceptions.py`.

```
D:/t/LOMA/rag/
├── .env
├── requirements.txt
├── main.py                          # subcommand dispatcher: serve|repl|ingest|index
├── loma_rag/
│   ├── __init__.py
│   ├── exceptions.py                # RagError, RetrievalError, IngestError, LLMError, WebFallbackError, VectorDBError
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py              # AzureConfig, QdrantConfig, RagConfig (load .env once)
│   ├── constant/
│   │   ├── __init__.py
│   │   ├── collections.py           # CHUNKS_COLL, NODES_COLL, DENSE_DIM=1536, COLBERT_DIM=128
│   │   ├── models.py                # default model names
│   │   ├── tokens.py                # INSUFFICIENT_TOKEN
│   │   └── thresholds.py            # PREFETCH_LIMIT, RRF/rerank/early-exit thresholds, cache sizes, MIN/MAX_CHUNK_CHARS
│   ├── model/
│   │   ├── __init__.py
│   │   ├── api_models.py            # Pydantic: ChatRequest, ChatResponse, Citation, GraphNode
│   │   └── domain.py                # @dataclass: AnswerResult, RetrievedChunk, RetrievalResult, Chunk, IngestNode, Edge, Quiz, WebDoc
│   ├── prompt/
│   │   ├── __init__.py
│   │   ├── system.py                # LOMA_SYSTEM, WEB_SYSTEM
│   │   ├── analyzer.py              # _ANALYZE_QUERY_SYSTEM, _ANALYZE_RE, _LANG_DETECT_SYSTEM, _LANG_CODE_RE
│   │   ├── translate.py             # TRANSLATE_SYSTEM, TRANSLATE_FOR_SEARCH_SYSTEM
│   │   ├── builder.py               # build_loma_user_prompt, build_web_user_prompt
│   │   └── judge.py                 # judge prompts from eval.py
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── openai_client.py         # make_chat_client, make_async_chat_client, make_dense_client
│   │   ├── embedding.py             # embed_dense (batch + single-with-cache), embed_sparse, embed_colbert
│   │   ├── completion.py            # select_answer_model, analyze_query[_async], detect_language_llm, translate_to_english_query
│   │   └── streaming.py             # stream_with_sentinel_detect, complete_with_sentinel_detect
│   ├── db_vector/
│   │   ├── __init__.py
│   │   ├── client.py                # make_qdrant_client(url)
│   │   ├── collection.py            # ensure_collection (3 named vectors)
│   │   ├── upsert.py                # _upsert_with_retry, stable_id, NAMESPACE, EMBED_BATCH, UPSERT_BATCH
│   │   └── search.py                # hybrid Prefetch+RRF query, _search_nodes_direct
│   ├── chunking/
│   │   ├── __init__.py
│   │   ├── chunker.py               # chunk_docx, _merge_tiny, link_chunks_to_nodes
│   │   ├── text.py                  # _norm_ws, _SKIP_PARAS
│   │   └── docx_blocks.py           # iter_block_items, table_to_md
│   ├── ingest/
│   │   ├── __init__.py
│   │   ├── filename.py              # parse_filename, FILENAME_RE
│   │   ├── docx_parser.py           # thin orchestrator over chunking
│   │   ├── xlsx_parser.py           # parse_quiz, parse_nodes, _parse_syllabus_meta, _parse_schedule
│   │   ├── tree_scanner.py          # _scan_course_tree, build_toc_chunks, build_syllabus_chunks, _extract_learning_objectives
│   │   ├── graph_builder.py         # build_edges
│   │   ├── pipeline.py              # find_files, main (orchestrator)
│   │   └── indexer.py               # index_docs (Qdrant upsert orchestrator from index.py)
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── retriever.py             # Retriever class (orchestration only)
│   │   ├── reranker.py              # cross-encoder rerank
│   │   ├── fusion.py                # RRF helpers
│   │   ├── graph.py                 # KG Node, KG class, load_graph
│   │   ├── analyzer.py              # analyze_query orchestration
│   │   ├── pipeline.py              # run_query, stream_query, answer_query, _prep_web_docs
│   │   └── web_fallback.py          # WebFallback, search_searxng, fetch_page_text, enrich_pages, faiss_rank
│   ├── api/
│   │   ├── __init__.py
│   │   ├── app.py                   # FastAPI factory + lifespan
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── health.py            # GET /health
│   │       └── chat.py              # POST /chat, POST /chat/stream
│   └── util/
│       ├── __init__.py
│       ├── cache.py                 # _LRUCache (single canonical implementation)
│       ├── io.py                    # load_jsonl, write_jsonl
│       ├── retry.py                 # generic retry decorator
│       ├── concurrency.py           # _BG_POOL ThreadPoolExecutor
│       ├── language.py              # language_name, _detect_user_language, _LANG_NAMES, _LANG_CODE_RE
│       ├── citation.py              # _build_citations, _format_loma_citations, _format_web_citations
│       ├── sse.py                   # _sse_event encoder
│       └── printer.py               # print_loma_context/sources, print_web_sources
├── scripts/
│   ├── eval.py                      # was eval.py (imports updated)
│   └── bench.py                     # was bench.py (imports updated)
├── tests/
│   ├── __init__.py
│   ├── qa_cases.py                  # was qa_cases.py
│   └── test_qa.py                   # was test_qa.py (imports updated)
├── docs/
│   └── superpowers/specs/2026-05-06-rag-refactor-design.md   # this file
├── out/                             # unchanged (data outputs)
└── _legacy/                         # temporary backup of originals during migration; deleted after validation
```

### Convention: `util/` vs `helper/` (now merged)

All cross-cutting helpers live in `util/`. File naming makes intent clear: `cache.py`, `io.py`, `retry.py` are generic; `citation.py`, `language.py`, `printer.py`, `sse.py` are domain-aware. The folder boundary that originally separated them carried no behavioral weight.

## 4. Symbol-level mapping

The mapping below is the source of truth for what moves where. Each entry: `<old_path>:<line>` → `<new_path>`. Symbols not listed individually are moved with their containing region.

### config / constant / exceptions

| Source | Destination |
|--------|-------------|
| `chat.py:51-67` env reads | `config/settings.py` (consolidated) |
| `retrieval.py:64-77` env reads | `config/settings.py` |
| `index.py:50-65` env reads | `config/settings.py` |
| `web_fallback.py:34-35` env reads | `config/settings.py` |
| `chat.py:63` `INSUFFICIENT_TOKEN` | `constant/tokens.py` |
| `index.py:52` `DENSE_DIM`, `index.py:55` `COLBERT_DIM` | `constant/collections.py` |
| `index.py:57` `CHUNKS_COLL`, `index.py:58` `NODES_COLL` | `constant/collections.py` (canonical) |
| `retrieval.py:64-65` `CHUNKS_COLLECTION`, `NODES_COLLECTION` (duplicates) | removed; consumers import from `constant.collections` |
| `retrieval.py:77,92,93` `PREFETCH_LIMIT`, `NODE_RRF_THRESHOLD`, `DIRECT_NODE_TOP_K` | `constant/thresholds.py` |
| `chat.py:67` `SNIFF_CHARS`, `chat.py:108-114` early-exit/high-confidence thresholds | `constant/thresholds.py` |
| `ingest.py:36-37` `MIN_CHUNK_CHARS`, `MAX_CHUNK_CHARS` | `constant/thresholds.py` |
| `chat.py:122,237,333` cache size constants, `retrieval.py:91,97` cache sizes | `constant/thresholds.py` |
| Default model names (`text-embedding-3-small`, `gpt-4o`, `gpt-4o-mini`, `colbert-ir/colbertv2.0`, `Xenova/ms-marco-MiniLM-L-6-v2`) | `constant/models.py` |
| New: base `RagError` and subclasses (`RetrievalError`, `IngestError`, `LLMError`, `WebFallbackError`, `VectorDBError`) | `loma_rag/exceptions.py` |

### model

| Source | Destination |
|--------|-------------|
| `api.py:44-82` Pydantic classes | `model/api_models.py` |
| `chat.py:141` `AnswerResult` | `model/domain.py` |
| `retrieval.py:102` `RetrievedChunk`, `retrieval.py:118` `RetrievalResult` | `model/domain.py` |
| `ingest.py:96` `Chunk` | `model/domain.py` |
| `ingest.py:229` `Node` | `model/domain.py` (renamed `IngestNode` to disambiguate) |
| `ingest.py:245` `Edge` | `model/domain.py` |
| `ingest.py:368` `Quiz` | `model/domain.py` |
| `web_fallback.py:44` `WebDoc` | `model/domain.py` |
| `graph.py:26` `Node` | stays as `Node` in `rag/graph.py` (KG-specific; different shape from `IngestNode`) |

### prompt

| Source | Destination |
|--------|-------------|
| `chat.py:156` `LOMA_SYSTEM`, `chat.py:170` `WEB_SYSTEM` | `prompt/system.py` |
| `chat.py:181` `_LANG_DETECT_SYSTEM`, `chat.py:193` `_ANALYZE_QUERY_SYSTEM`, `chat.py:232` `_ANALYZE_RE`, `chat.py:334` `_LANG_CODE_RE` | `prompt/analyzer.py` |
| `chat.py:397` `TRANSLATE_SYSTEM`, `retrieval.py:82` `TRANSLATE_FOR_SEARCH_SYSTEM` | `prompt/translate.py` |
| `chat.py:406` `build_loma_user_prompt`, `chat.py:438` `build_web_user_prompt` | `prompt/builder.py` |
| Judge prompts from `eval.py` (`build_judge_user`, judge system) | `prompt/judge.py` |

### llm

| Source | Destination |
|--------|-------------|
| `chat.py:461` `make_chat_client` (canonical), `chat.py:471` `make_async_chat_client` | `llm/openai_client.py` |
| `eval.py` `make_chat_client` (duplicate) | removed; `scripts/eval.py` imports from `llm.openai_client` |
| `index.py:70` `make_dense_client` | `llm/openai_client.py` |
| `index.py:78` `embed_dense` (batch) | `llm/embedding.py` |
| `retrieval.py:220` `Retriever._embed_dense` (cached single) | `llm/embedding.py` (becomes free function with cache) |
| `retrieval.py:234` `Retriever._embed_sparse`, `retrieval.py:238` `Retriever._embed_colbert` | `llm/embedding.py` |
| `chat.py:86` `select_answer_model`, `chat.py:255` `analyze_query`, `chat.py:287` `analyze_query_async`, `chat.py:352` `detect_language_llm`, `chat.py:483` `translate_to_english_query` | `llm/completion.py` |
| `chat.py:509` `stream_with_sentinel_detect`, `chat.py:570` `complete_with_sentinel_detect` | `llm/streaming.py` |

### db_vector

| Source | Destination |
|--------|-------------|
| New: `make_qdrant_client(url)` factory wrapping `QdrantClient(url=url)` | `db_vector/client.py` |
| `index.py:94` `ensure_collection` | `db_vector/collection.py` |
| `index.py:158` `stable_id`, `index.py:65` `NAMESPACE`, `index.py:60,63` batch sizes, `index.py:162` `_upsert_with_retry` | `db_vector/upsert.py` |
| Hybrid Qdrant query (Prefetch + Query API), currently inline in `retrieval.py:243` (`Retriever.retrieve`) | extracted into `db_vector/search.py` as `hybrid_search(...)` |
| `retrieval.py:415` `Retriever._search_nodes_direct` | `db_vector/search.py` as `search_nodes_direct(...)` |

### chunking

| Source | Destination |
|--------|-------------|
| `ingest.py:42` `_norm_ws`, `ingest.py:46` `_SKIP_PARAS` | `chunking/text.py` |
| `ingest.py:63` `iter_block_items`, `ingest.py:78` `table_to_md` | `chunking/docx_blocks.py` |
| `ingest.py:111` `chunk_docx`, `ingest.py:179` `_merge_tiny`, `ingest.py:339` `link_chunks_to_nodes` | `chunking/chunker.py` |

### ingest

| Source | Destination |
|--------|-------------|
| `ingest.py:39` `FILENAME_RE`, `ingest.py:51` `parse_filename` | `ingest/filename.py` |
| Thin orchestrator wrapping `chunking.docx_blocks` + `chunking.chunker` | `ingest/docx_parser.py` |
| `ingest.py:251` `parse_nodes`, `ingest.py:383` `parse_quiz`, `ingest.py:638` `_parse_syllabus_meta`, `ingest.py:652` `_parse_schedule` | `ingest/xlsx_parser.py` |
| `ingest.py:469-471` `_COURSE/_MODULE/_LESSON_DIR_RE`, `ingest.py:474` `_extract_learning_objectives`, `ingest.py:508` `_scan_course_tree`, `ingest.py:545` `build_toc_chunks`, `ingest.py:710` `_find_syllabus`, `ingest.py:719` `build_syllabus_chunks`, `ingest.py:634-635` `_M_RE`, `_L_RE` | `ingest/tree_scanner.py` |
| `ingest.py:305` `build_edges` | `ingest/graph_builder.py` |
| `ingest.py:887` `find_files`, `ingest.py:903` `write_jsonl`, `ingest.py:909` `main` | `ingest/pipeline.py` (note: `write_jsonl` re-exported via `util/io.py`; pipeline calls `util.io.write_jsonl`) |
| `index.py:179` `index_docs`, `index.py:248` `main` | `ingest/indexer.py` |

### rag

| Source | Destination |
|--------|-------------|
| `retrieval.py:127` `Retriever` class | `rag/retriever.py` (body becomes orchestrator: calls `db_vector.search`, `llm.embedding`, `rag.reranker`, `rag.fusion`) |
| Cross-encoder rerank logic inline in `Retriever.retrieve`, plus `retrieval.py:72` `RERANK_MODEL` | `rag/reranker.py` |
| RRF fusion math inline in `Retriever.retrieve` | `rag/fusion.py` |
| `graph.py` entire | `rag/graph.py` |
| Wrapper around `llm.completion.analyze_query` using `prompt.analyzer` | `rag/analyzer.py` |
| `chat.py:501` `_prep_web_docs`, `chat.py:620` `run_query`, `chat.py:757` `stream_query`, `chat.py:970` `answer_query` | `rag/pipeline.py` |
| `web_fallback.py` entire | `rag/web_fallback.py` |

### api

| Source | Destination |
|--------|-------------|
| `api.py:85` `lifespan`, FastAPI app construction | `api/app.py` (factory `create_app()`) |
| `api.py:125` `GET /health` | `api/routes/health.py` |
| `api.py:130` `POST /chat`, `api.py:197` `POST /chat/stream` | `api/routes/chat.py` |

### util (incl. former helper)

| Source | Destination |
|--------|-------------|
| `retrieval.py:36` `_LRUCache` | `util/cache.py` (single canonical) |
| `chat.py:122-345` ad-hoc dict caches `_translate_cache_*`, `_analyze_cache_*`, `_lang_cache_*` | replaced by `util.cache._LRUCache` instances |
| `index.py:242` `load_jsonl`, `ingest.py:903` `write_jsonl` | `util/io.py` |
| Retry loops in `index.py:78` (`embed_dense`) and `index.py:162` (`_upsert_with_retry`) | factored into `util/retry.py` decorator/helper |
| `chat.py:118` `_BG_POOL` | `util/concurrency.py` |
| `chat.py:316` `_LANG_NAMES`, `chat.py:380` `language_name`, `chat.py:384` `_detect_user_language` | `util/language.py` |
| `api.py:119` `_build_citations`, `chat.py:728` `_format_loma_citations`, `chat.py:745` `_format_web_citations` | `util/citation.py` |
| `api.py:192` `_sse_event` | `util/sse.py` |
| `chat.py:589` `print_loma_context`, `chat.py:604` `print_loma_sources`, `chat.py:611` `print_web_sources` | `util/printer.py` |

### main / scripts / tests

| Source | Destination |
|--------|-------------|
| `chat.py:1052` `main` (REPL) | `main.py repl` subcommand |
| `ingest.py:909` `main` | `main.py ingest` subcommand |
| `index.py:248` `main` | `main.py index` subcommand |
| FastAPI launcher | `main.py serve` subcommand |
| `eval.py` | `scripts/eval.py` (imports updated) |
| `bench.py` | `scripts/bench.py` (imports updated) |
| `qa_cases.py` | `tests/qa_cases.py` |
| `test_qa.py` | `tests/test_qa.py` (imports updated) |

## 5. Public interface contract

The following symbols/paths are the cross-package public surface. Internal helpers (prefixed `_`) remain private to their module.

```python
# Configuration
from loma_rag.config.settings import AzureConfig, QdrantConfig, RagConfig

# Pipelines
from loma_rag.rag.pipeline import run_query, stream_query, answer_query
from loma_rag.rag.retriever import Retriever
from loma_rag.rag.web_fallback import WebFallback

# Ingest
from loma_rag.ingest.pipeline import main as run_ingest
from loma_rag.ingest.indexer import index_docs

# API
from loma_rag.api.app import create_app

# Domain types
from loma_rag.model.domain import AnswerResult, RetrievalResult, RetrievedChunk, Chunk, IngestNode, Edge, Quiz, WebDoc
from loma_rag.model.api_models import ChatRequest, ChatResponse, Citation, GraphNode

# Errors
from loma_rag.exceptions import RagError, RetrievalError, IngestError, LLMError, WebFallbackError, VectorDBError
```

## 6. Migration order

Leaf-first to keep every intermediate state importable.

**Phase 1 — Foundations (no internal deps).** `config/settings.py`, `constant/*` (4 files), `exceptions.py`, `model/domain.py`, `model/api_models.py`, `util/{cache,io,retry,concurrency,language,sse}.py`.

**Phase 2 — Providers (depend on Phase 1).** `prompt/*` (5 files), `llm/openai_client.py`, `llm/embedding.py`, `db_vector/{client,collection,upsert}.py`.

**Phase 3 — Domain logic.** `chunking/*`; `ingest/{filename,docx_parser,xlsx_parser,tree_scanner,graph_builder,pipeline,indexer}.py`; `db_vector/search.py`; `llm/{completion,streaming}.py`; `util/{citation,printer}.py`.

**Phase 4 — Orchestration.** `rag/{graph,web_fallback,reranker,fusion,analyzer,retriever,pipeline}.py`.

**Phase 5 — Interface.** `api/app.py`, `api/routes/*`, `main.py`, `scripts/{eval,bench}.py`, `tests/*`.

**Phase 6 — Cleanup.** Move originals to `_legacy/` (do not delete). Run validation. After user confirms pass, delete `_legacy/`.

## 7. Validation

**Step 1 — Static** (no external deps required):

```bash
python -c "import loma_rag.api.app, loma_rag.rag.pipeline, loma_rag.rag.retriever, loma_rag.ingest.pipeline, loma_rag.ingest.indexer"
python -c "from loma_rag.api.app import create_app; create_app()"
```

Both must exit 0.

**Step 2 — Smoke** (requires `.env`):

```bash
python main.py repl --help
python -c "from loma_rag.rag.retriever import Retriever; r = Retriever(); print(r)"
```

**Step 3 — End-to-end** (requires Qdrant at `http://localhost:6333` + valid Azure OpenAI key + indexed `out/` JSONL files):

```bash
python -m tests.test_qa
```

Pass-rate must be ≥ pre-refactor baseline (≥94% per `SESSION_NOTES.md`). Any drop triggers rollback from `_legacy/`.

**Pre-flight checklist** (confirmed with user before Step 3): Qdrant running, `.env` valid, `out/` populated.

## 8. Risks and mitigations

| Risk | Mitigation |
|------|-----------|
| Import cycle when splitting `chat.py` into `rag/pipeline.py` + `llm/{completion,streaming}.py` | One-way deps: `rag/pipeline` imports `llm/*`. `completion` and `streaming` do not import each other. |
| Module-level cache state lost across modules after relocating `_LRUCache` instances | Each cache instance remains a module-level global in its consumer module. `util.cache._LRUCache` is just the class. |
| `Retriever` constructor signature drift breaks `api.py` lifespan | Constructor signature preserved exactly; only body is refactored. |
| Behavior drift from prompt-string reformatting | Prompts copied verbatim (no whitespace changes). `tests/test_qa.py` pass-rate is the regression gate. |
| Two `Node` classes (ingest vs KG) collide if both moved to `model/domain.py` | Ingest version renamed to `IngestNode`. KG version stays in `rag/graph.py`. |
| Repo is not a git repo, so no `git checkout` rollback available | Originals copied to `_legacy/` before any deletion. Deletion only after Step 3 passes. |

## 9. Decisions log

- **Approach 3** (sub-package `loma_rag/`): chosen over flat-root (Approach 1, naming collision) and `src/` layout (Approach 2, overkill without packaging).
- **Option A consolidation**: chosen over heavy layer-merge (Option B) to keep all user-requested folder names visible.
- **`exception/` → `exceptions.py`**: 5–6 error classes do not warrant a folder.
- **`Node` rename**: ingest `Node` becomes `IngestNode` to coexist with KG `Node`.
- **Judge prompts**: moved into `prompt/judge.py` to keep all prompt text in one place; `scripts/eval.py` imports them.
- **Validation level**: full E2E (Step 3) per user choice.
- **Dedup scope**: yes — `make_chat_client`, `make_dense_client`, `_LRUCache`, collection-name constants, model-name env reads.
