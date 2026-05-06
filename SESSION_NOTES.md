# LOMA RAG — Session Notes & Handoff

Last updated: 2026-05-07.
Read this first when resuming work. Companion notes in user-memory at
`C:\Users\huudan\.claude\projects\D--t-LOMA\memory\`.

---

## TL;DR — pipeline state

Production-ready RAG chatbot for LOMA 281 (Meeting Customer Needs with
Insurance and Annuities) and LOMA 291 (Insurance Company Operations)
self-study courses. Multilingual (vi/en/fr/ja tested).

- **Quality**: `test_qa.py` → 48–49/51 PASS (94–96%) on 51 curated cases.
- **Latency**: median 4.5s, 76% < 5s, 84% < 7s.
- **API**: FastAPI on port 8000. `POST /chat`, `POST /chat/stream` (SSE),
  `GET /health`.

## How to resume

```powershell
# 1. Verify Qdrant + SearXNG are up
curl http://localhost:6333/collections          # need 'loma_chunks' + 'loma_nodes'
curl http://localhost:8080/healthz              # SearXNG health

# 2. If Qdrant collections missing (no persistent volume), rebuild:
cd D:\t\LOMA\rag
python ingest.py   # ~1 min  (produces out/*.jsonl)
python index.py    # ~2 min  (Azure embed + ColBERT + 3-vector upsert)

# 3. Start API
python api.py      # http://127.0.0.1:8000

# 4. Smoke test
python test_qa.py --id vi_def_antiselection -v   # one case
python test_qa.py                                # full 51-case suite (~7-8 min)
python bench.py                                  # latency benchmark
```

## Architecture overview

```
            User query (any language)
                      │
              analyze_query (gpt-4o-mini, 1 LLM call)
              ─ returns (lang_code, english_translation)
              ─ cached LRU 512
                      │
       ┌──────────────┴──────────────┐
       │ embed in parallel (ThreadPool)
       │  • dense   text-embedding-3-small (Azure, 1536d)
       │  • sparse  BM42 (local, IDF)
       │  • colbert colbert-ir/colbertv2.0 (local, 128d MaxSim)
       └──────────────┬──────────────┘
                      │
          Qdrant 3-way prefetch + RRF fusion
          (loma_chunks collection, top 100/vector)
          ─ dense uses INT8 quant + rescore + hnsw_ef=128
                      │
          Cross-encoder rerank (Xenova/ms-marco-MiniLM-L-6-v2)
          ─ top 12 candidates → top 5-7 chunks
          ─ RRF reservation: 2 slots for high-RRF chunks
                      │
          Direct KG semantic search (loma_nodes coll, same embeddings)
          ─ adds seed node IDs
                      │
          Graph expansion (NetworkX, 1-hop)
          ─ returns up to 12 related Node objects
                      │
       ┌──────────────┴──────────────┐
       │ select_answer_model(query)   │
       │  ─ default gpt-4o-mini       │
       │  ─ escalate gpt-4o for       │
       │    "step by step" / >40 word │
       └──────────────┬──────────────┘
                      │
          stream_with_sentinel_detect (sniff_chars=25)
          ─ if "[INSUFFICIENT_CONTEXT]" emitted → fallback
                      │
       ┌─── LOMA answer (cite [chunk_id]) ── return
       │
       └─── Web fallback (speculative-prep ran in parallel)
            ─ SearXNG search → page fetch → embed → FAISS rerank
            ─ Web LLM with [WEB_N] cites
```

## Files

```
D:\t\LOMA\rag\
  ingest.py           # docx + xlsx → chunks/nodes/edges/quizzes JSONL
  index.py            # JSONL → Qdrant (3 vectors per point)
  retrieval.py        # Retriever class: 3-way RRF + rerank + KG
  graph.py            # NetworkX KG load + helpers
  chat.py             # analyze_query, run_query, stream_query, CLI
  web_fallback.py     # SearXNG + page fetch + FAISS
  api.py              # FastAPI endpoints
  qa_cases.py         # 51 test cases (25 categories)
  test_qa.py          # quality test runner
  bench.py            # latency benchmark
  .env                # Azure + Qdrant + SearXNG credentials
  requirements.txt
  out/
    chunks.jsonl      # 583 chunks
    nodes.jsonl       # 343 KG nodes
    edges.jsonl       # 395 edges
    quizzes.jsonl     # empty (03_Quiz folder removed by user)
    qa_results_*.jsonl   # eval records
    *.log             # run logs
```

## Tunables (env vars)

```env
# Models
OPENAI_API_BASE=https://apim-maya-gpt-fnt-dev.azure-api.net/
OPENAI_API_KEY=...
OPENAI_API_VERSION=2025-01-01-preview
OPENAI_CHAT_MODEL=gpt-4o
OPENAI_DETECT_MODEL=gpt-4o-mini
OPENAI_SHORT_ANSWER_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_PREFER_BIG=                # set "1" to force gpt-4o for all answers
RERANK_MODEL=Xenova/ms-marco-MiniLM-L-6-v2

# Services
QDRANT_URL=http://localhost:6333
SEARXNG_URL=http://localhost:8080

# Retrieval
PREFETCH_LIMIT=100                # per vector type in 3-way prefetch
NODE_RRF_THRESHOLD=0.4            # min RRF for direct KG hits
DIRECT_NODE_TOP_K=5

# Decision thresholds (see chat.py)
EARLY_EXIT_RERANK=-2.0            # AND
EARLY_EXIT_RRF=0.10               # both must be below to skip LOMA
HIGH_CONFIDENCE_RERANK=2.0        # OR
HIGH_CONFIDENCE_RRF=0.50          # either above → no spec web prep

# Streaming
SNIFF_CHARS=25                    # TTFT vs sentinel-detect tradeoff

# API
API_HOST=127.0.0.1
API_PORT=8000
```

## Known stable test fails (out of 51 cases)

| Test | Cause | Workaround |
|---|---|---|
| `lang_override_vi_to_en` | gpt-4o-mini occasionally misses Vietnamese override phrases despite few-shot examples | Use gpt-4o for analyze_query (slower) — `OPENAI_DETECT_MODEL=gpt-4o` |
| `schedule_lesson_sections_vi` | TOC chunks rank higher than SCHEDULE chunks on this paraphrasing | Tune RRF reservation, or rephrase test |
| `concept_group_insurance_vi` (sometimes) | LLM emits sentinel on borderline relevance | Lower confidence threshold |

## Latency status

| Tier | Cases | % |
|---|---:|---:|
| **< 5s** | 10/13 | **76%** |
| **< 6s** | 11/13 | 84% |
| **< 7s** | 11/13 | 84% |
| **> 7s** | 2/13 | 16% (long-form synthesis: compare/list) |

Stages (median):
- analyze_query (combined detect+translate): ~700ms gpt-4o-mini
- embed (parallel dense+sparse+colbert): ~500ms
- Qdrant 3-way RRF: ~50ms
- Cross-encoder rerank: ~400ms
- Graph + direct KG node: ~30ms
- Answer LLM: 2–7s (depends on length, model)

## What was tried & rejected (don't re-introduce)

- **HyDE** — measured to hurt on English MCQ-style queries; current "translate-for-search via gpt-4o-mini" replaces it more cleanly.
- **Local embedding** (bge-small) — user explicitly wants Azure embedding stays.
- **ColBERT as final ranker** — over-ranks short overview chunks. Current 3-way RRF is better.

## Optimizations applied this session

1. ✅ Combined detect+translate into 1 LLM call
2. ✅ gpt-4o-mini for detect+translate
3. ✅ AsyncAzureOpenAI client added (not yet wired into main flow — sync still default)
4. ✅ Rerank candidates 20→12 (with RRF reservation for SYLLABUS chunks)
5. ✅ gpt-4o-mini default for answer (escalate only for >40-word or "step by step")
6. ✅ Speculative web threshold tuned (more aggressive)
7. ✅ sniff_chars 60→25 (faster TTFT)
8. ✅ Concise prompt (80–250 words) in LOMA + WEB system
9. ✅ Few-shot examples in analyze_query system prompt for language override detection

## Open / not done

- ❌ True speculative LLM start (start LLM during rerank). Skipped — needs full async refactor.
- ❌ Pre-warmed answer cache for top-K common queries. Would push synthesis queries < 7s.
- ❌ Multilingual reranker (e.g., jina-reranker-v2-base-multilingual). Current Xenova MiniLM is English-only — translate-for-search compensates.
- ❌ Redis-backed shared cache for multi-worker production.
- ❌ Web UI (Gradio / Next.js).
- ❌ Closed-book eval baseline (asked earlier, deprioritized when user removed quizzes).

## Suggested next steps (pick from these)

1. **Push synthesis queries < 7s consistently** — tighten prompt to 80–150 words, OR add answer cache.
2. **Switch to streaming UX** for compare/list — TTFT ~2s feels faster even if total stays at 7-8s.
3. **Production deployment** — Docker compose with Qdrant persistent volume, Redis for shared cache, multi-worker uvicorn.
4. **Web UI** — simple Gradio chat with related_nodes graph visualization.
5. **Multilingual reranker swap** — verify with bench whether quality + speed improves.

## Test commands cheat sheet

```powershell
# Single curl test
echo '{"query":"Antiselection là gì?"}' > q.json
curl -X POST http://127.0.0.1:8000/chat `
  -H "Content-Type: application/json; charset=utf-8" --data-binary @q.json

# Streaming
curl -N -X POST http://127.0.0.1:8000/chat/stream `
  -H "Content-Type: application/json; charset=utf-8" --data-binary @q.json

# Filter test cases by id
python test_qa.py --id vi_def_antiselection --id syllabus_281_hours_vi -v

# Filter by category
python test_qa.py --category concept_vi --category structural_vi

# Single CLI debug
python chat.py --show-context "Insurance Contract có những loại nào?"
python retrieval.py "Insurance Contract types"

# Graph self-test
python graph.py
```

## Pipeline restart checklist

When picking back up:
1. `tasklist | grep python` → kill old python processes if any
2. Verify Qdrant: `curl http://localhost:6333/collections` should show `loma_chunks` and `loma_nodes`. If not → `python ingest.py && python index.py`.
3. Verify SearXNG: `curl http://localhost:8080/search?q=test&format=json | head`.
4. Start API: `cd D:\t\LOMA\rag && python api.py`.
5. Smoke: `python test_qa.py --id vi_def_antiselection -v`.

---

## Session 2026-05-07 — `/quiz/chat` endpoint shipped

Merged `feat/quiz-chat` into main as merge commit `6119fe0` (16 commits +
merge). Spec/plan committed alongside the code.

### What changed in `/chat` response shape (earlier in the same session)
The `/chat` envelope was reshaped to a wrapper template:
```json
{"success": bool, "data": {path, answer, citations, related_nodes,
 en_search_query, intent, web_search_used}, "error": str}
```
- Removed `query` and `timings_ms` from the data field.
- Each LOMA citation now carries a `source` field (original chunk filename).
- New `intent` field added with value `"quiz"` when the user asks for a
  quiz/test (e.g. "cho tôi quiz về underwriting"). Pipeline gate
  `is_quiz_query` short-circuits before retrieval.

### What `/quiz/chat` is
A new endpoint for the **live quiz session UX**. Frontend passes the
current quiz question + 4 options + the learner's prompt; backend
classifies into one of:
- `intent="answer"` (with `answer="A"|"B"|"C"|"D"`) — letter/digit/fuzzy/LLM
- `intent="hint"` — keyword fuzzy match
- `intent="finish"` — keyword fuzzy match (incl. typos like `kent thuc`)
- `intent="question"` — discussion via RAG, but with a no-leak guardrail
  that prevents the AI from revealing which option is correct
- `intent="off_topic"` / `"unsupported_language"` — same canned responses
  as `/chat`

Both `POST /quiz/chat` (non-streaming) and `POST /quiz/chat/stream` (SSE)
are wired. Response shape mirrors `/chat`'s wrapper template.

### Files (all additive — `pipeline.py` and `chat.py` are bit-identical to before)
- New: `loma_rag/prompt/quiz_discussion.py` (no-leak system prompts +
  few-shot answer-parser prompt).
- New: `loma_rag/rag/quiz_intent.py` (pure-function intent helpers:
  `normalize_text`, `detect_end_session`, `detect_hint_request`,
  `parse_answer_letter` / `_fuzzy` / `_llm`).
- New: `loma_rag/rag/quiz_chat.py` (orchestrator: `run_quiz_chat` +
  `stream_quiz_chat`; mirrors `pipeline.run_query` shape but with
  quiz-specific intent gates and no-leak prompt injection).
- New: `loma_rag/api/routes/quiz_chat.py` (FastAPI handlers).
- Additive: `loma_rag/model/api_models.py` (Quiz* models with strict
  validation: option ids `^[A-D]$`, max_length on question/query/content
  to bound prompt-injection / DoS surface).
- Additive: `loma_rag/api/app.py` (2 lines: import + include_router).
- Tests: `tests/test_quiz_chat_intent.py` (87/87 unit cases including
  LLM ones) and `tests/test_quiz_chat_pipeline.py` (6/6 end-to-end with
  no-leak heuristic).

### Acceptance verified
- `python tests/test_quiz_chat_intent.py` → 87/87
- `python tests/test_quiz_chat_pipeline.py` → 6/6 in 15.4s
- `tests/test_qa.py --filter "vi_def_*" --filter "quiz_*"` → 9/9
  (regression check: existing `/chat` behaviour intact)

### Reference docs
- Spec: `docs/superpowers/specs/2026-05-06-quiz-chat-design.md`
- Plan: `docs/superpowers/plans/2026-05-07-quiz-chat.md` (12 TDD tasks,
  full code in each step — useful as a template for future features)

### Known v1 trade-offs
- `detect_end_session` / `detect_hint_request` use whole-token
  containment, which accepts a few false-positives like `"Don't finish
  this"` or `"I'd rather not have a hint"`. Documented in spec §10.
- Three trivial helpers (`_off_topic_response`, `_no_result_response`,
  `_prep_web_docs`) and ~120 LOC of streaming-loop logic are duplicated
  from `pipeline.py` to honour the user-imposed "do not modify chat
  code" constraint. When that constraint is lifted, lift them to a
  shared module and import from both.
- Spec said `options` `max_length=6`; implementation uses `max_length=4`
  because the id whitelist is A/B/C/D — pragmatic deviation, not a bug.
- `parse_answer_llm` uses a few-shot prompt to coax `gpt-4o-mini` into
  mapping conceptual paraphrases (e.g. "adverse selection" → option A
  "Antiselection"). Tuning was needed in-task to get from 3/6 to 6/6.
