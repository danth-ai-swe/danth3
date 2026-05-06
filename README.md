# danth3

LOMA RAG — Retrieval-Augmented Generation chatbot for the LOMA 281 / LOMA 291
insurance courses, with hybrid Qdrant search, knowledge-graph expansion,
cross-encoder reranking, and SearXNG + FAISS web fallback.

## Layout

```
loma_rag/                package: config, constant, model, prompt,
                         llm, db_vector, chunking, ingest, rag, api, util
main.py                  CLI dispatcher: serve | repl | ingest | index
scripts/                 eval.py, bench.py
tests/                   qa_cases.py, test_qa.py
docs/superpowers/        spec + implementation plan
```

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env       # fill in Azure OpenAI + Qdrant config
python main.py ingest      # docx/xlsx -> out/*.jsonl
python main.py index       # JSONL -> Qdrant collections
python main.py serve       # FastAPI on http://127.0.0.1:8000
python main.py repl        # interactive chat
```

## Pipeline gates

1. Unsupported language (outside `vi` / `en` / `ja`) → English-only canned response.
2. Off-topic (not insurance / LOMA) → canned response in user's language.
3. Hybrid Qdrant retrieval → cross-encoder rerank → KG expand.
4. LLM answer with `[INSUFFICIENT_CONTEXT]` sentinel.
5. SearXNG + FAISS web fallback if LOMA chunks insufficient.
6. No-result fallback → canned response in user's language.
