"""RAG answer pipeline: LOMA retrieval -> LLM answer -> web fallback.

Pipeline per query:
    [user query, any language]
        |
        v
  Retriever  ---(pre-translated English query for embed/sparse/rerank)
        |     ---> Qdrant hybrid (dense+sparse, RRF)
        |     ---> cross-encoder rerank
        |     ---> graph expand (KG nodes referenced by chunks)
        v
  GPT-4o answer attempt   <-- if context insufficient, model emits
        |                     [INSUFFICIENT_CONTEXT] sentinel
        |
        +--- sufficient ---> stream answer (LOMA citations)
        |
        +--- insufficient -> Translate query to EN
                              SearXNG -> fetch pages -> embed -> FAISS top-k
                              Stream answer (web citations)

Final answer is always in the user's language (system prompt rule).
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import Future

from loma_rag.config.settings import azure, rag as rag_cfg
from loma_rag.constant.responses import (
    LANG_FULL_NAME,
    NO_RESULT_RESPONSE_MAP,
    OFF_TOPIC_RESPONSE_MAP,
    SUPPORTED_LANGS,
    UNSUPPORTED_LANGUAGE_MSG,
)
from loma_rag.constant.tokens import INSUFFICIENT_TOKEN
from loma_rag.llm.completion import analyze_query, select_answer_model, translate_to_english_query
from loma_rag.llm.streaming import complete_with_sentinel_detect, stream_with_sentinel_detect
from loma_rag.llm.topic import is_insurance_topic
from loma_rag.model.domain import AnswerResult, RetrievalResult
from loma_rag.prompt.builder import build_loma_user_prompt, build_web_user_prompt
from loma_rag.prompt.system import LOMA_SYSTEM, WEB_SYSTEM
from loma_rag.rag.retriever import Retriever
from loma_rag.rag.web_fallback import WebFallback
from loma_rag.util.citation import format_loma_citations, format_web_citations
from loma_rag.util.concurrency import BG_POOL
from loma_rag.util.language import detect_user_language
from loma_rag.util.printer import print_loma_context, print_loma_sources, print_web_sources

EARLY_EXIT_RERANK_THRESHOLD = rag_cfg.early_exit_rerank
EARLY_EXIT_RRF_THRESHOLD = rag_cfg.early_exit_rrf
HIGH_CONFIDENCE_RERANK_THRESHOLD = rag_cfg.high_confidence_rerank
HIGH_CONFIDENCE_RRF_THRESHOLD = rag_cfg.high_confidence_rrf
SNIFF_CHARS = rag_cfg.sniff_chars
CHAT_MODEL = None  # resolved lazily via select_answer_model


def prep_web_docs(query: str, web_fallback: WebFallback, chat_client, web_k: int):
    """Background-callable: translate -> SearXNG -> fetch -> embed -> rank.
    Returns (en_query, web_docs). Used for speculative fallback prep."""
    en_query = translate_to_english_query(chat_client, query)
    docs = web_fallback.retrieve(en_query, top_k=web_k)
    return en_query, docs


def _off_topic_response(user_lang: str) -> str:
    """Pick the canned off-topic response for the user's language."""
    return OFF_TOPIC_RESPONSE_MAP[LANG_FULL_NAME[user_lang]]


def _no_result_response(user_lang: str) -> str:
    """Pick the canned no-result response for the user's language."""
    return NO_RESULT_RESPONSE_MAP[LANG_FULL_NAME[user_lang]]


# ---- orchestration (programmatic) ----

def run_query(
    retriever: Retriever,
    chat_client,
    web_fallback: WebFallback | None,
    query: str,
    top_k: int = 5,
    web_k: int = 5,
) -> AnswerResult:
    """Run the full pipeline non-interactively. No printing. Returns structured result."""
    res = AnswerResult(query=query, used_path="")

    # Single combined call: detect language + translate to English. Caches both.
    user_lang, search_text = analyze_query(chat_client, query)

    # Gate 1: language outside vi/en/ja → English-only canned response.
    if user_lang not in SUPPORTED_LANGS:
        res.used_path = "unsupported_language"
        res.answer_text = UNSUPPORTED_LANGUAGE_MSG
        return res

    # Gate 2: off-topic (not insurance / LOMA) → canned response in user lang.
    if not is_insurance_topic(chat_client, query):
        res.used_path = "off_topic"
        res.answer_text = _off_topic_response(user_lang)
        return res

    # Stage 1: retrieve LOMA chunks (skip retriever's internal translate by
    # passing the pre-translated search_text).
    retr = retriever.retrieve(query, top_k=top_k, user_lang=user_lang,
                              search_text=search_text)
    res.loma_chunks = list(retr.chunks)
    res.related_nodes = list(retr.related_nodes)
    res.direct_node_ids = list(getattr(retr, "direct_node_ids", []) or [])
    res.timings = dict(retr.timings)

    top_rerank = retr.chunks[0].rerank_score if retr.chunks else float("-inf")
    top_rrf = retr.chunks[0].fused_score if retr.chunks else 0.0

    # Early-exit: only when BOTH signals say "off-topic" — catches obvious
    # off-corpus queries without misfiring on borderline-RRF good matches.
    early_exit = (
        web_fallback is not None
        and top_rerank < EARLY_EXIT_RERANK_THRESHOLD
        and top_rrf < EARLY_EXIT_RRF_THRESHOLD
    )

    # Speculative web prep: when NEITHER signal indicates high confidence,
    # fire off web-fallback prep in the background while LOMA LLM runs.
    high_confidence = (
        top_rerank >= HIGH_CONFIDENCE_RERANK_THRESHOLD
        or top_rrf >= HIGH_CONFIDENCE_RRF_THRESHOLD
    )
    spec_future: "Future | None" = None
    if web_fallback is not None and not early_exit and not high_confidence:
        spec_future = BG_POOL.submit(prep_web_docs, query, web_fallback, chat_client, web_k)

    answer_model = select_answer_model(query)

    if early_exit:
        text, insufficient = "", True
    else:
        # Stage 2: try LOMA answer.
        user_prompt = build_loma_user_prompt(retr, user_lang=user_lang)
        text, insufficient = complete_with_sentinel_detect(
            chat_client, LOMA_SYSTEM, user_prompt, model=answer_model,
        )

    if not insufficient:
        # LOMA was good — discard speculative work (best-effort cancel).
        if spec_future is not None:
            spec_future.cancel()
        res.used_path = "loma"
        res.answer_text = text
        return res

    # Stage 3: web fallback.
    if web_fallback is None:
        res.used_path = "refused"
        return res

    if spec_future is not None:
        # Reuse the speculative result if it's done (or wait for it — cheaper
        # than starting from scratch since translate+search already ran).
        try:
            en_query, web_docs = spec_future.result(timeout=60)
        except Exception as e:  # noqa: BLE001
            res.used_path = "no_result"
            res.answer_text = _no_result_response(user_lang)
            res.error = f"{type(e).__name__}: {e}"
            return res
    else:
        en_query = translate_to_english_query(chat_client, query)
        try:
            web_docs = web_fallback.retrieve(en_query, top_k=web_k)
        except Exception as e:  # noqa: BLE001
            res.used_path = "no_result"
            res.answer_text = _no_result_response(user_lang)
            res.error = f"{type(e).__name__}: {e}"
            return res
    res.en_search_query = en_query

    if not web_docs:
        res.used_path = "no_result"
        res.answer_text = _no_result_response(user_lang)
        return res

    res.web_docs = list(web_docs)
    web_user = build_web_user_prompt(query, web_docs, user_lang=user_lang)
    resp = chat_client.chat.completions.create(
        model=answer_model,
        messages=[
            {"role": "system", "content": WEB_SYSTEM},
            {"role": "user", "content": web_user},
        ],
        temperature=0.2,
    )
    res.used_path = "web"
    res.answer_text = resp.choices[0].message.content or ""
    return res


# ---- orchestration (programmatic streaming, generator) ----

def stream_query(
    retriever: Retriever,
    chat_client,
    web_fallback: WebFallback | None,
    query: str,
    top_k: int = 5,
    web_k: int = 5,
):
    """Synchronous generator yielding event dicts. Caller serialises (e.g. SSE).

    Event shapes:
      {"type": "stage",     "stage": "retrieving" | "answering_loma" | "translating" | "web_search" | "answering_web"}
      {"type": "delta",     "text": "<chunk>"}                (answer tokens)
      {"type": "done",      "path": "loma"|"web"|"refused"|"no_web_results"|"error",
                            "citations": [...], "timings": {...}, "answer_text": "..."}
    """
    timings: dict[str, float] = {}
    t_overall = time.time()

    # Combined detect-language + translate-to-English in a single LLM call.
    yield {"type": "stage", "stage": "analyze_query"}
    t0 = time.time()
    user_lang, search_text = analyze_query(chat_client, query)
    timings["analyze_query"] = time.time() - t0

    # Gate 1: language outside vi/en/ja → English-only canned response.
    if user_lang not in SUPPORTED_LANGS:
        timings["total"] = time.time() - t_overall
        yield {"type": "delta", "text": UNSUPPORTED_LANGUAGE_MSG}
        yield {"type": "done", "path": "unsupported_language",
               "answer_text": UNSUPPORTED_LANGUAGE_MSG, "citations": [],
               "timings": {k: round(v * 1000) for k, v in timings.items()}}
        return

    # Gate 2: off-topic (not insurance / LOMA) → canned response in user lang.
    yield {"type": "stage", "stage": "topic_check"}
    t0 = time.time()
    on_topic = is_insurance_topic(chat_client, query)
    timings["topic_check"] = time.time() - t0
    if not on_topic:
        canned = _off_topic_response(user_lang)
        timings["total"] = time.time() - t_overall
        yield {"type": "delta", "text": canned}
        yield {"type": "done", "path": "off_topic",
               "answer_text": canned, "citations": [],
               "timings": {k: round(v * 1000) for k, v in timings.items()}}
        return

    # Stage 1: retrieve LOMA chunks (pass pre-translated search_text).
    yield {"type": "stage", "stage": "retrieving"}
    t0 = time.time()
    retr = retriever.retrieve(query, top_k=top_k, user_lang=user_lang,
                              search_text=search_text)
    timings["retrieve"] = time.time() - t0
    timings.update({f"retrieve.{k}": v for k, v in retr.timings.items()})

    top_rerank = retr.chunks[0].rerank_score if retr.chunks else float("-inf")
    top_rrf = retr.chunks[0].fused_score if retr.chunks else 0.0
    skip_loma_llm = (
        web_fallback is not None
        and top_rerank < EARLY_EXIT_RERANK_THRESHOLD
        and top_rrf < EARLY_EXIT_RRF_THRESHOLD
    )

    high_confidence = (
        top_rerank >= HIGH_CONFIDENCE_RERANK_THRESHOLD
        or top_rrf >= HIGH_CONFIDENCE_RRF_THRESHOLD
    )
    spec_future: "Future | None" = None
    if web_fallback is not None and not skip_loma_llm and not high_confidence:
        spec_future = BG_POOL.submit(prep_web_docs, query, web_fallback, chat_client, web_k)

    sniff_chars = SNIFF_CHARS
    buf: list[str] = []
    decided = False
    insufficient = False
    answer_full: list[str] = []

    if skip_loma_llm:
        insufficient = True
        timings["loma_answer"] = 0.0
        yield {"type": "stage", "stage": "early_exit_to_web",
               "rrf_top1": top_rrf}

    answer_model = select_answer_model(query)

    if not skip_loma_llm:
        # Stage 2: try LOMA answer (stream, sentinel-detect).
        yield {"type": "stage", "stage": "answering_loma", "model": answer_model}
        user_prompt = build_loma_user_prompt(retr, user_lang=user_lang)
        t0 = time.time()
        stream = chat_client.chat.completions.create(
            model=answer_model,
            messages=[
                {"role": "system", "content": LOMA_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            stream=True,
        )

        for ev in stream:
            if not ev.choices:
                continue
            delta = ev.choices[0].delta.content or ""
            if not delta:
                continue
            buf.append(delta)
            text = "".join(buf)
            if not decided:
                if INSUFFICIENT_TOKEN in text:
                    insufficient = True
                    decided = True
                    break
                if len(text) >= sniff_chars or "\n" in text:
                    decided = True
                    yield {"type": "delta", "text": text}
                    answer_full.append(text)
            else:
                yield {"type": "delta", "text": delta}
                answer_full.append(delta)

        if not decided:
            text = "".join(buf)
            if INSUFFICIENT_TOKEN in text:
                insufficient = True
            elif text:
                yield {"type": "delta", "text": text}
                answer_full.append(text)

        timings["loma_answer"] = time.time() - t0

    if not insufficient:
        if spec_future is not None:
            spec_future.cancel()
        loma_text = "".join(answer_full)
        timings["total"] = time.time() - t_overall
        direct_ids = set(getattr(retr, "direct_node_ids", []) or [])
        yield {
            "type": "done",
            "path": "loma",
            "answer_text": loma_text,
            "citations": format_loma_citations(loma_text, retr.chunks),
            "related_nodes": [
                {
                    "node_id": n.node_id, "name": n.name, "category": n.category,
                    "definition": n.definition, "lesson_id": n.lesson_id,
                    "direct_hit": n.node_id in direct_ids,
                }
                for n in (retr.related_nodes or [])
            ],
            "timings": {k: round(v * 1000) for k, v in timings.items()},
        }
        return

    # Stage 3: fallback.
    if web_fallback is None:
        timings["total"] = time.time() - t_overall
        yield {"type": "done", "path": "refused", "answer_text": "",
               "citations": [], "timings": {k: round(v * 1000) for k, v in timings.items()}}
        return

    if spec_future is not None:
        # Wait for the speculative work (it may already be done).
        yield {"type": "stage", "stage": "awaiting_speculative_web"}
        t0 = time.time()
        try:
            en_query, web_docs = spec_future.result(timeout=60)
        except Exception as e:  # noqa: BLE001
            timings["web_speculative"] = time.time() - t0
            timings["total"] = time.time() - t_overall
            canned = _no_result_response(user_lang)
            yield {"type": "delta", "text": canned}
            yield {"type": "done", "path": "no_result", "answer_text": canned,
                   "error": f"{type(e).__name__}: {e}", "citations": [],
                   "timings": {k: round(v * 1000) for k, v in timings.items()}}
            return
        timings["web_speculative"] = time.time() - t0
    else:
        yield {"type": "stage", "stage": "translating"}
        t0 = time.time()
        en_query = translate_to_english_query(chat_client, query)
        timings["translate"] = time.time() - t0

        yield {"type": "stage", "stage": "web_search"}
        t0 = time.time()
        try:
            web_docs = web_fallback.retrieve(en_query, top_k=web_k)
        except Exception as e:  # noqa: BLE001
            timings["web_search"] = time.time() - t0
            timings["total"] = time.time() - t_overall
            canned = _no_result_response(user_lang)
            yield {"type": "delta", "text": canned}
            yield {"type": "done", "path": "no_result", "answer_text": canned,
                   "error": f"{type(e).__name__}: {e}", "citations": [],
                   "timings": {k: round(v * 1000) for k, v in timings.items()}}
            return
        timings["web_search"] = time.time() - t0

    if not web_docs:
        timings["total"] = time.time() - t_overall
        canned = _no_result_response(user_lang)
        yield {"type": "delta", "text": canned}
        yield {"type": "done", "path": "no_result", "answer_text": canned,
               "citations": [],
               "timings": {k: round(v * 1000) for k, v in timings.items()}}
        return

    yield {"type": "stage", "stage": "answering_web", "model": answer_model}
    web_user = build_web_user_prompt(query, web_docs, user_lang=user_lang)
    t0 = time.time()
    web_stream = chat_client.chat.completions.create(
        model=answer_model,
        messages=[
            {"role": "system", "content": WEB_SYSTEM},
            {"role": "user", "content": web_user},
        ],
        temperature=0.2,
        stream=True,
    )
    web_full: list[str] = []
    for ev in web_stream:
        if not ev.choices:
            continue
        delta = ev.choices[0].delta.content or ""
        if delta:
            yield {"type": "delta", "text": delta}
            web_full.append(delta)
    timings["web_answer"] = time.time() - t0
    timings["total"] = time.time() - t_overall

    yield {
        "type": "done",
        "path": "web",
        "answer_text": "".join(web_full),
        "citations": format_web_citations(web_docs),
        "en_search_query": en_query,
        "timings": {k: round(v * 1000) for k, v in timings.items()},
    }


# ---- orchestration (CLI, with streaming) ----

def answer_query(
    retriever: Retriever,
    chat_client,
    web_fallback: WebFallback | None,
    query: str,
    top_k: int,
    web_k: int,
    show_context: bool,
    stream: bool,
) -> None:
    print(f"\n[?] {query}")

    # Stage 1: combined detect+translate, then retrieve.
    user_lang, search_text = analyze_query(chat_client, query)

    # Gate 1: language outside vi/en/ja → English-only canned response.
    if user_lang not in SUPPORTED_LANGS:
        print(UNSUPPORTED_LANGUAGE_MSG)
        return

    # Gate 2: off-topic → canned response in user lang.
    if not is_insurance_topic(chat_client, query):
        print(_off_topic_response(user_lang))
        return

    result = retriever.retrieve(query, top_k=top_k, user_lang=user_lang,
                                search_text=search_text)
    if show_context:
        print_loma_context(result)

    # Stage 2: try answering from LOMA. If model emits sentinel, go to fallback.
    user_prompt = build_loma_user_prompt(result, user_lang=user_lang)
    print()
    if stream:
        text, insufficient = stream_with_sentinel_detect(
            chat_client, LOMA_SYSTEM, user_prompt
        )
    else:
        text, insufficient = complete_with_sentinel_detect(
            chat_client, LOMA_SYSTEM, user_prompt
        )
        if not insufficient:
            print(text)

    if not insufficient:
        print_loma_sources(result)
        return

    # Stage 3: fallback to web search.
    if web_fallback is None:
        print("[fallback disabled — no answer]")
        return

    print("\n[no LOMA answer; falling back to web search]")
    en_query = translate_to_english_query(chat_client, query)
    print(f"  english search query: {en_query!r}")

    try:
        web_docs = web_fallback.retrieve(en_query, top_k=web_k)
    except Exception as e:  # noqa: BLE001
        print(f"  ! web search failed: {e}", file=sys.stderr)
        print(_no_result_response(user_lang))
        return

    if not web_docs:
        print(_no_result_response(user_lang))
        return

    if show_context:
        for i, d in enumerate(web_docs, 1):
            print(f"  WEB_{i} score={d.score:.3f}  {d.title[:80]}")

    print()
    web_user = build_web_user_prompt(query, web_docs, user_lang=user_lang)
    if stream:
        # No sentinel needed in fallback path; fall back to plain stream.
        # Reuse stream_with_sentinel_detect with an unmatchable sentinel.
        stream_with_sentinel_detect(chat_client, WEB_SYSTEM, web_user, sentinel="\x00\x01\x02")
    else:
        resp = chat_client.chat.completions.create(
            model=azure.chat_model,
            messages=[
                {"role": "system", "content": WEB_SYSTEM},
                {"role": "user", "content": web_user},
            ],
            temperature=0.2,
        )
        print(resp.choices[0].message.content or "")
    print_web_sources(web_docs)
