"""Quiz-chat orchestrator. Reuses chat building blocks but never
modifies pipeline.py / chat.py. Tiny private helpers
(_off_topic_response, _no_result_response, _prep_web_docs) are
duplicated rather than imported from pipeline.py to keep that module
untouched.
"""
from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Optional

from loma_rag.config.settings import rag as rag_cfg
from loma_rag.constant.responses import (
    LANG_FULL_NAME,
    NO_RESULT_RESPONSE_MAP,
    OFF_TOPIC_RESPONSE_MAP,
    SUPPORTED_LANGS,
    UNSUPPORTED_LANGUAGE_MSG,
)
from loma_rag.llm.completion import (
    analyze_query,
    select_answer_model,
    translate_to_english_query,
)
from loma_rag.llm.streaming import complete_with_sentinel_detect
from loma_rag.llm.topic import is_insurance_topic
from loma_rag.prompt.builder import build_loma_user_prompt, build_web_user_prompt
from loma_rag.prompt.quiz_discussion import QUIZ_DISCUSSION_SYSTEM, QUIZ_WEB_SYSTEM
from loma_rag.rag.quiz_intent import (
    detect_end_session,
    detect_hint_request,
    parse_answer_fuzzy,
    parse_answer_letter,
    parse_answer_llm,
)
from loma_rag.rag.retriever import Retriever
from loma_rag.rag.web_fallback import WebFallback
from loma_rag.util.concurrency import BG_POOL


@dataclass
class QuizAnswerResult:
    intent: str = ""                  # "answer"|"hint"|"finish"|"question"|"off_topic"|"unsupported_language"
    path: str = ""                    # "answer"|"hint"|"finish"|"loma"|"web"|"no_result"|"refused"|"off_topic"|"unsupported_language"
    answer: Optional[str] = None      # "A"/"B"/"C"/"D" when intent=="answer"
    message: str = ""
    en_search_query: str = ""
    loma_chunks: list = field(default_factory=list)
    related_nodes: list = field(default_factory=list)
    direct_node_ids: list = field(default_factory=list)
    web_docs: list = field(default_factory=list)
    web_search_used: bool = False
    error: str = ""


def _off_topic_response(user_lang: str) -> str:
    return OFF_TOPIC_RESPONSE_MAP[LANG_FULL_NAME[user_lang]]


def _no_result_response(user_lang: str) -> str:
    return NO_RESULT_RESPONSE_MAP[LANG_FULL_NAME[user_lang]]


def _prep_web_docs(query: str, web_fallback: WebFallback, chat_client, web_k: int):
    en_query = translate_to_english_query(chat_client, query)
    docs = web_fallback.retrieve(en_query, top_k=web_k)
    return en_query, docs


def _options_pairs(options) -> list[tuple[str, str]]:
    """Convert QuizOption pydantic objects (or dicts) into (id, content) tuples."""
    out: list[tuple[str, str]] = []
    for o in options:
        oid = getattr(o, "id", None) or o["id"]
        content = getattr(o, "content", None) or o["content"]
        out.append((oid, content))
    return out


def _detect_intent(
    chat_client,
    question: str,
    options_pairs: list[tuple[str, str]],
    query: str,
    user_lang: str,
) -> tuple[str, Optional[str]]:
    """Return (intent, answer_letter_or_None). Empty intent means caller
    should fall through to off-topic + discussion."""
    if detect_end_session(query):
        return ("finish", None)
    if detect_hint_request(query):
        return ("hint", None)

    letter = parse_answer_letter(query)
    if letter:
        return ("answer", letter)

    # Lang-gated free-text answer parsing.
    question_lang, _ = analyze_query(chat_client, question)
    if user_lang == question_lang:
        letter = parse_answer_fuzzy(query, options_pairs)
        if letter:
            return ("answer", letter)
        letter = parse_answer_llm(chat_client, question, query, options_pairs)
        if letter:
            return ("answer", letter)
    return ("", None)


EARLY_EXIT_RERANK_THRESHOLD = rag_cfg.early_exit_rerank
EARLY_EXIT_RRF_THRESHOLD = rag_cfg.early_exit_rrf
HIGH_CONFIDENCE_RERANK_THRESHOLD = rag_cfg.high_confidence_rerank
HIGH_CONFIDENCE_RRF_THRESHOLD = rag_cfg.high_confidence_rrf


def _inject_quiz_context(
    base_system: str, question: str, options_pairs: list[tuple[str, str]]
) -> str:
    opts_block = "\n".join(f"{oid}. {content}" for oid, content in options_pairs)
    header = f"QUIZ QUESTION:\n{question}\n\nOPTIONS:\n{opts_block}\n\n"
    return header + base_system


def _run_discussion(
    retriever: Retriever,
    chat_client,
    web_fallback: Optional[WebFallback],
    question: str,
    options_pairs: list[tuple[str, str]],
    query: str,
    user_lang: str,
    search_text: str,
    top_k: int,
    web_k: int,
    res: QuizAnswerResult,
) -> None:
    retr = retriever.retrieve(query, top_k=top_k, user_lang=user_lang,
                              search_text=search_text)
    res.loma_chunks = list(retr.chunks)
    res.related_nodes = list(retr.related_nodes)
    res.direct_node_ids = list(getattr(retr, "direct_node_ids", []) or [])

    top_rerank = retr.chunks[0].rerank_score if retr.chunks else float("-inf")
    top_rrf = retr.chunks[0].fused_score if retr.chunks else 0.0
    early_exit = (
        web_fallback is not None
        and top_rerank < EARLY_EXIT_RERANK_THRESHOLD
        and top_rrf < EARLY_EXIT_RRF_THRESHOLD
    )
    high_confidence = (
        top_rerank >= HIGH_CONFIDENCE_RERANK_THRESHOLD
        or top_rrf >= HIGH_CONFIDENCE_RRF_THRESHOLD
    )

    spec_future: "Future | None" = None
    if web_fallback is not None and not early_exit and not high_confidence:
        spec_future = BG_POOL.submit(_prep_web_docs, query, web_fallback,
                                     chat_client, web_k)

    answer_model = select_answer_model(query)
    quiz_system = _inject_quiz_context(QUIZ_DISCUSSION_SYSTEM, question, options_pairs)

    if early_exit:
        text, insufficient = "", True
    else:
        user_prompt = build_loma_user_prompt(retr, user_lang=user_lang)
        text, insufficient = complete_with_sentinel_detect(
            chat_client, quiz_system, user_prompt, model=answer_model,
        )

    if not insufficient:
        if spec_future is not None:
            spec_future.cancel()
        res.intent = "question"
        res.path = "loma"
        res.message = text
        return

    if web_fallback is None:
        res.intent = "question"
        res.path = "refused"
        res.message = ""
        return

    if spec_future is not None:
        try:
            en_query, web_docs = spec_future.result(timeout=60)
        except Exception as e:  # noqa: BLE001
            res.intent = "question"
            res.path = "no_result"
            res.message = _no_result_response(user_lang)
            res.error = f"{type(e).__name__}: {e}"
            return
    else:
        en_query = translate_to_english_query(chat_client, query)
        try:
            web_docs = web_fallback.retrieve(en_query, top_k=web_k)
        except Exception as e:  # noqa: BLE001
            res.intent = "question"
            res.path = "no_result"
            res.message = _no_result_response(user_lang)
            res.error = f"{type(e).__name__}: {e}"
            return

    res.en_search_query = en_query
    if not web_docs:
        res.intent = "question"
        res.path = "no_result"
        res.message = _no_result_response(user_lang)
        return

    res.web_docs = list(web_docs)
    res.web_search_used = True
    quiz_web_system = _inject_quiz_context(QUIZ_WEB_SYSTEM, question, options_pairs)
    web_user = build_web_user_prompt(query, web_docs, user_lang=user_lang)
    resp = chat_client.chat.completions.create(
        model=answer_model,
        messages=[
            {"role": "system", "content": quiz_web_system},
            {"role": "user", "content": web_user},
        ],
        temperature=0.2,
    )
    res.intent = "question"
    res.path = "web"
    res.message = resp.choices[0].message.content or ""


def run_quiz_chat(
    retriever: Retriever,
    chat_client,
    web_fallback: Optional[WebFallback],
    question: str,
    options,
    query: str,
    top_k: int = 7,
    web_k: int = 5,
) -> QuizAnswerResult:
    """Non-streaming quiz-chat orchestrator. See spec §4."""
    res = QuizAnswerResult()
    options_pairs = _options_pairs(options)

    # Step 1: language gate.
    user_lang, search_text = analyze_query(chat_client, query)
    if user_lang not in SUPPORTED_LANGS:
        res.intent = "unsupported_language"
        res.path = "unsupported_language"
        res.message = UNSUPPORTED_LANGUAGE_MSG
        return res

    # Steps 2-4: keyword + answer parsing.
    intent, answer_letter = _detect_intent(
        chat_client, question, options_pairs, query, user_lang,
    )
    if intent == "finish":
        res.intent = "finish"
        res.path = "finish"
        return res
    if intent == "hint":
        res.intent = "hint"
        res.path = "hint"
        return res
    if intent == "answer":
        res.intent = "answer"
        res.path = "answer"
        res.answer = answer_letter
        return res

    # Step 5: off-topic.
    if not is_insurance_topic(chat_client, query):
        res.intent = "off_topic"
        res.path = "off_topic"
        res.message = _off_topic_response(user_lang)
        return res

    # Step 6: discussion.
    _run_discussion(
        retriever, chat_client, web_fallback,
        question, options_pairs, query,
        user_lang, search_text, top_k, web_k, res,
    )
    return res
