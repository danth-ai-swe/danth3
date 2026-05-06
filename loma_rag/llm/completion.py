"""Synchronous + async helpers wrapping Azure OpenAI chat completions."""
from __future__ import annotations

import os

from loma_rag.config.settings import azure
from loma_rag.constant.thresholds import ANALYZE_CACHE_MAX, LANG_CACHE_MAX, TRANSLATE_CACHE_MAX
from loma_rag.prompt.analyzer import ANALYZE_QUERY_SYSTEM, ANALYZE_RE, LANG_DETECT_SYSTEM
from loma_rag.prompt.translate import TRANSLATE_SYSTEM
from loma_rag.util.cache import LRUCache
from loma_rag.util.language import LANG_CODE_RE

_translate_cache = LRUCache(TRANSLATE_CACHE_MAX)
_analyze_cache = LRUCache(ANALYZE_CACHE_MAX)
_lang_cache = LRUCache(LANG_CACHE_MAX)

_PREFER_BIG_MODEL = (
    os.environ.get("OPENAI_PREFER_BIG") == "1"
    or os.environ.get("OPENAI_SHORT_ANSWER_DISABLED") == "1"
)
# Only escalate to gpt-4o for queries that need genuine multi-step reasoning,
# not just lists or comparisons (mini handles those fine).
_BIG_MODEL_KEYWORDS = (
    "step by step", "step-by-step", "multi-step", "phân tích sâu",
    "deep analysis", "detailed analysis", "explain in detail",
    "comprehensive overview", "tổng quan toàn diện",
)


def select_answer_model(query: str) -> str:
    """Default to gpt-4o-mini; escalate to gpt-4o for deep-reasoning queries."""
    if _PREFER_BIG_MODEL:
        return azure.chat_model
    q = (query or "").lower()
    if any(k in q for k in _BIG_MODEL_KEYWORDS):
        return azure.chat_model
    # Long, free-form queries (>40 words) likely need full model.
    if len(q.split()) > 40:
        return azure.chat_model
    return azure.short_answer_model


def analyze_query(client, text: str) -> tuple[str, str]:
    """Single-call combined language detect + translate-to-English.
    Returns (lang_code, english_search_query). On parse failure, returns
    ('en', text) — safe default. Cached LRU by input text."""
    if not text or len(text.strip()) < 2:
        return ("en", text)
    sample = text.strip()
    cached = _analyze_cache.get(sample)
    if cached is not None:
        return cached
    try:
        resp = client.chat.completions.create(
            model=azure.detect_model,
            messages=[
                {"role": "system", "content": ANALYZE_QUERY_SYSTEM},
                {"role": "user", "content": sample},
            ],
            temperature=0.0,
            max_tokens=300,
        )
        out = (resp.choices[0].message.content or "").strip()
        m = ANALYZE_RE.match(out)
        if m:
            code = m.group(1).lower()
            translation = m.group(2).strip().strip("\"' ")
            _analyze_cache.put(sample, (code, translation))
            return (code, translation)
    except Exception:  # noqa: BLE001
        pass
    return ("en", text)


async def analyze_query_async(async_client, text: str) -> tuple[str, str]:
    """Async variant of analyze_query — same interface, uses AsyncAzureOpenAI."""
    if not text or len(text.strip()) < 2:
        return ("en", text)
    sample = text.strip()
    cached = _analyze_cache.get(sample)
    if cached is not None:
        return cached
    try:
        resp = await async_client.chat.completions.create(
            model=azure.detect_model,
            messages=[
                {"role": "system", "content": ANALYZE_QUERY_SYSTEM},
                {"role": "user", "content": sample},
            ],
            temperature=0.0,
            max_tokens=300,
        )
        out = (resp.choices[0].message.content or "").strip()
        m = ANALYZE_RE.match(out)
        if m:
            code = m.group(1).lower()
            translation = m.group(2).strip().strip("\"' ")
            _analyze_cache.put(sample, (code, translation))
            return (code, translation)
    except Exception:  # noqa: BLE001
        pass
    return ("en", text)


def detect_language_llm(client, text: str, fallback: str = "en") -> str:
    """Ask the chat model to return an ISO 639-1 code for the text's language.
    Cached by input text. Returns `fallback` on parse failure or empty input."""
    if not text or len(text.strip()) < 2:
        return fallback
    sample = text.strip()[:600]  # cap to keep tokens cheap
    cached = _lang_cache.get(sample)
    if cached is not None:
        return cached
    try:
        resp = client.chat.completions.create(
            model=azure.chat_model,
            messages=[
                {"role": "system", "content": LANG_DETECT_SYSTEM},
                {"role": "user", "content": sample},
            ],
            temperature=0.0,
            max_tokens=4,
        )
        code = (resp.choices[0].message.content or "").strip().lower()
        if LANG_CODE_RE.match(code):
            _lang_cache.put(sample, code)
            return code
    except Exception:  # noqa: BLE001
        pass
    return fallback


def translate_to_english_query(client, query: str) -> str:
    cached = _translate_cache.get(query)
    if cached is not None:
        return cached
    resp = client.chat.completions.create(
        model=azure.chat_model,
        messages=[
            {"role": "system", "content": TRANSLATE_SYSTEM},
            {"role": "user", "content": query},
        ],
        temperature=0.0,
        max_tokens=40,
    )
    text = (resp.choices[0].message.content or "").strip().strip("\"' ")
    _translate_cache.put(query, text)
    return text
