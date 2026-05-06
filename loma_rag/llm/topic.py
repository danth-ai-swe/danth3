"""Insurance-topic classifier wrapping the OpenAI chat completions API."""
from __future__ import annotations

from loma_rag.config.settings import azure
from loma_rag.constant.thresholds import ANALYZE_CACHE_MAX
from loma_rag.prompt.topic import TOPIC_CLASSIFIER_SYSTEM
from loma_rag.util.cache import LRUCache

_topic_cache = LRUCache(ANALYZE_CACHE_MAX)


def is_insurance_topic(client, query: str) -> bool:
    """Return True iff the query is on-topic for the insurance / LOMA assistant.

    Uses the cheaper detect-class model (gpt-4o-mini by default). Falls back to
    True on any LLM error so that ambiguous classification never silently
    blocks a legitimate user question.
    """
    if not query or not query.strip():
        return True
    cached = _topic_cache.get(query)
    if cached is not None:
        return cached
    try:
        resp = client.chat.completions.create(
            model=azure.detect_model,
            messages=[
                {"role": "system", "content": TOPIC_CLASSIFIER_SYSTEM},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
            max_tokens=4,
        )
        out = (resp.choices[0].message.content or "").strip().upper()
        on_topic = out.startswith("YES")
    except Exception:  # noqa: BLE001
        # Fail-open: don't block users when the classifier itself misbehaves.
        on_topic = True
    _topic_cache.put(query, on_topic)
    return on_topic
