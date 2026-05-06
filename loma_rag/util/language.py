"""Language code/name helpers and heuristic-based user-language detection."""
from __future__ import annotations

import re

LANG_NAMES = {
    "en": "English",
    "vi": "Vietnamese (tiếng Việt)",
    "fr": "French",
    "ja": "Japanese",
    "zh": "Chinese (Mandarin)",
    "es": "Spanish",
    "de": "German",
    "ko": "Korean",
    "th": "Thai",
    "pt": "Portuguese",
    "ru": "Russian",
    "it": "Italian",
}

LANG_CODE_RE = re.compile(r"^[a-z]{2}$")


def language_name(code: str) -> str:
    return LANG_NAMES.get(code, code)


def detect_user_language(text: str) -> str:
    """Heuristic Vietnamese-vs-English detection from character set.

    Cheap fallback for code paths without an LLM client; prefer
    `loma_rag.llm.completion.detect_language_llm()` when one is available.
    """
    vn = set(
        "ảẩẳẻểỉỏổởủửỷ"
        "ãẫẵẽễĩõỗỡũữỹ"
        "ạậặẹệịọộợụựỵ"
        "ăằắẳẵặ"
        "ơờớợởỡưừứửữự"
        "đ"
    )
    return "vi" if any(c in vn for c in (text or "").lower()) else "en"
