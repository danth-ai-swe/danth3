"""Pure intent-detection helpers for the quiz-chat endpoint.

These are stdlib-only utilities (no LLM, no I/O) — except parse_answer_llm
which is added in a later task. Keep them deterministic and side-effect
free so they can be unit-tested without the network.
"""
from __future__ import annotations

import re
import unicodedata


_PUNCT_TRIM_RE = re.compile(r'^[\s\.\,\!\?\;\:\"\'\(\)\[\]]+|[\s\.\,\!\?\;\:\"\'\(\)\[\]]+$')
_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Lowercase, collapse whitespace, strip leading/trailing punctuation,
    fold diacritics (NFD + drop combining marks).

    Used to make user prompts comparable to fixed keyword sets in
    detect_end_session and detect_hint_request, and to compute fuzzy
    similarity in parse_answer_fuzzy.
    """
    if not text:
        return ""
    s = text.strip().lower()
    s = _WS_RE.sub(" ", s)
    # Strip leading/trailing punctuation per-token so e.g. "Option (A)." -> "option a".
    s = " ".join(_PUNCT_TRIM_RE.sub("", tok) for tok in s.split(" ") if tok)
    # Fold diacritics: "kết thúc" -> "ket thuc", "đố" -> "do".
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    # Vietnamese đ/Đ are not combining; map separately.
    s = s.replace("đ", "d").replace("Đ", "d")
    return s
