"""Pure intent-detection helpers for the quiz-chat endpoint.

These are stdlib-only utilities (no LLM, no I/O) — except parse_answer_llm
which is added in a later task. Keep them deterministic and side-effect
free so they can be unit-tested without the network.
"""
from __future__ import annotations

import re
import unicodedata


_WS_RE = re.compile(r"\s+")


def _is_edge_strippable(ch: str) -> bool:
    """True for whitespace, separators, and any Unicode punctuation."""
    cat = unicodedata.category(ch)
    return cat[0] in ("P", "Z") or cat == "Cc"


def _strip_token(tok: str) -> str:
    i, j = 0, len(tok)
    while i < j and _is_edge_strippable(tok[i]):
        i += 1
    while j > i and _is_edge_strippable(tok[j - 1]):
        j -= 1
    return tok[i:j]


def normalize_text(text: str) -> str:
    """Lowercase, collapse whitespace, strip leading/trailing punctuation,
    fold diacritics (NFD + drop combining marks).

    Used to make user prompts comparable to fixed keyword sets in
    detect_end_session and detect_hint_request, and to compute fuzzy
    similarity in parse_answer_fuzzy.
    """
    if not text:
        return ""
    # Vietnamese đ/Đ are not combining; fold both cases before lowercasing
    # so the subsequent NFD path handles only diacritic marks.
    s = text.replace("Đ", "d").replace("đ", "d")
    s = s.strip().lower()
    s = _WS_RE.sub(" ", s)
    # Strip leading/trailing punctuation per-token (Unicode-aware) so e.g.
    # "Option (A)." -> "option a" and "Kết thúc。" -> "ket thuc".
    s = " ".join(t for t in (_strip_token(tok) for tok in s.split(" ")) if t)
    # Fold Latin diacritics: "kết thúc" -> "ket thuc". Restrict to the
    # "Combining Diacritical Marks" block (U+0300-U+036F) so that CJK
    # voicing marks (e.g. dakuten on ズ) survive — otherwise "クイズ"
    # would collapse to "クイス".
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if not 0x0300 <= ord(ch) <= 0x036F)
    s = unicodedata.normalize("NFC", s)
    return s
