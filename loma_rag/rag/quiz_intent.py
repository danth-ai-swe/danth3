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


import difflib

END_SESSION_KEYWORDS = (
    # multi-word phrases first
    "kết thúc phiên",
    "kết thúc buổi học",
    "submit and finish",
    "end session",
    "end quiz",
    "kết thúc",
    "nộp bài",
    "thoát",
    "dừng",
    "finish",
    "done",
    "quit",
    "exit",
    "stop",
)
_END_KEYWORDS_NORM = tuple(normalize_text(k) for k in END_SESSION_KEYWORDS)

# Tokens that, when present as a whole word, count as end-session intent.
# Multi-word phrases are matched as substrings instead.
_END_TOKEN_KEYWORDS = {
    normalize_text(k) for k in
    ("finish", "done", "quit", "exit", "stop", "thoát", "dừng", "nộp bài",
     "kết thúc")
}

# Question-mark heuristic: if the original text ends with a question mark,
# treat as a question, not a command. Keeps "what does 'finish' mean?" out.
_QUESTION_MARK_RE = re.compile(r"[?？]\s*$")


def detect_end_session(text: str) -> bool:
    """True iff the user's prompt is an end-session command.

    Match procedure:
    1. Reject if text ends with a question mark.
    2. Exact equality of the normalized text with any keyword.
    3. difflib.SequenceMatcher ratio >= 0.85 against any keyword
       (catches typos like 'kent thuc', 'ngp bai').
    4. Whole-token containment for short imperative keywords.
    """
    if not text or _QUESTION_MARK_RE.search(text):
        return False
    n = normalize_text(text)
    if not n:
        return False
    if n in _END_KEYWORDS_NORM:
        return True
    for kw in _END_KEYWORDS_NORM:
        if difflib.SequenceMatcher(None, n, kw).ratio() >= 0.85:
            return True
    tokens = set(n.split())
    for kw in _END_TOKEN_KEYWORDS:
        if " " in kw:
            if kw in n:
                return True
        elif kw in tokens:
            return True
    return False


HINT_KEYWORDS = (
    "cho tôi gợi ý",
    "cho toi mot hint",
    "give me a hint",
    "show me a hint",
    "can i have a hint",
    "gợi ý",
    "goi y",
    "hint",
    "hin",
    "help",
)
_HINT_KEYWORDS_NORM = tuple(normalize_text(k) for k in HINT_KEYWORDS)

# Token-set with Vietnamese pre-normalised so set membership works against
# the normalised user text. Multi-word keywords use substring containment.
_HINT_TOKEN_KEYWORDS = {
    normalize_text(k) for k in ("hint", "hin", "help", "goi y", "gợi ý")
}


def detect_hint_request(text: str) -> bool:
    """True iff the user's prompt is a hint request.

    Procedure mirrors detect_end_session (exact / fuzzy / token-containment)
    but does NOT skip on a trailing question mark — hint requests are often
    phrased politely as questions ('Can I have a hint?'). The cost is one
    accepted false-positive: 'what does "hint" mean in poker?' is
    classified as a hint request.
    """
    if not text:
        return False
    n = normalize_text(text)
    if not n:
        return False
    if n in _HINT_KEYWORDS_NORM:
        return True
    for kw in _HINT_KEYWORDS_NORM:
        if difflib.SequenceMatcher(None, n, kw).ratio() >= 0.85:
            return True
    tokens = set(n.split())
    for kw in _HINT_TOKEN_KEYWORDS:
        if " " in kw:
            if kw in n:
                return True
        elif kw in tokens:
            return True
    return False
