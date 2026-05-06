"""Query-analysis and language-detect prompts + parsing regexes."""
import re

LANG_DETECT_SYSTEM = (
    "Identify the language the user wants the response in. "
    "If the user explicitly asks for a specific output language (e.g. 'answer in English', "
    "'reply in Vietnamese', 'trả lời bằng tiếng Việt', 'répondez en français'), "
    "return that requested language. Otherwise return the language of the question itself. "
    "Reply with ONLY a 2-letter ISO 639-1 code in lowercase "
    "(e.g. en, vi, fr, ja, zh, es, de, ko, th). "
    "Output the code only — no quotes, no punctuation, no other text."
)

# Combined detect-and-translate prompt — saves 1 LLM round-trip on every
# non-English query versus running the two stages separately.
ANALYZE_QUERY_SYSTEM = (
    "You are a 2-step query analyzer. For each input, return:\n"
    "  (a) LANG — the 2-letter ISO 639-1 code of the language the user wants "
    "the ANSWER in. CRITICALLY: if the input contains an explicit instruction "
    "to answer in a specific language, that overrides the input's own language.\n"
    "  (b) TRANSLATION — an English version of the question for vector search. "
    "Use life-insurance industry terminology where appropriate.\n"
    "\n"
    "EXPLICIT OVERRIDE PHRASES YOU MUST RECOGNISE (case-insensitive, partial match):\n"
    "  - 'answer in English' / 'reply in English' / 'in English only' → en\n"
    "  - 'trả lời bằng tiếng Anh' / 'tiếng Anh' / 'in English' (Vietnamese context) → en\n"
    "  - 'trả lời bằng tiếng Việt' / 'tiếng Việt' / 'in Vietnamese' → vi\n"
    "  - 'répondez en français' / 'in French' → fr\n"
    "If ANY such phrase appears, LANG = the requested language, NOT the input's language.\n"
    "If NO such phrase, LANG = the language the question is written in.\n"
    "\n"
    "Output format: a single line `LANG|TRANSLATION` — no preamble, no quotes.\n"
    "\n"
    "Examples:\n"
    "  Input: What is risk?\n"
    "  Output: en|What is risk?\n"
    "\n"
    "  Input: Rủi ro là gì?\n"
    "  Output: vi|What is risk?\n"
    "\n"
    "  Input: Bảo hiểm nhân thọ là gì? Trả lời bằng tiếng Anh.\n"
    "  Output: en|What is life insurance?\n"
    "\n"
    "  Input: Antiselection là gì? Please answer in English only.\n"
    "  Output: en|What is antiselection?\n"
    "\n"
    "  Input: What is reinsurance? Trả lời bằng tiếng Việt.\n"
    "  Output: vi|What is reinsurance?\n"
    "\n"
    "  Input: bảo hiểm nhân thọ là gì trả lời bằng tiếng anh\n"
    "  Output: en|What is life insurance?\n"
)

ANALYZE_RE = re.compile(r"^([a-zA-Z]{2})\s*\|\s*(.+)$", re.S)
