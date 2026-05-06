"""Translation prompts: user-language → English query, and search-translation."""

TRANSLATE_SYSTEM = (
    "Convert the user's question into a concise English search query "
    "(3-12 words) using insurance industry terminology when relevant. "
    "Output ONLY the query — no quotes, no preamble."
)

TRANSLATE_FOR_SEARCH_SYSTEM = (
    "Translate the user's question into English while preserving its meaning "
    "and question form. Use life-insurance industry terminology where "
    "appropriate (e.g., 'rủi ro' -> 'risk', 'bảo hiểm nhân thọ' -> 'life "
    "insurance', 'tử kỳ' -> 'term', 'niên kim' -> 'annuity'). "
    "If the input is already English, return it unchanged. "
    "Output ONLY the English question — no quotes, no preamble, no explanation."
)
