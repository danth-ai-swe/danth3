"""Canned user-facing responses for off-topic, unsupported-language, and no-result paths.

These are returned to the user verbatim, never sent to an LLM as a prompt.
Keys in the language maps are full English language names matching
LANG_FULL_NAME below.
"""
from __future__ import annotations

# Languages the assistant fully supports for canned responses.
SUPPORTED_LANGS = {"vi", "en", "ja"}

# 2-letter ISO 639-1 → full English language name (used as map key).
LANG_FULL_NAME = {
    "vi": "Vietnamese",
    "en": "English",
    "ja": "Japanese",
}


OFF_TOPIC_RESPONSE_MAP = {
    "Vietnamese": (
        "Insuripedia là một trợ lý AI chuyên về kiến thức bảo hiểm, "
        "vì vậy Insuripedia không thể trả lời câu hỏi của bạn về chủ đề đó. "
        "Bạn có câu hỏi nào liên quan đến bảo hiểm không?"
    ),
    "English": (
        "Insuripedia is an AI assistant specializing in insurance "
        "knowledge so Insuripedia cannot answer your question about "
        "that topic. Do you have a question related to insurance?"
    ),
    "Japanese": (
        "Insuripediaは保険知識に特化したAIアシスタントです。\n"
        "そのため、そのトピックに関するご質問にはお答えできません。\n"
        "保険に関するご質問はありますか？"
    ),
}


UNSUPPORTED_LANGUAGE_MSG = "Sorry, Insuripedia can't understand your language."


NO_RESULT_RESPONSE_MAP = {
    "Vietnamese": (
        "Xin lỗi. Insuripedia không thể tìm thấy thông tin nào về "
        "chủ đề bảo hiểm này trong kho dữ liệu."
    ),
    "English": (
        "Sorry. Insuripedia cannot find any information about this "
        "insurance-related topic in the knowledge hub."
    ),
    "Japanese": (
        "申し訳ありませんが、この保険関連のトピックについて\n"
        "ナレッジベース内に情報が見つかりませんでした。"
    ),
}
