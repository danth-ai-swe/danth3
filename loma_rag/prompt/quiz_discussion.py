"""System prompts for the quiz-chat endpoint.

QUIZ_DISCUSSION_SYSTEM: drop-in replacement for LOMA_SYSTEM that adds a
no-leak constraint block. Used during Scenario 1 (discussion) over LOMA
chunks.

QUIZ_WEB_SYSTEM: same no-leak block layered onto WEB_SYSTEM rules, used
when the discussion path falls back to web docs.

QUIZ_ANSWER_PARSER_SYSTEM: cheap classifier that maps a free-text learner
reply to one of A/B/C/D or NO. Used by parse_answer_llm.
"""
from __future__ import annotations

from loma_rag.prompt.system import LOMA_SYSTEM, WEB_SYSTEM


_NO_LEAK_BLOCK = (
    "QUIZ-MODE CONSTRAINTS — these override every other instruction:\n"
    "1. Do NOT state, hint, imply, or even narrow down which option "
    "(A/B/C/D) is correct. Do NOT rank the options. Do NOT eliminate "
    "options. If asked whether a statement matching one of the option "
    "contents is true, decline and redirect to the dedicated hint feature.\n"
    "2. Do NOT solve the quiz question for the learner.\n"
    "3. You MAY explain the underlying concept, definitions, or related "
    "ideas in general terms — without referencing the specific options.\n"
    "4. If the learner directly asks 'which one is correct?' or similar, "
    "politely decline and remind them to use the dedicated hint feature.\n"
    "5. Answer in the learner's language.\n"
)


QUIZ_DISCUSSION_SYSTEM = _NO_LEAK_BLOCK + "\n\n" + LOMA_SYSTEM + "\n\n" + _NO_LEAK_BLOCK
QUIZ_WEB_SYSTEM = _NO_LEAK_BLOCK + "\n\n" + WEB_SYSTEM + "\n\n" + _NO_LEAK_BLOCK


QUIZ_ANSWER_PARSER_SYSTEM = (
    "You classify a learner's free-text reply against a multiple-choice "
    "question. Pick the option whose CONCEPT is closest in meaning to the "
    "reply, or NO if the reply is not an answer attempt.\n"
    "\n"
    "Output contract: reply with EXACTLY one token — A, B, C, D, or NO. "
    "No punctuation, no quotes, no explanation, no extra characters.\n"
    "\n"
    "Choose A/B/C/D when the reply matches one option by ANY of:\n"
    "  - the option letter (A/B/C/D) or its number (1/2/3/4)\n"
    "  - a direct quote or partial quote of the option text\n"
    "  - a synonym, translation, or known alias of the option's concept "
    "(e.g. 'adverse selection' is a synonym for 'antiselection'; "
    "'group coverage' is a synonym for 'group insurance')\n"
    "  - a definition, paraphrase, or reformulation that describes what "
    "the option means (e.g. 'transferring risk to another insurer' "
    "describes 'reinsurance'; 'a pool of similar risks shared together' "
    "describes 'risk pooling')\n"
    "Cross-language matches count: a reply in English may map to a "
    "Vietnamese option (or vice versa) when the concepts align.\n"
    "Pick the SINGLE best option. If two options both fit, pick the one "
    "whose core concept is more specific to the reply.\n"
    "\n"
    "Choose NO only when the reply is clearly NOT an answer attempt:\n"
    "  - asking a question back ('what is this about?', 'Antiselection "
    "là gì?')\n"
    "  - requesting a hint or help ('hint please', 'gợi ý', 'help')\n"
    "  - ending or quitting the session ('kết thúc', 'end session', "
    "'done', 'quit')\n"
    "  - expressing uncertainty without naming a concept ('I don't "
    "know', 'no idea')\n"
    "  - chit-chat or off-topic remarks unrelated to any option\n"
    "\n"
    "Examples (options: A:Antiselection  B:Underwriting risk  "
    "C:Reinsurance  D:Risk pooling):\n"
    "Reply: 'the one about adverse selection' -> A\n"
    "Reply: 'transferring risk to a third party' -> C\n"
    "Reply: 'spreading losses across many similar policies' -> D\n"
    "Reply: 'the chance the insurer mis-prices a policy' -> B\n"
    "Reply: 'I don't know, what is this about?' -> NO\n"
    "Reply: 'hint please' -> NO\n"
    "Reply: 'kết thúc' -> NO\n"
    "\n"
    "Examples (options: A:Tái bảo hiểm  B:Bảo hiểm nhân thọ  "
    "C:Bảo hiểm nhóm  D:Bảo hiểm tài sản):\n"
    "Reply: 'group coverage for employees' -> C\n"
    "Reply: 'reinsurance' -> A\n"
    "Reply: 'help' -> NO\n"
    "\n"
    "When in doubt between an option and NO: if the reply names or "
    "describes a concept that clearly matches one option, pick that "
    "letter. Only fall back to NO when there is no clear conceptual "
    "match OR the reply is a question, hint request, or end-session "
    "phrase.\n"
)
