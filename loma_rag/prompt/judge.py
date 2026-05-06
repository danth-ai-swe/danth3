"""Eval judge prompts (LLM-as-judge, closed/open book)."""
from __future__ import annotations

JUDGE_SYSTEM = (
    "You are answering a multiple-choice question from a LOMA insurance training "
    "course. Use ONLY the provided context excerpts. Reply with ONLY the digit of "
    "the correct option (1, 2, 3, or 4). No explanation, no other text."
)

CLOSED_BOOK_SYSTEM = (
    "You are answering a multiple-choice question from a LOMA insurance training "
    "course. Use your general knowledge of life insurance, annuities, and insurance "
    "company operations. Reply with ONLY the digit of the correct option "
    "(1, 2, 3, or 4). No explanation, no other text."
)


def build_judge_user(quiz: dict, chunks: list, related_nodes: list) -> str:
    parts = [f"Question:\n{quiz['question']}\n", "Options:"]
    for i, opt in enumerate(quiz["options"], 1):
        parts.append(f"{i}) {opt}")
    parts.append("\nContext excerpts:")
    for c in chunks:
        sub = f" > {c.subsection}" if c.subsection else ""
        parts.append(f"[{c.chunk_id} | {c.lesson_id} | {c.section}{sub}]\n{c.text}\n")
    if related_nodes:
        parts.append("Related concepts:")
        for n in related_nodes:
            parts.append(f"- {n.name}: {n.definition}")
    parts.append("\nReply with ONLY the digit (1, 2, 3, or 4).")
    return "\n".join(parts)
