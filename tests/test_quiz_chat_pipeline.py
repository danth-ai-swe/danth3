"""End-to-end smoke tests for run_quiz_chat.

Slow — boots Retriever + WebFallback. Run from the project root:

    .venv/Scripts/python.exe tests/test_quiz_chat_pipeline.py
    .venv/Scripts/python.exe tests/test_quiz_chat_pipeline.py --no-fallback
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
load_dotenv(HERE / ".env")

from loma_rag.llm.openai_client import make_chat_client  # noqa: E402
from loma_rag.rag.quiz_chat import run_quiz_chat          # noqa: E402
from loma_rag.rag.retriever import Retriever              # noqa: E402
from loma_rag.rag.web_fallback import WebFallback         # noqa: E402

# Heuristic phrases that would mean the model leaked the answer.
LEAK_PATTERNS = [
    "the answer is",
    "the correct answer",
    "correct option is",
    "đáp án đúng",
    "đáp án là",
    "answer:",
]

QUESTION_EN = "Which term describes the tendency of higher-risk applicants to seek insurance more aggressively?"
OPTS_EN = [
    {"id": "A", "content": "Antiselection"},
    {"id": "B", "content": "Underwriting risk"},
    {"id": "C", "content": "Reinsurance"},
    {"id": "D", "content": "Risk pooling"},
]

# Each case: dict with at minimum {id, query, expects: {intent, answer?, no_leak?}}
CASES: list[dict] = [
    {
        "id": "discussion_concept_en",
        "query": "Can you explain what antiselection means in life insurance?",
        "expects": {"intent": "question", "no_leak": True},
    },
    {
        "id": "discussion_request_for_correct_answer",
        "query": "Just tell me which one is correct.",
        "expects": {"intent": "question", "no_leak": True},
    },
    {
        "id": "answer_letter",
        "query": "B",
        "expects": {"intent": "answer", "answer": "B"},
    },
    {
        "id": "finish",
        "query": "Kết thúc phiên",
        "expects": {"intent": "finish"},
    },
    {
        "id": "hint",
        "query": "hint",
        "expects": {"intent": "hint"},
    },
    {
        "id": "off_topic",
        "query": "Cách nấu phở bò ngon nhất là gì?",
        "expects": {"intent": "off_topic"},
    },
]


def check(case: dict, result) -> list[str]:
    fails: list[str] = []
    exp = case["expects"]
    if result.intent != exp["intent"]:
        fails.append(f"intent: expected {exp['intent']!r}, got {result.intent!r}")
    if "answer" in exp and result.answer != exp["answer"]:
        fails.append(f"answer: expected {exp['answer']!r}, got {result.answer!r}")
    if exp.get("no_leak"):
        msg = (result.message or "").lower()
        for p in LEAK_PATTERNS:
            if p in msg:
                fails.append(f"no_leak: leaked phrase {p!r} in message")
                break
    return fails


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fallback", action="store_true")
    args = ap.parse_args()

    retriever = Retriever(rerank=True, expand_graph=True)
    chat_client = make_chat_client()
    web_fallback = None if args.no_fallback else WebFallback(dense_client=chat_client)

    n_pass = 0
    t0 = time.time()
    for case in CASES:
        try:
            result = run_quiz_chat(
                retriever, chat_client, web_fallback,
                QUESTION_EN, OPTS_EN, case["query"],
            )
        except Exception as e:  # noqa: BLE001
            print(f"[ERROR] {case['id']}: {type(e).__name__}: {e}")
            continue
        fails = check(case, result)
        ok = not fails
        n_pass += int(ok)
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {case['id']:32}  intent={result.intent}  answer={result.answer}")
        if fails:
            for f in fails:
                print(f"    - {f}")
    elapsed = time.time() - t0
    print(f"\n=== {n_pass}/{len(CASES)} passed in {elapsed:.1f}s ===")
    return 0 if n_pass == len(CASES) else 1


if __name__ == "__main__":
    sys.exit(main())
