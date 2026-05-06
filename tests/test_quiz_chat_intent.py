"""Unit tests for quiz-chat intent detection.

Pure-function tests. Most tests need only stdlib; LLM-backed tests
(parse_answer_llm) require a chat client and are marked with
`requires_llm=True`. Run from project root:

    .venv/Scripts/python.exe tests/test_quiz_chat_intent.py
    .venv/Scripts/python.exe tests/test_quiz_chat_intent.py -v
    .venv/Scripts/python.exe tests/test_quiz_chat_intent.py --no-llm
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
# Note: LLM-backed cases (parse_answer_llm) are added in Task 8; the
# `--no-llm` flag and dotenv import are wired up here in advance.

from loma_rag.rag.quiz_intent import normalize_text  # noqa: E402
from loma_rag.rag.quiz_intent import detect_end_session  # noqa: E402


# --- normalize_text cases: (input, expected) ---
NORMALIZE_CASES: list[tuple[str, str]] = [
    ("  HINT!  ", "hint"),
    ("Kết Thúc.", "ket thuc"),
    ("Option (A).", "option a"),
    ("nộp\tbài", "nop bai"),
    ("", ""),
    ("a   b", "a b"),
    ("option(a)", "option(a"),
    ("Kết thúc。", "ket thuc"),
    ("Đố là quiz?", "do la quiz"),
    ("  ...kết thúc...  ", "ket thuc"),
    ("クイズ！", "クイズ"),
]


def run_normalize_tests(verbose: bool) -> tuple[int, int]:
    n_pass = 0
    for raw, expected in NORMALIZE_CASES:
        actual = normalize_text(raw)
        ok = actual == expected
        n_pass += int(ok)
        if verbose or not ok:
            mark = "PASS" if ok else "FAIL"
            print(f"  [{mark}] normalize_text({raw!r}) -> {actual!r}  expected={expected!r}")
    return n_pass, len(NORMALIZE_CASES)


# (input, expected_is_end_session)
END_SESSION_CASES: list[tuple[str, bool]] = [
    # positive
    ("Kết thúc", True),
    ("kết thúc phiên", True),
    ("KENT THUC", True),                    # typo, fuzzy
    ("end session", True),
    ("End Quiz!", True),
    ("submit and finish", True),
    ("done", True),
    ("quit", True),
    ("exit", True),
    ("stop", True),
    ("nộp bài", True),
    ("ngp bai", True),                       # typo per spec
    ("thoát", True),
    ("dừng lại đi", True),                   # contains "dung" token
    ("I want to finish now", True),          # contains "finish" token
    # negative
    ("How do I end a contract?", False),
    ("what does 'finish' mean here?", False),
    ("Antiselection là gì?", False),
    ("A", False),
    ("hint", False),
]


def run_end_session_tests(verbose: bool) -> tuple[int, int]:
    n_pass = 0
    for raw, expected in END_SESSION_CASES:
        actual = detect_end_session(raw)
        ok = actual == expected
        n_pass += int(ok)
        if verbose or not ok:
            mark = "PASS" if ok else "FAIL"
            print(f"  [{mark}] detect_end_session({raw!r}) -> {actual}  expected={expected}")
    return n_pass, len(END_SESSION_CASES)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--no-llm", action="store_true",
                    help="skip LLM-backed cases")
    args = ap.parse_args()

    t0 = time.time()
    total_pass = 0
    total_count = 0

    p, n = run_normalize_tests(args.verbose)
    total_pass += p; total_count += n
    print(f"normalize_text: {p}/{n}")

    p, n = run_end_session_tests(args.verbose)
    total_pass += p; total_count += n
    print(f"detect_end_session: {p}/{n}")

    elapsed = time.time() - t0
    print(f"\n=== {total_pass}/{total_count} passed in {elapsed:.1f}s ===")
    return 0 if total_pass == total_count else 1


if __name__ == "__main__":
    sys.exit(main())
