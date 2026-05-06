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

from loma_rag.rag.quiz_intent import normalize_text  # noqa: E402


# --- normalize_text cases: (input, expected) ---
NORMALIZE_CASES: list[tuple[str, str]] = [
    ("  HINT!  ", "hint"),
    ("Kết Thúc.", "ket thuc"),
    ("Option (A).", "option a"),
    ("nộp\tbài", "nop bai"),
    ("", ""),
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

    elapsed = time.time() - t0
    print(f"\n=== {total_pass}/{total_count} passed in {elapsed:.1f}s ===")
    return 0 if total_pass == total_count else 1


if __name__ == "__main__":
    sys.exit(main())
