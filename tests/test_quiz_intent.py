"""Quiz-intent classifier tests for is_quiz_query.

Calls the classifier directly (no retriever / web) so iteration on the
prompt is fast and cheap. Run from the project root:

    # Bash / PowerShell — make sure the project root is on PYTHONPATH:
    PYTHONPATH=. python tests/test_quiz_intent.py
    PYTHONPATH=. python tests/test_quiz_intent.py -v   # print every case
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
# Make the package importable when run as a plain script (PYTHONPATH=.).
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
load_dotenv(HERE / ".env")

from loma_rag.llm.openai_client import make_chat_client  # noqa: E402
from loma_rag.llm.topic import is_quiz_query             # noqa: E402

# Each case: (id, query, expected_is_quiz)
# Positive: a real quiz / practice-question request -> True
# Negative: informational/definition/comparison about LOMA/insurance -> False
#           (including queries that mention "quiz"/"test" but aren't requests)
CASES: list[tuple[str, str, bool]] = [
    # ---------- positive (should classify as quiz) ----------
    ("vi_quiz_basic",
     "Cho tôi vài câu hỏi quiz về underwriting.",
     True),
    ("vi_quiz_module",
     "Tạo cho tôi bài kiểm tra trắc nghiệm về Module 1 LOMA 281.",
     True),
    ("vi_quiz_review",
     "Cho tôi 5 câu hỏi ôn tập về risk management.",
     True),
    ("vi_quiz_practice",
     "Mình muốn luyện đề trắc nghiệm phần annuity, ra đề giúp mình.",
     True),
    ("en_quiz_basic",
     "Give me a quiz on annuities.",
     True),
    ("en_quiz_test",
     "Test my knowledge on reinsurance with a few multiple-choice questions.",
     True),
    ("en_quiz_practice",
     "Generate practice questions about LOMA 291 module 2.",
     True),
    ("ja_quiz_basic",
     "保険についてクイズを出してください。",
     True),

    # ---------- negative (should NOT classify as quiz) ----------
    ("neg_def_antiselection",
     "Antiselection là gì? Tại sao underwriter cần quan tâm tới nó?",
     False),
    ("neg_def_reinsurance",
     "Reinsurance là gì và ai nhận rủi ro?",
     False),
    ("neg_compare_term_whole",
     "So sánh term life insurance và whole life insurance.",
     False),
    ("neg_explain_underwriting",
     "Explain the role of underwriting in insurance.",
     False),
    ("neg_meaning_of_quiz_word",
     "Trong khóa LOMA 281 có những bài quiz nào và chúng dùng để làm gì?",
     False),  # asks ABOUT quizzes, not asking the bot to give one
    ("neg_test_word_informational",
     "What does it mean when an underwriter says a case requires further testing?",
     False),  # contains "test" but is informational
    ("neg_course_structure",
     "LOMA 281 có bao nhiêu module và bao nhiêu lesson?",
     False),
    ("neg_advice",
     "Tôi 35 tuổi đã có gia đình, nên mua loại bảo hiểm nào?",
     False),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="print expected vs actual for every case")
    args = ap.parse_args()

    client = make_chat_client()

    n_pass = 0
    fails: list[tuple[str, str, bool, bool]] = []
    t0 = time.time()
    for cid, query, expected in CASES:
        actual = is_quiz_query(client, query)
        ok = actual == expected
        n_pass += int(ok)
        if not ok:
            fails.append((cid, query, expected, actual))
        if args.verbose or not ok:
            mark = "PASS" if ok else "FAIL"
            print(f"  [{mark}] {cid:32}  expected={expected}  actual={actual}")
            if not ok or args.verbose:
                print(f"         Q: {query}")

    elapsed = time.time() - t0
    print()
    print(f"=== Quiz-intent classifier: {n_pass}/{len(CASES)} "
          f"passed in {elapsed:.1f}s ===")
    if fails:
        print("\nFailures:")
        for cid, q, exp, got in fails:
            print(f"  - {cid}: expected={exp} got={got}\n      {q}")
    return 0 if n_pass == len(CASES) else 1


if __name__ == "__main__":
    sys.exit(main())
