"""Eval harness: measure retrieval recall + end-to-end answer accuracy.

Uses the 190 quiz items in out/quizzes.jsonl as a held-out test set.

Two metrics:
  1) lesson-recall@k — for each quiz, retrieve top-k chunks; success if any
     chunk's lesson_id matches the quiz's lesson_id. (Retrieval-only, no LLM.)
  2) answer-accuracy — feed question + 4 options + retrieved context to gpt-4o
     and check whether it picks the gold option. (Full pipeline; costs tokens.)

Results saved to out/eval_results.jsonl. Summary printed to stdout.

Run:
    python scripts/eval.py                 # full eval (retrieval + LLM judge)
    python scripts/eval.py --retrieval-only
    python scripts/eval.py --no-graph --no-rerank   # ablation
    python scripts/eval.py --limit 30      # quick run on first 30 items
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
load_dotenv(HERE / ".env")

from loma_rag.llm.openai_client import make_chat_client  # noqa: E402
from loma_rag.model.domain import RetrievalResult  # noqa: E402
from loma_rag.prompt.judge import (  # noqa: E402
    CLOSED_BOOK_SYSTEM,
    JUDGE_SYSTEM,
    build_judge_user,
)
from loma_rag.rag.retriever import Retriever  # noqa: E402

OUT = HERE / "out"
RECALL_KS = [1, 3, 5, 10, 20]
JUDGE_TOP_K = 5  # chunks fed to LLM judge
import os
CHAT_MODEL = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o")


@dataclass
class EvalRow:
    quiz_id: str
    lesson_id: str
    course: str
    difficulty: str
    correct_idx: int
    retrieved_lessons: list[str] = field(default_factory=list)
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    rank_of_lesson: int = -1   # 1-based rank of first chunk matching lesson_id, or -1
    llm_pick: int = 0          # 1..4, 0 if parse failure
    llm_correct: bool = False
    llm_raw: str = ""
    error: str = ""


# ---- LLM judge ----

_DIGIT_RE = re.compile(r"\b([1-4])\b")


def parse_pick(reply: str) -> int:
    m = _DIGIT_RE.search(reply.strip())
    return int(m.group(1)) if m else 0


def llm_judge(client, quiz: dict, result: RetrievalResult) -> tuple[int, str]:
    user = build_judge_user(quiz, result.chunks[:JUDGE_TOP_K], result.related_nodes)
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        max_tokens=8,
    )
    raw = resp.choices[0].message.content or ""
    return parse_pick(raw), raw


def llm_judge_closed_book(client, quiz: dict) -> tuple[int, str]:
    parts = [f"Question:\n{quiz['question']}\n", "Options:"]
    for i, opt in enumerate(quiz["options"], 1):
        parts.append(f"{i}) {opt}")
    parts.append("\nReply with ONLY the digit (1, 2, 3, or 4).")
    user = "\n".join(parts)
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": CLOSED_BOOK_SYSTEM},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        max_tokens=8,
    )
    raw = resp.choices[0].message.content or ""
    return parse_pick(raw), raw


# ---- driver ----

def load_quizzes() -> list[dict]:
    path = OUT / "quizzes.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing {path} — run ingest.py first")
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def evaluate(
    quizzes: list[dict],
    retriever: Retriever | None,
    chat_client,
    do_llm_judge: bool,
    verbose: bool,
    closed_book: bool = False,
) -> list[EvalRow]:
    rows: list[EvalRow] = []
    n = len(quizzes)
    t_start = time.time()
    max_k = max(RECALL_KS)

    for i, q in enumerate(quizzes, 1):
        row = EvalRow(
            quiz_id=q["quiz_id"],
            lesson_id=q["lesson_id"],
            course=q["course"],
            difficulty=q.get("difficulty", "").strip(),
            correct_idx=q["correct_idx"],
        )
        try:
            if closed_book:
                # No retrieval. LLM answers from prior knowledge.
                pick, raw = llm_judge_closed_book(chat_client, q)
                row.llm_pick = pick
                row.llm_raw = raw
                row.llm_correct = (pick == row.correct_idx) and pick > 0
            else:
                # Retrieve max_k for recall@1..max_k.
                result = retriever.retrieve(q["question"], top_k=max_k)
                row.retrieved_chunk_ids = [c.chunk_id for c in result.chunks]
                row.retrieved_lessons = [c.lesson_id for c in result.chunks]

                for rank, lid in enumerate(row.retrieved_lessons, 1):
                    if lid == row.lesson_id:
                        row.rank_of_lesson = rank
                        break

                if do_llm_judge:
                    # Reuse the SAME retrieval result; LLM gets top JUDGE_TOP_K only.
                    pick, raw = llm_judge(chat_client, q, result)
                    row.llm_pick = pick
                    row.llm_raw = raw
                    row.llm_correct = (pick == row.correct_idx) and pick > 0

        except Exception as e:  # noqa: BLE001
            row.error = f"{type(e).__name__}: {e}"

        rows.append(row)

        if verbose or i % 10 == 0 or i == n:
            elapsed = time.time() - t_start
            eta = (elapsed / i) * (n - i)
            tag = "✓" if row.rank_of_lesson == 1 else f"r={row.rank_of_lesson}"
            extra = ""
            if do_llm_judge:
                extra = "  llm✓" if row.llm_correct else (f"  llm={row.llm_pick}≠{row.correct_idx}" if not row.error else "")
            err = f"  ERR {row.error}" if row.error else ""
            print(f"  [{i:>3}/{n}] {row.quiz_id:24} {tag:>5}{extra}{err}  "
                  f"({elapsed:.0f}s, eta {eta:.0f}s)")

    return rows


# ---- summary ----

def summarize(rows: list[EvalRow], do_llm_judge: bool, closed_book: bool = False) -> None:
    total = len(rows)
    valid = [r for r in rows if not r.error]
    err_count = total - len(valid)

    print(f"\n=== Summary (n={total}, errors={err_count}) ===\n")

    if closed_book:
        # No retrieval to summarize.
        pass
    else:
        # Lesson-recall@k overall.
        print("Lesson-recall@k (does any of top-k chunks match the quiz's lesson_id?)")
        for k in RECALL_KS:
            hits = sum(1 for r in valid if 0 < r.rank_of_lesson <= k)
            print(f"  recall@{k:>2} = {hits/len(valid):.3f}  ({hits}/{len(valid)})")

    by_diff: dict[str, list[EvalRow]] = defaultdict(list)
    for r in valid:
        by_diff[r.difficulty or "unknown"].append(r)
    by_course: dict[str, list[EvalRow]] = defaultdict(list)
    for r in valid:
        by_course[r.course].append(r)

    if not closed_book:
        # MRR (mean reciprocal rank) over lessons.
        rrs = [1.0 / r.rank_of_lesson for r in valid if r.rank_of_lesson > 0]
        not_found = len(valid) - len(rrs)
        mrr = (sum(rrs) + 0.0 * not_found) / len(valid) if valid else 0.0
        print(f"  MRR    = {mrr:.3f}  (lesson never found in top-20: {not_found}/{len(valid)})")

        # By difficulty.
        print("\nLesson-recall@5 by difficulty:")
        for diff in sorted(by_diff):
            rs = by_diff[diff]
            hits = sum(1 for r in rs if 0 < r.rank_of_lesson <= 5)
            print(f"  {diff:<14} {hits/len(rs):.3f}  ({hits}/{len(rs)})")

        # By course.
        print("\nLesson-recall@5 by course:")
        for course in sorted(by_course):
            rs = by_course[course]
            hits = sum(1 for r in rs if 0 < r.rank_of_lesson <= 5)
            print(f"  {course:<10} {hits/len(rs):.3f}  ({hits}/{len(rs)})")

    if not do_llm_judge:
        return

    # End-to-end accuracy.
    label = "closed-book LLM accuracy" if closed_book else "LLM answer accuracy"
    print(f"\n{label}:")
    answered = [r for r in valid if r.llm_pick > 0]
    correct = sum(1 for r in valid if r.llm_correct)
    print(f"  overall    {correct/len(valid):.3f}  ({correct}/{len(valid)})")
    parse_fail = len(valid) - len(answered)
    if parse_fail:
        print(f"  parse-fail {parse_fail} (LLM reply did not contain a digit 1-4)")

    if not closed_book:
        # Confusion: lesson-found AND llm-correct?
        found = [r for r in valid if 0 < r.rank_of_lesson <= 5]
        notfound = [r for r in valid if not (0 < r.rank_of_lesson <= 5)]
        if found:
            c1 = sum(1 for r in found if r.llm_correct)
            print(f"  when lesson IS in top-5: accuracy = {c1/len(found):.3f}  ({c1}/{len(found)})")
        if notfound:
            c2 = sum(1 for r in notfound if r.llm_correct)
            print(f"  when lesson NOT in top-5: accuracy = {c2/len(notfound):.3f}  ({c2}/{len(notfound)})")

    # By difficulty.
    print("\nLLM accuracy by difficulty:")
    for diff in sorted(by_diff):
        rs = by_diff[diff]
        c = sum(1 for r in rs if r.llm_correct)
        print(f"  {diff:<14} {c/len(rs):.3f}  ({c}/{len(rs)})")

    # By course.
    print("\nLLM accuracy by course:")
    for course in sorted(by_course):
        rs = by_course[course]
        c = sum(1 for r in rs if r.llm_correct)
        print(f"  {course:<10} {c/len(rs):.3f}  ({c}/{len(rs)})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval-only", action="store_true",
                    help="skip LLM judge stage")
    ap.add_argument("--closed-book", action="store_true",
                    help="LLM answers from prior knowledge only — no retrieval")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--limit", type=int, default=0,
                    help="evaluate only the first N quizzes (0 = all)")
    ap.add_argument("--shuffle-seed", type=int, default=0,
                    help="if non-zero, shuffle quizzes with this seed before --limit")
    ap.add_argument("--verbose", action="store_true",
                    help="print per-quiz progress (default: every 10)")
    ap.add_argument("--out", default="eval_results.jsonl")
    args = ap.parse_args()

    quizzes = load_quizzes()
    print(f"loaded {len(quizzes)} quiz items")

    if args.shuffle_seed:
        import random
        random.Random(args.shuffle_seed).shuffle(quizzes)

    if args.limit > 0:
        quizzes = quizzes[: args.limit]
        print(f"  limited to first {len(quizzes)}")

    if args.closed_book:
        print("config: CLOSED-BOOK (no retrieval)")
        retriever = None
        chat_client = make_chat_client()
        do_judge = True
    else:
        print(
            f"config: rerank={not args.no_rerank} "
            f"graph={not args.no_graph} llm_judge={not args.retrieval_only}"
        )
        print("\ninit retriever (lazy: BM42 + reranker download/load on first call)…")
        retriever = Retriever(
            rerank=not args.no_rerank,
            expand_graph=not args.no_graph,
        )
        chat_client = make_chat_client() if not args.retrieval_only else None
        do_judge = not args.retrieval_only

    print("\nrunning eval…")
    rows = evaluate(
        quizzes, retriever, chat_client, do_judge, args.verbose,
        closed_book=args.closed_book,
    )

    out_path = OUT / args.out
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    print(f"\nwrote {out_path}")

    summarize(rows, do_llm_judge=do_judge, closed_book=args.closed_book)
    return 0


if __name__ == "__main__":
    sys.exit(main())
