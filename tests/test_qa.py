"""Q&A test runner for the LOMA RAG chatbot.

Loads test cases from tests.qa_cases.TESTS, runs each through loma_rag.rag.pipeline.run_query,
asserts the expected behaviour, and prints a pass/fail report.

Run all:        python tests/test_qa.py
Run subset:     python tests/test_qa.py --filter "vi_def_*" --filter "web_*"
                python tests/test_qa.py --category concept_vi
                python tests/test_qa.py --id vi_def_antiselection
Verbose:        python tests/test_qa.py -v          (print full answers)
                python tests/test_qa.py -vv         (also print retrieval details)
Persist:        python tests/test_qa.py --out out/qa_results.jsonl
Tweak retriever:
                python tests/test_qa.py --no-graph --no-rerank
                python tests/test_qa.py --no-fallback   # web fallback OFF (web cases will fail)
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
load_dotenv(HERE / ".env")

from loma_rag.llm.completion import detect_language_llm  # noqa: E402
from loma_rag.llm.openai_client import make_chat_client   # noqa: E402
from loma_rag.model.domain import AnswerResult             # noqa: E402
from loma_rag.rag.pipeline import run_query                # noqa: E402
from loma_rag.rag.retriever import Retriever               # noqa: E402
from loma_rag.rag.web_fallback import WebFallback          # noqa: E402
from tests.qa_cases import TESTS                           # noqa: E402

# ---- helpers ----

# Module-level chat client for language detection (set in main()).
_LANG_CLIENT = None


def detect_language(text: str) -> str:
    """LLM-based language detection on `text`. Falls back to 'en' on failure.
    Cached internally by detect_language_llm."""
    if not text:
        return "en"
    if _LANG_CLIENT is None:
        return "en"
    return detect_language_llm(_LANG_CLIENT, text)


def check_expectations(case: dict, result: AnswerResult) -> list[str]:
    """Return list of failure messages (empty == pass)."""
    fails: list[str] = []
    exp = case.get("expects", {}) or {}
    text_low = (result.answer_text or "").lower()

    # path
    expected_path = exp.get("path")
    if expected_path is not None:
        if isinstance(expected_path, str):
            if result.used_path != expected_path:
                fails.append(f"path: expected {expected_path!r}, got {result.used_path!r}")
        elif isinstance(expected_path, (list, tuple, set)):
            if result.used_path not in expected_path:
                fails.append(f"path: expected one of {sorted(expected_path)}, got {result.used_path!r}")

    # answer must exist (unless test allows refused/error). The newer
    # off_topic / unsupported_language / no_result paths always populate
    # answer_text with a canned response, so they are not "no answer" paths.
    if result.used_path in ("refused", "error") and exp.get("path") not in (
        result.used_path, list(exp.get("path") or [])
    ):
        fails.append(f"no answer (used_path={result.used_path}, error={result.error!r})")
        return fails  # downstream checks meaningless

    # language
    want_lang = exp.get("language")
    if want_lang:
        actual = detect_language(result.answer_text)
        if actual != want_lang:
            fails.append(f"language: expected {want_lang!r}, got {actual!r}")

    # must_contain (all)
    for needle in exp.get("must_contain", []) or []:
        if needle.lower() not in text_low:
            fails.append(f"must_contain: missing {needle!r}")

    # must_contain_any (at least one)
    any_list = exp.get("must_contain_any", []) or []
    if any_list and not any(n.lower() in text_low for n in any_list):
        fails.append(f"must_contain_any: none of {any_list} in answer")

    # citation regex
    pat = exp.get("must_cite_pattern")
    if pat and not re.search(pat, result.answer_text or ""):
        fails.append(f"must_cite_pattern: regex {pat!r} not matched")

    # cited lesson prefix
    prefix = exp.get("must_cite_lesson_starting")
    if prefix:
        cited_lessons = {c.lesson_id for c in result.loma_chunks if any(
            c.chunk_id in (result.answer_text or "") for c in [c]
        )}
        # Above is awkward; redo more robustly:
        cited = set()
        for c in result.loma_chunks:
            if c.chunk_id and c.chunk_id in (result.answer_text or ""):
                cited.add(c.lesson_id)
        if not any(lid.startswith(prefix) for lid in cited):
            fails.append(
                f"must_cite_lesson_starting: no cited chunk has lesson_id starting with "
                f"{prefix!r}; cited lessons={sorted(cited) or 'none'}"
            )

    # min counts
    n_chunks = len(result.loma_chunks)
    n_web = len(result.web_docs)
    n_rel = len(result.related_nodes)
    if exp.get("min_loma_chunks") and n_chunks < exp["min_loma_chunks"]:
        fails.append(f"min_loma_chunks: {n_chunks} < {exp['min_loma_chunks']}")
    if exp.get("min_web_docs") and n_web < exp["min_web_docs"]:
        fails.append(f"min_web_docs: {n_web} < {exp['min_web_docs']}")
    if exp.get("min_related_nodes") and n_rel < exp["min_related_nodes"]:
        fails.append(f"min_related_nodes: {n_rel} < {exp['min_related_nodes']}")

    return fails


# ---- runner ----

def select_cases(
    cases: list[dict],
    ids: list[str],
    categories: list[str],
    patterns: list[str],
) -> list[dict]:
    if not (ids or categories or patterns):
        return cases
    selected = []
    for c in cases:
        cid = c["id"]
        if ids and cid in ids:
            selected.append(c); continue
        if categories and c.get("category") in categories:
            selected.append(c); continue
        if patterns and any(fnmatch.fnmatch(cid, p) for p in patterns):
            selected.append(c); continue
    return selected


def short_preview(text: str, n: int = 200) -> str:
    t = (text or "").strip().replace("\n", " ")
    return t if len(t) <= n else t[:n] + "…"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", action="append", default=[],
                    help="glob match against case id; can be passed multiple times")
    ap.add_argument("--category", action="append", default=[])
    ap.add_argument("--id", action="append", default=[])
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--no-fallback", action="store_true")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--web-k", type=int, default=5)
    ap.add_argument("-v", "--verbose", action="count", default=0,
                    help="-v: print answers; -vv: also print retrieval detail")
    ap.add_argument("--out", default="",
                    help="optional path to write per-case results JSONL")
    args = ap.parse_args()

    cases = select_cases(TESTS, args.id, args.category, args.filter)
    if not cases:
        print("no test cases matched the filters", file=sys.stderr)
        return 2

    print(
        f"running {len(cases)}/{len(TESTS)} test cases  |  "
        f"rerank={not args.no_rerank}  graph={not args.no_graph}  "
        f"fallback={not args.no_fallback}\n"
    )

    retriever = Retriever(
        rerank=not args.no_rerank,
        expand_graph=not args.no_graph,
    )
    chat_client = make_chat_client()
    web_fallback = None if args.no_fallback else WebFallback(dense_client=chat_client)
    global _LANG_CLIENT
    _LANG_CLIENT = chat_client

    results: list[dict] = []
    n_pass = 0
    t0 = time.time()
    for i, case in enumerate(cases, 1):
        cid = case["id"]
        cat = case.get("category", "?")
        print(f"[{i:>2}/{len(cases)}] {cid:32}  ({cat})")
        print(f"    Q: {case['query']}")

        t_q = time.time()
        try:
            result = run_query(
                retriever, chat_client, web_fallback, case["query"],
                top_k=args.top_k, web_k=args.web_k,
            )
        except Exception as e:  # noqa: BLE001
            result = AnswerResult(query=case["query"], used_path="error",
                                  answer_text="", error=f"{type(e).__name__}: {e}")
        elapsed = time.time() - t_q

        fails = check_expectations(case, result)
        passed = not fails
        n_pass += int(passed)
        status = "PASS" if passed else "FAIL"

        meta = (
            f"{status}  path={result.used_path}  "
            f"chunks={len(result.loma_chunks)}  "
            f"web={len(result.web_docs)}  "
            f"nodes={len(result.related_nodes)}  "
            f"{elapsed:.1f}s"
        )
        print(f"    {meta}")
        if not passed:
            for f in fails:
                print(f"      - {f}")
        if args.verbose >= 1:
            print(f"    A: {short_preview(result.answer_text, 600)}")
        if args.verbose >= 2:
            if result.en_search_query:
                print(f"    en_search_query: {result.en_search_query!r}")
            if result.loma_chunks:
                print("    loma chunks:")
                for c in result.loma_chunks:
                    sub = f" > {c.subsection}" if c.subsection else ""
                    print(f"      - rerank={c.rerank_score:+.2f} {c.chunk_id} | {c.section}{sub}")
            if result.web_docs:
                print("    web docs:")
                for j, d in enumerate(result.web_docs, 1):
                    print(f"      - WEB_{j}  ({d.score:.3f})  {d.title[:80]}")
        print()

        results.append({
            "id": cid,
            "category": cat,
            "query": case["query"],
            "passed": passed,
            "fails": fails,
            "used_path": result.used_path,
            "answer_text": result.answer_text,
            "answer_language": detect_language(result.answer_text),
            "en_search_query": result.en_search_query,
            "loma_chunks": [
                {"chunk_id": c.chunk_id, "lesson_id": c.lesson_id,
                 "rerank_score": c.rerank_score, "section": c.section,
                 "subsection": c.subsection}
                for c in result.loma_chunks
            ],
            "related_nodes": [n.name for n in result.related_nodes],
            "web_docs": [{"title": d.title, "url": d.url, "score": d.score}
                          for d in result.web_docs],
            "elapsed_s": elapsed,
            "error": result.error,
        })

    total_elapsed = time.time() - t0
    print(f"\n=== Summary ===")
    print(f"  passed: {n_pass}/{len(cases)}  ({n_pass/len(cases)*100:.1f}%)")
    print(f"  total time: {total_elapsed:.1f}s")

    # Per-category breakdown
    from collections import defaultdict
    by_cat: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r["passed"])
    print("\nBy category:")
    for cat in sorted(by_cat):
        vals = by_cat[cat]
        print(f"  {cat:<22} {sum(vals)}/{len(vals)}")

    if args.out:
        out_path = HERE / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nwrote {out_path}")

    return 0 if n_pass == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
