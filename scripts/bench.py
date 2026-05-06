"""Benchmark the FastAPI endpoints on every Q&A test case.

Measures, per case:
    /chat        total wall-clock time + server-reported per-stage timings
    /chat/stream total wall-clock time, TTFT (Time To First Token),
                 chars/s once streaming starts, per-stage timings

Outputs a Markdown-friendly table to stdout and a JSONL with raw rows.

Run:
    python scripts/bench.py                       # default: localhost:8000, all cases
    python scripts/bench.py --base http://x:y     # custom server
    python scripts/bench.py --filter "vi_*"       # subset by glob on case id
    python scripts/bench.py --warmup              # send a dummy request first
    python scripts/bench.py --out bench.jsonl     # write raw rows
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import statistics
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
load_dotenv(HERE / ".env")

from tests.qa_cases import TESTS  # noqa: E402


def bench_nonstream(client: httpx.Client, base: str, query: str) -> dict:
    t0 = time.time()
    r = client.post(f"{base}/chat", json={"query": query}, timeout=120)
    elapsed_ms = (time.time() - t0) * 1000
    r.raise_for_status()
    data = r.json()
    return {
        "endpoint": "chat",
        "total_ms": elapsed_ms,
        "path": data.get("path"),
        "answer_chars": len(data.get("answer", "")),
        "n_citations": len(data.get("citations", [])),
        "timings_ms": data.get("timings_ms", {}),
    }


def bench_stream(client: httpx.Client, base: str, query: str) -> dict:
    t0 = time.time()
    ttft_ms = None
    last_delta_ms = 0.0
    chars = 0
    final: dict = {}
    with client.stream("POST", f"{base}/chat/stream",
                       json={"query": query}, timeout=120) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            ev = json.loads(line[6:])
            t_ev = (time.time() - t0) * 1000
            if ev.get("type") == "delta":
                if ttft_ms is None:
                    ttft_ms = t_ev
                chars += len(ev.get("text", ""))
                last_delta_ms = t_ev
            elif ev.get("type") == "done":
                final = ev
    total_ms = (time.time() - t0) * 1000
    streaming_window_ms = max(last_delta_ms - (ttft_ms or 0), 1.0)
    chars_per_s = chars / (streaming_window_ms / 1000) if chars else 0
    return {
        "endpoint": "stream",
        "total_ms": total_ms,
        "ttft_ms": ttft_ms,
        "answer_chars": chars,
        "chars_per_s": chars_per_s,
        "path": final.get("path"),
        "n_citations": len(final.get("citations", [])),
        "timings_ms": final.get("timings", {}),
    }


def run_bench(base: str, cases: list[dict], warmup: bool) -> list[dict]:
    rows: list[dict] = []
    with httpx.Client() as client:
        # Health check.
        h = client.get(f"{base}/health", timeout=10)
        h.raise_for_status()

        if warmup:
            print("[warmup] firing one query…")
            try:
                bench_nonstream(client, base, "What is insurance?")
            except Exception as e:  # noqa: BLE001
                print(f"  warmup failed: {e}")

        for i, case in enumerate(cases, 1):
            cid = case["id"]
            cat = case.get("category", "?")
            q = case["query"]
            print(f"[{i:>2}/{len(cases)}] {cid:32} ({cat})")

            try:
                ns = bench_nonstream(client, base, q)
            except Exception as e:  # noqa: BLE001
                ns = {"endpoint": "chat", "error": str(e)}
            try:
                st = bench_stream(client, base, q)
            except Exception as e:  # noqa: BLE001
                st = {"endpoint": "stream", "error": str(e)}

            print(
                f"    chat  : total={ns.get('total_ms', 0):>6.0f}ms  "
                f"path={ns.get('path')}  chars={ns.get('answer_chars')}  "
                f"timings={ns.get('timings_ms')}"
            )
            print(
                f"    stream: total={st.get('total_ms', 0):>6.0f}ms  "
                f"ttft={st.get('ttft_ms') and round(st.get('ttft_ms'))}ms  "
                f"path={st.get('path')}  chars={st.get('answer_chars')}  "
                f"chars/s={st.get('chars_per_s', 0):.0f}"
            )
            rows.append({"id": cid, "category": cat, "query": q,
                         "nonstream": ns, "stream": st})
    return rows


# ---- analysis ----

def median(xs: list[float]) -> float:
    return statistics.median(xs) if xs else 0.0


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * (p / 100)
    f = int(k); c = min(f + 1, len(s) - 1); frac = k - f
    return s[f] + (s[c] - s[f]) * frac


def summarize(rows: list[dict]) -> None:
    paths_by_id: dict[str, str] = {r["id"]: (r["nonstream"].get("path") or r["stream"].get("path") or "?") for r in rows}

    def split(rows: list[dict], pred) -> list[dict]:
        return [r for r in rows if pred(r)]

    def stats(label: str, vals: list[float]) -> None:
        if not vals:
            print(f"  {label:<24} (no data)")
            return
        print(
            f"  {label:<24} n={len(vals):<3}  "
            f"min={min(vals):>5.0f}  med={median(vals):>5.0f}  "
            f"p95={percentile(vals,95):>5.0f}  max={max(vals):>5.0f} ms"
        )

    print("\n=== Latency summary ===")
    print("\n[Non-streaming /chat] total_ms")
    stats("all", [r["nonstream"]["total_ms"] for r in rows if "total_ms" in r["nonstream"]])
    loma_rows = [r for r in rows if r["nonstream"].get("path") == "loma"]
    web_rows  = [r for r in rows if r["nonstream"].get("path") == "web"]
    stats("loma path",  [r["nonstream"]["total_ms"] for r in loma_rows])
    stats("web fallback", [r["nonstream"]["total_ms"] for r in web_rows])

    print("\n[Streaming /chat/stream] TTFT_ms (Time To First Token)")
    stats("all",          [r["stream"]["ttft_ms"] for r in rows if r["stream"].get("ttft_ms")])
    stats("loma path",    [r["stream"]["ttft_ms"] for r in loma_rows if r["stream"].get("ttft_ms")])
    stats("web fallback", [r["stream"]["ttft_ms"] for r in web_rows  if r["stream"].get("ttft_ms")])

    print("\n[Streaming /chat/stream] total_ms")
    stats("all", [r["stream"]["total_ms"] for r in rows if "total_ms" in r["stream"]])

    # Stage breakdown for LOMA path (averaging server-reported timings)
    print("\n=== Stage breakdown — server timings (median, ms) ===\n")
    print("Stage                     loma  web")
    stages = ["retrieve.embed", "retrieve.qdrant",
              "retrieve.rerank", "retrieve.graph", "loma_answer",
              "translate", "web_search", "web_answer", "total"]
    for stage in stages:
        loma_vals = [r["nonstream"]["timings_ms"].get(stage)
                     for r in loma_rows
                     if isinstance(r["nonstream"].get("timings_ms"), dict)
                        and r["nonstream"]["timings_ms"].get(stage) is not None]
        web_vals = [r["nonstream"]["timings_ms"].get(stage)
                    for r in web_rows
                    if isinstance(r["nonstream"].get("timings_ms"), dict)
                       and r["nonstream"]["timings_ms"].get(stage) is not None]
        l = f"{median(loma_vals):>5.0f}" if loma_vals else "    -"
        w = f"{median(web_vals):>5.0f}" if web_vals else "    -"
        print(f"  {stage:<24}  {l}   {w}")

    # Slowest 5 cases
    print("\n=== Slowest 5 cases (non-streaming) ===")
    by_total = sorted(rows, key=lambda r: r["nonstream"].get("total_ms", 0), reverse=True)[:5]
    for r in by_total:
        ns = r["nonstream"]
        print(f"  {r['id']:<32}  total={ns.get('total_ms',0):>5.0f}ms  path={ns.get('path')}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--filter", action="append", default=[])
    ap.add_argument("--id", action="append", default=[])
    ap.add_argument("--warmup", action="store_true")
    ap.add_argument("--out", default="out/bench_results.jsonl")
    args = ap.parse_args()

    cases = TESTS
    if args.id:
        cases = [c for c in cases if c["id"] in args.id]
    if args.filter:
        cases = [c for c in cases if any(fnmatch.fnmatch(c["id"], p) for p in args.filter)]
    if not cases:
        print("no cases matched filters", file=sys.stderr); return 2

    print(f"benchmarking {len(cases)} case(s) against {args.base}")
    rows = run_bench(args.base, cases, warmup=args.warmup)

    out_path = HERE / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {out_path}")

    summarize(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
