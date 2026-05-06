"""LOMA RAG CLI dispatcher.

Subcommands:
  main.py serve     — run the FastAPI server (uvicorn)
  main.py repl      — interactive chat REPL (was chat.py)
  main.py ingest    — run the ingestion pipeline (docx/xlsx → JSONL)
  main.py index     — build/rebuild Qdrant collections from JSONL
"""
from __future__ import annotations

import sys


def cmd_serve(argv: list[str]) -> int:
    import uvicorn
    from loma_rag.config.settings import api as api_cfg
    uvicorn.run("loma_rag.api.app:app", host=api_cfg.host, port=api_cfg.port)
    return 0


def cmd_repl(argv: list[str]) -> int:
    # Body extracted from chat.py main(), with imports updated.
    import argparse

    from loma_rag.llm.openai_client import make_chat_client
    from loma_rag.rag.pipeline import answer_query
    from loma_rag.rag.retriever import Retriever
    from loma_rag.rag.web_fallback import WebFallback

    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="*", help="single-shot question; omit for REPL")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--web-k", type=int, default=5)
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--no-fallback", action="store_true")
    ap.add_argument("--no-stream", action="store_true")
    ap.add_argument("--show-context", action="store_true")
    args = ap.parse_args()

    print(
        f"[init] rerank={not args.no_rerank}  graph={not args.no_graph}  "
        f"fallback={not args.no_fallback}"
    )
    retriever = Retriever(
        rerank=not args.no_rerank,
        expand_graph=not args.no_graph,
    )
    chat_client = make_chat_client()
    web_fallback = None if args.no_fallback else WebFallback(dense_client=chat_client)

    if args.query:
        answer_query(
            retriever, chat_client, web_fallback, " ".join(args.query),
            top_k=args.top_k, web_k=args.web_k,
            show_context=args.show_context, stream=not args.no_stream,
        )
        return 0

    print("[ready] enter your question (empty line to quit).")
    while True:
        try:
            q = input("\n>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            break
        try:
            answer_query(
                retriever, chat_client, web_fallback, q,
                top_k=args.top_k, web_k=args.web_k,
                show_context=args.show_context, stream=not args.no_stream,
            )
        except Exception as e:  # noqa: BLE001
            print(f"\n! error: {e}", file=sys.stderr)
    return 0


def cmd_ingest(argv: list[str]) -> int:
    from loma_rag.ingest.pipeline import main as ingest_main
    return ingest_main()


def cmd_index(argv: list[str]) -> int:
    from loma_rag.ingest.indexer import main as index_main
    return index_main()


COMMANDS = {
    "serve": cmd_serve,
    "repl": cmd_repl,
    "ingest": cmd_ingest,
    "index": cmd_index,
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print("usage: main.py {serve|repl|ingest|index} [args...]")
        return 2
    # Pop the subcommand so argparse in the subcommand sees clean argv.
    cmd = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    return COMMANDS[cmd](sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
