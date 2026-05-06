"""Web search fallback: SearXNG -> page fetch -> embed -> FAISS top-k.

Used when LOMA chunks don't contain enough info to answer the user's question.
The fallback retrieves up to N web results, fetches their page text, embeds
them with the same Azure OpenAI dense model, and ranks via FAISS (cosine, IP).

Public API:
    fb = WebFallback()
    docs = fb.retrieve(query_in_english, top_k=5)
    -> list[(WebDoc, score)]
"""
from __future__ import annotations

import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import requests
from bs4 import BeautifulSoup

from loma_rag.config.settings import azure, web
from loma_rag.model.domain import WebDoc

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_SEARXNG = web.searxng_url
DENSE_MODEL = azure.embedding_model
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
MAX_PAGE_CHARS = 4000


# ---- SearXNG ----

def search_searxng(
    query: str,
    n: int = 10,
    base_url: Optional[str] = None,
    categories: str = "general",
    language: str = "en",
    timeout: float = 15.0,
) -> list[WebDoc]:
    base = (base_url or DEFAULT_SEARXNG).rstrip("/")
    r = requests.get(
        f"{base}/search",
        params={
            "q": query,
            "format": "json",
            "categories": categories,
            "language": language,
        },
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
    )
    r.raise_for_status()
    data = r.json()
    out: list[WebDoc] = []
    for i, item in enumerate(data.get("results", [])[:n], 1):
        out.append(WebDoc(
            url=item.get("url", ""),
            title=item.get("title", ""),
            snippet=item.get("content", "") or "",
            engine=item.get("engine", ""),
            rank_in_search=i,
        ))
    return out


# ---- Page fetching ----

def fetch_page_text(url: str, timeout: float = 8.0) -> str:
    try:
        r = requests.get(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html,*/*"},
            timeout=timeout,
            allow_redirects=True,
        )
        ctype = r.headers.get("Content-Type", "")
        if not r.ok or "html" not in ctype.lower():
            return ""
        soup = BeautifulSoup(r.content, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
            tag.decompose()
        # prefer <article> / <main> if present
        main = soup.find("article") or soup.find("main") or soup.body or soup
        text = main.get_text(" ", strip=True)
        text = re.sub(r"\s+", " ", text)
        return text[:MAX_PAGE_CHARS]
    except Exception:
        return ""


def enrich_pages(docs: list[WebDoc], max_workers: int = 6) -> list[WebDoc]:
    if not docs:
        return docs
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_to_doc = {ex.submit(fetch_page_text, d.url): d for d in docs}
        for fut in future_to_doc:
            d = future_to_doc[fut]
            try:
                text = fut.result()
            except Exception:
                text = ""
            d.content = text or d.snippet
    return docs


# ---- FAISS rank ----

def faiss_rank(
    query_emb: list[float],
    doc_embs: list[list[float]],
    top_k: int,
) -> list[tuple[int, float]]:
    """Brute-force inner-product over normalized vectors == cosine."""
    qv = np.array([query_emb], dtype=np.float32)
    dv = np.array(doc_embs, dtype=np.float32)
    qv = qv / (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-9)
    dv = dv / (np.linalg.norm(dv, axis=1, keepdims=True) + 1e-9)
    index = faiss.IndexFlatIP(dv.shape[1])
    index.add(dv)
    k = min(top_k, dv.shape[0])
    D, I = index.search(qv, k)
    return [(int(I[0][j]), float(D[0][j])) for j in range(k)]


# ---- main class ----

class WebFallback:
    def __init__(
        self,
        dense_client=None,
        searxng_url: Optional[str] = None,
        n_search: int = 8,
        fetch_pages: bool = True,
    ) -> None:
        self.searxng_url = searxng_url or DEFAULT_SEARXNG
        self.n_search = n_search
        self.fetch_pages = fetch_pages
        self._dense_client = dense_client  # may be set externally to share

    def _ensure_client(self):
        if self._dense_client is None:
            from openai import AzureOpenAI

            self._dense_client = AzureOpenAI(
                azure_endpoint=azure.api_base.rstrip("/"),
                api_key=azure.api_key,
                api_version=azure.api_version,
            )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        self._ensure_client()
        resp = self._dense_client.embeddings.create(model=DENSE_MODEL, input=texts)
        return [d.embedding for d in resp.data]

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[WebDoc]:
        docs = search_searxng(query, n=self.n_search, base_url=self.searxng_url)
        if not docs:
            return []
        if self.fetch_pages:
            docs = enrich_pages(docs)
        # Filter out docs with no usable content.
        docs = [d for d in docs if (d.content or d.snippet).strip()]
        if not docs:
            return []

        # Embed: query + each doc's content (cap text length per doc).
        passage_texts = [(d.content or d.snippet)[:2000] for d in docs]
        embs = self._embed([query] + passage_texts)
        q_emb, doc_embs = embs[0], embs[1:]

        ranked = faiss_rank(q_emb, doc_embs, top_k=top_k)
        out: list[WebDoc] = []
        for idx, score in ranked:
            d = docs[idx]
            d.score = score
            out.append(d)
        return out


# ---- CLI smoke test ----

def _self_test() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="*", default=["What is antiselection in life insurance?"])
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--no-fetch", action="store_true")
    args = ap.parse_args()
    q = " ".join(args.query) if args.query else "What is antiselection in life insurance?"

    print(f"[SearXNG] {DEFAULT_SEARXNG}")
    print(f"[query]   {q!r}\n")

    fb = WebFallback(fetch_pages=not args.no_fetch)
    docs = fb.retrieve(q, top_k=args.top_k)
    if not docs:
        print("(no results)")
        return 1
    for i, d in enumerate(docs, 1):
        body = (d.content or d.snippet).replace("\n", " ")
        print(f"#{i}  score={d.score:.3f}  rank_in_search={d.rank_in_search}  engine={d.engine}")
        print(f"    {d.title}")
        print(f"    {d.url}")
        print(f"    {body[:240]}…\n")
    return 0


if __name__ == "__main__":
    sys.exit(_self_test())
