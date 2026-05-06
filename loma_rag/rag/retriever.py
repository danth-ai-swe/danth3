"""Hybrid retrieval pipeline: dense + sparse + cross-encoder rerank + graph expand.

  Query
    |
    +-- dense (text-embedding-3-small) ---+
    |                                     |
    +-- sparse (BM42) --------------------+--> Qdrant RRF fusion (top 20)
                                                   |
                                                   v
                                          Cross-encoder rerank
                                          (Xenova/ms-marco-MiniLM-L-6-v2)
                                                   |
                                                   v
                                              top-K chunks
                                                   |
                                                   v
                          Knowledge graph: chunks.node_refs --1 hop--> related Nodes
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor

from loma_rag.config.settings import azure, qdrant as qdrant_cfg, rag as rag_cfg
from loma_rag.constant.collections import CHUNKS_COLL as CHUNKS_COLLECTION, NODES_COLL as NODES_COLLECTION
from loma_rag.constant.thresholds import TRANSLATE_FOR_SEARCH_CACHE_SIZE
from loma_rag.db_vector.search import hybrid_search, search_nodes_direct
from loma_rag.llm.embedding import embed_colbert, embed_dense_single, embed_sparse
from loma_rag.model.domain import RetrievalResult, RetrievedChunk
from loma_rag.prompt.translate import TRANSLATE_FOR_SEARCH_SYSTEM
from loma_rag.rag.reranker import Reranker
from loma_rag.util.cache import LRUCache

# Windows consoles default to cp1252 — force UTF-8 so non-ASCII chars in chunks print fine.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

CHAT_MODEL = azure.chat_model
PREFETCH_LIMIT = rag_cfg.prefetch_limit
NODE_RRF_THRESHOLD = rag_cfg.node_rrf_threshold
DIRECT_NODE_TOP_K = rag_cfg.direct_node_top_k
RERANK_MODEL = rag_cfg.rerank_model
_TRANSLATE_CACHE_SIZE = TRANSLATE_FOR_SEARCH_CACHE_SIZE


class Retriever:
    """End-to-end retriever. Clients are lazy-initialised."""

    def __init__(
        self,
        candidates: int = 12,
        rerank: bool = True,
        expand_graph: bool = True,
    ) -> None:
        # candidates: post-RRF list size that gets fed to the reranker.
        # Bench showed reranker dominates retrieval latency; 12 keeps top-5
        # quality on tests while saving ~700ms vs 20.
        self.candidates = candidates
        self.rerank_enabled = rerank
        self.expand_graph_enabled = expand_graph

        self._dense_client = None
        self._reranker: Reranker | None = None
        self._qc = None
        self._kg = None

        # Cache for the query→English translation (max ~300ms LLM call saved).
        # Dense / sparse / colbert embedding singletons live in loma_rag.llm.embedding.
        self._translate_cache = LRUCache(_TRANSLATE_CACHE_SIZE)
        # Worker pool for parallel local-CPU sparse embed alongside the
        # blocking Azure dense embed call.
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="rag-embed")

    # ---- lazy clients ----

    def _ensure_clients(self) -> None:
        if self._dense_client is None:
            from openai import AzureOpenAI

            self._dense_client = AzureOpenAI(
                azure_endpoint=azure.api_base.rstrip("/"),
                api_key=azure.api_key,
                api_version=azure.api_version,
            )
        if self._qc is None:
            from qdrant_client import QdrantClient

            self._qc = QdrantClient(
                url=qdrant_cfg.url,
                timeout=30,
            )
        if self.rerank_enabled and self._reranker is None:
            self._reranker = Reranker(model_name=RERANK_MODEL)
        if self.expand_graph_enabled and self._kg is None:
            from loma_rag.rag.graph import load_graph

            self._kg = load_graph()

    # ---- pipeline ----

    def translate_for_search(self, query: str) -> str:
        """Translate a user query into English for consistent vector search.
        Cached. Returns the original on parse failure."""
        cached = self._translate_cache.get(query)
        if cached is not None:
            return cached  # type: ignore[return-value]
        self._ensure_clients()
        try:
            resp = self._dense_client.chat.completions.create(  # type: ignore[union-attr]
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": TRANSLATE_FOR_SEARCH_SYSTEM},
                    {"role": "user", "content": query},
                ],
                temperature=0.0,
                max_tokens=200,
            )
            out = (resp.choices[0].message.content or "").strip().strip("\"' ")
            if out:
                self._translate_cache.put(query, out)
                return out
        except Exception:  # noqa: BLE001
            pass
        return query

    def _embed_dense(self, text: str) -> list[float]:
        """Cached dense embedding via Azure text-embedding-3-small."""
        self._ensure_clients()
        return embed_dense_single(self._dense_client, text)

    def _embed_sparse(self, text: str):
        self._ensure_clients()
        return embed_sparse(text)

    def _embed_colbert(self, text: str):
        """Returns a numpy array shape (n_tokens, 128)."""
        self._ensure_clients()
        return embed_colbert(text)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        max_graph_nodes: int = 12,
        user_lang: str = "en",
        search_text: str | None = None,
    ) -> RetrievalResult:
        self._ensure_clients()
        timings: dict[str, float] = {}

        # Standardise to English for vector search. Corpus is English-only so
        # Vietnamese (or any other) input is translated first. The original
        # query is preserved for the answer LLM (so reply language is correct).
        # If the caller already pre-translated (e.g. via combined analyze_query
        # call), accept that and skip the internal translate step.
        if search_text is None:
            search_text = query
            if user_lang and user_lang != "en":
                t0 = time.time()
                search_text = self.translate_for_search(query)
                timings["translate_search"] = time.time() - t0

        # Run dense (Azure HTTP), sparse BM42 (local CPU) and ColBERT (local
        # CPU, late-interaction) in parallel.
        t0 = time.time()
        f_dense = self._pool.submit(self._embed_dense, search_text)
        f_sparse = self._pool.submit(self._embed_sparse, search_text)
        f_colbert = self._pool.submit(self._embed_colbert, search_text)
        emb = f_dense.result()
        sparse_emb = f_sparse.result()
        colbert_vecs = f_colbert.result()
        timings["embed"] = time.time() - t0

        # Three-way prefetch + RRF fusion:
        #   - Dense (INT8-rescored) and sparse BM42 each retrieve top-N.
        #   - ColBERT MaxSim contributes a third independent ranking.
        #   - RRF merges the three lists fairly. ColBERT used as a prefetch
        #     (not the final ranker) keeps short overview chunks from
        #     dominating long detailed chunks on token-density signal.
        t0 = time.time()
        res = hybrid_search(
            self._qc,
            emb,
            sparse_emb,
            colbert_vecs,
            prefetch_limit=PREFETCH_LIMIT,
            final_limit=self.candidates,
        )
        timings["qdrant"] = time.time() - t0

        # Build candidate chunks from Qdrant payloads.
        candidates: list[RetrievedChunk] = []
        for p in res.points:
            pl = p.payload or {}
            candidates.append(
                RetrievedChunk(
                    chunk_id=pl.get("chunk_id", ""),
                    text=pl.get("text", ""),
                    section=pl.get("section", ""),
                    subsection=pl.get("subsection", ""),
                    course=pl.get("course", ""),
                    module=pl.get("module", ""),
                    lesson=pl.get("lesson", ""),
                    lesson_id=pl.get("lesson_id", ""),
                    source=pl.get("source", ""),
                    node_refs=list(pl.get("node_refs") or []),
                    fused_score=float(p.score),
                )
            )

        # Cross-encoder rerank against the (English) search text — gives the
        # English-only reranker a fair chance regardless of query language.
        if self.rerank_enabled and candidates:
            t0 = time.time()
            scores = self._reranker.rerank(search_text, [c.text for c in candidates])
            for c, s in zip(candidates, scores):
                c.rerank_score = float(s)
            candidates.sort(key=lambda c: c.rerank_score, reverse=True)
            timings["rerank"] = time.time() - t0

        # Hybrid top-K: keep most slots for rerank winners, but reserve a
        # couple of slots for chunks Qdrant's RRF strongly favoured. Long
        # metadata chunks (SYLLABUS, SCHEDULE) often score high in RRF but
        # low in cross-encoder rerank because the relevant fact is buried;
        # this reservation ensures those chunks still reach the LLM.
        rrf_reserve = min(2, max(0, top_k - 3)) if candidates else 0
        if rrf_reserve:
            primary = candidates[: top_k - rrf_reserve]
            seen = {c.chunk_id for c in primary}
            extras: list = []
            for c in sorted(candidates, key=lambda x: x.fused_score, reverse=True):
                if c.chunk_id in seen:
                    continue
                extras.append(c)
                seen.add(c.chunk_id)
                if len(extras) >= rrf_reserve:
                    break
            chunks = primary + extras
            # Re-sort by rerank for presentation; RRF reserves keep their place via inclusion.
            chunks.sort(key=lambda c: c.rerank_score, reverse=True)
        else:
            chunks = candidates[:top_k]

        # Direct KG node lookup: query loma_nodes Qdrant collection with the
        # SAME embeddings (no extra Azure calls). Catches definitional questions
        # where the chunk corpus uses descriptive English ("the risk of outliving
        # your assets") but the KG has the canonical entry ("Longevity Risk:
        # the risk of outliving one's financial resources").
        direct_node_ids: list[str] = []
        if self.expand_graph_enabled:
            t0 = time.time()
            try:
                direct_node_ids = self._search_nodes_direct(emb, sparse_emb, colbert_vecs)
            except Exception:  # noqa: BLE001
                direct_node_ids = []
            timings["nodes"] = time.time() - t0

        # Graph expansion: pull definitions for nodes referenced by retrieved
        # chunks PLUS the direct KG hits, BFS-expanded by 1 hop, deduped.
        related_nodes: list = []
        if self.expand_graph_enabled:
            t0 = time.time()
            chunk_payloads = [{"node_refs": c.node_refs} for c in chunks]
            # Inject direct hits as additional seeds.
            if direct_node_ids:
                chunk_payloads.append({"node_refs": direct_node_ids})
            related_nodes = self._kg.expand_for_chunks(
                chunk_payloads,
                hops=1,
                max_nodes=max_graph_nodes,
            )
            timings["graph"] = time.time() - t0

        return RetrievalResult(
            query=query,
            chunks=chunks,
            related_nodes=related_nodes,
            timings=timings,
            search_text=search_text,
            direct_node_ids=direct_node_ids,
        )

    def _search_nodes_direct(self, dense_emb, sparse_emb, colbert_vecs) -> list[str]:
        """Three-way prefetch + RRF fusion against the loma_nodes collection.
        Returns up to DIRECT_NODE_TOP_K node_ids whose RRF score clears
        NODE_RRF_THRESHOLD."""
        return search_nodes_direct(
            self._qc,
            dense_emb,
            sparse_emb,
            colbert_vecs,
            top_k=DIRECT_NODE_TOP_K,
            node_rrf_threshold=NODE_RRF_THRESHOLD,
        )
