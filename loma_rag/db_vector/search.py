"""Hybrid search abstraction over Qdrant.

Inputs are already-computed embeddings. Outputs Qdrant scored-point lists.
"""
from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Fusion, FusionQuery, Prefetch,
    QuantizationSearchParams, SearchParams, SparseVector,
)

from loma_rag.constant.collections import CHUNKS_COLL, NODES_COLL


def hybrid_search(
    qc: QdrantClient,
    dense_emb: list[float],
    sparse_emb,
    colbert_vecs,
    prefetch_limit: int,
    final_limit: int,
):
    """Run dense+sparse+ColBERT prefetch with RRF fusion against the chunks collection."""
    dense_search_params = SearchParams(
        quantization=QuantizationSearchParams(rescore=True),
        hnsw_ef=128,
        exact=False,
    )
    res = qc.query_points(
        collection_name=CHUNKS_COLL,
        prefetch=[
            Prefetch(
                query=dense_emb,
                using="dense",
                limit=prefetch_limit,
                params=dense_search_params,
            ),
            Prefetch(
                query=SparseVector(
                    indices=sparse_emb.indices.tolist(),
                    values=sparse_emb.values.tolist(),
                ),
                using="sparse",
                limit=prefetch_limit,
            ),
            Prefetch(
                query=colbert_vecs.tolist(),
                using="colbert",
                limit=prefetch_limit,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=final_limit,
        with_payload=True,
    )
    return res


def search_nodes_direct(
    qc: QdrantClient,
    dense_emb: list[float],
    sparse_emb,
    colbert_vecs,
    top_k: int,
    node_rrf_threshold: float = 0.4,
) -> list[str]:
    """Three-way prefetch + RRF fusion against the loma_nodes collection.
    Returns up to top_k node_ids whose RRF score clears node_rrf_threshold."""
    dense_search_params = SearchParams(
        quantization=QuantizationSearchParams(rescore=True),
        hnsw_ef=128,
        exact=False,
    )
    res = qc.query_points(
        collection_name=NODES_COLL,
        prefetch=[
            Prefetch(
                query=dense_emb,
                using="dense",
                limit=top_k * 4,
                params=dense_search_params,
            ),
            Prefetch(
                query=SparseVector(
                    indices=sparse_emb.indices.tolist(),
                    values=sparse_emb.values.tolist(),
                ),
                using="sparse",
                limit=top_k * 4,
            ),
            Prefetch(
                query=colbert_vecs.tolist(),
                using="colbert",
                limit=top_k * 4,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )
    ids: list[str] = []
    for p in res.points:
        if float(p.score) < node_rrf_threshold:
            continue
        nid = (p.payload or {}).get("node_id")
        if nid:
            ids.append(nid)
    return ids
