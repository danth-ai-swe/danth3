"""Index chunks + nodes into Qdrant with named vectors (dense + sparse + ColBERT).

Inputs:  out/chunks.jsonl, out/nodes.jsonl
Output:  Qdrant collections "loma_chunks", "loma_nodes" at ${QDRANT_URL}

Vectors:
  dense  = Azure OpenAI text-embedding-3-small (1536 dim, cosine)
  sparse = fastembed Qdrant/bm42-all-minilm-l6-v2-attentions, with IDF modifier
  colbert = fastembed colbert-ir/colbertv2.0 (late-interaction multi-vector)

Run: python -m loma_rag.ingest.indexer            # rebuilds both collections from scratch
     python -m loma_rag.ingest.indexer --keep     # skip if collection already exists
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, SparseVector

from loma_rag.config.settings import OUT_DIR as OUT, qdrant as qdrant_cfg
from loma_rag.constant.collections import CHUNKS_COLL, COLBERT_DIM, DENSE_DIM, NODES_COLL
from loma_rag.constant.models import BM42_MODEL, COLBERT_MODEL
from loma_rag.constant.thresholds import EMBED_BATCH, UPSERT_BATCH
from loma_rag.db_vector.collection import ensure_collection
from loma_rag.db_vector.upsert import stable_id, upsert_with_retry
from loma_rag.llm.embedding import embed_dense_batch
from loma_rag.llm.openai_client import make_dense_client
from loma_rag.util.io import load_jsonl


def index_docs(
    qc: QdrantClient,
    dense_client,
    sparse_model,
    colbert_model,
    coll: str,
    docs: list[dict],
    text_key: str,
    id_key: str,
) -> None:
    """Batch-embed and upsert documents into a Qdrant collection.

    Indexer uses batch-capable model instances directly; the single-text helpers
    in llm.embedding are for the retriever hot path.
    """
    if not docs:
        print(f"  [{coll}] no docs to index")
        return

    print(f"  [{coll}] indexing {len(docs)} docs (embed batch={EMBED_BATCH})")
    pending: list[PointStruct] = []
    done = 0
    t0 = time.time()
    for i in range(0, len(docs), EMBED_BATCH):
        batch = docs[i : i + EMBED_BATCH]
        texts = [d[text_key] for d in batch]

        dense_vecs = embed_dense_batch(dense_client, texts)
        sparse_vecs = list(sparse_model.embed(texts))
        colbert_vecs = list(colbert_model.embed(texts))

        for d, dv, sv, cv in zip(batch, dense_vecs, sparse_vecs, colbert_vecs):
            payload = {k: v for k, v in d.items() if not k.startswith("_")}
            pending.append(
                PointStruct(
                    id=stable_id(d[id_key]),
                    vector={
                        "dense": dv,
                        "sparse": SparseVector(
                            indices=sv.indices.tolist(),
                            values=sv.values.tolist(),
                        ),
                        # ColBERT: list of per-token 128-d vectors.
                        "colbert": cv.tolist(),
                    },
                    payload=payload,
                )
            )

        if len(pending) >= UPSERT_BATCH:
            upsert_with_retry(qc, coll, pending)
            done += len(pending)
            pending = []
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            print(f"    upserted {done}/{len(docs)}  ({rate:.1f} docs/s)")

    if pending:
        upsert_with_retry(qc, coll, pending)
        done += len(pending)
        print(f"    upserted {done}/{len(docs)}")

    info = qc.get_collection(coll)
    print(f"  [{coll}] done — points_count={info.points_count}")


def main() -> int:
    # Import model classes here (only needed at index time, not import time).
    from fastembed import LateInteractionTextEmbedding, SparseTextEmbedding

    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="skip collections that already exist")
    ap.add_argument("--only", choices=["chunks", "nodes"], help="index only one collection")
    args = ap.parse_args()

    chunks = load_jsonl(OUT / "chunks.jsonl")
    nodes = load_jsonl(OUT / "nodes.jsonl")
    print(f"Loaded: {len(chunks)} chunks, {len(nodes)} nodes from {OUT}")

    print(f"\nConnecting Qdrant @ {qdrant_cfg.url}")
    qc = QdrantClient(url=qdrant_cfg.url, prefer_grpc=False, timeout=60)
    qc.get_collections()  # ping

    print("\nDense client smoke test (Azure OpenAI)")
    dense_client = make_dense_client()
    sample = embed_dense_batch(dense_client, ["hello world"])
    actual_dim = len(sample[0])
    print(f"  ok — dim={actual_dim}")
    if actual_dim != DENSE_DIM:
        print(f"  ! adjusting DENSE_DIM from {DENSE_DIM} to {actual_dim}")

    print("\nLoading BM42 sparse model (downloaded on first run)…")
    sparse_model = SparseTextEmbedding(model_name=BM42_MODEL)
    _ = list(sparse_model.embed(["warmup"]))
    print("  ok")

    print("\nLoading ColBERT v2 late-interaction model…")
    colbert_model = LateInteractionTextEmbedding(model_name=COLBERT_MODEL)
    _ = list(colbert_model.embed(["warmup"]))
    print("  ok")

    print("\nEnsuring collections")
    if args.only != "nodes":
        ensure_collection(qc, CHUNKS_COLL, recreate=not args.keep)
    if args.only != "chunks":
        ensure_collection(qc, NODES_COLL, recreate=not args.keep)

    print("\nIndexing")
    if args.only != "nodes":
        index_docs(qc, dense_client, sparse_model, colbert_model,
                   CHUNKS_COLL, chunks, "text", "chunk_id")
    if args.only != "chunks":
        # For nodes: combine name + definition as embed text.
        for n in nodes:
            n["_text"] = f"{n.get('name','')}\n{n.get('definition','')}".strip()
        index_docs(qc, dense_client, sparse_model, colbert_model,
                   NODES_COLL, nodes, "_text", "node_id")

    print("\nAll done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
