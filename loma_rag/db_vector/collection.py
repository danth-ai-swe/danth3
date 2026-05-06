"""Qdrant collection schema setup (3 named vectors: dense + sparse + ColBERT)."""
from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    Modifier,
    MultiVectorComparator,
    MultiVectorConfig,
    OptimizersConfigDiff,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    SparseIndexParams,
    SparseVectorParams,
    StrictModeConfig,
    VectorParams,
)

from loma_rag.constant.collections import COLBERT_DIM, DENSE_DIM


def ensure_collection(qc: QdrantClient, name: str, recreate: bool) -> None:
    exists = qc.collection_exists(name)
    if exists and not recreate:
        print(f"  [{name}] exists, keeping (--keep)")
        return
    if exists:
        print(f"  [{name}] dropping existing collection")
        qc.delete_collection(name)
    qc.create_collection(
        collection_name=name,
        vectors_config={
            "dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE),
            # Late-interaction ColBERT: stored as multivector (one 128d vector
            # per token); MaxSim comparator. HNSW disabled (m=0) — used as a
            # rerank stage on prefetch candidates, not for nearest-neighbour
            # search on its own (saves memory).
            "colbert": VectorParams(
                size=COLBERT_DIM,
                distance=Distance.COSINE,
                multivector_config=MultiVectorConfig(
                    comparator=MultiVectorComparator.MAX_SIM,
                ),
                hnsw_config=HnswConfigDiff(m=0),
            ),
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(
                index=SparseIndexParams(on_disk=False),
                modifier=Modifier.IDF,
            )
        },
        # Collection-level config (applies to dense vectors).
        optimizers_config=OptimizersConfigDiff(
            indexing_threshold=20_000,
            default_segment_number=5,
            max_segment_size=5_000_000,
        ),
        hnsw_config=HnswConfigDiff(
            m=10,
            on_disk=False,
        ),
        strict_mode_config=StrictModeConfig(
            enabled=True,
            max_timeout=10,
            upsert_max_batchsize=1_000,
            read_rate_limit=5_000,
            write_rate_limit=5_000,
        ),
        quantization_config=ScalarQuantization(
            scalar=ScalarQuantizationConfig(
                type=ScalarType.INT8,
                always_ram=True,
                quantile=0.99,
            ),
        ),
    )
    print(
        f"  [{name}] created  (dense={DENSE_DIM}d cosine + INT8 quant, "
        f"sparse=BM42+IDF, colbert={COLBERT_DIM}d MaxSim)"
    )
