"""LOMA knowledge graph: load nodes + edges, expose retrieval helpers.

Source of truth: out/nodes.jsonl, out/edges.jsonl (produced by ingest.py).
Optional dependency: Qdrant collection 'loma_nodes' for semantic node lookup
(populated by index.py). All semantic methods lazy-init clients.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Iterable, Optional

import networkx as nx

from loma_rag.config.settings import OUT_DIR as OUT, azure, qdrant as qdrant_cfg
from loma_rag.constant.collections import NODES_COLL
from loma_rag.constant.models import BM42_MODEL


@dataclass
class Node:
    node_id: str
    raw_id: str
    name: str
    definition: str
    category: str
    domain_tags: str
    related_raw: str
    course: str
    module: str
    lesson: str
    lesson_id: str
    source: str

    @classmethod
    def from_dict(cls, d: dict) -> "Node":
        keys = {f.name for f in fields(cls)}
        return cls(**{k: d.get(k, "") for k in keys})


class KG:
    """In-memory knowledge graph backed by NetworkX (undirected)."""

    def __init__(self, graph: nx.Graph, nodes_by_id: dict[str, Node]) -> None:
        self.g = graph
        self._nodes = nodes_by_id
        self._by_name: dict[str, list[str]] = {}
        for n in nodes_by_id.values():
            if n.name:
                self._by_name.setdefault(n.name.lower(), []).append(n.node_id)

        # lazy semantic-search clients
        self._dense_client = None
        self._sparse_model = None
        self._qc = None

    # ---- structural ----

    def get(self, node_id: str) -> Optional[Node]:
        return self._nodes.get(node_id)

    def neighbors(self, node_id: str, hops: int = 1) -> list[Node]:
        """BFS up to `hops` hops, excluding the seed itself."""
        if node_id not in self.g:
            return []
        visited = {node_id}
        frontier = {node_id}
        for _ in range(hops):
            nxt: set[str] = set()
            for nid in frontier:
                for nb in self.g.neighbors(nid):
                    if nb not in visited:
                        nxt.add(nb)
                        visited.add(nb)
            frontier = nxt
        visited.discard(node_id)
        return [self._nodes[n] for n in visited if n in self._nodes]

    def find_by_name(self, name: str, lesson_id: Optional[str] = None) -> list[Node]:
        ids = self._by_name.get(name.lower(), [])
        out = [self._nodes[i] for i in ids]
        if lesson_id:
            out = [n for n in out if n.lesson_id == lesson_id]
        return out

    def expand_for_chunks(
            self,
            chunk_payloads: Iterable[dict],
            hops: int = 1,
            max_nodes: int = 30,
    ) -> list[Node]:
        """Pull nodes referenced by chunks, plus their k-hop neighborhood.

        Returns a list including the seed nodes themselves first, then
        BFS-discovered relatives, capped at max_nodes.
        """
        seeds: list[str] = []
        seen: set[str] = set()
        for c in chunk_payloads:
            for nid in c.get("node_refs") or []:
                if nid not in seen and nid in self.g:
                    seeds.append(nid)
                    seen.add(nid)
        return self._expand(seeds, hops, max_nodes)

    def _expand(self, seeds: list[str], hops: int, max_nodes: int) -> list[Node]:
        result: list[str] = []
        visited: set[str] = set()
        frontier: set[str] = set()
        for s in seeds:
            if s in self.g and s not in visited:
                result.append(s)
                visited.add(s)
                frontier.add(s)
                if len(result) >= max_nodes:
                    return [self._nodes[n] for n in result]
        for _ in range(hops):
            if len(result) >= max_nodes:
                break
            nxt: set[str] = set()
            for nid in frontier:
                for nb in self.g.neighbors(nid):
                    if nb in visited:
                        continue
                    visited.add(nb)
                    nxt.add(nb)
                    result.append(nb)
                    if len(result) >= max_nodes:
                        break
                if len(result) >= max_nodes:
                    break
            frontier = nxt
        return [self._nodes[n] for n in result if n in self._nodes]

    # ---- semantic (uses Qdrant 'loma_nodes' collection) ----

    def _ensure_clients(self) -> None:
        if self._dense_client is None:
            from openai import AzureOpenAI

            self._dense_client = AzureOpenAI(
                azure_endpoint=azure.api_base.rstrip("/"),
                api_key=azure.api_key,
                api_version=azure.api_version,
            )
        if self._sparse_model is None:
            from fastembed import SparseTextEmbedding

            self._sparse_model = SparseTextEmbedding(model_name=BM42_MODEL)
        if self._qc is None:
            from qdrant_client import QdrantClient

            self._qc = QdrantClient(url=qdrant_cfg.url, timeout=30)

    def semantic_search(self, query: str, top_k: int = 5) -> list[tuple[Node, float]]:
        """Hybrid (dense + BM42) lookup against the loma_nodes collection."""
        from qdrant_client.models import Fusion, FusionQuery, Prefetch, SparseVector

        self._ensure_clients()
        emb = (
            self._dense_client.embeddings.create(
                model=azure.embedding_model,
                input=[query],
            )
            .data[0]
            .embedding
        )
        sv = next(iter(self._sparse_model.embed([query])))
        res = self._qc.query_points(
            collection_name=NODES_COLL,
            prefetch=[
                Prefetch(query=emb, using="dense", limit=20),
                Prefetch(
                    query=SparseVector(
                        indices=sv.indices.tolist(),
                        values=sv.values.tolist(),
                    ),
                    using="sparse",
                    limit=20,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )
        out: list[tuple[Node, float]] = []
        for p in res.points:
            nid = (p.payload or {}).get("node_id")
            if nid and nid in self._nodes:
                out.append((self._nodes[nid], float(p.score)))
        return out


# ---- loader ----

def load_graph() -> KG:
    nodes_path = OUT / "nodes.jsonl"
    edges_path = OUT / "edges.jsonl"
    if not nodes_path.exists() or not edges_path.exists():
        raise FileNotFoundError(
            f"missing {nodes_path.name} or {edges_path.name} - run ingest.py first"
        )

    nodes_by_id: dict[str, Node] = {}
    for line in nodes_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        n = Node.from_dict(d)
        nodes_by_id[n.node_id] = n

    g = nx.Graph()
    for nid in nodes_by_id:
        g.add_node(nid)

    n_edges = 0
    for line in edges_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        s, t = d.get("src"), d.get("dst")
        if s in nodes_by_id and t in nodes_by_id:
            g.add_edge(s, t, relation=d.get("relation", "related_to"))
            n_edges += 1

    print(f"[graph] loaded {len(nodes_by_id)} nodes, {n_edges} edges from {OUT}")
    return KG(g, nodes_by_id)


# ---- self-test ----

def _self_test() -> int:
    kg = load_graph()
    print(f"\n=== KG STATS ===")
    print(f"nodes: {len(kg._nodes)}  edges: {kg.g.number_of_edges()}")
    print(f"connected components: {nx.number_connected_components(kg.g)}")

    by_degree = sorted(kg.g.nodes, key=lambda n: kg.g.degree[n], reverse=True)
    print(f"\nTop-5 highest-degree nodes:")
    for nid in by_degree[:5]:
        n = kg._nodes[nid]
        print(f"  deg={kg.g.degree[nid]:>3}  {n.name!r:35} ({n.lesson_id})")

    # 1-hop demo on the highest-degree node
    seed_id = by_degree[0]
    seed = kg._nodes[seed_id]
    print(f"\n--- neighbors of {seed.name!r} (1 hop) ---")
    for n in kg.neighbors(seed_id, hops=1)[:10]:
        print(f"  - {n.name!r:35} ({n.category})")

    # find_by_name demo
    pick = "Annuity"
    print(f"\n--- find_by_name({pick!r}) ---")
    matches = kg.find_by_name(pick)
    print(f"  {len(matches)} hit(s)")
    for n in matches:
        print(f"    {n.node_id}  ({n.lesson_id})  cat={n.category!r}")
        for nb in kg.neighbors(n.node_id, hops=1)[:5]:
            print(f"        -> {nb.name}")

    # expand_for_chunks demo
    print(f"\n--- expand_for_chunks (seed = '{seed.name}') ---")
    fake = [{"chunk_id": "demo", "node_refs": [seed_id]}]
    rel = kg.expand_for_chunks(fake, hops=1, max_nodes=12)
    print(f"  -> {len(rel)} nodes (seed + 1-hop):")
    for n in rel:
        marker = "*" if n.node_id == seed_id else " "
        print(f"   {marker} {n.name!r:35}  -- {n.definition[:70]}")

    # semantic_search demo (skipped if Qdrant unavailable)
    print(f"\n--- semantic_search via loma_nodes ---")
    queries = [
        "people who need insurance more tend to apply for it more",
        "outliving your savings in retirement",
        "company that reinsures risk for an insurer",
        "fee charged when withdrawing from an annuity early",
    ]
    try:
        for q in queries:
            print(f"\n  q: {q!r}")
            for n, sc in kg.semantic_search(q, top_k=3):
                print(f"    [{sc:.3f}] {n.name!r:35} ({n.lesson_id})")
                print(f"            -- {n.definition[:90]}")
    except Exception as e:
        print(f"  ! semantic search skipped: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(_self_test())
