"""Retrieval arms. Each takes an index handle and a query; none knows the corpus.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .fusion import Hit, alpha_fuse
from .index import LocalIndex
from .protocol import ScoredChunk

RRF_K = 60


def _to_chunks(index: LocalIndex, hits: Sequence[Hit], scores_by_leg: Mapping[str, Mapping[str, float]] | None = None) -> list[ScoredChunk]:
    out: list[ScoredChunk] = []
    for rank, (chunk_id, score) in enumerate(hits, start=1):
        record = index.chunks([chunk_id]).get(chunk_id)
        if record is None:
            continue
        detail = {}
        if scores_by_leg:
            detail = {leg: v[chunk_id] for leg, v in scores_by_leg.items() if chunk_id in v}
        out.append(
            ScoredChunk(
                chunk_id=chunk_id,
                doc_id=record.doc_id,
                score=float(score),
                rank=rank,
                text=record.text,
                scores=detail,
            )
        )
    return out


@dataclass
class BM25Retriever:
    index: LocalIndex
    retriever_id: str = "bm25"

    def raw(self, query: str, k: int, scope: list[str] | None = None) -> list[Hit]:
        if self.index.bm25 is None:
            return []
        allowed = self.index.scope_positions(scope)
        hits = self.index.bm25.search(query, k, allowed)
        return [(self.index.record_at(position).chunk_id, score) for position, score in hits]

    def retrieve(self, query: str, k: int, scope: list[str] | None = None) -> list[ScoredChunk]:
        return _to_chunks(self.index, self.raw(query, k, scope))

    def params(self) -> Mapping[str, Any]:
        bm25 = self.index.bm25
        return {
            "k1": getattr(bm25, "k1", None),
            "b": getattr(bm25, "b", None),
            "analyzer": getattr(bm25, "analyzer_name", None),
            "scope_mode": "in_scoring_v2",
        }


@dataclass
class DenseRetriever:
    index: LocalIndex
    retriever_id: str = "dense"

    def raw(self, query: str, k: int, scope: list[str] | None = None) -> list[Hit]:
        if not self.index.has_dense:
            return []
        vector = np.asarray(self.index.encode_query(query).require("dense"), dtype=np.float32)
      
        if self.index.metric == "cosine":
            vector = vector / max(float(np.linalg.norm(vector)), 1e-12)
        scores = self.index.vectors @ vector
        mask = self.index.scope_mask(scope)
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
        count = min(k, int(np.isfinite(scores).sum()))
        if count <= 0:
            return []
        top = np.argpartition(-scores, count - 1)[:count]
        top = top[np.argsort(-scores[top])]
        return [(self.index.record_at(int(i)).chunk_id, float(scores[i])) for i in top]

    def retrieve(self, query: str, k: int, scope: list[str] | None = None) -> list[ScoredChunk]:
        return _to_chunks(self.index, self.raw(query, k, scope))

    def params(self) -> Mapping[str, Any]:
        return {"embeddings_model": self.index.embeddings_model, "metric": self.index.metric}


@dataclass
class RrfRetriever:

    index: LocalIndex
    depth: int = 200
    k_constant: int = RRF_K
    retriever_id: str = "rrf"

    def __post_init__(self) -> None:
        self.sparse = BM25Retriever(self.index)
        self.dense = DenseRetriever(self.index)

    def raw(self, query: str, k: int, scope: list[str] | None = None) -> list[Hit]:
        runs = [
            self.sparse.raw(query, self.depth, scope),
            self.dense.raw(query, self.depth, scope),
        ]
        fused: dict[str, float] = {}
        for run in runs:
            for rank, (chunk_id, _) in enumerate(run, start=1):
                fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (self.k_constant + rank)
        return sorted(fused.items(), key=lambda item: -item[1])[:k]

    def retrieve(self, query: str, k: int, scope: list[str] | None = None) -> list[ScoredChunk]:
        legs = {
            "bm25": dict(self.sparse.raw(query, self.depth, scope)),
            "dense": dict(self.dense.raw(query, self.depth, scope)),
        }
        return _to_chunks(self.index, self.raw(query, k, scope), legs)

    def params(self) -> Mapping[str, Any]:
        return {"depth": self.depth, "k": self.k_constant, **self.sparse.params()}


@dataclass
class AlphaRetriever:
  

    index: LocalIndex
    alpha: float = 0.7
    depth: int = 200
    retriever_id: str = "alpha"

    def __post_init__(self) -> None:
        self.sparse = BM25Retriever(self.index)
        self.dense = DenseRetriever(self.index)
        self.retriever_id = f"alpha{self.alpha:g}"

    def raw(self, query: str, k: int, scope: list[str] | None = None) -> list[Hit]:
        return alpha_fuse(
            self.dense.raw(query, self.depth, scope),
            self.sparse.raw(query, self.depth, scope),
            self.alpha,
            k,
        )

    def retrieve(self, query: str, k: int, scope: list[str] | None = None) -> list[ScoredChunk]:
        legs = {
            "bm25": dict(self.sparse.raw(query, self.depth, scope)),
            "dense": dict(self.dense.raw(query, self.depth, scope)),
        }
        return _to_chunks(self.index, self.raw(query, k, scope), legs)

    def params(self) -> Mapping[str, Any]:
        return {"alpha": self.alpha, "depth": self.depth, **self.sparse.params()}


BUILDERS = {
    "bm25": lambda index, **kw: BM25Retriever(index),
    "dense": lambda index, **kw: DenseRetriever(index),
    "rrf": lambda index, **kw: RrfRetriever(index, **kw),
    "alpha": lambda index, **kw: AlphaRetriever(index, **kw),
}


def build(name: str, index: LocalIndex, **kwargs: Any) -> Any:
    if name.startswith("alpha") and name != "alpha":
        return AlphaRetriever(index, alpha=float(name[5:]))
    if name not in BUILDERS:
        raise ValueError(f"Unknown retriever {name!r}. Known: {sorted(BUILDERS)}")
    return BUILDERS[name](index, **kwargs)
