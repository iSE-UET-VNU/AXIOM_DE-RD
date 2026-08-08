"""Corpus-wide retrieval: BM25, dense, fusion, and chunk->document aggregation.

Main ships BM25 *inputs* at per-document scope and no scorer, so the corpus-wide
statistics and the ranking function live here. The analyzer is imported from
``chunking_embedding.lexical`` so tokens stay identical to what the pipeline
already writes; the Vietnamese word-segmentation arm wraps that analyzer rather
than replacing it.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence
import math
import sys
from pathlib import Path

from src.utils.paths import repo_root

import numpy as np

PROJECT_ROOT = repo_root(__file__)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chunking_embedding.lexical import analyze  # noqa: E402

Analyzer = Callable[[str], list[str]]


def plain_analyzer(text: str) -> list[str]:
    return list(analyze(text))


def segmented_analyzer(text: str) -> list[str]:
    """Vietnamese word segmentation before the shared analyzer.

    pyvi joins compound words with "_", but the shared token pattern excludes
    underscore, so passing its output straight to analyze() splits every compound
    back apart and silently reduces to plain_analyzer. Normalize each segmented
    unit separately and rejoin, so "Đại_học" survives as one term.
    """
    from pyvi import ViTokenizer

    tokens: list[str] = []
    for unit in ViTokenizer.tokenize(text or "").split():
        parts = analyze(unit)
        if parts:
            tokens.append("_".join(parts))
    return tokens


ANALYZERS: dict[str, Analyzer] = {"plain": plain_analyzer, "segmented": segmented_analyzer}


@dataclass
class BM25Index:
    k1: float = 1.2
    b: float = 0.75
    analyzer: Analyzer = plain_analyzer
    ids: list[str] = field(default_factory=list)
    doc_ids: list[str] = field(default_factory=list)
    postings: dict[str, list[tuple[int, int]]] = field(default_factory=lambda: defaultdict(list))
    lengths: list[int] = field(default_factory=list)
    avgdl: float = 0.0

    def build(self, chunks: Sequence[Any]) -> "BM25Index":
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for position, chunk in enumerate(chunks):
            tokens = self.analyzer(chunk.index_text)
            counts: dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            for term, count in counts.items():
                postings[term].append((position, count))
            self.ids.append(chunk.chunk_id)
            self.doc_ids.append(chunk.doc_id)
            self.lengths.append(len(tokens))
        self.postings = postings
        self.avgdl = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        return self

    def search(self, query: str, top_k: int = 100) -> list[tuple[int, float]]:
        n = len(self.lengths)
        if not n:
            return []
        scores = np.zeros(n, dtype=np.float32)
        for term in self.analyzer(query):
            posting = self.postings.get(term)
            if not posting:
                continue
            df = len(posting)
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            for position, freq in posting:
                length = self.lengths[position] or 1
                denominator = freq + self.k1 * (1 - self.b + self.b * length / (self.avgdl or 1.0))
                scores[position] += idf * (freq * (self.k1 + 1)) / denominator
        return _top(scores, top_k)


@dataclass
class DenseIndex:
    matrix: np.ndarray | None = None
    ids: list[str] = field(default_factory=list)
    doc_ids: list[str] = field(default_factory=list)

    def build(self, chunks: Sequence[Any], vectors: np.ndarray) -> "DenseIndex":
        normalized = vectors.astype(np.float32)
        norms = np.linalg.norm(normalized, axis=1, keepdims=True)
        self.matrix = normalized / np.clip(norms, 1e-12, None)
        self.ids = [chunk.chunk_id for chunk in chunks]
        self.doc_ids = [chunk.doc_id for chunk in chunks]
        return self

    def search(self, query_vector: np.ndarray, top_k: int = 100) -> list[tuple[int, float]]:
        if self.matrix is None:
            return []
        vector = query_vector.astype(np.float32)
        vector = vector / max(float(np.linalg.norm(vector)), 1e-12)
        return _top(self.matrix @ vector, top_k)


def _top(scores: np.ndarray, top_k: int) -> list[tuple[int, float]]:
    if scores.size == 0:
        return []
    count = min(top_k, scores.size)
    candidates = np.argpartition(-scores, count - 1)[:count]
    ordered = candidates[np.argsort(-scores[candidates])]
    return [(int(i), float(scores[i])) for i in ordered if scores[i] > 0]


def rrf(runs: Iterable[list[tuple[int, float]]], k: int = 60, top_k: int = 100) -> list[tuple[int, float]]:
    fused: dict[int, float] = defaultdict(float)
    for run in runs:
        for rank, (position, _) in enumerate(run, start=1):
            fused[position] += 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda item: -item[1])[:top_k]


def alpha_fuse(
    lexical: list[tuple[int, float]],
    dense: list[tuple[int, float]],
    alpha: float = 0.5,
    top_k: int = 100,
) -> list[tuple[int, float]]:
    """Min-max normalize each run over the fused pool, then weight and add."""
    left, right = _minmax(lexical), _minmax(dense)
    fused: dict[int, float] = defaultdict(float)
    for position, score in left.items():
        fused[position] += (1.0 - alpha) * score
    for position, score in right.items():
        fused[position] += alpha * score
    return sorted(fused.items(), key=lambda item: -item[1])[:top_k]


def _minmax(run: list[tuple[int, float]]) -> dict[int, float]:
    if not run:
        return {}
    values = [score for _, score in run]
    low, high = min(values), max(values)
    span = high - low
    if span <= 0:
        return {position: 1.0 for position, _ in run}
    return {position: (score - low) / span for position, score in run}


def aggregate(
    ranked: list[tuple[int, float]],
    doc_ids: Sequence[str],
    method: str = "maxp",
    top_k: int = 2,
) -> list[tuple[str, float]]:
    """Collapse a chunk ranking into a document ranking."""
    grouped: dict[str, list[float]] = defaultdict(list)
    for position, score in ranked:
        grouped[doc_ids[position]].append(score)
    if method == "maxp":
        scored = {doc: max(values) for doc, values in grouped.items()}
    elif method == "sum_topk":
        scored = {doc: sum(sorted(values, reverse=True)[:top_k]) for doc, values in grouped.items()}
    elif method == "sum":
        scored = {doc: sum(values) for doc, values in grouped.items()}
    else:
        raise ValueError(f"Unknown aggregation {method!r}")
    return sorted(scored.items(), key=lambda item: -item[1])
