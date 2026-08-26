"""Score fusion and document-level aggregation.

* Normalization for BM25
* alpha weights the DENSE side
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

Hit = tuple[str, float]


def minmax(hits: Sequence[Hit]) -> dict[str, float]:
    if not hits:
        return {}
    scores = [score for _, score in hits]
    low, high = min(scores), max(scores)
    if high - low < 1e-12:
        return {key: 1.0 for key, _ in hits}
    return {key: (score - low) / (high - low) for key, score in hits}


def alpha_fuse(dense: Sequence[Hit], sparse: Sequence[Hit], alpha: float, top_k: int) -> list[Hit]:
    """fused = alpha * dense_norm + (1 - alpha) * sparse_norm."""
    left, right = minmax(dense), minmax(sparse)
    fused: dict[str, float] = defaultdict(float)
    for key, score in left.items():
        fused[key] += alpha * score
    for key, score in right.items():
        fused[key] += (1.0 - alpha) * score
    return sorted(fused.items(), key=lambda item: -item[1])[:top_k]


def distinct_documents(
    hits: Sequence[Hit], doc_of: dict[str, str], depth: int
) -> tuple[list[Hit], list[Hit]]:
    """Split into ``depth`` distinct documents (best chunk each) plus the rest.
    """
    head: list[Hit] = []
    rest: list[Hit] = []
    seen: set[str] = set()
    for chunk_id, score in hits:
        doc = doc_of.get(chunk_id, chunk_id)
        if doc not in seen and len(head) < depth:
            seen.add(doc)
            head.append((chunk_id, score))
        else:
            rest.append((chunk_id, score))
    return head, rest


def merge_reranked(reranked: Sequence[Hit], tail: Sequence[Hit]) -> list[Hit]:
    floor = min((score for _, score in reranked), default=0.0)
    below = [(chunk_id, floor - 1.0 - rank) for rank, (chunk_id, _) in enumerate(tail)]
    return list(reranked) + below
