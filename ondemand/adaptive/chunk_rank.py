from time import perf_counter

import numpy as np

from ..text import ChunkBM25, windows
from .registry import register

ALPHA, DEPTH = 0.7, 100


class Chunk:
    def __init__(self, page_id, text, page_rank, position):
        self.page_id, self.text, self.page_rank, self.position = page_id, text, page_rank, position


@register("chunker", "fixed")
class FixedChunker:
    name = "fixed"

    def __init__(self, argument, n_words=512, overlap=128, **_):
        parts = [p for p in (argument or "").split(":") if p]
        self.n_words = int(parts[0]) if parts else int(n_words)
        self.overlap = int(parts[1]) if len(parts) > 1 else int(overlap)

    def __call__(self, passages):
        out = []
        for passage in passages:
            for start, end in windows(passage.text, self.n_words, self.overlap):
                piece = passage.text[start:end]
                if piece.strip():
                    out.append(Chunk(passage.page_id, piece, passage.rank, len(out)))
        return out

    def describe(self):
        return f"fixed:{self.n_words}:{self.overlap}"


def minmax(values):
    if not values:
        return {}
    low, high = min(values.values()), max(values.values())
    if high - low <= 1e-12:
        return {k: 1.0 for k in values}
    return {k: (v - low) / (high - low) for k, v in values.items()}


class Ranker:
    name = "base"

    def rank(self, query, chunks, top_k):
        raise NotImplementedError

    def describe(self):
        return self.name


@register("ranker", "page_order")
class PageOrderRanker(Ranker):
    name = "page_order"

    def __init__(self, argument, **_):
        pass

    def rank(self, query, chunks, top_k):
        return chunks[:top_k], {}


@register("ranker", "hybrid")
class HybridRanker(Ranker):
    name = "hybrid"

    def __init__(self, argument, embedder=None, alpha=ALPHA, depth=DEPTH, **_):
        self.alpha = float(argument) if argument else float(alpha)
        self.depth = int(depth)
        self.embedder = embedder

    def rank(self, query, chunks, top_k):
        if not chunks:
            return [], {}
        bm25 = ChunkBM25([c.text for c in chunks])
        lexical = minmax(dict(bm25.search(query, self.depth)))
        vectors = self.embedder([c.text for c in chunks])
        qvec = self.embedder([query])[0]
        dense = minmax({i: float(s) for i, s in enumerate(vectors @ qvec)})
        fused = {}
        for position in range(len(chunks)):
            fused[position] = ((1 - self.alpha) * lexical.get(position, 0.0)
                               + self.alpha * dense.get(position, 0.0))
        order = sorted(fused, key=lambda p: (-fused[p], p))[:top_k]
        return [chunks[p] for p in order], {"alpha_dense": self.alpha, "n_chunks": len(chunks),
                                            "scores": [round(fused[p], 6) for p in order]}

    def describe(self):
        return f"hybrid:{self.alpha}"


class ChunkRank:
    def __init__(self, chunker, ranker, top_chunks=10):
        self.chunker, self.ranker, self.top_chunks = chunker, ranker, top_chunks

    def run(self, query, passages):
        started = perf_counter()
        chunks = self.chunker(passages)
        top, notes = self.ranker.rank(query, chunks, self.top_chunks)
        return top, {"seconds": perf_counter() - started, "chunker": self.chunker.describe(),
                     "ranker": self.ranker.describe(), **notes}
