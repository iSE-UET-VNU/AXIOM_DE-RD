from time import perf_counter

import numpy as np

from ..text import ChunkBM25, real_text, windows
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
            text = real_text(passage.text)
            for start, end in windows(text, self.n_words, self.overlap):
                piece = text[start:end]
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


def shingles(text, n=3):
    words = real_text(text or "").lower().split()
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def dedupe(items, text_of, threshold, limit):
    kept, sets, replaced, skipped = [], [], 0, 0
    for item in items:
        if len(kept) >= limit:
            break
        s = shingles(text_of(item))
        twin = next((j for j, t in enumerate(sets)
                     if s and t and len(s & t) >= threshold * min(len(s), len(t))), None)
        if twin is None:
            kept.append(item)
            sets.append(s)
        elif len(s) > len(sets[twin]):
            kept[twin], sets[twin] = item, s
            replaced += 1
        else:
            skipped += 1
    return kept, {"dedupe_threshold": threshold, "dedupe_skipped": skipped, "dedupe_replaced": replaced}


def reading_order(chunk):
    doc, _, page = chunk.page_id.partition("#page=")
    return doc, int(page) if page.isdigit() else 0, chunk.position


class ChunkRank:
    def __init__(self, chunker, ranker, top_chunks=10, dedupe=0.0):
        self.chunker, self.ranker, self.top_chunks, self.dedupe = chunker, ranker, top_chunks, dedupe

    def run(self, query, passages):
        started = perf_counter()
        chunks = self.chunker(passages)
        if self.dedupe:
            ranked, notes = self.ranker.rank(query, chunks, len(chunks))
            scores = notes.pop("scores", [None] * len(ranked))
            pairs, info = dedupe(list(zip(ranked, scores)), lambda pair: pair[0].text, self.dedupe, self.top_chunks)
            top = [c for c, _ in pairs]
            notes.update(info, scores=[s for _, s in pairs])
        else:
            top, notes = self.ranker.rank(query, chunks, self.top_chunks)
        top = sorted(top, key=reading_order)
        return top, {"seconds": perf_counter() - started, "chunker": self.chunker.describe(),
                     "ranker": self.ranker.describe(), "order": "reading", **notes}
