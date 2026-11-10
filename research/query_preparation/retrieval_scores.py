"""Per-page score breakdown of the saved light retriever (BM25 + dense + file prior).

Re-computes the ranking stored in
``data/work/adaptive_retrieval_20261020_router_te3s_top100/retrieval_scores.jsonl``
(variant ``hier_dense_pool_k103_enfrbm250.30_te3s0.70_all``) so the UI can show
*why* a page is ranked where it is. The formula was recovered by matching the stored
scores (max abs error ~2e-7 on probed queries; ``verify`` checks all 220):

    b      = max_norm(BM25 raw over the 5,091 PP-OCRv5 pages)       # enfr analyzer
    d      = max_norm(page dense) with page dense = max cosine over its 512-word
             chunks (text-embedding-3-small, query = raw query text)
    base   = 0.30 * b + 0.70 * d
    file   = max_norm( max_{page in file} d )                       # "dense_pool", direct BM25 weight 0
    final  = 0.85 * base + 0.15 * file[page's file]

``max_norm(x) = clip(x, 0) / max(x)``. No language routing is applied in this variant.

Run ``python research/query_preparation/retrieval_scores.py verify`` to re-check.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/baselines/hierarchical_light"))
from ondemand.text import PageBM25, enfr, real_text, windows  # noqa: E402

HANDOFF = (
    ROOT
    / "data/work/adaptive_routers_handoff/adaptive_routers_handoff/data/work/ondemand_v2/ef7f11811803"
)
PAGES_PATH = HANDOFF / "light_prep/pages_ppocrv5.jsonl"
EMBED_CACHE = HANDOFF / "embedding_cache/te3s_ppocr"
SAVED_SCORES = ROOT / "data/work/adaptive_retrieval_20261020_router_te3s_top100/retrieval_scores.jsonl"
CHUNK_CACHE = ROOT / "research/query_preparation/results/cache/te3s_chunks.npz"
EMBED_MODEL = "openai/text-embedding-3-small"
W_DENSE, PARENT_WEIGHT = 0.70, 0.15
CHUNK_WORDS, CHUNK_OVERLAP = 512, 128

TOKEN = re.compile(r"[^\W_]+(?:'[^\W_]+)*", flags=re.UNICODE)


@lru_cache(maxsize=200_000)
def token_stems(token: str) -> tuple[str, ...]:
    return tuple(enfr(token))


def query_terms(query: str) -> dict[str, str]:
    """Analyzer stem -> first query word that produced it."""
    terms: dict[str, str] = {}
    for match in TOKEN.finditer(query):
        for stem in token_stems(match.group(0)):
            terms.setdefault(stem, match.group(0))
    return terms


def max_norm(values: np.ndarray) -> np.ndarray:
    top = float(values.max()) if values.size else 0.0
    return np.clip(values, 0, None) / top if top > 0 else np.zeros_like(values, dtype=np.float64)


def ranks_desc(values: np.ndarray) -> np.ndarray:
    """1-based rank of every item by descending value (ties broken by index)."""
    order = np.argsort(-values, kind="stable")
    ranks = np.empty(len(values), dtype=np.int32)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


class MissingEmbedding(RuntimeError):
    """The query text has no cached text-embedding-3-small vector."""


@dataclass
class PageScores:
    """Score arrays over all pages for one query (index = page position in corpus)."""

    query: str
    bm25_raw: np.ndarray
    bm25_norm: np.ndarray
    dense: np.ndarray           # max chunk cosine per page (-1 for pages without text)
    dense_norm: np.ndarray
    base: np.ndarray
    file_prior: np.ndarray      # per page: normalised best dense page of its file
    final: np.ndarray
    bm25_rank: np.ndarray
    dense_rank: np.ndarray
    final_rank: np.ndarray
    chunk_sims: np.ndarray      # cosine of every chunk with the query

    def contributions(self, i: int) -> tuple[float, float, float]:
        """Additive parts of ``final``: (BM25, dense, file prior)."""
        keep = 1.0 - PARENT_WEIGHT
        return (
            keep * (1.0 - W_DENSE) * float(self.bm25_norm[i]),
            keep * W_DENSE * float(self.dense_norm[i]),
            PARENT_WEIGHT * float(self.file_prior[i]),
        )


class RetrieverScorer:
    """Corpus, BM25 index and chunk vectors needed to explain the saved ranking."""

    def __init__(self) -> None:
        rows = []
        with PAGES_PATH.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        self.page_ids = [str(row["page_id"]) for row in rows]
        self.texts = [real_text(str(row.get("text") or "")) for row in rows]
        self.page_index = {page_id: i for i, page_id in enumerate(self.page_ids)}
        docs = [page_id.split("#", 1)[0] for page_id in self.page_ids]
        self.files = sorted(set(docs))
        file_index = {doc: i for i, doc in enumerate(self.files)}
        self.page_file = np.array([file_index[doc] for doc in docs])
        self.bm25 = PageBM25([enfr(text) for text in self.texts])
        self._build_chunks()

    # ---- chunks / dense

    @staticmethod
    def _key(text: str) -> Path:
        return EMBED_CACHE / (hashlib.sha256((EMBED_MODEL + "\n" + text).encode()).hexdigest() + ".json")

    def _build_chunks(self) -> None:
        owners, spans = [], []
        for i, text in enumerate(self.texts):
            for start, end in windows(text, CHUNK_WORDS, CHUNK_OVERLAP):
                if text[start:end].strip():
                    owners.append(i)
                    spans.append((start, end))
        self.chunk_owner = np.asarray(owners, dtype=np.int32)
        self.chunk_span = np.asarray(spans, dtype=np.int32)
        if CHUNK_CACHE.exists():
            cached = np.load(CHUNK_CACHE)
            if cached["vectors"].shape[0] == len(owners) and np.array_equal(cached["owners"], self.chunk_owner):
                self.vectors = cached["vectors"]
                return
        vectors = np.asarray(
            [json.loads(self._key(self.texts[o][s:e]).read_text()) for o, (s, e) in zip(owners, spans)],
            dtype=np.float32,
        )
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        CHUNK_CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez(CHUNK_CACHE, vectors=vectors, owners=self.chunk_owner)
        self.vectors = vectors

    def query_vector(self, query: str) -> np.ndarray | None:
        path = self._key(query)
        if not path.exists():
            return None
        vector = np.asarray(json.loads(path.read_text()), dtype=np.float32)
        return vector / np.linalg.norm(vector)

    # ---- scoring

    def score(self, query: str) -> PageScores:
        vector = self.query_vector(query)
        if vector is None:
            raise MissingEmbedding(f"no cached embedding for query: {query[:80]!r}")
        bm25_raw = self.bm25.scores(enfr(query)).astype(np.float64)
        chunk_sims = self.vectors @ vector
        dense = np.full(len(self.page_ids), -1.0, dtype=np.float64)
        np.maximum.at(dense, self.chunk_owner, chunk_sims)
        bm25_norm, dense_norm = max_norm(bm25_raw), max_norm(dense)
        base = (1.0 - W_DENSE) * bm25_norm + W_DENSE * dense_norm
        pool = np.zeros(len(self.files))
        np.maximum.at(pool, self.page_file, dense_norm)
        file_prior = max_norm(pool)[self.page_file]
        final = (1.0 - PARENT_WEIGHT) * base + PARENT_WEIGHT * file_prior
        return PageScores(
            query=query, bm25_raw=bm25_raw, bm25_norm=bm25_norm, dense=dense, dense_norm=dense_norm,
            base=base, file_prior=file_prior, final=final,
            bm25_rank=ranks_desc(bm25_raw), dense_rank=ranks_desc(dense), final_rank=ranks_desc(final),
            chunk_sims=chunk_sims,
        )

    def best_chunk(self, scores: PageScores, page_idx: int) -> tuple[str, float]:
        """Chunk of the page closest to the query (the one that sets the dense score)."""
        indices = np.flatnonzero(self.chunk_owner == page_idx)
        if not len(indices):
            return "", float("nan")
        best = indices[int(np.argmax(scores.chunk_sims[indices]))]
        start, end = self.chunk_span[best]
        return self.texts[page_idx][start:end], float(scores.chunk_sims[best])

    def bm25_terms(self, query: str, page_idx: int) -> list[dict[str, Any]]:
        """BM25 contribution of each analysed query term on one page."""
        rows = []
        for stem, word in query_terms(query).items():
            column = self.bm25.vocab.get(stem)
            if column is None:
                rows.append({"word": word, "idf": None, "df": 0, "contribution": 0.0})
                continue
            weight = float(self.bm25.W[page_idx, column])
            rows.append({
                "word": word,
                "idf": float(self.bm25.idf[column]),
                "df": int(self.bm25.W.indptr[column + 1] - self.bm25.W.indptr[column]),
                "contribution": float(self.bm25.idf[column]) * weight,
            })
        return rows


def verify() -> int:
    scorer = RetrieverScorer()
    saved = {}
    with SAVED_SCORES.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            saved[row["qid"]] = row
    queries = {}
    with (ROOT / "data/raw/0.1. BENCHMARK/queries.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            queries[row["query_id"]] = row["query"]
    worst, exact, same_set = 0.0, 0, 0
    for qid, row in saved.items():
        scores = scorer.score(queries[qid])
        pages = [scorer.page_index[p] for p in row["ranked_pages"]]
        worst = max(worst, float(np.max(np.abs(scores.final[pages] - np.asarray(row["scores"])))))
        top = [scorer.page_ids[i] for i in np.argsort(-scores.final, kind="stable")[:100]]
        exact += top == row["ranked_pages"]
        same_set += set(top) == set(row["ranked_pages"])
    print(f"{len(saved)} queries | exact top-100 order {exact} | same top-100 set {same_set} | max |score error| {worst:.2e}")
    return 0 if same_set == len(saved) and worst < 1e-5 else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        raise SystemExit(verify())
    print(__doc__)
