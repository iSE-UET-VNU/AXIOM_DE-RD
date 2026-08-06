"""BM25 over our own chunk manifest.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence
import json
import math

from ..chunking_embedding.lexical import analyze

K1 = 1.2
B = 0.75


_CJK = (
    (0x3040, 0x30FF),
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xAC00, 0xD7AF),
    (0xF900, 0xFAFF),
)
CJK_RATIO = 0.15


def _is_cjk(char: str) -> bool:
    code = ord(char)
    return any(low <= code <= high for low, high in _CJK)


def analyze_cjk(text: str) -> list[str]:
    """Character bigrams for CJK runs, ordinary tokens for everything else.

    Chinese has no spaces, so ``analyze`` emits one token per phrase and a query
    only scores on an exact phrase match. Bigrams are the standard fix and restore
    partial matching.
    """
    tokens: list[str] = []
    for token in analyze(text):
        if sum(1 for ch in token if _is_cjk(ch)) / max(len(token), 1) > 0.5:
            if len(token) == 1:
                tokens.append(token)
            else:
                tokens.extend(token[i:i + 2] for i in range(len(token) - 1))
        else:
            tokens.append(token)
    return tokens


ANALYZERS: dict[str, Callable[[str], list[str]]] = {
    "plain": analyze,
    "cjk_bigram": analyze_cjk,
}


def cjk_ratio(text: str) -> float:
    stripped = str(text)
    return sum(1 for ch in stripped if _is_cjk(ch)) / len(stripped) if stripped else 0.0


def resolve_analyzer(texts: Iterable[str]) -> str:
    for text in texts:
        if any(_is_cjk(ch) for ch in str(text)):
            return "cjk_bigram"
    return "plain"


@dataclass
class BM25Index:
    analyzer_name: str = "auto"
    k1: float = K1
    b: float = B
    chunk_ids: list[str] = field(default_factory=list)
    doc_ids: list[str] = field(default_factory=list)
    postings: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    lengths: list[int] = field(default_factory=list)
    avgdl: float = 0.0

    @property
    def analyzer(self) -> Callable[[str], list[str]]:
        if self.analyzer_name not in ANALYZERS:
            raise ValueError(
                f"Analyzer {self.analyzer_name!r} was never resolved to a concrete "
                "tokenizer. Build resolves 'auto' over the corpus; an index still "
                "holding it would tokenize queries differently from documents."
            )
        return ANALYZERS[self.analyzer_name]

    def build(self, chunks: Iterable[dict[str, Any]]) -> "BM25Index":
        chunks = list(chunks)
        if self.analyzer_name == "auto":
            self.analyzer_name = resolve_analyzer(chunk["text"] for chunk in chunks)
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for position, chunk in enumerate(chunks):
            tokens = self.analyzer(chunk["text"])
            counts: dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            for term, count in counts.items():
                postings[term].append((position, count))
            self.chunk_ids.append(chunk["chunk_id"])
            self.doc_ids.append(chunk["doc_id"])
            self.lengths.append(len(tokens))
        self.postings = dict(postings)
        self.avgdl = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        return self

    def search(
        self, query: str, top_k: int, allowed: set[int] | None = None
    ) -> list[tuple[int, float]]:
        total = len(self.lengths)
        if not total:
            return []
        scores: dict[int, float] = defaultdict(float)
        for term in self.analyzer(query):
            posting = self.postings.get(term)
            if not posting:
                continue
            df = len(posting)
            idf = math.log(1.0 + (total - df + 0.5) / (df + 0.5))
            for position, freq in posting:
                if allowed is not None and position not in allowed:
                    continue
                length = self.lengths[position] or 1
                denominator = freq + self.k1 * (1 - self.b + self.b * length / (self.avgdl or 1.0))
                scores[position] += idf * (freq * (self.k1 + 1)) / denominator
        ranked = sorted(scores.items(), key=lambda item: -item[1])
        return [(position, score) for position, score in ranked[:top_k] if score > 0]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "analyzer": self.analyzer_name,
            "k1": self.k1,
            "b": self.b,
            "avgdl": self.avgdl,
            "chunk_ids": self.chunk_ids,
            "doc_ids": self.doc_ids,
            "lengths": self.lengths,
            "postings": self.postings,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        payload = json.loads(path.read_text(encoding="utf-8"))
        index = cls(analyzer_name=payload["analyzer"], k1=payload["k1"], b=payload["b"])
        index.avgdl = payload["avgdl"]
        index.chunk_ids = payload["chunk_ids"]
        index.doc_ids = payload["doc_ids"]
        index.lengths = payload["lengths"]
        index.postings = {term: [tuple(p) for p in plist] for term, plist in payload["postings"].items()}
        return index


def positions_to_ids(index: BM25Index, hits: Sequence[tuple[int, float]]) -> list[tuple[str, float]]:
    return [(index.chunk_ids[position], score) for position, score in hits]
