"""Listwise LLM reranking through the AXIOM Model Service (RankGPT-style).

A cross-encoder reranker is the textbook fix for gold that is retrieved but
ranked below the cutoff. With no Cohere key available, an LLM asked to order
passages by relevance is the free substitute: one call per query instead of one
per (query, passage) pair.

Failure is always non-fatal. A reranker that raises would destroy a multi-minute
benchmark run, and a reranker that silently mangles order would be worse than
none, so a failed or unparseable response falls back to first-stage order and is
counted in ``stats``.
"""

from __future__ import annotations

from typing import Any, Sequence
import os
import re

import requests

MODEL_SERVICE_URL = "http://localhost:8006/api/v1"
MAX_PASSAGE_CHARS = 1200

PROMPT = """Rank the passages by how well each answers the query.

Query: {query}

{passages}

Reply with the passage numbers from most to least relevant, separated by ">".
Example: 4 > 1 > 7 > 2
Include every number exactly once. Output only the ranking, no explanation."""


class LLMReranker:
    def __init__(
        self,
        model: str = "llm-rerank",
        base_url: str = "",
        depth: int = 20,
        timeout: float = 180.0,
        max_retries: int = 2,
    ) -> None:
        self.model = model
        self.base_url = (base_url or os.getenv("AXIOM_MODEL_SERVICE_URL") or MODEL_SERVICE_URL).rstrip("/")
        self.depth = depth
        self.timeout = timeout
        self.max_retries = max(1, max_retries)
        self.stats: dict[str, int] = {"calls": 0, "failed": 0, "unparsed": 0, "partial": 0}

    def rerank(
        self,
        query: str,
        hits: Sequence[tuple[int, float]],
        texts: Sequence[str],
        doc_ids: Sequence[str] | None = None,
    ) -> list[tuple[int, float]]:
        head, tail = self._head(hits, doc_ids)
        if len(head) < 2:
            return list(hits)
        order = self._ask(query, [texts[index] for index, _ in head])
        if order is None:
            return list(hits)
        # Any position the model omitted keeps its first-stage rank, appended in order.
        seen = set(order)
        order = order + [i for i in range(len(head)) if i not in seen]
        if len(seen) < len(head):
            self.stats["partial"] += 1
        scored = [(head[position][0], float(len(order) - rank)) for rank, position in enumerate(order)]
        floor = -1.0
        return scored + [(index, floor - rank) for rank, (index, _) in enumerate(tail)]

    def _head(
        self, hits: Sequence[tuple[int, float]], doc_ids: Sequence[str] | None
    ) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
        """Select what to rerank. Scoring is document-level, so spend the budget on
        ``depth`` distinct documents (each represented by its best chunk) rather than
        ``depth`` chunks that may all belong to a handful of documents."""
        if doc_ids is None:
            return list(hits[: self.depth]), list(hits[self.depth :])
        head: list[tuple[int, float]] = []
        rest: list[tuple[int, float]] = []
        seen: set[str] = set()
        for hit in hits:
            doc = doc_ids[hit[0]]
            if doc not in seen and len(head) < self.depth:
                seen.add(doc)
                head.append(hit)
            else:
                rest.append(hit)
        return head, rest

    def _ask(self, query: str, passages: list[str]) -> list[int] | None:
        listing = "\n\n".join(
            f"[{i + 1}] {text[:MAX_PASSAGE_CHARS]}" for i, text in enumerate(passages)
        )
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": PROMPT.format(query=query, passages=listing)}],
            "temperature": 0.0,
            # Generous: reasoning models spend this budget before emitting content,
            # and a truncated reply parses as a partial ranking or not at all.
            "max_output_tokens": 700,
        }
        for _ in range(self.max_retries):
            self.stats["calls"] += 1
            try:
                response = requests.post(
                    f"{self.base_url}/inference/responses", json=body, timeout=self.timeout
                )
            except requests.RequestException:
                self.stats["failed"] += 1
                continue
            if response.status_code != 200:
                self.stats["failed"] += 1
                continue
            order = _parse_ranking(_content(response.json()), len(passages))
            if order:
                return order
            self.stats["unparsed"] += 1
        return None


def _content(body: Any) -> str:
    output = body.get("output") if isinstance(body, dict) else None
    if isinstance(output, dict):
        return str(output.get("content") or "")
    return ""


def _parse_ranking(text: str, count: int) -> list[int] | None:
    """1-based numbers in the reply -> 0-based positions, deduplicated in order."""
    seen: list[int] = []
    for match in re.findall(r"\d+", text or ""):
        value = int(match) - 1
        if 0 <= value < count and value not in seen:
            seen.append(value)
    return seen or None
