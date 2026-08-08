"""Cross-encoder reranking through the AXIOM Model Service.

The failure analysis attributed 6 of 24 misses to gold documents retrieved but
ranked 11-100 — the exact band a reranker addresses. Running it through the
gateway sidesteps the local torch/numpy conflict: the model lives in a container
with its own dependencies, and we only send text.

Reranking is applied to the top ``depth`` chunks of an existing run, so it can
only reorder what first-stage retrieval already found. Recall@depth is the
ceiling; nothing below the cutoff can be rescued.
"""

from __future__ import annotations

from typing import Any, Sequence
import os

import requests

MODEL_SERVICE_URL = "http://localhost:8006/api/v1"


class GatewayReranker:
    def __init__(
        self,
        model: str = "cohere-reranker",
        base_url: str = "",
        depth: int = 50,
        batch_size: int = 200,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.base_url = (base_url or os.getenv("AXIOM_MODEL_SERVICE_URL") or MODEL_SERVICE_URL).rstrip("/")
        self.depth = depth
        self.batch_size = batch_size
        self.timeout = timeout
        self.stats: dict[str, int] = {"calls": 0, "documents": 0}

    def rerank(
        self,
        query: str,
        hits: Sequence[tuple[int, float]],
        texts: Sequence[str],
        doc_ids: Sequence[str] | None = None,
    ) -> list[tuple[int, float]]:
        """Reorder ``hits`` by relevance to ``query``; tail beyond depth is kept as-is.

        With ``doc_ids``, the budget is spent on distinct documents (best chunk each)
        because scoring is document-level.
        """
        head, tail = _distinct_docs(hits, doc_ids, self.depth)
        if not head:
            return list(hits)
        scored: list[tuple[int, float]] = []
        for start in range(0, len(head), self.batch_size):
            window = head[start:start + self.batch_size]
            documents = [{"id": str(i), "text": texts[index][:4000]} for i, (index, _) in enumerate(window)]
            for position, score in self._call(query, documents):
                scored.append((window[position][0], score))
        scored.sort(key=lambda pair: -pair[1])
        # Tail keeps first-stage order, pushed below every reranked chunk.
        floor = scored[-1][1] if scored else 0.0
        return scored + [(index, floor - 1.0 - rank) for rank, (index, _) in enumerate(tail)]

    def _call(self, query: str, documents: list[dict[str, str]]) -> list[tuple[int, float]]:
        headers = {}
        token = os.getenv("AXIOM_MODEL_SERVICE_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        response = requests.post(
            f"{self.base_url}/inference/reranks",
            headers=headers,
            json={"model": self.model, "query": query, "documents": documents},
            timeout=self.timeout,
        )
        self.stats["calls"] += 1
        self.stats["documents"] += len(documents)
        if response.status_code != 200:
            raise RuntimeError(f"rerank failed HTTP {response.status_code}: {_message(response)}")
        results = response.json().get("results") or response.json().get("data") or []
        return [(int(item["id"]), float(item.get("relevance_score", item.get("score", 0.0)))) for item in results]


def _distinct_docs(
    hits: Sequence[tuple[int, float]], doc_ids: Sequence[str] | None, depth: int
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    if doc_ids is None:
        return list(hits[:depth]), list(hits[depth:])
    head: list[tuple[int, float]] = []
    rest: list[tuple[int, float]] = []
    seen: set[str] = set()
    for hit in hits:
        doc = doc_ids[hit[0]]
        if doc not in seen and len(head) < depth:
            seen.add(doc)
            head.append(hit)
        else:
            rest.append(hit)
    return head, rest


def _message(response: Any) -> str:
    try:
        body = response.json()
    except ValueError:
        return (response.text or "")[:300]
    for key in ("detail", "message", "error"):
        value = body.get(key) if isinstance(body, dict) else None
        if isinstance(value, str) and value:
            return value
    return str(body)[:300]
