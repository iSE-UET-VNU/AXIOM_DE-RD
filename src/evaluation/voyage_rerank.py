"""Voyage AI cross-encoder reranking.

A purpose-built reranker rather than an LLM asked to sort: deterministic, an
order of magnitude cheaper, and it returns calibrated relevance scores instead of
a permutation we have to parse. The LLM reranker remains in ``llm_rerank`` for
comparison.

Called directly rather than through the Model Service because the gateway has no
Voyage adapter — its ``cohere_compatible`` adapter reads ``payload["results"]``
and sends ``top_n``, while Voyage returns ``data`` and takes ``top_k``. Adding a
``VoyageAdapter`` to the gateway is the productionization path.
"""

from __future__ import annotations

from typing import Any, Sequence
import os
import time

import requests

VOYAGE_URL = "https://api.voyageai.com/v1/rerank"
MAX_DOCUMENTS = 1000
# An account without a payment method is capped at 3 RPM / 10K TPM. Passages are
# clipped so one request stays inside the token budget, and requests are paced so
# we never rely on retry-after to discover the limit.
MAX_DOC_CHARS = 1200
DEFAULT_RPM = 3
DEFAULT_TPM = 10_000


class VoyageReranker:
    def __init__(
        self,
        model: str = "rerank-2.5",
        api_key_env: str = "VOYAGE_API_KEY",
        url: str = "",
        depth: int = 20,
        timeout: float = 120.0,
        max_retries: int = 6,
        rpm: int = DEFAULT_RPM,
        tpm: int = DEFAULT_TPM,
    ) -> None:
        self.model = model
        self.api_key_env = api_key_env
        self.url = url or os.getenv("VOYAGE_RERANK_URL") or VOYAGE_URL
        self.depth = min(depth, MAX_DOCUMENTS)
        self.timeout = timeout
        self.max_retries = max(1, max_retries)
        self.rpm = max(1, rpm)
        self.tpm = max(1, tpm)
        self._history: list[tuple[float, int]] = []
        self.stats: dict[str, int] = {"calls": 0, "documents": 0, "tokens": 0, "retries": 0,
                                      "rate_limited": 0, "failed": 0, "waited_s": 0}

    def rerank(
        self,
        query: str,
        hits: Sequence[tuple[int, float]],
        texts: Sequence[str],
        doc_ids: Sequence[str] | None = None,
    ) -> list[tuple[int, float]]:
        from .rerank import _distinct_docs

        head, tail = _distinct_docs(hits, doc_ids, self.depth)
        if len(head) < 2:
            return list(hits)
        documents = [texts[index][:MAX_DOC_CHARS] or " " for index, _ in head]
        scored = self._call(query, documents)
        if scored is None:
            self.stats["failed"] += 1
            return list(hits)
        reordered = [(head[position][0], score) for position, score in scored]
        floor = min((score for _, score in reordered), default=0.0)
        return reordered + [(index, floor - 1.0 - rank) for rank, (index, _) in enumerate(tail)]

    def _throttle(self, tokens: int) -> None:
        """Block until this request fits both the per-minute request and token caps."""
        while True:
            now = time.monotonic()
            self._history = [(t, n) for t, n in self._history if now - t < 60.0]
            requests_used = len(self._history)
            tokens_used = sum(n for _, n in self._history)
            if requests_used < self.rpm and tokens_used + tokens <= self.tpm:
                self._history.append((now, tokens))
                return
            oldest = min(t for t, _ in self._history)
            wait = max(0.5, 60.0 - (now - oldest) + 0.5)
            self.stats["waited_s"] += int(wait)
            time.sleep(wait)

    def _call(self, query: str, documents: list[str]) -> list[tuple[int, float]] | None:
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"{self.api_key_env} is not set; cannot call Voyage.")
        body = {
            "query": query,
            "documents": documents,
            "model": self.model,
            "top_k": len(documents),
        }
        estimate = sum(len(d) for d in documents) // 3 + len(query) // 3
        for attempt in range(self.max_retries):
            if attempt:
                self.stats["retries"] += 1
            self._throttle(estimate)
            self.stats["calls"] += 1
            self.stats["documents"] += len(documents)
            try:
                response = requests.post(
                    self.url,
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json=body,
                    timeout=self.timeout,
                )
            except requests.RequestException:
                continue
            if response.status_code == 429 or response.status_code >= 500:
                self.stats["rate_limited"] += response.status_code == 429
                wait = float(response.headers.get("retry-after") or 20.0)
                self.stats["waited_s"] += int(wait)
                time.sleep(wait)
                continue
            if response.status_code != 200:
                raise RuntimeError(f"voyage rerank HTTP {response.status_code}: {response.text[:300]}")
            payload = response.json()
            self.stats["tokens"] += int((payload.get("usage") or {}).get("total_tokens") or 0)
            return [
                (int(item["index"]), float(item["relevance_score"]))
                for item in sorted(payload["data"], key=lambda i: -float(i["relevance_score"]))
            ]
        return None
