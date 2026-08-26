"""Reranking arms behind one Protocol.

"""

from __future__ import annotations

from typing import Mapping, Protocol, Sequence
import logging
import re

import requests

from .settings import Settings

logger = logging.getLogger(__name__)

Hit = tuple[str, float]

MAX_PASSAGE_CHARS = 1200

RERANKS_PATH = "/inference/reranks"
RESPONSES_PATH = "/inference/responses"


class RerankUnavailable(RuntimeError):
    """The reranker could not score. Caller keeps fused order and degrades."""


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, hits: Sequence[Hit], texts: Mapping[str, str]) -> list[Hit]:
        """Reorder ``hits``. Returns the same chunk ids with new scores.

        Must never return chunks the first stage did not find: a reranker may
        only reorder what retrieval surfaced.
        """
        ...


class NoReranker:
    name = "none"

    def rerank(self, query: str, hits: Sequence[Hit], texts: Mapping[str, str]) -> list[Hit]:
        return list(hits)


class GatewayReranker:
    """model-service ``POST /api/v1/inference/reranks``.
    """

    name = "gateway"

    def __init__(self, base_url: str, model: str, timeout_s: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def rerank(self, query: str, hits: Sequence[Hit], texts: Mapping[str, str]) -> list[Hit]:
        if not hits:
            return []
        ids = [chunk_id for chunk_id, _ in hits]
        documents = [texts.get(chunk_id, "")[:MAX_PASSAGE_CHARS] for chunk_id in ids]
        body = {"model": self.model, "query": query, "documents": documents, "top_n": len(ids)}
        try:
            response = requests.post(
                f"{self.base_url}{RERANKS_PATH}", json=body, timeout=self.timeout_s
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as error:
            raise RerankUnavailable(f"{self.name}: {error}") from error

        results = payload.get("results")
        if not isinstance(results, list):
            raise RerankUnavailable(f"{self.name}: no 'results' in response")

        # Index-addressed, so a malformed index must not silently shift the
        # ranking by one. Out-of-range is a protocol violation, not a hint.
        out: list[Hit] = []
        seen: set[int] = set()
        for item in results:
            if not isinstance(item, Mapping):
                continue
            index = item.get("index")
            score = item.get("relevance_score", item.get("score"))
            if not isinstance(index, int) or not 0 <= index < len(ids) or index in seen:
                raise RerankUnavailable(f"{self.name}: bad index {index!r} for {len(ids)} docs")
            seen.add(index)
            out.append((ids[index], float(score if score is not None else 0.0)))

        if not out:
            raise RerankUnavailable(f"{self.name}: empty ranking")
        # A reranker may only reorder. Anything it dropped keeps fused order
        # behind what it ranked, so retrieval's recall is never reduced.
        out.extend((ids[i], hits[i][1]) for i in range(len(ids)) if i not in seen)
        return out


class LlmReranker:
    """Listwise reranking via model-service ``POST /api/v1/inference/responses``.
    """

    name = "llm"

    def __init__(self, base_url: str, model: str, timeout_s: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def rerank(self, query: str, hits: Sequence[Hit], texts: Mapping[str, str]) -> list[Hit]:
        if not hits:
            return []
        ids = [chunk_id for chunk_id, _ in hits]
        passages = "\n\n".join(
            f"[{position}] {texts.get(chunk_id, '')[:MAX_PASSAGE_CHARS]}"
            for position, chunk_id in enumerate(ids, start=1)
        )
        prompt = (
            "Rank the passages by how well they answer the query. Reply with "
            "passage numbers separated by '>' and nothing else, e.g. 3 > 1 > 2. "
            "Include every number exactly once.\n\n"
            f"Query: {query}\n\n{passages}\n\nRanking:"
        )
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_output_tokens": 256,
        }
        try:
            response = requests.post(
                f"{self.base_url}{RESPONSES_PATH}", json=body, timeout=self.timeout_s
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as error:
            raise RerankUnavailable(f"{self.name}: {error}") from error

        output = payload.get("output")
        text = output.get("content", "") if isinstance(output, Mapping) else ""
        order = parse_ranking(str(text), ids)
        if not order:
            raise RerankUnavailable(f"{self.name}: unparseable ranking {str(text)[:120]!r}")

        # Descending synthetic scores: the LLM gives an order, not a magnitude,
        # and inventing calibrated-looking numbers would let them be compared
        # across queries as if they meant something.
        ranked = [(chunk_id, float(len(order) - position))
                  for position, chunk_id in enumerate(order)]
        fused = dict(hits)
        ranked.extend((chunk_id, fused[chunk_id]) for chunk_id in ids if chunk_id not in set(order))
        return ranked


def parse_ranking(raw: str, ids: Sequence[str]) -> list[str]:
    """Parse an LLM's ``4 > 1 > 7 > 2`` ranking into chunk ids.
    """
    seen: set[int] = set()
    order: list[str] = []
    for token in re.findall(r"\d+", raw or ""):
        index = int(token)
        if 1 <= index <= len(ids) and index not in seen:
            seen.add(index)
            order.append(ids[index - 1])
    return order


def build(settings: Settings) -> Reranker:
    """Construct the configured reranker.

    Unknown names are rejected in ``load_settings``; this mapping stays total
    so a future name cannot fall through to a silent no-op.
    """
    if settings.reranker == "gateway":
        return GatewayReranker(
            settings.model_service_url, settings.rerank_model, settings.rerank_timeout_s
        )
    if settings.reranker == "llm":
        return LlmReranker(
            settings.model_service_url, settings.rerank_llm_model, settings.rerank_timeout_s
        )
    return NoReranker()
