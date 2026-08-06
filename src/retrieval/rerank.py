"""Reranking arms behind one Protocol.

"""

from __future__ import annotations

from typing import Mapping, Protocol, Sequence
import logging

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
        raise NotImplementedError


class GatewayReranker:
    """model-service ``POST /api/v1/inference/reranks``.
    """

    name = "gateway"

    def __init__(self, base_url: str, model: str, timeout_s: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def rerank(self, query: str, hits: Sequence[Hit], texts: Mapping[str, str]) -> list[Hit]:
        raise NotImplementedError


class LlmReranker:
    """Listwise reranking via model-service ``POST /api/v1/inference/responses``.
    """

    name = "llm"

    def __init__(self, base_url: str, model: str, timeout_s: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def rerank(self, query: str, hits: Sequence[Hit], texts: Mapping[str, str]) -> list[Hit]:
        raise NotImplementedError


def parse_ranking(raw: str, ids: Sequence[str]) -> list[str]:
    """Parse an LLM's ``4 > 1 > 7 > 2`` ranking into chunk ids.
    """
    raise NotImplementedError


def build(settings: Settings) -> Reranker:
    raise NotImplementedError
