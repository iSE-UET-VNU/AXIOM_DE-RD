"""AXIOM Model Service embedder — the platform's internal model gateway.

The gateway normalizes across OpenRouter/OpenAI/Cohere behind a logical alias,
so its wire format is not OpenAI's: input items carry an ``id`` that the
response echoes back, and ``input_type`` selects asymmetric query vs passage
encoding. We match vectors by that id rather than trusting list order.

    OPENROUTER_BASE_URL is irrelevant here; set AXIOM_MODEL_SERVICE_URL.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
import hashlib
import json
import os
import time

import requests

from ..registry import embedder
from . import MAX_REQUEST_TOKENS, sanitize_text, token_batches

MODEL_SERVICE_URL = "http://localhost:8006/api/v1"
MAX_BATCH = 256


@embedder("axiom_gateway")
class AxiomGatewayEmbedder:
    name = "axiom_gateway"

    def __init__(
        self,
        model: str = "openrouter-embedding",
        dimension: int = 1536,
        input_type: str = "document",
        base_url: str = "",
        api_key_env: str = "AXIOM_MODEL_SERVICE_TOKEN",
        cache_model: str = "",
        max_request_tokens: int = MAX_REQUEST_TOKENS,
        batch_size: int = 128,
        cache_dir: str | Path = "data/work/embedding_cache",
        timeout: float = 120.0,
        max_retries: int = 4,
    ) -> None:
        # A vector is identified by the model that produced it, not the route it
        # arrived by. ``route`` is the gateway alias we call; ``model`` is what
        # the vector is stamped with and what every consumer filters on.
        #
        # These must not be conflated. corpus-service stores ``model`` in
        # document_embeddings.embeddings_model, and Methods-Hub sends
        # DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small" on every
        # vector search. Stamping vectors with the alias instead makes the
        # platform's own retrieval tools match zero rows for our documents --
        # HTTP 200, empty list, no error anywhere.
        self.route = model
        self.model = cache_model or model
        self.dim = dimension
        self.input_type = input_type
        self.base_url = (base_url or os.getenv("AXIOM_MODEL_SERVICE_URL") or MODEL_SERVICE_URL).rstrip("/")
        self.api_key_env = api_key_env
        self.max_request_tokens = max_request_tokens
        self.batch_size = max(1, min(batch_size, MAX_BATCH))
        self.cache_dir = Path(cache_dir)
        self.timeout = timeout
        self.max_retries = max(1, max_retries)
        self.stats: dict[str, int] = {"tokens": 0, "api_calls": 0, "cache_hits": 0, "retries": 0}
        self._sleep: Callable[[float], None] = time.sleep

    def embed(self, texts: list[str]) -> list[list[float]]:
        sanitized = [sanitize_text(t) for t in texts]
        vectors: list[list[float] | None] = [None] * len(sanitized)
        todo: list[int] = []
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        for i, text in enumerate(sanitized):
            cached = self._cache_read(text)
            if cached is not None:
                vectors[i] = cached
                self.stats["cache_hits"] += 1
            else:
                todo.append(i)
        batches, tokens = token_batches(todo, sanitized, self.batch_size, self.max_request_tokens)
        self.stats["tokens"] += tokens
        for batch_idx in batches:
            batch = [sanitized[i] for i in batch_idx]
            for i, vector in zip(batch_idx, self._embed_batch(batch)):
                vectors[i] = vector
                self._cache_write(sanitized[i], vector)
        return [v for v in vectors if v is not None]

    def embed_query(self, texts: list[str]) -> list[list[float]]:
        """Encode as queries. Asymmetric models put queries in a different space."""
        passage_type, self.input_type = self.input_type, "query"
        try:
            return self.embed(texts)
        finally:
            self.input_type = passage_type

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        headers = {}
        token = os.getenv(self.api_key_env)
        if token:
            headers["Authorization"] = f"Bearer {token}"
        payload = {
            # The gateway routes on the alias, not the upstream id.
            "model": self.route,
            "input": [{"id": str(i), "text": text} for i, text in enumerate(batch)],
            "input_type": self.input_type,
        }

        last_error = "no response"
        for attempt in range(self.max_retries):
            if attempt:
                self.stats["retries"] += 1
                self._sleep(min(2.0 * 2 ** (attempt - 1), 30.0))
            # Retried: an uncaught ReadTimeout aborts the whole run mid-corpus.
            try:
                response = requests.post(
                    f"{self.base_url}/inference/embeddings",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
            except requests.RequestException as error:
                last_error = f"{type(error).__name__}: {error}"
                continue
            self.stats["api_calls"] += 1
            try:
                body = response.json()
            except ValueError:
                body = None
            if response.status_code == 200 and isinstance(body, dict) and isinstance(body.get("data"), list):
                by_id = {str(item["id"]): item["embedding"] for item in body["data"]}
                if len(by_id) != len(batch):
                    last_error = f"expected {len(batch)} embeddings, got {len(by_id)}"
                    continue
                self.dim = int(body.get("dimensions") or self.dim)
                return [[float(x) for x in by_id[str(i)]] for i in range(len(batch))]
            last_error = f"HTTP {response.status_code}: {_gateway_error(body, response.text)}"
        raise RuntimeError(f"{self.name} embedding failed after {self.max_retries} attempts — {last_error}")

    def _cache_path(self, text: str) -> Path:
        # Keyed on the producing model, so a direct client and the gateway share
        # vectors and switching routes costs nothing.
        identity = f"{self.model}|{self.input_type}"
        key = hashlib.sha1(f"{identity}|{text}".encode("utf-8", "ignore")).hexdigest()
        return self.cache_dir / f"emb_{key}.json"

    def _cache_read(self, text: str) -> list[float] | None:
        path = self._cache_path(text)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return None

    def _cache_write(self, text: str, vector: list[float]) -> None:
        self._cache_path(text).write_text(json.dumps(vector), encoding="utf-8")


def _gateway_error(body: Any, raw_text: str) -> str:
    if isinstance(body, dict):
        for key in ("detail", "message", "error"):
            value = body.get(key)
            if isinstance(value, str) and value:
                return value
            if isinstance(value, dict) and value.get("message"):
                return str(value["message"])
    return (raw_text or "")[:300]
