"""OpenRouter embedder — hardened OpenAI-wire-format client (production default).

Disk cache keyed by sha1(model|text), batch 64, input sanitization, and retry
with exponential backoff on the HTTP-200-with-error-body responses OpenRouter
occasionally returns. Raises with the server's own error message once exhausted.
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
from ._sanitize import sanitize_text

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@embedder("openrouter_te3s")
class OpenRouterEmbedder:
    name = "openrouter_te3s"

    def __init__(
        self,
        model: str = "openai/text-embedding-3-small",
        dimension: int = 1536,
        api_key_env: str = "OPENROUTER_API_KEY",
        base_url: str = OPENROUTER_BASE_URL,
        batch_size: int = 64,
        cache_dir: str | Path = "data/work/embedding_cache",
        timeout: float = 120.0,
        max_retries: int = 4,
        app_title: str | None = None,
    ) -> None:
        self.model = model
        self.dim = dimension
        self.api_key_env = api_key_env
        self.base_url = base_url.rstrip("/")
        self.batch_size = max(1, batch_size)
        self.cache_dir = Path(cache_dir)
        self.timeout = timeout
        self.max_retries = max(1, max_retries)
        self.app_title = app_title
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
        for batch_start in range(0, len(todo), self.batch_size):
            batch_idx = todo[batch_start:batch_start + self.batch_size]
            batch = [sanitized[i] for i in batch_idx]
            for i, vector in zip(batch_idx, self._embed_batch(batch)):
                vectors[i] = vector
                self._cache_write(sanitized[i], vector)
                self.stats["tokens"] += max(1, len(sanitized[i]) // 4)
        return [v for v in vectors if v is not None]

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"{self.api_key_env} is not set; cannot call {self.base_url}.")
        headers = {"Authorization": f"Bearer {api_key}"}
        if self.app_title:
            headers["X-Title"] = self.app_title

        last_error = "no response"
        for attempt in range(self.max_retries):
            if attempt:
                self.stats["retries"] += 1
                self._sleep(min(2.0 * 2 ** (attempt - 1), 30.0))
            response = requests.post(
                f"{self.base_url}/embeddings",
                headers=headers,
                json={"model": self.model, "input": batch},
                timeout=self.timeout,
            )
            self.stats["api_calls"] += 1
            try:
                payload = response.json()
            except ValueError:
                payload = None
            if response.status_code == 200 and isinstance(payload, dict) and isinstance(payload.get("data"), list):
                items = sorted(payload["data"], key=lambda item: item.get("index", 0))
                embeddings = [[float(x) for x in item["embedding"]] for item in items]
                if len(embeddings) != len(batch):
                    last_error = f"expected {len(batch)} embeddings, got {len(embeddings)}"
                    continue
                return embeddings
            last_error = f"HTTP {response.status_code}: {_server_error_message(payload, response.text)}"
        raise RuntimeError(f"{self.name} embedding failed after {self.max_retries} attempts — {last_error}")

    def _cache_path(self, text: str) -> Path:
        key = hashlib.sha1(f"{self.model}|{text}".encode("utf-8", "ignore")).hexdigest()
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


def _server_error_message(payload: Any, raw_text: str) -> str:
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
        if isinstance(error, str) and error:
            return error
    return (raw_text or "")[:300]
