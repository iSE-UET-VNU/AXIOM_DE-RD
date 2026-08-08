"""OpenRouter embedding and chat clients.

Both clients use deterministic disk caches and bounded retry behavior.  The
embedding client is the production default; chat is an optional resource for
chunkers that declare an LLM dependency.
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

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterLLM:
    """Cached OpenRouter chat client with offline-safe fallback behavior."""

    def __init__(
        self,
        model: str = "google/gemini-2.5-flash",
        api_key_env: str = "OPENROUTER_API_KEY",
        base_url: str = OPENROUTER_BASE_URL,
        cache_dir: str | Path = "data/work/llm_cache",
        timeout: float = 120.0,
        max_retries: int = 6,
        app_title: str | None = None,
    ) -> None:
        self.model = model
        self.api_key_env = api_key_env
        self.base_url = base_url.rstrip("/")
        self.cache_dir = Path(cache_dir)
        self.timeout = timeout
        self.max_retries = max(1, max_retries)
        self.app_title = app_title
        self.stats: dict[str, int] = {
            "calls": 0,
            "cache_hits": 0,
            "fails": 0,
            "tokens": 0,
        }
        self._sleep: Callable[[float], None] = time.sleep

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 1200,
        temperature: float = 0.0,
    ) -> str:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha1(
            f"{self.model}|{system}|{prompt}|{max_tokens}|{temperature}".encode(
                "utf-8",
                "ignore",
            )
        ).hexdigest()
        path = self.cache_dir / f"llm_{key}.txt"
        if path.exists():
            self.stats["cache_hits"] += 1
            return path.read_text(encoding="utf-8")
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            return ""
        headers = {"Authorization": f"Bearer {api_key}"}
        if self.app_title:
            headers["X-Title"] = self.app_title
        messages = (
            ([{"role": "system", "content": system}] if system else [])
            + [{"role": "user", "content": prompt}]
        )
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=body,
                    timeout=self.timeout,
                )
            except requests.RequestException:
                self._sleep(min(2 ** attempt * 2, 40))
                continue
            self.stats["calls"] += 1
            if response.status_code in (429, 500, 502, 503):
                self._sleep(min(2 ** attempt * 2, 40))
                continue
            if response.status_code >= 400:
                self.stats["fails"] += 1
                return ""
            try:
                output = response.json()["choices"][0]["message"]["content"] or ""
            except (ValueError, KeyError, IndexError):
                self.stats["fails"] += 1
                return ""
            self.stats["tokens"] += max(1, len(prompt) // 4) + max(1, len(output) // 4)
            path.write_text(output, encoding="utf-8")
            return output
        self.stats["fails"] += 1
        return ""


@embedder("openrouter_te3s")
class OpenRouterEmbedder:
    name = "openrouter_te3s"

    def __init__(
        self,
        model: str = "openai/text-embedding-3-small",
        dimension: int = 1536,
        api_key_env: str = "OPENROUTER_API_KEY",
        base_url: str = "",
        batch_size: int = 64,
        max_request_tokens: int = MAX_REQUEST_TOKENS,
        cache_dir: str | Path = "data/work/embedding_cache",
        timeout: float = 120.0,
        max_retries: int = 4,
        app_title: str | None = None,
    ) -> None:
        self.model = model
        self.dim = dimension
        self.api_key_env = api_key_env
        self.base_url = (base_url or os.getenv("OPENROUTER_BASE_URL") or OPENROUTER_BASE_URL).rstrip("/")
        self.batch_size = max(1, batch_size)
        self.max_request_tokens = max_request_tokens
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
        batches, tokens = token_batches(todo, sanitized, self.batch_size, self.max_request_tokens)
        self.stats["tokens"] += tokens
        for batch_idx in batches:
            batch = [sanitized[i] for i in batch_idx]
            for i, vector in zip(batch_idx, self._embed_batch(batch)):
                vectors[i] = vector
                self._cache_write(sanitized[i], vector)
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
            # Retried like the chat client above: an uncaught ReadTimeout here
            # aborts the whole run and every later arm loses its embeddings.
            try:
                response = requests.post(
                    f"{self.base_url}/embeddings",
                    headers=headers,
                    json={"model": self.model, "input": batch},
                    timeout=self.timeout,
                )
            except requests.RequestException as error:
                last_error = f"{type(error).__name__}: {error}"
                continue
            self.stats["api_calls"] += 1
            try:
                payload = response.json()
            except ValueError:
                payload = None
            if (
                response.status_code == 200
                and isinstance(payload, dict)
                and isinstance(payload.get("data"), list)
            ):
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


__all__ = ["OpenRouterEmbedder", "OpenRouterLLM"]
