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
import logging
import os
import time
import uuid

import requests

from ..registry import embedder
from . import MAX_REQUEST_TOKENS, sanitize_text, token_batches

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LOGGER = logging.getLogger(__name__)


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
        started = time.perf_counter()
        sanitized = [sanitize_text(t) for t in texts]
        vectors: list[list[float] | None] = [None] * len(sanitized)
        todo: list[int] = []
        cache_hits = 0
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        for i, text in enumerate(sanitized):
            cached = self._cache_read(text)
            if cached is not None:
                vectors[i] = cached
                self.stats["cache_hits"] += 1
                cache_hits += 1
            else:
                todo.append(i)
        batches, tokens = token_batches(todo, sanitized, self.batch_size, self.max_request_tokens)
        self.stats["tokens"] += tokens
        LOGGER.info(
            "Embedding call prepared: model=%s input_count=%d cache_hits=%d "
            "remote_count=%d batch_count=%d",
            self.model,
            len(sanitized),
            cache_hits,
            len(todo),
            len(batches),
        )
        for batch_idx in batches:
            batch = [sanitized[i] for i in batch_idx]
            for i, vector in zip(batch_idx, self._embed_batch(batch)):
                vectors[i] = vector
                self._cache_write(sanitized[i], vector)
        result = [v for v in vectors if v is not None]
        LOGGER.info(
            "Embedding call completed: model=%s input_count=%d vectors=%d "
            "cache_hits=%d elapsed_ms=%.1f",
            self.model,
            len(sanitized),
            len(result),
            cache_hits,
            (time.perf_counter() - started) * 1000.0,
        )
        return result

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"{self.api_key_env} is not set; cannot call {self.base_url}.")
        headers = {"Authorization": f"Bearer {api_key}"}
        if self.app_title:
            headers["X-Title"] = self.app_title

        last_error = "no response"
        request_id = uuid.uuid4().hex[:12]
        request_started = time.perf_counter()
        for attempt_index in range(self.max_retries):
            attempt = attempt_index + 1
            if attempt_index:
                backoff = min(2.0 * 2 ** (attempt_index - 1), 30.0)
                self.stats["retries"] += 1
                LOGGER.info(
                    "Embedding retry waiting: request_id=%s attempt=%d/%d "
                    "backoff_seconds=%.1f",
                    request_id,
                    attempt,
                    self.max_retries,
                    backoff,
                )
                self._sleep(backoff)
            attempt_started = time.perf_counter()
            LOGGER.info(
                "Embedding HTTP request started: request_id=%s attempt=%d/%d "
                "model=%s input_count=%d timeout_seconds=%.1f url=%s",
                request_id,
                attempt,
                self.max_retries,
                self.model,
                len(batch),
                self.timeout,
                f"{self.base_url}/embeddings",
            )
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
                LOGGER.warning(
                    "Embedding HTTP request failed: request_id=%s attempt=%d/%d "
                    "error_type=%s error=%s elapsed_ms=%.1f retrying=%s",
                    request_id,
                    attempt,
                    self.max_retries,
                    type(error).__name__,
                    str(error),
                    (time.perf_counter() - attempt_started) * 1000.0,
                    attempt < self.max_retries,
                )
                continue
            self.stats["api_calls"] += 1
            elapsed_ms = (time.perf_counter() - attempt_started) * 1000.0
            LOGGER.info(
                "Embedding HTTP response received: request_id=%s attempt=%d/%d "
                "status=%d elapsed_ms=%.1f",
                request_id,
                attempt,
                self.max_retries,
                response.status_code,
                elapsed_ms,
            )
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
                    LOGGER.warning(
                        "Embedding response invalid: request_id=%s attempt=%d/%d "
                        "reason=%s retrying=%s",
                        request_id,
                        attempt,
                        self.max_retries,
                        last_error,
                        attempt < self.max_retries,
                    )
                    continue
                LOGGER.info(
                    "Embedding request succeeded: request_id=%s attempt=%d/%d "
                    "vectors=%d total_elapsed_ms=%.1f",
                    request_id,
                    attempt,
                    self.max_retries,
                    len(embeddings),
                    (time.perf_counter() - request_started) * 1000.0,
                )
                return embeddings
            last_error = f"HTTP {response.status_code}: {_server_error_message(payload, response.text)}"
            LOGGER.warning(
                "Embedding HTTP response unsuccessful: request_id=%s attempt=%d/%d "
                "status=%d error=%s retrying=%s",
                request_id,
                attempt,
                self.max_retries,
                response.status_code,
                last_error,
                attempt < self.max_retries,
            )
        LOGGER.error(
            "Embedding request exhausted retries: request_id=%s attempts=%d "
            "input_count=%d total_elapsed_ms=%.1f last_error=%s",
            request_id,
            self.max_retries,
            len(batch),
            (time.perf_counter() - request_started) * 1000.0,
            last_error,
        )
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
