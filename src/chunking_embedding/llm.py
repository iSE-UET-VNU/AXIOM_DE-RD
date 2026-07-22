"""OpenRouter chat client for LLM-guided chunkers — cached, offline-safe.

No API key or any HTTP failure returns "" so callers fall back to heuristics.
Every call is disk-cached by (model, system, prompt, params) hash, so a re-run of
the same chunker over the same document spends nothing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
import hashlib
import os
import time

import requests

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterLLM:
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
        self.stats: dict[str, int] = {"calls": 0, "cache_hits": 0, "fails": 0, "tokens": 0}
        self._sleep: Callable[[float], None] = time.sleep

    def complete(self, prompt: str, system: str | None = None, max_tokens: int = 1200, temperature: float = 0.0) -> str:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        key = _sha1(f"{self.model}|{system}|{prompt}|{max_tokens}|{temperature}")
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
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
        for attempt in range(self.max_retries):
            try:
                response = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=body, timeout=self.timeout)
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
                out = response.json()["choices"][0]["message"]["content"] or ""
            except (ValueError, KeyError, IndexError):
                self.stats["fails"] += 1
                return ""
            self.stats["tokens"] += max(1, len(prompt) // 4) + max(1, len(out) // 4)
            path.write_text(out, encoding="utf-8")
            return out
        self.stats["fails"] += 1
        return ""


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()
