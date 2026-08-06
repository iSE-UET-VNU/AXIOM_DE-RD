"""Chat completion through the AXIOM Model Service.

Generation and judging both need the same call, and they must not drift apart:
if the judge and the generator reach the model by different code paths, a change
to one silently changes the comparison. One function, both callers.

Temperature is a required argument rather than a default. Every experimental
result here depends on decoding being frozen, so the caller states it.
"""

from __future__ import annotations

from typing import Any
import os

import requests

MODEL_SERVICE_URL = "http://localhost:8006/api/v1"


class ModelUnavailable(RuntimeError):
    """The call did not produce text. Never swallowed: an unanswered question
    scored as wrong would look like a retrieval failure in the results."""


def complete(
    model: str,
    prompt: str,
    *,
    temperature: float,
    max_output_tokens: int = 1024,
    base_url: str = "",
    timeout: float = 180.0,
    max_retries: int = 3,
) -> str:
    url = (base_url or os.getenv("AXIOM_MODEL_SERVICE_URL") or MODEL_SERVICE_URL).rstrip("/")
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
    }
    last = ""
    for _ in range(max(1, max_retries)):
        try:
            response = requests.post(f"{url}/inference/responses", json=body, timeout=timeout)
        except requests.RequestException as error:
            last = str(error)
            continue
        if response.status_code != 200:
            last = f"HTTP {response.status_code}: {response.text[:200]}"
            continue
        text = _content(response.json())
        if text.strip():
            return text
        last = "empty content"
    raise ModelUnavailable(f"{model}: {last}")


def _content(body: Any) -> str:
    output = body.get("output") if isinstance(body, dict) else None
    if isinstance(output, dict):
        return str(output.get("content") or "")
    return ""
