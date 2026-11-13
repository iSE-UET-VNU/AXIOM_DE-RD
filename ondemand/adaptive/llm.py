import json
import os
import time
from time import perf_counter

import requests

from ..openrouter import chat

TEMPERATURE, TOP_P, SEED = 0.0, 1.0, 0


class OpenRouterChat:
    backend = "openrouter"

    def __init__(self, model, provider, max_tokens, response_format=None):
        if not provider:
            raise SystemExit(f"{model}: no pinned provider; set one so every call is served by the same backend")
        self.model, self.max_tokens, self.response_format = model, max_tokens, response_format
        self.provider = provider if isinstance(provider, dict) else \
            {"order": [provider], "allow_fallbacks": False, "require_parameters": True}

    def settings(self):
        return {"backend": self.backend, "model": self.model, "provider": self.provider, "temperature": TEMPERATURE,
                "top_p": TOP_P, "seed": SEED, "max_tokens": self.max_tokens, "response_format": self.response_format}

    def key(self):
        return json.dumps(self.settings(), sort_keys=True) + "\n"

    def __call__(self, prompt):
        started = perf_counter()
        payload = chat(self.model, prompt, max_tokens=self.max_tokens, temperature=TEMPERATURE, seed=SEED,
                       top_p=TOP_P, provider=self.provider, response_format=self.response_format, full=True)
        return {"text": payload["choices"][0]["message"]["content"].strip(), "seconds": perf_counter() - started,
                "served_model": payload.get("model"), "served_provider": payload.get("provider"),
                "usage": payload.get("usage")}


class EndpointChat:
    backend = "endpoint"

    def __init__(self, url, served_name, max_tokens, response_format=None):
        self.url = url.rstrip("/") if url else None
        self.model, self.max_tokens, self.response_format = served_name, max_tokens, response_format

    def settings(self):
        return {"backend": self.backend, "model": self.model, "temperature": TEMPERATURE, "top_p": TOP_P,
                "seed": SEED, "max_tokens": self.max_tokens, "response_format": self.response_format}

    def key(self):
        return json.dumps(self.settings(), sort_keys=True) + "\n"

    def __call__(self, prompt):
        if not self.url:
            raise SystemExit(f"{self.model}: no endpoint URL; start the vLLM server (colab/serve_qwen3_vl.sh) "
                             f"and set QWEN_VL_ENDPOINT to its /v1 URL")
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "temperature": TEMPERATURE,
                "top_p": TOP_P, "seed": SEED, "max_tokens": self.max_tokens}
        if self.response_format:
            body["response_format"] = self.response_format
        headers = {"Authorization": f"Bearer {os.getenv('QWEN_VL_API_KEY', 'EMPTY')}"}
        started, last = perf_counter(), ""
        for attempt in range(4):
            if attempt:
                time.sleep(2 ** attempt)
            try:
                response = requests.post(f"{self.url}/chat/completions", json=body, headers=headers, timeout=600)
                payload = response.json()
            except (requests.RequestException, ValueError) as error:
                last = str(error)
                continue
            text = ((payload.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            if response.status_code == 200 and text.strip():
                return {"text": text.strip(), "seconds": perf_counter() - started, "served_model": payload.get("model"),
                        "served_provider": self.url, "usage": payload.get("usage")}
            last = str(payload.get("error") or payload)[:300]
        raise RuntimeError(f"{self.model} at {self.url}: {last}")
