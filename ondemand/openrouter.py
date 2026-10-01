import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv

from .bench import ROOT

load_dotenv(ROOT / ".env")

URL = (os.getenv("OPENROUTER_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://openrouter.ai/api/v1").rstrip("/")


def _get_api_key():
    key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("No LLM API key found. Please set OPENROUTER_API_KEY or OPENAI_API_KEY in your .env file.")
    return key


def _post(path, body, timeout=180, retries=4):
    last = ""
    api_key = _get_api_key()
    headers = {"Authorization": f"Bearer {api_key}"}
    referer = os.getenv("OPENROUTER_HTTP_REFERER")
    if referer:
        headers["HTTP-Referer"] = referer

    for attempt in range(retries):
        if attempt:
            time.sleep(min(2 ** attempt, 30))
        try:
            response = requests.post(f"{URL}/{path}", json=body, timeout=timeout, headers=headers)
            payload = response.json()
        except (requests.RequestException, ValueError) as error:
            last = str(error)
            continue
        if response.status_code == 200 and not payload.get("error"):
            return payload
        last = str(payload.get("error") or payload)[:300]
    raise RuntimeError(last)


def embed(texts, model, cache_dir, batch=32):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = lambda t: cache_dir / (hashlib.sha256((model + "\n" + t).encode()).hexdigest() + ".json")
    out = [None] * len(texts)
    todo = []
    for i, t in enumerate(texts):
        if key(t).exists():
            out[i] = json.loads(key(t).read_text())
        else:
            todo.append(i)
    if todo:
        total_batches = (len(todo) + batch - 1) // batch
        for b_idx, s in enumerate(range(0, len(todo), batch), 1):
            idx = todo[s:s + batch]
            print(f"\r[embed] embedding batch {b_idx}/{total_batches} ({len(idx)} texts via {model})...", end="", flush=True)
            data = _post("embeddings", {"model": model, "input": [texts[i] for i in idx]})["data"]
            for i, item in zip(idx, sorted(data, key=lambda d: d["index"])):
                out[i] = item["embedding"]
                key(texts[i]).write_text(json.dumps(item["embedding"]))
        print("\r" + " " * 70 + "\r", end="", flush=True)
    return np.asarray(out, dtype=np.float32)


def chat(model, content, max_tokens=512, temperature=0.0, seed=None):
    body = {"model": model, "temperature": temperature, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": content}]}
    if seed is not None:
        body["seed"] = seed
    for attempt in range(3):
        if attempt:
            time.sleep(2 ** attempt)
        payload = _post("chat/completions", body)
        text = ((payload.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        if text.strip():
            return text
    raise RuntimeError(f"{model}: empty content after 3 attempts")
