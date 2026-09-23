import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv

from .bench import ROOT

URL = "https://openrouter.ai/api/v1"
load_dotenv(ROOT / ".env")


def _post(path, body, timeout=180, retries=4):
    last = ""
    for attempt in range(retries):
        if attempt:
            time.sleep(min(2 ** attempt, 30))
        try:
            response = requests.post(f"{URL}/{path}", json=body, timeout=timeout,
                                     headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"})
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
    for s in range(0, len(todo), batch):
        idx = todo[s:s + batch]
        data = _post("embeddings", {"model": model, "input": [texts[i] for i in idx]})["data"]
        for i, item in zip(idx, sorted(data, key=lambda d: d["index"])):
            out[i] = item["embedding"]
            key(texts[i]).write_text(json.dumps(item["embedding"]))
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
