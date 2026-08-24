"""Task 1: resolve every alias the benchmark touches, and prove it responds.

Two routes exist and they are not the same thing:
  * ``openrouter_te3s``  -- the harness calls OpenRouter DIRECTLY (openrouter.py)
  * ``llm-*``            -- these go to the AXIOM Model Service gateway on :8006

Secrets are never printed; only whether a key is present.
"""
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GATEWAY = os.getenv("AXIOM_MODEL_SERVICE_URL", "http://localhost:8006/api/v1")
OR_BASE = "https://openrouter.ai/api/v1"


def load_env():
    path = REPO / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env()
KEY = os.getenv("OPENROUTER_API_KEY", "")
print(f"OPENROUTER_API_KEY present: {bool(KEY)}  (len {len(KEY)})\n")


def post(url, body, headers=None, timeout=90):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r), round(time.time() - t, 2)
    except urllib.error.HTTPError as e:
        raw = e.read().decode()[:400]
        try:
            raw = json.loads(raw)
        except ValueError:
            pass
        return e.code, raw, round(time.time() - t, 2)
    except Exception as e:  # noqa: BLE001
        return 0, str(e), round(time.time() - t, 2)


# ---------------------------------------------------------------- OpenRouter catalogue
print("=" * 78)
print("OpenRouter catalogue check (is the provider string still served?)")
print("=" * 78)
catalogue = {}
try:
    with urllib.request.urlopen(f"{OR_BASE}/models", timeout=60) as r:
        for m in json.load(r)["data"]:
            catalogue[m["id"]] = m
    print(f"catalogue size: {len(catalogue)}")
except Exception as e:  # noqa: BLE001
    print("could not fetch catalogue:", e)

for mid in ("openai/text-embedding-3-small", "openai/gpt-4o", "openai/gpt-4o-mini"):
    m = catalogue.get(mid)
    if not m:
        print(f"  {mid:35s} NOT IN CATALOGUE")
        continue
    p = m.get("pricing", {})
    print(f"  {mid:35s} ctx={m.get('context_length')} "
          f"prompt=${p.get('prompt')} completion=${p.get('completion')}")

# ---------------------------------------------------------------- direct OpenRouter embed
print()
print("=" * 78)
print("openrouter_te3s -> OpenRouter DIRECT /embeddings (what the harness does)")
print("=" * 78)
status, body, dt = post(
    f"{OR_BASE}/embeddings",
    {"model": "openai/text-embedding-3-small", "input": ["dimension probe"]},
    {"Authorization": f"Bearer {KEY}"},
)
print(f"  HTTP {status}  {dt}s")
if status == 200 and isinstance(body, dict):
    vec = body["data"][0]["embedding"]
    print(f"  RETURNED DIMENSION: {len(vec)}   usage={body.get('usage')}")
else:
    print(f"  body: {str(body)[:400]}")

# ---------------------------------------------------------------- gateway aliases
print()
print("=" * 78)
print("Gateway aliases -> POST /inference/responses")
print("=" * 78)
for alias in ("llm-default", "llm-judge", "llm-rerank", "llm-rerank-strong",
              "agent-llm-default", "openrouter-llm-free"):
    status, body, dt = post(
        f"{GATEWAY}/inference/responses",
        {"model": alias,
         "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
         "temperature": 0.0, "max_output_tokens": 8},
    )
    out = ""
    if isinstance(body, dict):
        o = body.get("output")
        out = (o.get("content") if isinstance(o, dict) else "") or ""
        usage = body.get("usage") or (o or {}).get("usage")
    else:
        usage = None
    verdict = "OK" if status == 200 and out.strip() else "FAIL"
    print(f"  {alias:22s} HTTP {status:3d}  {dt:5.2f}s  {verdict:4s} "
          f"out={out.strip()[:30]!r} usage={usage}")
    if verdict == "FAIL":
        print(f"      -> {str(body)[:300]}")

# ---------------------------------------------------------------- gateway embeddings
print()
print("=" * 78)
print("Gateway embedding aliases -> POST /inference/embeddings")
print("=" * 78)
for alias in ("openrouter-embedding", "embedding-default"):
    status, body, dt = post(
        f"{GATEWAY}/inference/embeddings",
        {"model": alias, "input": ["dimension probe"]},
    )
    dim = None
    if status == 200 and isinstance(body, dict):
        data = body.get("data") or body.get("embeddings") or []
        if data:
            first = data[0]
            vec = first.get("embedding") if isinstance(first, dict) else first
            dim = len(vec) if vec else None
    print(f"  {alias:22s} HTTP {status:3d}  {dt:5.2f}s  dim={dim}")
    if dim is None:
        print(f"      -> {str(body)[:300]}")

# ---------------------------------------------------------------- manifests
print()
print("=" * 78)
print("Embedding dimension recorded in artifacts / manifests")
print("=" * 78)
hits = 0
for path in list(REPO.glob("**/*.json"))[:6000]:
    if any(p in path.parts for p in (".git", "node_modules", "__pycache__")):
        continue
    if path.stat().st_size > 3_000_000:
        continue
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        continue
    if '"dim' not in text and "embedding_dim" not in text:
        continue
    try:
        payload = json.loads(text)
    except ValueError:
        continue
    for key in ("dim", "dimension", "embedding_dim", "embeddings_dim"):
        if isinstance(payload, dict) and key in payload:
            print(f"  {path.relative_to(REPO)}: {key}={payload[key]} "
                  f"embedder={payload.get('embedder') or payload.get('embeddings_model')}")
            hits += 1
if not hits:
    print("  no manifest carrying an explicit dimension found")
