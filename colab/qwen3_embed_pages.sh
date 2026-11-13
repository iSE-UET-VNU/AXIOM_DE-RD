#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-Embedding-8B}"
PORT="${PORT:-8010}"
STEM="${STEM:-dense_qwen8b_local_pages_ppocrv5}"
PY="${PY:-python}"

pip install -q "vllm>=0.10"
vllm serve "$MODEL" --task embed --dtype bfloat16 --seed 0 --max-model-len 8192 --max-num-seqs 1 \
  --served-model-name qwen3-embedding-8b --port "$PORT" > vllm_embed.log 2>&1 &
SERVER=$!
until curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null; do
  kill -0 "$SERVER" || { tail -50 vllm_embed.log; exit 1; }
  sleep 5
done

export QWEN_EMBED_ENDPOINT="http://127.0.0.1:$PORT/v1" QWEN_EMBED_SERVED=qwen3-embedding-8b
PYTHONPATH=. "$PY" - "$STEM" <<'EOF'
import json
import sys

import numpy as np

from ondemand.adaptive.embedders import DEFAULT_STEM, Qwen3Embedding8B
from ondemand.bench import load_jsonl, work

rows = load_jsonl(work("light_prep") / "pages_ppocrv5.jsonl")
local = Qwen3Embedding8B(sys.argv[1], online=True)
local.prepare(rows, "pages_ppocrv5.jsonl")
folder = work("light_v2")
report = {"local": sys.argv[1], "api": DEFAULT_STEM}
api_pages = folder / f"{DEFAULT_STEM}_pages.npy"
if api_pages.exists():
    for part in ("pages", "queries"):
        a = np.load(folder / f"{sys.argv[1]}_{part}.npy")
        b = np.load(folder / f"{DEFAULT_STEM}_{part}.npy")
        live = np.linalg.norm(a, axis=1) > 0
        a, b = a[live], b[live]
        cos = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
        report[part] = {"n": int(live.sum()), "min_cos": float(cos.min()), "mean_cos": float(cos.mean())}
print(json.dumps(report, indent=1))
EOF
kill "$SERVER"
zip -qr qwen3_embed_local.zip "$(PYTHONPATH=. "$PY" -c 'from ondemand.bench import work; print(work("light_v2"))')/${STEM}_"*
echo "download qwen3_embed_local.zip and unzip it into the same light_v2 folder locally"
