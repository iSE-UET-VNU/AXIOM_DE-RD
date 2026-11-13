#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-VL-32B-Instruct-FP8}"
PORT="${PORT:-8020}"
SERVED="${SERVED:-qwen3-vl-32b}"
TAG="${TAG:?set TAG to a finished run under results/}"
PY="${PY:-python}"

pip install -q "vllm>=0.11"
vllm serve "$MODEL" --seed 0 --max-model-len 32768 --max-num-seqs 1 --served-model-name "$SERVED" \
  --port "$PORT" > vllm_qwen3_vl.log 2>&1 &
SERVER=$!
until curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null; do
  kill -0 "$SERVER" || { tail -50 vllm_qwen3_vl.log; exit 1; }
  sleep 10
done

export QWEN_VL_ENDPOINT="http://127.0.0.1:$PORT/v1" QWEN_VL_SERVED="$SERVED"
PYTHONPATH=. "$PY" -m ondemand.adaptive qa --tag "$TAG" --reader qwen3_vl_32b:endpoint --judge legacy_4omini \
  --stage-mode qa_reader=online
kill "$SERVER"
ANSWERS="$(PYTHONPATH=. "$PY" -c 'from ondemand.bench import work; print(work("qa_cache", "answers_qwen3_vl_32b_endpoint"))')"
zip -qr qwen3_vl_answers.zip "$ANSWERS"
echo "download qwen3_vl_answers.zip, unzip into the same qa_cache folder, then judge locally:"
echo "python -m ondemand.adaptive qa --tag $TAG --reader qwen3_vl_32b:endpoint --stage-mode qa_judge=online"
