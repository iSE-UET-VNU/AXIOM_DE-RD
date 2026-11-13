#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
EVIE_DIR="${EVIE_DIR:-$REPO_DIR/../EVIE}"
EVIE_PIN="70b8a2be14de7285970a05ac0f670f768abad966"

cd "$REPO_DIR"
pip install -q -e . 2>/dev/null || pip install -q numpy requests python-dotenv tiktoken tokenizers xgboost scikit-learn pymupdf
if [ ! -d "$EVIE_DIR" ]; then
  git clone -q https://github.com/Tencent/EVIE.git "$EVIE_DIR"
fi
git -C "$EVIE_DIR" checkout -q "$EVIE_PIN"
pip install -q -r "$EVIE_DIR/requirements.txt" "transformers==5.13.1" pymupdf
echo "export EVIE_DIR=$EVIE_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv
