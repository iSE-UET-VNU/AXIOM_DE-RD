#!/usr/bin/env bash
set -euo pipefail

TAG="${TAG:-evie_live_gpu}"
PY="${PY:-python}"
: "${EVIE_DIR:?run colab/setup_gpu.sh and export EVIE_DIR first}"

PYTHONPATH=. "$PY" - <<'EOF'
import json

from ondemand.adaptive.services.evie import EVIE
from ondemand.bench import FLAWED_QIDS, queries
from ondemand.adaptive.retrievers import load_gate

gate = load_gate("gate_ppocrv5_ungated_xgbK214.json")
q = queries()
qids = [x for x in sorted(gate["gate"]) if x not in FLAWED_QIDS][:2]
service = EVIE("local")
result = service.verify([(x, q[x]["query"], gate["gate"][x][:5]) for x in qids])
print(json.dumps(result, indent=1))
if not result.get("reproduces"):
    raise SystemExit("live EVIE does not reproduce the cached V2 scores; stopping")
EOF

PYTHONPATH=. "$PY" -m ondemand.adaptive run --tag "$TAG" --exclude-flawed --stage-mode visual_encoder=online "$@"
LIVE="$(PYTHONPATH=. "$PY" -c 'from ondemand.bench import work; print(work("evie", "live"))')"
zip -qr evie_live.zip "$LIVE"
echo "download evie_live.zip and unzip it into the same evie/live folder locally; offline runs then read it"
