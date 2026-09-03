"""Convert the KDL SEP+ColVec pool into Metarank bootstrap events + config."""
import json
import re
import sys
from pathlib import Path

ROOT = Path("/Users/khoatran/Desktop/ISE/Axiom-DE-RnD/AXIOM_DE-RD")
sys.path.insert(0, str(ROOT))
SCRATCH = Path(__file__).resolve().parent

import numpy as np
from src.utils.env import load_dotenv_file
load_dotenv_file(ROOT)

from research.experiments.physics_ltr import kdl_pool_scored, features, FEAT_NAMES

qrels, pool = kdl_pool_scored()
vdir = ROOT / "data/work/vidore_physics_colvec"
matrix = np.load(vdir / "physics_colvec_scores.npy")
ck = json.loads((vdir / "physics_colvec_keys.json").read_text())
cq = json.loads((vdir / "physics_colvec_qids.json").read_text())
ki = {k: i for i, k in enumerate(ck)}
ri = {q: i for i, q in enumerate(cq)}
vis = {q: {c: float(matrix[ri[q], ki[c]]) for c in pool[q]["cand"] if c in ki}
       for q in pool if q in ri}

X, y, groups, meta = features(qrels, pool, vis)
sanitize = lambda s: re.sub(r"[^A-Za-z0-9]", "_", s)

by_q = {}
for row, (qid, uid) in enumerate(meta):
    by_q.setdefault(qid, []).append(row)

events = []
ts = 1_600_000_000_000
seen_items = set()
for row, (qid, uid) in enumerate(meta):
    s = sanitize(uid)
    if s in seen_items:
        continue
    seen_items.add(s)
    ts += 1
    events.append({"event": "item", "id": f"m_{len(seen_items)}", "item": s,
                   "timestamp": str(ts), "fields": [{"name": "kind", "value": "page"}]})

for qi, (qid, rows) in enumerate(by_q.items()):
    ts += 1000
    items = []
    for r in rows:
        uid = meta[r][1]
        flds = [{"name": FEAT_NAMES[k], "value": float(X[r][k])} for k in range(len(FEAT_NAMES))]
        items.append({"id": sanitize(uid), "label": float(y[r]), "fields": flds})
    rid = f"r_{qi}"
    events.append({"event": "ranking", "id": rid, "timestamp": str(ts),
                   "fields": [{"name": "query", "value": pool[qid]["query"][:120]}],
                   "items": items})
    for r in rows:
        if y[r] > 0:
            ts += 1
            events.append({"event": "interaction", "id": f"i_{qi}_{meta[r][1][-8:]}",
                           "ranking": rid, "timestamp": str(ts), "type": "click",
                           "item": sanitize(meta[r][1])})

out = SCRATCH / "mr_events.jsonl"
with out.open("w") as f:
    for e in events:
        f.write(json.dumps(e) + "\n")

feats_yaml = "\n".join(
    f"  - name: {n}\n    type: number\n    scope: item\n    field: ranking.{n}" for n in FEAT_NAMES)
config = f"""state:
  type: memory
train:
  type: memory
features:
{feats_yaml}
models:
  ltr:
    type: lambdamart
    backend:
      type: xgboost
      iterations: 300
      seed: 0
    weights:
      click: 1
    features:
{chr(10).join(f"      - {n}" for n in FEAT_NAMES)}
"""
(SCRATCH / "mr_config.yml").write_text(config)
n_rank = sum(1 for e in events if e["event"] == "ranking")
n_int = sum(1 for e in events if e["event"] == "interaction")
print(f"{n_rank} ranking + {n_int} interaction events -> {out}")
print(f"config -> {SCRATCH / 'mr_config.yml'}  ({len(FEAT_NAMES)} features)")
