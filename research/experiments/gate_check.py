"""Exact per-question comparison against the frozen gate reference.

An aggregate can match while individual questions move, so the check is a
SHA-256 over the sorted per-question NDCG map. Any single question differing
changes the hash.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LADDER = ROOT / "data/benchmark/vidore_v3/results/physics_retrieval_ladder.json"


def digest(per_question: dict) -> str:
    return hashlib.sha256(json.dumps(per_question, sort_keys=True).encode()).hexdigest()


def summarize(rows: list[dict], indexes: set[str]) -> dict:
    return {
        f'{r["index"]}::{r["arm"]}': {
            "ndcg@10": r["ndcg@10"], "recall@10": r["recall@10"],
            "per_question_sha256": digest(r["per_question"]),
        }
        for r in rows if r["index"] in indexes
    }


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("reference", help="frozen gate3_baseline.json")
parser.add_argument("--ladder", default=str(LADDER))
parser.add_argument("--write", default="", help="write the current summary here instead of comparing")
args = parser.parse_args()

reference = json.loads(Path(args.reference).read_text())
indexes = {key.split("::")[0] for key in reference}
current = summarize(json.loads(Path(args.ladder).read_text()), indexes)

if args.write:
    Path(args.write).write_text(json.dumps(current, indent=1))
    raise SystemExit(f"wrote {len(current)} arms to {args.write}")

missing = sorted(set(reference) - set(current))
drift = [
    (key, reference[key], current[key])
    for key in sorted(set(reference) & set(current))
    if reference[key]["per_question_sha256"] != current[key]["per_question_sha256"]
]

print(f"{'arm':28s} {'ndcg ref':>9s} {'ndcg now':>9s}  {'sha ref':16s} {'sha now':16s}")
for key in sorted(reference):
    ref = reference[key]
    now = current.get(key)
    mark = "MISSING" if now is None else ("ok" if now["per_question_sha256"] == ref["per_question_sha256"] else "DRIFT")
    print(f"{key:28s} {ref['ndcg@10']:9.2f} "
          f"{(now['ndcg@10'] if now else float('nan')):9.2f}  "
          f"{ref['per_question_sha256'][:16]} {(now['per_question_sha256'][:16] if now else '-'):16s} {mark}")

if missing or drift:
    raise SystemExit(f"\nFAIL: {len(missing)} missing, {len(drift)} drifted.")
print(f"\nPASS: {len(reference)} arms identical per-question.")
