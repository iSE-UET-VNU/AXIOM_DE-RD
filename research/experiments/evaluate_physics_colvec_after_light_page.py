"""Evaluate light page retrieval followed by cached ColVec reranking.

For each fixed Kf/Kp pair, this experiment takes the top-Kp pages from the
existing light hierarchical run for that Kf and lets ColVec rerank only those
pages.  It does not use qrels to build the pool or the ranking.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import math
import sys
from time import perf_counter
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.benchmarks.vidore_v3 import ViDoreV3


SCORE_DIR = ROOT / "data/work/physics_colvec_export"
LIGHT_DIR = ROOT / "data/benchmark/vidore_v3/results/physics_topk_budget_ablation/runs"
OUTPUT_DIR = ROOT / "data/benchmark/vidore_v3/results/physics_colvec_after_light_page"
KF_VALUES = (3, 5, 10, 15, 20)
KP_VALUES = (10, 20, 30, 40, 50, 100)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _ndcg_at_10(ranked: list[str], qrel: dict[str, int]) -> float:
    def gain(value: int) -> float:
        return float(2 ** int(value) - 1)

    dcg = sum(
        gain(qrel.get(page, 0)) / math.log2(rank + 1)
        for rank, page in enumerate(ranked[:10], 1)
        if qrel.get(page, 0) > 0
    )
    ideal = sorted((int(value) for value in qrel.values()), reverse=True)[:10]
    idcg = sum(gain(value) / math.log2(rank + 1) for rank, value in enumerate(ideal, 1))
    return dcg / idcg if idcg else 0.0


def _file_id(page_id: str) -> str:
    return page_id.split("#page=", 1)[0]


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _load_colvec() -> tuple[np.ndarray, list[str], list[str], dict[str, int], dict[str, int]]:
    scores = np.load(SCORE_DIR / "physics_colvec_scores.npy", mmap_mode="r")
    keys = [str(value) for value in _read_json(SCORE_DIR / "physics_colvec_keys.json")]
    qids = [str(value) for value in _read_json(SCORE_DIR / "physics_colvec_qids.json")]
    if scores.shape != (len(qids), len(keys)):
        raise ValueError(f"ColVec score shape {scores.shape} disagrees with IDs")
    if len(set(keys)) != len(keys) or len(set(qids)) != len(qids):
        raise ValueError("Duplicate ColVec page/query IDs")
    return scores, keys, qids, {key: i for i, key in enumerate(keys)}, {
        qid: i for i, qid in enumerate(qids)
    }


def _load_light_runs(qids: list[str], key_set: set[str]) -> dict[int, dict[str, list[str]]]:
    result: dict[int, dict[str, list[str]]] = {}
    for kf in KF_VALUES:
        path = LIGHT_DIR / f"full_kf{kf}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing light run: {path}")
        rows: dict[str, list[str]] = {}
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                qid = str(row["qid"])
                if qid in rows:
                    raise ValueError(f"Duplicate qid {qid} in {path}:{line_number}")
                pages = [str(item["chunk_id"]) for item in row["results"]]
                # A low-Kf query can have fewer than 100 pages in its selected
                # files.  For Kp=100 the correct pool is then all available
                # pages, not an artificial padded list.
                if len(pages) < min(KP_VALUES) or len(pages) != len(set(pages)):
                    raise ValueError(f"Light run {path} has invalid page depth for {qid}")
                if any(page not in key_set for page in pages):
                    raise ValueError(f"Light page not found in ColVec keys for {qid}")
                rows[qid] = pages
        if set(rows) != set(qids):
            raise ValueError(f"QID set mismatch for {path}")
        result[kf] = rows
    return result


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    scores, keys, qids, page_to_col, qid_to_row = _load_colvec()
    light_runs = _load_light_runs(qids, set(keys))
    benchmark = ViDoreV3(
        root=ROOT / "data/benchmark/vidore_v3",
        subset="physics",
        language="french",
    )
    qrels = benchmark.qrels()
    if set(qrels) != set(qids):
        raise ValueError("QID set mismatch between ColVec export and benchmark qrels")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "runs").mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    arms: list[dict[str, Any]] = []
    started = perf_counter()

    # File recall is determined at the Kf stage; it is repeated for each Kp.
    file_recall_by_kf: dict[int, float] = {}
    scope_files: dict[int, dict[str, list[str]]] = {}
    for kf in KF_VALUES:
        path = (
            ROOT
            / "data/benchmark/vidore_v3/exports/physics_file_retrieval_for_colembed"
            / f"queries_kf{kf}.jsonl"
        )
        scope_files[kf] = {}
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    item = json.loads(line)
                    scope_files[kf][str(item["qid"])] = [
                        str(file_id) for file_id in item["selected_file_ids"]
                    ]
        if set(scope_files[kf]) != set(qids):
            raise ValueError(f"Scope qid mismatch for Kf={kf}")
        file_recall_by_kf[kf] = _mean(
            [
                len(
                    set(scope_files[kf][qid])
                    & {_file_id(page) for page in qrels[qid]}
                )
                / len({_file_id(page) for page in qrels[qid]})
                for qid in qids
            ]
        )

    for kf in KF_VALUES:
        for kp in KP_VALUES:
            arm_started = perf_counter()
            per_query: list[dict[str, Any]] = []
            run_rows: list[dict[str, Any]] = []
            for qid in qids:
                light_pool = light_runs[kf][qid][:kp]
                columns = np.asarray([page_to_col[page] for page in light_pool], dtype=np.int64)
                values = np.asarray(scores[qid_to_row[qid], columns], dtype=np.float32)
                order = np.argsort(-values, kind="stable")[:10]
                ranked = [light_pool[int(position)] for position in order]
                gold_pages = set(qrels[qid])
                pool_found = set(light_pool) & gold_pages
                found = set(ranked) & gold_pages
                run_rows.append({
                    "qid": qid,
                    "retriever_id": f"physics-colvec-after-light-kf{kf}-kp{kp}",
                    "index_id": f"physics-light-page-pool-kf{kf}-kp{kp}",
                    "chunks": [
                        {
                            "chunk_id": page,
                            "doc_id": page,
                            "score": float(values[int(position)]),
                            "rank": rank,
                        }
                        for rank, position in enumerate(order, 1)
                        for page in [light_pool[int(position)]]
                    ],
                })
                per_query.append({
                    "qid": qid,
                    "light_pool_pages": len(light_pool),
                    "light_pool_recall_ceiling": len(pool_found) / len(gold_pages),
                    "colvec_page_recall@10": len(found) / len(gold_pages),
                    "colvec_page_hit@10": bool(found),
                    "colvec_page_precision@10": len(found) / 10,
                    "colvec_nDCG@10": _ndcg_at_10(ranked, qrels[qid]),
                })

            metrics = {
                "k_files": kf,
                "k_pages": kp,
                "file_recall@kf": file_recall_by_kf[kf],
                "light_page_recall_ceiling@kp": _mean(
                    [row["light_pool_recall_ceiling"] for row in per_query]
                ),
                "colvec_page_recall@10": _mean(
                    [row["colvec_page_recall@10"] for row in per_query]
                ),
                "colvec_page_hit@10": _mean(
                    [float(row["colvec_page_hit@10"]) for row in per_query]
                ),
                "colvec_page_precision@10": _mean(
                    [row["colvec_page_precision@10"] for row in per_query]
                ),
                "colvec_nDCG@10": _mean(
                    [row["colvec_nDCG@10"] for row in per_query]
                ),
                "avg_pages_sent_to_colvec": _mean(
                    [float(row["light_pool_pages"]) for row in per_query]
                ),
                "retrieval_seconds": perf_counter() - arm_started,
            }
            arm_name = f"Kf={kf},Kp={kp}"
            arms.append({"name": arm_name, "metrics": metrics})
            rows.extend({"arm": arm_name, **row} for row in per_query)
            _write_jsonl(
                args.output_dir / "runs" / f"colvec_after_light_kf{kf}_kp{kp}_top10.jsonl",
                run_rows,
            )

    report = {
        "dataset": "vidore_v3/physics",
        "language": "french",
        "method": "light hierarchical top-Kp page retrieval followed by cached ColVec top-10 reranking",
        "config": {
            "k_files": list(KF_VALUES),
            "k_pages": list(KP_VALUES),
            "colvec_top_k": 10,
            "light_runs": str(LIGHT_DIR),
            "qrels_used_for_ranking": False,
        },
        "inputs": {
            "score_shape": list(scores.shape),
            "queries": len(qids),
            "pages": len(keys),
            "files": len({_file_id(page) for page in keys}),
            "total_seconds": perf_counter() - started,
        },
        "arms": arms,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_jsonl(args.output_dir / "per_query.jsonl", rows)

    lines = [
        "# Physics: light page retrieval before ColVec",
        "",
        "Each arm selects Kf files, keeps the existing light page ranker's top-Kp pages, "
        "then reranks only that page pool with cached ColVec and returns top-10.",
        "",
        "| Kf | Kp | Light recall ceiling | ColVec page recall@10 | ColVec hit@10 | nDCG@10 | File recall@Kf | Avg pages sent |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in arms:
        m = arm["metrics"]
        lines.append(
            f"| {m['k_files']} | {m['k_pages']} | {m['light_page_recall_ceiling@kp']:.2%} | "
            f"{m['colvec_page_recall@10']:.2%} | {m['colvec_page_hit@10']:.2%} | "
            f"{m['colvec_nDCG@10']:.2%} | {m['file_recall@kf']:.2%} | "
            f"{m['avg_pages_sent_to_colvec']:.2f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- `Light recall ceiling` is the maximum page recall possible after the top-Kp light filter.",
        "- `ColVec page recall@10` is measured after reranking that filtered pool.",
        "- This is a ranking-only experiment over the exported ColVec score matrix; no GPU encoding is repeated.",
        "- The light page run files are the existing fixed-protocol `full_kf*.jsonl` outputs.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "arms": arms}, indent=2))


if __name__ == "__main__":
    main()
