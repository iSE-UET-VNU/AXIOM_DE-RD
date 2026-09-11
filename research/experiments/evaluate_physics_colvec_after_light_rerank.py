"""Evaluate Cohere-reranked light pages followed by cached ColVec.

This is the Kf=3 follow-up for Kp=30 and Kp=40.  The existing Cohere
``or_only_r50`` run is read as page IDs/ranks only; no page text is copied to
the new artifacts.  ColVec then reranks the selected Kp page IDs and returns
top-10.
"""

from __future__ import annotations

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
LIGHT_RUN = ROOT / "data/benchmark/vidore_v3/results/physics_topk_budget_ablation/runs/full_kf3.jsonl"
RERANK_RUN = ROOT / "data/benchmark/vidore_v3/results/physics_fielded_kf3_openrouter_cohere_rerank/runs/or_only_r50.jsonl"
SCOPE_FILE = ROOT / "data/benchmark/vidore_v3/exports/physics_file_retrieval_for_colembed/queries_kf3.jsonl"
OUTPUT_DIR = ROOT / "data/benchmark/vidore_v3/results/physics_colvec_after_light_rerank"
KP_VALUES = (30, 40)
RERANK_DEPTH = 50
COLVEC_TOP_K = 10


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_id(page_id: str) -> str:
    return page_id.split("#page=", 1)[0]


def _ndcg(ranked: list[str], qrel: dict[str, int]) -> float:
    def gain(value: int) -> float:
        return float(2 ** int(value) - 1)

    dcg = sum(
        gain(qrel.get(page, 0)) / math.log2(rank + 1)
        for rank, page in enumerate(ranked[:COLVEC_TOP_K], 1)
        if qrel.get(page, 0) > 0
    )
    ideal = sorted((int(value) for value in qrel.values()), reverse=True)[:COLVEC_TOP_K]
    idcg = sum(gain(value) / math.log2(rank + 1) for rank, value in enumerate(ideal, 1))
    return dcg / idcg if idcg else 0.0


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _load_run(path: Path, *, key: str, qids: set[str], page_set: set[str]) -> dict[str, list[str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    result: dict[str, list[str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row["qid"])
            if qid in result:
                raise ValueError(f"Duplicate qid {qid} in {path}:{line_number}")
            page_ids = [str(item["chunk_id"]) for item in row[key]]
            if len(page_ids) < min(KP_VALUES) or len(page_ids) != len(set(page_ids)):
                raise ValueError(f"Invalid page list in {path}:{line_number}")
            if any(page_id not in page_set for page_id in page_ids):
                raise ValueError(f"Unknown page ID in {path}:{line_number}")
            result[qid] = page_ids
    if set(result) != qids:
        raise ValueError(f"QID set mismatch in {path}")
    return result


def main() -> None:
    scores = np.load(SCORE_DIR / "physics_colvec_scores.npy", mmap_mode="r")
    keys = [str(value) for value in _read_json(SCORE_DIR / "physics_colvec_keys.json")]
    qids = [str(value) for value in _read_json(SCORE_DIR / "physics_colvec_qids.json")]
    qid_set = set(qids)
    page_set = set(keys)
    if scores.shape != (len(qids), len(keys)):
        raise ValueError(f"Invalid ColVec shape {scores.shape}")
    page_to_col = {page_id: index for index, page_id in enumerate(keys)}
    qid_to_row = {qid: index for index, qid in enumerate(qids)}

    light = _load_run(LIGHT_RUN, key="results", qids=qid_set, page_set=page_set)
    reranked = _load_run(RERANK_RUN, key="chunks", qids=qid_set, page_set=page_set)
    for qid in qids:
        if set(light[qid]) != set(reranked[qid]):
            raise ValueError(f"Reranker changed candidate membership for {qid}")

    scope_files: dict[str, list[str]] = {}
    with SCOPE_FILE.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                scope_files[str(row["qid"])] = [str(value) for value in row["selected_file_ids"]]
    if set(scope_files) != qid_set:
        raise ValueError("Kf=3 scope qids do not match ColVec qids")

    benchmark = ViDoreV3(root=ROOT / "data/benchmark/vidore_v3", subset="physics", language="french")
    qrels = benchmark.qrels()
    if set(qrels) != qid_set:
        raise ValueError("Benchmark qrels do not match ColVec qids")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "runs").mkdir(parents=True, exist_ok=True)
    arms: list[dict[str, Any]] = []
    per_query: list[dict[str, Any]] = []
    started = perf_counter()

    for kp in KP_VALUES:
        arm_started = perf_counter()
        arm_rows: list[dict[str, Any]] = []
        run_rows: list[dict[str, Any]] = []
        for qid in qids:
            base_pool = light[qid][:kp]
            reranked_pool = reranked[qid][:kp]
            columns = np.asarray([page_to_col[page] for page in reranked_pool], dtype=np.int64)
            values = np.asarray(scores[qid_to_row[qid], columns], dtype=np.float32)
            order = np.argsort(-values, kind="stable")[:COLVEC_TOP_K]
            final_pages = [reranked_pool[int(position)] for position in order]
            gold_pages = set(qrels[qid])
            base_found = set(base_pool) & gold_pages
            rerank_found = set(reranked_pool) & gold_pages
            final_found = set(final_pages) & gold_pages
            run_rows.append({
                "qid": qid,
                "retriever_id": f"physics-cohere-r50-then-colvec-kf3-kp{kp}",
                "file_ids": scope_files[qid],
                "chunks": [
                    {
                        "page_id": page,
                        "file_id": _file_id(page),
                        "score": float(values[int(position)]),
                        "rank": rank,
                    }
                    for rank, position in enumerate(order, 1)
                    for page in [reranked_pool[int(position)]]
                ],
            })
            arm_rows.append({
                "qid": qid,
                "base_light_pool_recall_ceiling": len(base_found) / len(gold_pages),
                "reranked_light_pool_recall_ceiling": len(rerank_found) / len(gold_pages),
                "cohere_rerank_recall@10": len(set(reranked[qid][:10]) & gold_pages) / len(gold_pages),
                "colvec_page_recall@10": len(final_found) / len(gold_pages),
                "colvec_page_hit@10": bool(final_found),
                "colvec_nDCG@10": _ndcg(final_pages, qrels[qid]),
                "light_pool_pages_sent_to_colvec": len(reranked_pool),
                "final_page_ids": final_pages,
            })

        metrics = {
            "k_files": 3,
            "k_pages": kp,
            "reranker": "cohere/rerank-v3.5",
            "rerank_depth": RERANK_DEPTH,
            "file_recall@3": _mean([
                len(set(scope_files[qid]) & {_file_id(page) for page in qrels[qid]})
                / len({_file_id(page) for page in qrels[qid]})
                for qid in qids
            ]),
            "base_light_pool_recall_ceiling": _mean([row["base_light_pool_recall_ceiling"] for row in arm_rows]),
            "reranked_light_pool_recall_ceiling": _mean([row["reranked_light_pool_recall_ceiling"] for row in arm_rows]),
            "cohere_rerank_page_recall@10": _mean([row["cohere_rerank_recall@10"] for row in arm_rows]),
            "colvec_page_recall@10": _mean([row["colvec_page_recall@10"] for row in arm_rows]),
            "colvec_page_hit@10": _mean([float(row["colvec_page_hit@10"]) for row in arm_rows]),
            "colvec_nDCG@10": _mean([row["colvec_nDCG@10"] for row in arm_rows]),
            "avg_pages_sent_to_colvec": _mean([float(row["light_pool_pages_sent_to_colvec"]) for row in arm_rows]),
            "retrieval_seconds": perf_counter() - arm_started,
        }
        arms.append({"name": f"Kf=3,Kp={kp}", "metrics": metrics})
        per_query.extend({"arm": f"Kf=3,Kp={kp}", **row} for row in arm_rows)
        _write_jsonl(OUTPUT_DIR / "runs" / f"cohere_r50_then_colvec_kf3_kp{kp}_top10.jsonl", run_rows)

    report = {
        "dataset": "vidore_v3/physics",
        "language": "french",
        "method": "Cohere rerank depth 50 -> top-Kp page pool -> cached ColVec top-10",
        "config": {"k_files": 3, "k_pages": list(KP_VALUES), "rerank_depth": RERANK_DEPTH, "colvec_top_k": COLVEC_TOP_K, "qrels_used_for_ranking": False},
        "inputs": {"score_shape": list(scores.shape), "queries": len(qids), "pages": len(keys), "files": len({_file_id(page) for page in keys}), "total_seconds": perf_counter() - started},
        "artifacts_store_page_text": False,
        "arms": arms,
    }
    (OUTPUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_jsonl(OUTPUT_DIR / "per_query.jsonl", per_query)
    lines = [
        "# Physics: Cohere-reranked light pages before ColVec",
        "",
        "The existing Cohere rerank-v3.5 depth-50 run is used as page order. Only page/file IDs and scores are written to new artifacts.",
        "",
        "| Kf | Kp | Base light ceiling | Reranked pool ceiling | Cohere recall@10 | ColVec recall@10 | ColVec nDCG@10 | File recall@3 | Pages sent |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in arms:
        m = arm["metrics"]
        lines.append(
            f"| 3 | {m['k_pages']} | {m['base_light_pool_recall_ceiling']:.2%} | "
            f"{m['reranked_light_pool_recall_ceiling']:.2%} | {m['cohere_rerank_page_recall@10']:.2%} | "
            f"{m['colvec_page_recall@10']:.2%} | {m['colvec_nDCG@10']:.2%} | "
            f"{m['file_recall@3']:.2%} | {m['avg_pages_sent_to_colvec']:.2f} |"
        )
    lines += [
        "",
        "- The reranker can only reorder its existing depth-50 light candidate band.",
        "- ColVec can only rank pages that remain in the reranked top-Kp pool.",
        "- No page text is stored in this experiment's new artifacts.",
    ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


if __name__ == "__main__":
    main()
