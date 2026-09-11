"""Evaluate light top-Kp pages followed by Cohere top-10, without ColVec.

The light runs contain 100 pages for each query and Kf.  A Cohere R=100
cache already contains one query-page relevance score for that band.  This
experiment restricts the light band to its first Kp pages and reorders only
those pages using the cached Cohere scores, then returns the first 10.

This is an offline cached-score evaluation.  It is intentionally separate
from the ColVec experiments and does not persist page text or call ColVec.
Because the Cohere scores were obtained on the R=100 request, this measures a
score-projection of "light top-Kp -> Cohere top-10".  It is not a fresh API
request for every Kp; the distinction is recorded in report.json/report.md.
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

from src.evaluation.benchmarks import load


LIGHT_DIR = ROOT / "data/benchmark/vidore_v3/results/physics_topk_budget_ablation/runs"
COHERE_DIR = ROOT / "data/benchmark/vidore_v3/results/physics_all_kf_kp_cohere_then_colvec/rerank_cache"
SCOPE_DIR = ROOT / "data/benchmark/vidore_v3/exports/physics_file_retrieval_for_colembed"
OUTPUT_DIR = ROOT / "data/benchmark/vidore_v3/results/physics_light_then_cohere"
KF_VALUES = (3, 5, 10, 15, 20)
KP_VALUES = (10, 20, 30, 40, 50, 100)
LIGHT_DEPTH = 100
FINAL_DEPTH = 10


def _file_id(page_id: str) -> str:
    return page_id.split("#page=", 1)[0]


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _ndcg(ranked: list[str], qrel: dict[str, int]) -> float:
    def gain(value: int) -> float:
        return float(2 ** int(value) - 1)

    dcg = sum(
        gain(qrel.get(page, 0)) / math.log2(rank + 1)
        for rank, page in enumerate(ranked[:FINAL_DEPTH], 1)
        if qrel.get(page, 0) > 0
    )
    ideal = sorted((int(value) for value in qrel.values()), reverse=True)[:FINAL_DEPTH]
    idcg = sum(gain(value) / math.log2(rank + 1) for rank, value in enumerate(ideal, 1))
    return dcg / idcg if idcg else 0.0


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    temporary.replace(path)


def _load_light(qids: list[str]) -> dict[int, dict[str, list[str]]]:
    output: dict[int, dict[str, list[str]]] = {}
    for kf in KF_VALUES:
        path = LIGHT_DIR / f"full_kf{kf}.jsonl"
        rows: dict[str, list[str]] = {}
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                qid = str(row["qid"])
                pages = [str(item["chunk_id"]) for item in row["results"]]
                if qid in rows or len(pages) < min(KP_VALUES) or len(pages) != len(set(pages)):
                    raise ValueError(f"Invalid light run {path}:{line_number}")
                rows[qid] = pages[:LIGHT_DEPTH]
        if set(rows) != set(qids):
            raise ValueError(f"QID mismatch in {path}")
        output[kf] = rows
    return output


def _load_cohere(
    qids: list[str], light: dict[int, dict[str, list[str]]]
) -> dict[int, dict[str, list[dict[str, Any]]]]:
    output: dict[int, dict[str, list[dict[str, Any]]]] = {}
    for kf in KF_VALUES:
        path = COHERE_DIR / f"cohere_r100_kf{kf}.jsonl"
        rows: dict[str, list[dict[str, Any]]] = {}
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                qid = str(row["qid"])
                ranked = [
                    {"page_id": str(item["page_id"]), "score": float(item["score"]), "rank": int(item["rank"])}
                    for item in row["ranked"]
                ]
                pages = [str(item["page_id"]) for item in ranked]
                if qid in rows or len(pages) < min(KP_VALUES) or len(pages) != len(set(pages)):
                    raise ValueError(f"Invalid Cohere cache {path}:{line_number}")
                if set(pages) != set(light[kf][qid]):
                    raise ValueError(f"Cohere cache membership mismatch for Kf={kf}, qid={qid}")
                rows[qid] = ranked
        if set(rows) != set(qids):
            raise ValueError(f"QID mismatch in {path}")
        output[kf] = rows
    return output


def _load_scope(qids: list[str]) -> dict[int, dict[str, list[str]]]:
    output: dict[int, dict[str, list[str]]] = {}
    for kf in KF_VALUES:
        path = SCOPE_DIR / f"queries_kf{kf}.jsonl"
        rows: dict[str, list[str]] = {}
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    rows[str(row["qid"])] = [str(value) for value in row["selected_file_ids"]]
        if set(rows) != set(qids):
            raise ValueError(f"Scope QID mismatch in {path}")
        output[kf] = rows
    return output


def main() -> None:
    started = perf_counter()
    benchmark = load("vidore_v3", subset="physics", language="french")
    questions = list(benchmark.questions())
    qids = [str(question.qid) for question in questions]
    if len(qids) != 302 or len(set(qids)) != 302:
        raise RuntimeError(f"Expected 302 unique Physics qids, got {len(qids)}")
    qrels = benchmark.qrels()
    if set(qrels) != set(qids):
        raise RuntimeError("QID mismatch between benchmark qrels and questions")

    all_pages = {page_id for qrel in qrels.values() for page_id in qrel}
    light = _load_light(qids)
    cohere = _load_cohere(qids, light)
    scopes = _load_scope(qids)

    arms: list[dict[str, Any]] = []
    per_query: list[dict[str, Any]] = []
    (OUTPUT_DIR / "runs").mkdir(parents=True, exist_ok=True)
    for kf in KF_VALUES:
        file_recall = _mean([
            len(set(scopes[kf][qid]) & {_file_id(page) for page in qrels[qid]})
            / len({_file_id(page) for page in qrels[qid]})
            for qid in qids
        ])
        for kp in KP_VALUES:
            arm_started = perf_counter()
            rows: list[dict[str, Any]] = []
            run_rows: list[dict[str, Any]] = []
            for qid in qids:
                light_pool = light[kf][qid][:kp]
                light_ids = set(light_pool)
                # Preserve the cached Cohere order for tie-stable projection.
                ranked = [item for item in cohere[kf][qid] if item["page_id"] in light_ids]
                effective_kp = min(kp, len(light_pool))
                if len(ranked) != effective_kp:
                    raise RuntimeError(f"Cohere projection size mismatch for {qid}, Kf={kf}, Kp={kp}")
                final = [str(item["page_id"]) for item in ranked[:FINAL_DEPTH]]
                gold = set(qrels[qid])
                light_found = set(light_pool) & gold
                final_found = set(final) & gold
                rows.append({
                    "qid": qid,
                    "light_pool_recall_ceiling": len(light_found) / len(gold),
                    "cohere_page_recall@10": len(final_found) / len(gold),
                    "cohere_page_hit@10": bool(final_found),
                    "cohere_page_precision@10": len(final_found) / FINAL_DEPTH,
                    "cohere_nDCG@10": _ndcg(final, qrels[qid]),
                    "candidate_pages": effective_kp,
                    "final_page_ids": final,
                })
                run_rows.append({
                    "qid": qid,
                    "chunks": [
                        {
                            "page_id": page,
                            "file_id": _file_id(page),
                            "rank": rank,
                            "cohere_score": float(item["score"]),
                        }
                        for rank, item in enumerate(ranked[:FINAL_DEPTH], 1)
                        for page in [str(item["page_id"])]
                    ],
                })
            metrics = {
                "k_files": kf,
                "k_pages": kp,
                "light_pool_recall_ceiling": _mean([row["light_pool_recall_ceiling"] for row in rows]),
                "cohere_page_recall@10": _mean([row["cohere_page_recall@10"] for row in rows]),
                "cohere_page_hit@10": _mean([float(row["cohere_page_hit@10"]) for row in rows]),
                "cohere_page_precision@10": _mean([row["cohere_page_precision@10"] for row in rows]),
                "cohere_nDCG@10": _mean([row["cohere_nDCG@10"] for row in rows]),
                "file_recall@kf": file_recall,
                "avg_light_pages_retrieved": _mean([float(row["candidate_pages"]) for row in rows]),
                "retrieval_seconds": perf_counter() - arm_started,
            }
            name = f"Kf={kf},Kp={kp}"
            arms.append({"name": name, "metrics": metrics})
            per_query.extend({"arm": name, **row} for row in rows)
            _write_jsonl(OUTPUT_DIR / "runs" / f"cohere_top10_kf{kf}_kp{kp}.jsonl", run_rows)

    report = {
        "dataset": "vidore_v3/physics",
        "language": "french",
        "method": "light top-Kp -> cached Cohere score projection -> top-10",
        "config": {
            "k_files": list(KF_VALUES),
            "k_pages": list(KP_VALUES),
            "light_retrieval_depth": LIGHT_DEPTH,
            "final_depth": FINAL_DEPTH,
            "reranker": "cohere/rerank-v3.5",
            "rerank_cache_depth": LIGHT_DEPTH,
            "qrels_used_for_ranking": False,
            "fresh_api_request_per_kp": False,
        },
        "inputs": {
            "queries": len(qids),
            "qrel_pages": len(all_pages),
            "cache_dir": str(COHERE_DIR.relative_to(ROOT)),
        },
        "artifacts_store_page_text": False,
        "timing_seconds": {"total": perf_counter() - started},
        "arms": arms,
    }
    _write_json(OUTPUT_DIR / "report.json", report)
    _write_jsonl(OUTPUT_DIR / "per_query.jsonl", per_query)

    lines = [
        "# Physics: light top-Kp followed by Cohere top-10",
        "",
        "The light run is truncated to top-Kp pages for each query/Kf. Those pages are then ordered by the cached `cohere/rerank-v3.5` scores and the first 10 are returned. ColVec is not used.",
        "",
        "> Important: this is a cached-score projection. The scores came from one Cohere R=100 request per query; the experiment does not issue a fresh API request for each Kp. It is therefore suitable for the budget curve, but should not be presented as a separate API call at every Kp.",
        "",
        "| Kf | Kp | Light ceiling | Cohere page recall@10 | Cohere nDCG@10 | Cohere hit@10 | File recall@Kf | Avg pages |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in arms:
        m = arm["metrics"]
        lines.append(
            f"| {m['k_files']} | {m['k_pages']} | {m['light_pool_recall_ceiling']:.2%} | "
            f"{m['cohere_page_recall@10']:.2%} | {m['cohere_nDCG@10']:.2%} | "
            f"{m['cohere_page_hit@10']:.2%} | {m['file_recall@kf']:.2%} | "
            f"{m['avg_light_pages_retrieved']:.0f} |"
        )
    lines += [
        "",
        "- Cohere can only reorder pages that are present in the light top-Kp pool; `Light ceiling` is the upper bound for that row.",
        "- The file stage is unchanged from `full_kf*.jsonl`; `File recall@Kf` is reported as metadata, not used for ranking.",
        "- New artifacts contain page/file IDs and scores only; page text is not persisted.",
    ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(OUTPUT_DIR), "arms": arms}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
