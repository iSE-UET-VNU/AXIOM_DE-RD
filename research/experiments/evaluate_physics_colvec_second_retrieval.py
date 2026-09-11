"""Evaluate ColVec as the second retriever over exported Physics file scopes.

The ColVec notebook has already produced a query-by-page score matrix.  This
experiment deliberately does not encode or score again: it filters that matrix
by the light-retrieval file scopes and ranks the surviving pages.  This is the
offline equivalent of serving ColVec over a per-query page candidate scope.

The report keeps the stages separate:

* file-scope recall: recall of the files selected by light retrieval;
* page-pool ceiling: gold-page recall before ColVec ranking;
* second-retrieval ranking: page recall/nDCG/hit at Kp.

No qrels are read while building a ranking.  Qrels are loaded only by the
evaluation step after all score/scope validation has completed.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.benchmarks.vidore_v3 import ViDoreV3


DEFAULT_SCORE_DIR = ROOT / "data/work/physics_colvec_export"
DEFAULT_SCOPE_DIR = (
    ROOT / "data/benchmark/vidore_v3/exports/physics_file_retrieval_for_colembed"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "data/benchmark/vidore_v3/results/physics_colvec_second_retrieval"
)
KF_VALUES = (3, 5, 10, 15, 20)
KP_VALUES = (10, 50, 100)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def _file_id(page_id: str) -> str:
    return page_id.split("#page=", 1)[0]


def _ndcg_at_k(ranked: list[str], qrel: dict[str, int], k: int) -> float:
    def gain(value: int) -> float:
        return float((2 ** int(value)) - 1)

    dcg = sum(
        gain(qrel.get(page, 0)) / math.log2(rank + 1)
        for rank, page in enumerate(ranked[:k], 1)
        if qrel.get(page, 0) > 0
    )
    ideal = sorted((int(value) for value in qrel.values()), reverse=True)[:k]
    idcg = sum(gain(value) / math.log2(rank + 1) for rank, value in enumerate(ideal, 1))
    return dcg / idcg if idcg else 0.0


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _load_inputs(
    score_dir: Path,
    scope_dir: Path,
) -> tuple[np.ndarray, list[str], list[str], dict[str, Any], dict[int, dict[str, list[str]]]]:
    required = (
        "physics_colvec_scores.npy",
        "physics_colvec_keys.json",
        "physics_colvec_qids.json",
        "physics_colvec_meta.json",
    )
    missing = [name for name in required if not (score_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing ColVec export files: {missing}")

    scores = np.load(score_dir / "physics_colvec_scores.npy", mmap_mode="r")
    keys = [str(value) for value in _read_json(score_dir / "physics_colvec_keys.json")]
    qids = [str(value) for value in _read_json(score_dir / "physics_colvec_qids.json")]
    meta = _read_json(score_dir / "physics_colvec_meta.json")

    if scores.ndim != 2 or scores.shape != (len(qids), len(keys)):
        raise ValueError(
            f"Score shape {scores.shape} disagrees with qids/pages "
            f"({len(qids)}, {len(keys)})"
        )
    if not np.isfinite(scores).all():
        raise ValueError("ColVec score matrix contains non-finite values")
    if len(set(keys)) != len(keys) or len(set(qids)) != len(qids):
        raise ValueError("Duplicate page IDs or query IDs in ColVec export")
    if int(meta.get("n_pages", -1)) != len(keys) or int(meta.get("n_queries", -1)) != len(qids):
        raise ValueError("ColVec metadata counts disagree with exported arrays")

    key_set = set(keys)
    qid_set = set(qids)
    scopes: dict[int, dict[str, list[str]]] = {}
    for kf in KF_VALUES:
        path = scope_dir / f"queries_kf{kf}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing file scope export: {path}")
        rows: dict[str, list[str]] = {}
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                qid = str(row["qid"])
                if qid in rows:
                    raise ValueError(f"Duplicate qid {qid!r} in {path}:{line_number}")
                page_ids = [str(value) for value in row["candidate_page_ids"]]
                if len(page_ids) != len(set(page_ids)):
                    raise ValueError(f"Duplicate candidate page in {path}:{line_number}")
                if any(page_id not in key_set for page_id in page_ids):
                    raise ValueError(f"Unknown candidate page in {path}:{line_number}")
                if int(row["k_files"]) != kf:
                    raise ValueError(f"Wrong Kf metadata in {path}:{line_number}")
                rows[qid] = page_ids
        if set(rows) != qid_set:
            raise ValueError(f"QID set mismatch in {path}")
        scopes[kf] = rows

    return scores, keys, qids, meta, scopes


def _scope_file_ids(scope_dir: Path, kf: int) -> dict[str, list[str]]:
    path = scope_dir / f"queries_kf{kf}.jsonl"
    out: dict[str, list[str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                out[str(row["qid"])] = [str(value) for value in row["selected_file_ids"]]
    return out


def _rank_scope(
    scores: np.ndarray,
    keys: list[str],
    qids: list[str],
    scope: dict[str, list[str]] | None,
    top_k: int,
) -> dict[str, list[dict[str, Any]]]:
    key_to_col = {page_id: index for index, page_id in enumerate(keys)}
    qid_to_row = {qid: index for index, qid in enumerate(qids)}
    runs: dict[str, list[dict[str, Any]]] = {}
    for qid in qids:
        page_ids = keys if scope is None else scope[qid]
        columns = np.asarray([key_to_col[page_id] for page_id in page_ids], dtype=np.int64)
        values = np.asarray(scores[qid_to_row[qid], columns], dtype=np.float32)
        # Stable tie handling makes the run deterministic while preserving the
        # score matrix's original page order for exact score ties.
        order = np.argsort(-values, kind="stable")[:top_k]
        runs[qid] = [
            {
                "chunk_id": page_ids[int(position)],
                "doc_id": page_ids[int(position)],
                "score": float(values[int(position)]),
                "rank": rank,
            }
            for rank, position in enumerate(order, 1)
        ]
    return runs


def _evaluate_arm(
    runs: dict[str, list[dict[str, Any]]],
    qids: list[str],
    qrels: dict[str, dict[str, int]],
    scope: dict[str, list[str]] | None,
    selected_files: dict[str, list[str]] | None,
    kf: int | None,
    kps: tuple[int, ...],
    full_page_count: int,
    full_file_count: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    per_query: list[dict[str, Any]] = []
    for qid in qids:
        gold_pages = set(qrels[qid])
        gold_files = {_file_id(page_id) for page_id in gold_pages}
        candidate_pages = scope[qid] if scope is not None else None
        candidate_count = len(candidate_pages) if candidate_pages is not None else None
        ceiling = (
            len(set(candidate_pages) & gold_pages) / len(gold_pages)
            if candidate_pages is not None and gold_pages
            else 1.0
        )
        file_ids = selected_files[qid] if selected_files is not None else None
        file_recall = (
            len(set(file_ids) & gold_files) / len(gold_files)
            if file_ids is not None and gold_files
            else 1.0
        )
        file_hit = bool(set(file_ids or []) & gold_files) if file_ids is not None else True
        row: dict[str, Any] = {
            "qid": qid,
            "gold_page_count": len(gold_pages),
            "gold_file_count": len(gold_files),
            "candidate_page_count": candidate_count,
            "page_pool_recall_ceiling": ceiling,
            "selected_file_count": len(file_ids) if file_ids is not None else None,
            "file_scope_recall": file_recall,
            "file_scope_hit": file_hit,
            "ranked_page_ids": [str(item["chunk_id"]) for item in runs[qid]],
        }
        for kp in kps:
            ranked = row["ranked_page_ids"][:kp]
            found = set(ranked) & gold_pages
            row[f"page_hit@{kp}"] = bool(found)
            row[f"page_recall@{kp}"] = len(found) / len(gold_pages) if gold_pages else 0.0
            row[f"page_precision@{kp}"] = len(found) / kp
            row[f"nDCG@{kp}"] = _ndcg_at_k(ranked, qrels[qid], kp)
        per_query.append(row)

    metrics: dict[str, Any] = {
        "queries": len(qids),
        "scope_kf": kf,
        "avg_candidate_pages": _mean(
            [float(row["candidate_page_count"]) for row in per_query if row["candidate_page_count"] is not None]
        ) if scope is not None else float(full_page_count),
        "avg_selected_files": _mean(
            [float(row["selected_file_count"]) for row in per_query if row["selected_file_count"] is not None]
        ) if selected_files is not None else float(full_file_count),
        "file_scope_recall": _mean([float(row["file_scope_recall"]) for row in per_query]),
        "file_scope_hit": _mean([float(row["file_scope_hit"]) for row in per_query]),
        "page_pool_recall_ceiling": _mean(
            [float(row["page_pool_recall_ceiling"]) for row in per_query]
        ),
        "k_values": {},
    }
    for kp in kps:
        metrics["k_values"][str(kp)] = {
            "nDCG": _mean([float(row[f"nDCG@{kp}"]) for row in per_query]),
            "page_hit": _mean([float(row[f"page_hit@{kp}"]) for row in per_query]),
            "page_recall": _mean([float(row[f"page_recall@{kp}"]) for row in per_query]),
            "page_precision": _mean([float(row[f"page_precision@{kp}"]) for row in per_query]),
        }
    return metrics, per_query


def _write_report(
    output_dir: Path,
    report: dict[str, Any],
) -> None:
    lines = [
        "# Physics ColVec second retrieval",
        "",
        "ColVec ranking over the page scopes selected by light file retrieval.",
        "The ColVec score matrix was produced by the Colab notebook; this run only filters and ranks cached scores.",
        "",
        "## Results",
        "",
        "| Arm | Avg pages/query | File scope recall | Page-pool ceiling | Kp | nDCG | Page hit | Page recall | Page precision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in report["arms"]:
        base = arm["metrics"]
        for kp in report["config"]["k_values"]:
            value = base["k_values"][str(kp)]
            lines.append(
                f"| {arm['name']} | {base['avg_candidate_pages']:.2f} | "
                f"{base['file_scope_recall']:.2%} | {base['page_pool_recall_ceiling']:.2%} | "
                f"{kp} | {value['nDCG']:.2%} | {value['page_hit']:.2%} | "
                f"{value['page_recall']:.2%} | {value['page_precision']:.2%} |"
            )
    lines += [
        "",
        "## Interpretation",
        "",
        "- `file_scope_recall` measures the light file-retrieval stage.",
        "- `page_pool_recall_ceiling` is the maximum page recall possible after the file scope is fixed.",
        "- `page_recall` and `nDCG` measure the ColVec second-retrieval ranking inside that scope.",
        "- The full-corpus arm is an upper reference: it has no light-retrieval pruning.",
        "- This is ColVec1.1 4B from the export metadata, not the Nemotron ColEmbed8B arm.",
        "",
        "## Validation",
        "",
        f"- Score shape: `{report['inputs']['score_shape']}`; finite: `{report['inputs']['scores_finite']}`.",
        f"- Queries/pages: `{report['inputs']['queries']}` / `{report['inputs']['pages']}`.",
        "- Candidate page IDs were checked against the score-matrix page IDs.",
        "- Qrels were read only after score and scope validation.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-dir", type=Path, default=DEFAULT_SCORE_DIR)
    parser.add_argument("--scope-dir", type=Path, default=DEFAULT_SCOPE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--k-files", type=int, nargs="*", default=list(KF_VALUES))
    parser.add_argument("--k-pages", type=int, nargs="*", default=list(KP_VALUES))
    args = parser.parse_args()
    if tuple(args.k_files) != KF_VALUES:
        raise ValueError(f"This fixed experiment expects Kf values {KF_VALUES}")
    if not args.k_pages or any(value <= 0 for value in args.k_pages):
        raise ValueError("Kp values must be positive")

    scores, keys, qids, meta, scopes = _load_inputs(args.score_dir, args.scope_dir)
    benchmark = ViDoreV3(
        root=ROOT / "data/benchmark/vidore_v3",
        subset="physics",
        language="french",
    )
    qrels = benchmark.qrels()
    if set(qrels) != set(qids):
        raise ValueError("QID set mismatch between ColVec export and benchmark qrels")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = args.output_dir / "runs"
    arms: list[dict[str, Any]] = []
    all_per_query: list[dict[str, Any]] = []

    full_runs = _rank_scope(scores, keys, qids, None, max(args.k_pages))
    full_metrics, full_rows = _evaluate_arm(
        full_runs, qids, qrels, None, None, None, tuple(args.k_pages),
        len(keys), len({_file_id(page_id) for page_id in keys})
    )
    _write_jsonl(runs_dir / "colvec_full_corpus_top100.jsonl", [
        {"qid": qid, "retriever_id": "physics-colvec-second-full", "index_id": "physics-pages-full", "chunks": full_runs[qid]}
        for qid in qids
    ])
    arms.append({"name": "full_corpus", "metrics": full_metrics})
    all_per_query.extend(
        [{"arm": "full_corpus", **row} for row in full_rows]
    )

    for kf in KF_VALUES:
        scope = scopes[kf]
        selected_files = _scope_file_ids(args.scope_dir, kf)
        runs = _rank_scope(scores, keys, qids, scope, max(args.k_pages))
        metrics, rows = _evaluate_arm(
            runs, qids, qrels, scope, selected_files, kf, tuple(args.k_pages),
            len(keys), len({_file_id(page_id) for page_id in keys})
        )
        _write_jsonl(runs_dir / f"colvec_kf{kf}_top100.jsonl", [
            {"qid": qid, "retriever_id": f"physics-colvec-second-kf{kf}", "index_id": f"physics-pages-kf{kf}", "chunks": runs[qid]}
            for qid in qids
        ])
        arms.append({"name": f"Kf={kf}", "metrics": metrics})
        all_per_query.extend(
            [{"arm": f"Kf={kf}", **row} for row in rows]
        )

    report = {
        "dataset": "vidore_v3/physics",
        "language": "french",
        "method": "cached ColVec score matrix filtered by light file scopes",
        "config": {
            "k_files": list(KF_VALUES),
            "k_values": list(args.k_pages),
            "score_model": meta.get("model"),
            "score_dpi": meta.get("dpi"),
            "score_dim": meta.get("dim"),
            "qrels_used_for_ranking": False,
        },
        "inputs": {
            "score_dir": str(args.score_dir),
            "scope_dir": str(args.scope_dir),
            "score_shape": list(scores.shape),
            "scores_finite": bool(np.isfinite(scores).all()),
            "queries": len(qids),
            "pages": len(keys),
            "files": len({_file_id(page_id) for page_id in keys}),
        },
        "arms": arms,
    }
    _write_json(args.output_dir / "report.json", report)
    _write_jsonl(args.output_dir / "per_query.jsonl", all_per_query)
    _write_report(args.output_dir, report)
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "report": str(args.output_dir / "report.md"),
        "arms": {
            arm["name"]: {
                "page_recall@10": round(100.0 * arm["metrics"]["k_values"]["10"]["page_recall"], 2),
                "nDCG@10": round(100.0 * arm["metrics"]["k_values"]["10"]["nDCG"], 2),
                "page_pool_ceiling": round(100.0 * arm["metrics"]["page_pool_recall_ceiling"], 2),
            }
            for arm in arms
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
