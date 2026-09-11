"""Fixed RRF ensemble of two already-OOF light-retrieval runs."""

from __future__ import annotations

from pathlib import Path
import json
import sys
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics_adaptive_evidence_fusion import _load_run  # noqa: E402
from physics_hierarchical_retrieval import _derived_metrics, _write_run  # noqa: E402
from src.evaluation.benchmarks.vidore_v3 import ViDoreV3  # noqa: E402

BENCHMARK_ROOT = ROOT / "data/benchmark/vidore_v3"
RESULT_ROOT = BENCHMARK_ROOT / "results"
DEFAULT_OUTPUT = RESULT_ROOT / "physics_ce_dual_rrf"
RRF_K = 20


def _rrf(left: Mapping[str, list[dict[str, Any]]], right: Mapping[str, list[dict[str, Any]]], qids: Sequence[str]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for qid in qids:
        scores: dict[str, float] = {}
        for run in (left, right):
            for rank, item in enumerate(run[qid][:100], 1):
                page = str(item["chunk_id"])
                scores[page] = scores.get(page, 0.0) + 1.0 / (RRF_K + rank)
        ordered = sorted(scores, key=lambda page: (-scores[page], page))[:100]
        output[qid] = [{"chunk_id": page, "doc_id": page, "score": float(scores[page]), "rank": rank} for rank, page in enumerate(ordered, 1)]
    return output


def _metric_row(metrics: Mapping[str, Any]) -> dict[str, float]:
    return {"page_recall@10": metrics["page_recall@10"], "ndcg@10": metrics["ndcg@10"], "file_recall@3": metrics["file_metrics_by_k"]["3"]["file_recall"]}


def main() -> None:
    benchmark = ViDoreV3(root=BENCHMARK_ROOT, subset="physics", language="french")
    questions = sorted(list(benchmark.questions()), key=lambda q: int(q.qid.rsplit("::", 1)[1]))
    qids = [q.qid for q in questions]
    qrels = benchmark.qrels()
    ce = _load_run(RESULT_ROOT / "physics_cross_encoder_rerank/oof_run.jsonl")
    dual = _load_run(RESULT_ROOT / "physics_dual_encoder_fusion/oof_run.jsonl")
    if set(ce) != set(qids) or set(dual) != set(qids):
        raise RuntimeError("OOF input runs do not contain exactly the 302 Physics qids")
    fused = _rrf(ce, dual, qids)
    metrics = {"ce_oof": _derived_metrics(ce, qids, qrels), "dual_oof": _derived_metrics(dual, qids, qrels), "ce_dual_rrf20_oof": _derived_metrics(fused, qids, qrels)}
    output = DEFAULT_OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    _write_run(output / "oof_run.jsonl", fused, qids, queries={q.qid: q.query for q in questions})
    report = {"dataset": "vidore_v3/physics", "queries": len(qids), "rrf_constant": RRF_K, "inputs": {"cross_encoder_oof": str(RESULT_ROOT / "physics_cross_encoder_rerank/oof_run.jsonl"), "dual_encoder_oof": str(RESULT_ROOT / "physics_dual_encoder_fusion/oof_run.jsonl")}, "metrics": {name: _metric_row(value) for name, value in metrics.items()}, "success": metrics["ce_dual_rrf20_oof"]["page_recall@10"] > 0.4747, "notes": ["Both input runs are already out-of-fold; this fixed RRF combination performs no qrel-based refit.", "Cross-encoder candidate generation is limited to the c014/cross-Kf top-30 union; RRF is applied to the resulting OOF page runs.", "All metrics use canonical full page IDs."]}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Physics CE + dual encoder RRF", "", "| Run | Page recall@10 | nDCG@10 | File recall@3 |", "|---|---:|---:|---:|"]
    for name, value in metrics.items():
        row = _metric_row(value)
        lines.append(f"| {name} | {row['page_recall@10']:.2%} | {row['ndcg@10']:.2f} | {row['file_recall@3']:.2%} |")
    lines += ["", f"Fixed RRF constant: **{RRF_K}**.", f"Success (>47.47% OOF page recall): **{report['success']}**."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "metrics": {name: round(row["page_recall@10"] * 100, 2) for name, row in metrics.items()}, "success": report["success"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
