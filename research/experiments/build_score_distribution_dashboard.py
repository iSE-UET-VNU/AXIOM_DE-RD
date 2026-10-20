"""Build an interactive score debugger from full BM25/dense cached matrices."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/work/ohr_replacement_20260920/retrieval/combined_220_base"
INSPECTOR = ROOT / "data/work/ohr_replacement_20260920/retrieval/score_inspector"
TEMPLATE = ROOT / "research/experiments/score_distribution_dashboard.template.html"
OUTPUT = INSPECTOR / "score-distribution-dashboard.html"
OLD_INPUT = ROOT / "data/work/light_debug_bundle_benchmark_default_bm25_tesseract_kf3_top20/bundle_debug_light-retrieval/input"
OHR_INPUT = ROOT / "data/work/ohr_replacement_20260920/inputs"
HYBRID_RUN = ROOT / "data/work/ohr_replacement_20260920/retrieval/merged_220_all_arms/runs/hybrid_bm25_embeddings.jsonl"
TOP = 30


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def dataset_for(qid: str) -> str:
    if qid.startswith("vidore_physics::"):
        return "physics"
    if qid.startswith("vidore_industrial::"):
        return "industrial"
    return "ohr"


def page_id_from_qrel(row: dict) -> str:
    return row.get("page_id", row["source_evidence_id"])


def ranks_and_curve(scores: np.ndarray, page_ids: list[str], gold: set[str]) -> tuple[list[float], list[float], list[dict]]:
    order = np.argsort(-scores, kind="stable")
    inverse = np.empty(order.size, dtype=np.int32)
    inverse[order] = np.arange(1, order.size + 1)
    peak = float(scores[order[0]])
    raw_curve = scores[order[:TOP]].round(5).tolist()
    curve = (scores[order[:TOP]] / peak).clip(min=0).round(5).tolist() if peak else [0.0] * TOP
    gold_rows = [
        {"page_id": pid, "rank": int(inverse[i]), "score": round(float(scores[i] / peak), 5) if peak else 0.0, "raw": round(float(scores[i]), 5)}
        for i, pid in enumerate(page_ids) if pid in gold
    ]
    return curve, raw_curve, gold_rows


def main() -> None:
    metadata = json.loads((BASE / "metadata.json").read_text(encoding="utf-8"))
    qids, page_ids = metadata["query_ids"], metadata["page_ids"]
    bm = np.load(BASE / "page_bm25_raw.npy", mmap_mode="r")
    dense = np.load(BASE / "query_page_dense.npy", mmap_mode="r")
    if bm.shape != dense.shape or bm.shape != (len(qids), len(page_ids)):
        raise RuntimeError(f"Unexpected matrices: bm={bm.shape}, dense={dense.shape}")
    query_text = {row["query_id"]: row["query"] for row in read_jsonl(OLD_INPUT / "queries.jsonl")}
    query_text.update({row["query_id"]: row["query"] for row in read_jsonl(OHR_INPUT / "queries.jsonl")})
    qrels: dict[str, set[str]] = {}
    for row in read_jsonl(OLD_INPUT / "qrels_page_level.jsonl"):
        if row["query_id"].startswith(("vidore_physics::", "vidore_industrial::")):
            qrels.setdefault(row["query_id"], set()).add(page_id_from_qrel(row))
    for row in read_jsonl(OHR_INPUT / "qrels.jsonl"):
        qrels.setdefault(row["query_id"], set()).add(page_id_from_qrel(row))
    hybrid = {row["query_id"]: row for row in read_jsonl(HYBRID_RUN)}
    global_bm, global_dense, global_raw_bm, global_raw_dense, queries = [], [], [], [], []
    for qi, qid in enumerate(qids):
        gold = qrels.get(qid, set())
        bm_curve, bm_raw_curve, bm_gold = ranks_and_curve(np.asarray(bm[qi]), page_ids, gold)
        dense_curve, dense_raw_curve, dense_gold = ranks_and_curve(np.asarray(dense[qi]), page_ids, gold)
        run = hybrid[qid]
        candidate = {pid: rank for rank, pid in enumerate(run.get("candidate_pages", []), 1)}
        returned = {pid: rank for rank, pid in enumerate(run.get("returned_pages", []), 1)}
        hybrid_gold = [{"page_id": pid, "rank": rank} for pid, rank in candidate.items() if pid in gold]
        returned_gold = [{"page_id": pid, "rank": rank} for pid, rank in returned.items() if pid in gold]
        queries.append({
            "query_id": qid, "dataset": dataset_for(qid), "query": query_text.get(qid, "(query text unavailable)"),
            "goldCount": len(gold), "bmCurve": bm_curve, "denseCurve": dense_curve, "bmRawCurve": bm_raw_curve, "denseRawCurve": dense_raw_curve,
            "bmGold": bm_gold, "denseGold": dense_gold, "hybridGold": hybrid_gold, "returnedGold": returned_gold,
            "bmAt20": bm_curve[19], "denseAt20": dense_curve[19], "bmRawMax": float(np.max(bm[qi])), "denseRawMax": float(np.max(dense[qi])),
        })
        global_bm.append(bm_curve)
        global_dense.append(dense_curve)
        global_raw_bm.append(bm_raw_curve)
        global_raw_dense.append(dense_raw_curve)
    for q in queries:
        if q["bmAt20"] < 0.08 and q["denseAt20"] > 0.35:
            q["outlier"] = "BM25 rơi nhanh, dense còn mịn"
        elif q["bmAt20"] > 0.45 and q["denseAt20"] > 0.45:
            q["outlier"] = "cả hai signal mịn"
        elif q["bmAt20"] < 0.10 and q["denseAt20"] < 0.15:
            q["outlier"] = "cả hai signal rơi nhanh"
        else:
            q["outlier"] = ""
    def bands(matrix: list[list[float]]) -> list[list[float]]:
        return np.quantile(np.asarray(matrix), [0.25, 0.5, 0.75], axis=0).T.round(5).tolist()
    default = next((i for i, q in enumerate(queries) if q["query_id"] == "vidore_physics::524"), 0)
    data = {"queries": queries, "globalBm": bands(global_bm), "globalDense": bands(global_dense), "globalRawBm": bands(global_raw_bm), "globalRawDense": bands(global_raw_dense), "defaultIndex": default}
    html = TEMPLATE.read_text(encoding="utf-8").replace("__SCORE_DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    if "__SCORE_DATA__" in html or "<!doctype" in html.lower() or "<html" in html.lower():
        raise RuntimeError("Invalid fragment")
    OUTPUT.write_text(html, encoding="utf-8")
    if OUTPUT.stat().st_size >= 1_000_000:
        raise RuntimeError(f"Fragment exceeds 1 MB: {OUTPUT.stat().st_size}")
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
