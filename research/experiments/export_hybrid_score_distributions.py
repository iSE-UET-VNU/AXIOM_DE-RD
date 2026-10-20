"""Export readable distributions from the cached 220x5091 hybrid score matrices.

The full raw matrices stay in the original ``.npy`` cache.  This utility writes
small per-query diagnostics and top-score probes, avoiding a 1.12M-row CSV.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE = ROOT / "data/work/ohr_replacement_20260920/retrieval/combined_220_base"
OUT = ROOT / "data/work/ohr_replacement_20260920/retrieval/score_inspector"


def norm(x: np.ndarray) -> np.ndarray:
    maximum = float(x.max()) if x.size else 0.0
    return x / maximum if maximum > 0 else np.zeros_like(x)


def stats(x: np.ndarray) -> dict[str, float | int]:
    nonzero = x[x > 0]
    return {
        "min": float(x.min(initial=0.0)), "max": float(x.max(initial=0.0)),
        "nonzero": int(nonzero.size),
        "p50_nonzero": float(np.quantile(nonzero, 0.50)) if nonzero.size else 0.0,
        "p90_nonzero": float(np.quantile(nonzero, 0.90)) if nonzero.size else 0.0,
        "p99_nonzero": float(np.quantile(nonzero, 0.99)) if nonzero.size else 0.0,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.loads((BASE / "metadata.json").read_text(encoding="utf-8"))
    page_ids = [str(x) for x in meta["page_ids"]]
    query_ids = [str(x) for x in meta["query_ids"]]
    bm = np.load(BASE / "page_bm25_raw.npy", mmap_mode="r")
    dense = np.load(BASE / "query_page_dense.npy", mmap_mode="r")
    if bm.shape != dense.shape or bm.shape != (220, 5091):
        raise RuntimeError(f"Unexpected cached shapes: bm={bm.shape}, dense={dense.shape}")
    bins = np.linspace(0.0, 1.0, 41)
    summary, probes = [], []
    for qi, qid in enumerate(query_ids):
        b_raw, d_raw = np.asarray(bm[qi], dtype=np.float64), np.asarray(dense[qi], dtype=np.float64)
        b, d = norm(b_raw), norm(d_raw)
        summary.append({
            "query_id": qid,
            "bm25_raw": stats(b_raw), "dense_raw": stats(d_raw),
            "bm25_norm_histogram": np.histogram(b, bins=bins)[0].tolist(),
            "dense_norm_histogram": np.histogram(d, bins=bins)[0].tolist(),
            "histogram_bins": bins.tolist(),
            "pearson_raw_on_bm25_nonzero": float(np.corrcoef(b_raw[b_raw > 0], d_raw[b_raw > 0])[0, 1]) if int((b_raw > 0).sum()) > 1 else 0.0,
        })
        bm_order, dense_order = np.argsort(-b, kind="stable"), np.argsort(-d, kind="stable")
        bm_rank, dense_rank = np.empty(len(page_ids), dtype=np.int32), np.empty(len(page_ids), dtype=np.int32)
        bm_rank[bm_order] = np.arange(1, len(page_ids) + 1)
        dense_rank[dense_order] = np.arange(1, len(page_ids) + 1)
        positions = set(bm_order[:100]) | set(dense_order[:100])
        for pos in sorted(positions, key=lambda i: (-max(b[i], d[i]), i)):
            probes.append({"query_id": qid, "page_id": page_ids[int(pos)], "file_id": page_ids[int(pos)].split("#page=", 1)[0],
                           "bm25_raw": float(b_raw[pos]), "bm25_max_norm": float(b[pos]),
                           "dense_raw": float(d_raw[pos]), "dense_max_norm": float(d[pos]),
                           "bm25_rank": int(bm_rank[pos]), "dense_rank": int(dense_rank[pos])})
    (OUT / "query_distribution_summary.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in summary), encoding="utf-8")
    (OUT / "top100_union_score_probes.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in probes), encoding="utf-8")
    (OUT / "README.md").write_text(
        "# Hybrid score inspector\n\n"
        "Raw full-corpus arrays remain in `../combined_220_base/`: `page_bm25_raw.npy` and `query_page_dense.npy`. "
        "Both are `(220, 5091)` and align with `metadata.json.query_ids` and `metadata.json.page_ids`.\n\n"
        "`query_distribution_summary.jsonl` has 40-bin max-normalized histograms and raw-score quantiles for every query. "
        "`top100_union_score_probes.jsonl` provides the union of each signal's top 100 pages with raw score, max-normalized score and per-signal rank. "
        "It is a readable diagnostic subset; it is not the final hierarchical rank.\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(summary)} query distributions and {len(probes)} score probes to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
