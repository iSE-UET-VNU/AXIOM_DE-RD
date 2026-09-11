"""Measure cached ColVec retrieval time for every Kf/Kp arm.

The ColVec export is a precomputed query-by-page score matrix.  Consequently
there is one shared ColVec cache/index boundary, not a new index per Kf/Kp.
This script records that shared boundary and times only the per-arm page
scope extraction plus top-10 ranking.  It does not run model inference.
"""

from __future__ import annotations

from pathlib import Path
import json
from time import perf_counter

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCORE_DIR = ROOT / "data/work/physics_colvec_export"
RERANK_CACHE_DIR = ROOT / "data/benchmark/vidore_v3/results/physics_all_kf_kp_cohere_then_colvec/rerank_cache"
OUTPUT = ROOT / "data/benchmark/vidore_v3/results/physics_all_kf_kp_cohere_then_colvec/colvec_timing.json"
KF_VALUES = (3, 5, 10, 15, 20)
KP_VALUES = (10, 20, 30, 40, 50, 100)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _file_id(page_id: str) -> str:
    return page_id.split("#page=", 1)[0]


def main() -> None:
    started = perf_counter()
    score_path = SCORE_DIR / "physics_colvec_scores.npy"
    scores = np.load(score_path, mmap_mode="r")
    keys = [str(value) for value in _read_json(SCORE_DIR / "physics_colvec_keys.json")]
    qids = [str(value) for value in _read_json(SCORE_DIR / "physics_colvec_qids.json")]
    page_to_col = {page_id: index for index, page_id in enumerate(keys)}
    qid_to_row = {qid: index for index, qid in enumerate(qids)}
    shared_cache_open_seconds = perf_counter() - started

    arms = []
    for kf in KF_VALUES:
        cache_path = RERANK_CACHE_DIR / f"cohere_r100_kf{kf}.jsonl"
        reranked: dict[str, list[str]] = {}
        cache_started = perf_counter()
        with cache_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    reranked[str(row["qid"])] = [
                        str(item["page_id"]) for item in row["ranked"]
                    ]
        rerank_cache_load_seconds = perf_counter() - cache_started
        for kp in KP_VALUES:
            arm_started = perf_counter()
            pages_per_query = []
            for qid in qids:
                pool = reranked[qid][:kp]
                columns = np.asarray([page_to_col[page] for page in pool], dtype=np.int64)
                values = np.asarray(scores[qid_to_row[qid], columns], dtype=np.float32)
                np.argsort(-values, kind="stable")[:10]
                pages_per_query.append(len(pool))
            retrieval_seconds = perf_counter() - arm_started
            arms.append({
                "k_files": kf,
                "k_pages": kp,
                "colvec_indexing_seconds": 0.0,
                "colvec_indexing_note": "Shared precomputed score matrix; no per-arm index build.",
                "score_matrix_bytes": score_path.stat().st_size,
                "rerank_cache_load_seconds": rerank_cache_load_seconds,
                "colvec_cached_retrieval_seconds": retrieval_seconds,
                "milliseconds_per_query": retrieval_seconds / len(qids) * 1000,
                "avg_pages_scored": sum(pages_per_query) / len(pages_per_query),
            })

    result = {
        "dataset": "vidore_v3/physics",
        "model": _read_json(SCORE_DIR / "physics_colvec_meta.json").get("model"),
        "score_shape": list(scores.shape),
        "queries": len(qids),
        "pages": len(keys),
        "files": len({_file_id(page) for page in keys}),
        "shared_colvec_cache_open_seconds": shared_cache_open_seconds,
        "gpu_indexing_seconds": None,
        "gpu_indexing_note": "Original Colab page/query encoding timing was not exported.",
        "arms": arms,
        "total_timing_pass_seconds": perf_counter() - started,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
