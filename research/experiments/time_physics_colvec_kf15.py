"""Time the cached ColVec Kf=15 retrieval path.

The exported Physics artifact contains a query-by-page score matrix, not the
original GPU embedding/indexing telemetry.  This script therefore measures
only the local cache-open, scope-load, and ranking phases and records that
boundary explicitly.
"""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCORE_DIR = ROOT / "data/work/physics_colvec_export"
SCOPE_PATH = (
    ROOT
    / "data/benchmark/vidore_v3/exports/physics_file_retrieval_for_colembed"
    / "queries_kf15.jsonl"
)
OUTPUT = (
    ROOT
    / "data/benchmark/vidore_v3/results/physics_colvec_second_retrieval"
    / "timing_kf15.json"
)


def main() -> None:
    started = perf_counter()
    scores = np.load(SCORE_DIR / "physics_colvec_scores.npy", mmap_mode="r")
    keys = [
        str(value)
        for value in json.loads(
            (SCORE_DIR / "physics_colvec_keys.json").read_text(encoding="utf-8")
        )
    ]
    qids = [
        str(value)
        for value in json.loads(
            (SCORE_DIR / "physics_colvec_qids.json").read_text(encoding="utf-8")
        )
    ]
    meta = json.loads(
        (SCORE_DIR / "physics_colvec_meta.json").read_text(encoding="utf-8")
    )
    cache_open_seconds = perf_counter() - started

    started = perf_counter()
    scopes: dict[str, list[str]] = {}
    with SCOPE_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                scopes[str(row["qid"])] = [
                    str(page_id) for page_id in row["candidate_page_ids"]
                ]
    scope_load_seconds = perf_counter() - started

    key_to_column = {page_id: index for index, page_id in enumerate(keys)}
    qid_to_row = {qid: index for index, qid in enumerate(qids)}
    started = perf_counter()
    for qid in qids:
        page_ids = scopes[qid]
        columns = np.asarray(
            [key_to_column[page_id] for page_id in page_ids], dtype=np.int64
        )
        values = np.asarray(scores[qid_to_row[qid], columns], dtype=np.float32)
        np.argsort(-values, kind="stable")[:100]
    rank_seconds = perf_counter() - started

    result = {
        "dataset": "vidore_v3/physics",
        "scope": "Kf=15",
        "model": meta.get("model"),
        "score_shape": list(scores.shape),
        "queries": len(qids),
        "pages": len(keys),
        "avg_scope_pages_per_query": sum(
            len(scopes[qid]) for qid in qids
        )
        / len(qids),
        "gpu_indexing_seconds": None,
        "gpu_indexing_note": (
            "Not present in the Colab export; score matrix and model metadata "
            "were exported without encode timing."
        ),
        "local_cached_retrieval": {
            "cache_open_and_metadata_seconds": cache_open_seconds,
            "scope_load_seconds": scope_load_seconds,
            "rank_top100_seconds": rank_seconds,
            "total_seconds": cache_open_seconds + scope_load_seconds + rank_seconds,
            "milliseconds_per_query": (
                (cache_open_seconds + scope_load_seconds + rank_seconds)
                / len(qids)
                * 1000
            ),
            "machine_note": "Local cache filtering/ranking; no ColVec model inference.",
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
