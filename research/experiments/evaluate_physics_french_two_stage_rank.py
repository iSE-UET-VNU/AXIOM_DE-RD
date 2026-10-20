"""Test hybrid file discovery with dense-only page ranking for French Physics."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.experiments.evaluate_physics_native_french_language_route import (  # noqa: E402
    BASE,
    BM25_ALPHA,
    CANDIDATE_DEPTH,
    FILE_K,
    OUTPUT as NATIVE_OUTPUT,
    SAVED_DEPTH,
    file_id,
    load_file_languages,
    load_inputs,
    max_norm,
    metric_rows,
    read_jsonl,
    source_name,
    write_jsonl,
)


OUTPUT = NATIVE_OUTPUT.parent / "physics_french_two_stage_rank"


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((BASE / "metadata.json").read_text(encoding="utf-8"))
    page_ids = [str(value) for value in metadata["page_ids"]]
    file_ids = [str(value) for value in metadata["file_ids"]]
    queries, qrels, _ = load_inputs()
    file_languages, language_provenance = load_file_languages(file_ids)

    page_bm25_en = np.load(BASE / "page_bm25_raw.npy", mmap_mode="r")
    file_bm25_en = np.load(BASE / "file_bm25_raw.npy", mmap_mode="r")
    dense_en = np.load(BASE / "query_page_dense.npy", mmap_mode="r")
    native_dir = NATIVE_OUTPUT
    page_bm25_fr = np.load(native_dir / "native_french_page_bm25_raw.npy", mmap_mode="r")
    file_bm25_fr = np.load(native_dir / "native_french_file_bm25_raw.npy", mmap_mode="r")
    dense_fr = np.load(native_dir / "native_french_page_dense_raw.npy", mmap_mode="r")
    qid_to_native = {
        str(row["query_id"]): index
        for index, row in enumerate(
            query for query in queries if str(query["query_id"]).startswith("vidore_physics::")
        )
    }
    file_position = {fid: index for index, fid in enumerate(file_ids)}
    page_file_pos = np.asarray([file_position[file_id(page)] for page in page_ids], dtype=np.int32)
    french_page_mask = np.asarray(
        [
            file_languages[file_id(page)] == "fr"
            and file_id(page).startswith("vidore_physics::")
            for page in page_ids
        ],
        dtype=bool,
    )
    french_file_mask = np.asarray(
        [fid.startswith("vidore_physics::") and file_languages[fid] == "fr" for fid in file_ids],
        dtype=bool,
    )

    def rank_query(qi: int, qid: str, page_policy: str) -> dict[str, Any]:
        native_index = qid_to_native.get(qid)
        dense_en_norm = max_norm(np.asarray(dense_en[qi], dtype=np.float64))
        bm25_en_norm = max_norm(np.asarray(page_bm25_en[qi], dtype=np.float64))
        direct_en_norm = max_norm(np.asarray(file_bm25_en[qi], dtype=np.float64))
        dense = dense_en_norm.copy()
        bm25 = bm25_en_norm.copy()
        direct_bm25 = direct_en_norm.copy()
        if native_index is not None:
            dense_fr_norm = max_norm(np.asarray(dense_fr[native_index], dtype=np.float64))
            bm25_fr_norm = max_norm(np.asarray(page_bm25_fr[native_index], dtype=np.float64))
            direct_fr_norm = max_norm(np.asarray(file_bm25_fr[native_index], dtype=np.float64))
            dense[french_page_mask] = dense_fr_norm[french_page_mask]
            bm25[french_page_mask] = bm25_fr_norm[french_page_mask]
            direct_bm25[french_file_mask] = direct_fr_norm[french_file_mask]

        selection_page_score = BM25_ALPHA * bm25 + (1.0 - BM25_ALPHA) * dense
        direct_score = np.zeros(len(file_ids), dtype=np.float64)
        for fi in range(len(file_ids)):
            mask = page_file_pos == fi
            dense_file_max = float(dense[mask].max()) if np.any(mask) else 0.0
            direct_score[fi] = BM25_ALPHA * float(direct_bm25[fi]) + (1.0 - BM25_ALPHA) * dense_file_max
        selection_page_max = np.zeros(len(file_ids), dtype=np.float64)
        np.maximum.at(selection_page_max, page_file_pos, selection_page_score)
        file_score = 0.50 * direct_score + 0.50 * max_norm(selection_page_max)
        selected_positions = np.argsort(-file_score, kind="stable")[:FILE_K]
        selected_set = {int(position) for position in selected_positions}

        if page_policy == "dense":
            # Keep the canonical hybrid page rank outside French Physics.
            page_base = selection_page_score.copy()
            page_base[french_page_mask] = dense[french_page_mask]
        elif page_policy == "hybrid":
            page_base = selection_page_score.copy()
        else:
            raise ValueError(page_policy)
        final_score = 0.85 * page_base + 0.15 * max_norm(file_score)[page_file_pos]
        allowed = np.flatnonzero(np.isin(page_file_pos, list(selected_set)))
        order = allowed[np.argsort(-final_score[allowed], kind="stable")]
        candidate = [page_ids[int(position)] for position in order[:CANDIDATE_DEPTH]]
        return {
            "query_id": qid,
            "query": str(queries[qi]["query"]),
            "arm": f"hybrid_file_selection_{page_policy}_page_rank",
            "selected_files": [file_ids[int(position)] for position in selected_positions],
            "candidate_pages": candidate,
            "returned_pages": candidate[:SAVED_DEPTH],
            "returned_scores": [float(final_score[int(position)]) for position in order[:SAVED_DEPTH]],
            "selected_file_languages": dict(Counter(file_languages[file_ids[int(position)]] for position in selected_positions)),
        }

    runs: dict[str, list[dict[str, Any]]] = {}
    reports: dict[str, Any] = {}
    physics_reports: dict[str, Any] = {}
    for policy in ("hybrid", "dense"):
        name = f"hybrid_file_selection_{policy}_page_rank"
        rows = [rank_query(qi, str(query["query_id"]), policy) for qi, query in enumerate(queries)]
        runs[name] = rows
        write_jsonl(OUTPUT / "runs" / f"{name}.jsonl", rows)
        reports[name] = {"@10": metric_rows(rows, qrels, 10), "@20": metric_rows(rows, qrels, 20)}
        subset = [row for row in rows if source_name(str(row["query_id"])) == "vidore_physics"]
        subset_qrels = {str(row["query_id"]): qrels[str(row["query_id"])] for row in subset}
        physics_reports[name] = {"@10": metric_rows(subset, subset_qrels, 10), "@20": metric_rows(subset, subset_qrels, 20)}

    protocol = {
        "contract_version": "physics-french-two-stage-rank-v1",
        "benchmark": {"queries": 220, "physics_queries": 80, "industrial_queries": 75, "ohr_queries": 65, "files": 103, "pages": 5091},
        "file_k": FILE_K,
        "candidate_depth": CANDIDATE_DEPTH,
        "saved_depth": SAVED_DEPTH,
        "qrels_used_for_ranking": False,
        "selection_formula": "hybrid alpha=0.70 for all files; native French dense query on French Physics files",
        "page_policies": ["hybrid", "dense"],
        "language_provenance_counts": dict(Counter(language_provenance.values())),
    }
    report = {"protocol": protocol, "arms": reports, "physics_only": physics_reports}
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Physics French two-stage hybrid file selection",
        "",
        "Hybrid selects the three files; page ranking is then either hybrid or native-French dense-only. The corpus and protocol are fixed.",
        "",
        "| Arm | Recall@10 | nDCG@10 | Recall@20 | nDCG@20 | File recall@3 | Candidate recall@200 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in reports.items():
        m10, m20 = values["@10"], values["@20"]
        lines.append(
            f"| {name} | {m10['page_recall@10']:.2%} | {m10['ndcg@10']:.2f} | {m20['page_recall@20']:.2%} | {m20['ndcg@20']:.2f} | {m10['file_recall@3']:.2%} | {m10['candidate_pool_recall@200']:.2%} |"
        )
    lines.extend(["", "## Physics-only result", "", "| Arm | Recall@10 | nDCG@10 | Recall@20 | nDCG@20 | File recall@3 | Candidate recall@200 |", "|---|---:|---:|---:|---:|---:|---:|"])
    for name, values in physics_reports.items():
        m10, m20 = values["@10"], values["@20"]
        lines.append(
            f"| {name} | {m10['page_recall@10']:.2%} | {m10['ndcg@10']:.2f} | {m20['page_recall@20']:.2%} | {m20['ndcg@20']:.2f} | {m10['file_recall@3']:.2%} | {m10['candidate_pool_recall@200']:.2%} |"
        )
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"Output: {OUTPUT.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
