"""Quick comparison of three French Physics retrieval combinations."""

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
    source_name,
    write_jsonl,
)


OUTPUT = NATIVE_OUTPUT.parent / "physics_french_quick_combinations"
LOW_BM25_ALPHA = 0.30


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
    page_bm25_fr = np.load(NATIVE_OUTPUT / "native_french_page_bm25_raw.npy", mmap_mode="r")
    file_bm25_fr = np.load(NATIVE_OUTPUT / "native_french_file_bm25_raw.npy", mmap_mode="r")
    dense_fr = np.load(NATIVE_OUTPUT / "native_french_page_dense_raw.npy", mmap_mode="r")
    physics_queries = [query for query in queries if str(query["query_id"]).startswith("vidore_physics::")]
    native_index = {str(query["query_id"]): index for index, query in enumerate(physics_queries)}

    file_position = {fid: index for index, fid in enumerate(file_ids)}
    page_file_pos = np.asarray([file_position[file_id(page)] for page in page_ids], dtype=np.int32)
    french_page_mask = np.asarray(
        [file_languages[file_id(page)] == "fr" and file_id(page).startswith("vidore_physics::") for page in page_ids],
        dtype=bool,
    )
    french_file_mask = np.asarray(
        [fid.startswith("vidore_physics::") and file_languages[fid] == "fr" for fid in file_ids],
        dtype=bool,
    )

    def relation_scores(qi: int, qid: str) -> dict[str, np.ndarray]:
        dense_en_norm = max_norm(np.asarray(dense_en[qi], dtype=np.float64))
        bm25_en_norm = max_norm(np.asarray(page_bm25_en[qi], dtype=np.float64))
        direct_en_norm = max_norm(np.asarray(file_bm25_en[qi], dtype=np.float64))
        dense = dense_en_norm.copy()
        bm25 = bm25_en_norm.copy()
        direct = direct_en_norm.copy()
        native_qi = native_index.get(qid)
        if native_qi is not None:
            dense_fr_norm = max_norm(np.asarray(dense_fr[native_qi], dtype=np.float64))
            bm25_fr_norm = max_norm(np.asarray(page_bm25_fr[native_qi], dtype=np.float64))
            direct_fr_norm = max_norm(np.asarray(file_bm25_fr[native_qi], dtype=np.float64))
            dense[french_page_mask] = dense_fr_norm[french_page_mask]
            bm25[french_page_mask] = bm25_fr_norm[french_page_mask]
            direct[french_file_mask] = direct_fr_norm[french_file_mask]
        return {"dense": dense, "bm25": bm25, "direct": direct}

    def blend(scores: dict[str, np.ndarray], french_alpha: float) -> tuple[np.ndarray, np.ndarray]:
        """Blend French files with a supplied alpha; keep other files canonical."""
        page = BM25_ALPHA * scores["bm25"] + (1.0 - BM25_ALPHA) * scores["dense"]
        french_page = french_alpha * scores["bm25"] + (1.0 - french_alpha) * scores["dense"]
        page[french_page_mask] = french_page[french_page_mask]
        direct = np.zeros(len(file_ids), dtype=np.float64)
        for fi in range(len(file_ids)):
            page_mask = page_file_pos == fi
            dense_file_max = float(scores["dense"][page_mask].max()) if np.any(page_mask) else 0.0
            alpha = french_alpha if french_file_mask[fi] else BM25_ALPHA
            direct[fi] = alpha * scores["direct"][fi] + (1.0 - alpha) * dense_file_max
        return page, direct

    def select_files(page_score: np.ndarray, direct_score: np.ndarray) -> tuple[set[int], np.ndarray]:
        page_max = np.zeros(len(file_ids), dtype=np.float64)
        np.maximum.at(page_max, page_file_pos, page_score)
        file_score = 0.50 * direct_score + 0.50 * max_norm(page_max)
        selected_positions = np.argsort(-file_score, kind="stable")[:FILE_K]
        return {int(position) for position in selected_positions}, file_score

    def make_row(qi: int, arm: str, page_base: np.ndarray, file_score: np.ndarray, selected: set[int], rerank_candidates: np.ndarray | None = None) -> dict[str, Any]:
        final_score = 0.85 * page_base + 0.15 * max_norm(file_score)[page_file_pos]
        allowed = np.flatnonzero(np.isin(page_file_pos, list(selected)))
        if rerank_candidates is None:
            order = allowed[np.argsort(-final_score[allowed], kind="stable")]
        else:
            order = rerank_candidates[np.argsort(-final_score[rerank_candidates], kind="stable")]
        candidate = [page_ids[int(position)] for position in order[:CANDIDATE_DEPTH]]
        returned = candidate[:SAVED_DEPTH]
        return {
            "query_id": str(queries[qi]["query_id"]),
            "query": str(queries[qi]["query"]),
            "arm": arm,
            "selected_files": [file_ids[position] for position in sorted(selected, key=lambda p: (-file_score[p], p))],
            "candidate_pages": candidate,
            "returned_pages": returned,
            "returned_scores": [float(final_score[int(position)]) for position in order[:SAVED_DEPTH]],
            "selected_file_languages": dict(Counter(file_languages[file_ids[position]] for position in selected)),
        }

    runs: dict[str, list[dict[str, Any]]] = {"keyword03_dense07": [], "hybrid_file_dense_page": [], "dense_candidate_hybrid_rerank": []}
    for qi, query in enumerate(queries):
        qid = str(query["query_id"])
        scores = relation_scores(qi, qid)
        current_hybrid_page, current_hybrid_direct = blend(scores, BM25_ALPHA)
        low_keyword_page, low_keyword_direct = blend(scores, LOW_BM25_ALPHA)

        selected_low, low_file_score = select_files(low_keyword_page, low_keyword_direct)
        runs["keyword03_dense07"].append(make_row(qi, "keyword03_dense07", low_keyword_page, low_file_score, selected_low))

        selected_hybrid, hybrid_file_score = select_files(current_hybrid_page, current_hybrid_direct)
        dense_page_base = current_hybrid_page.copy()
        dense_page_base[french_page_mask] = scores["dense"][french_page_mask]
        runs["hybrid_file_dense_page"].append(make_row(qi, "hybrid_file_dense_page", dense_page_base, hybrid_file_score, selected_hybrid))

        selected_allowed = np.flatnonzero(np.isin(page_file_pos, list(selected_hybrid)))
        dense_candidate_order = selected_allowed[np.argsort(-dense_page_base[selected_allowed], kind="stable")]
        rerank_candidates = dense_candidate_order[:CANDIDATE_DEPTH]
        runs["dense_candidate_hybrid_rerank"].append(make_row(qi, "dense_candidate_hybrid_rerank", current_hybrid_page, hybrid_file_score, selected_hybrid, rerank_candidates))

    reports: dict[str, Any] = {}
    physics_reports: dict[str, Any] = {}
    for arm, rows in runs.items():
        write_jsonl(OUTPUT / "runs" / f"{arm}.jsonl", rows)
        reports[arm] = {"@10": metric_rows(rows, qrels, 10), "@20": metric_rows(rows, qrels, 20)}
        subset = [row for row in rows if source_name(str(row["query_id"])) == "vidore_physics"]
        subset_qrels = {str(row["query_id"]): qrels[str(row["query_id"])] for row in subset}
        physics_reports[arm] = {"@10": metric_rows(subset, subset_qrels, 10), "@20": metric_rows(subset, subset_qrels, 20)}

    protocol = {
        "contract_version": "physics-french-quick-combinations-v1",
        "benchmark": {"queries": 220, "physics_queries": 80, "industrial_queries": 75, "ohr_queries": 65, "files": 103, "pages": 5091},
        "file_k": FILE_K,
        "candidate_depth": CANDIDATE_DEPTH,
        "saved_depth": SAVED_DEPTH,
        "qrels_used_for_ranking": False,
        "native_french_query_on_french_physics_files": True,
        "arm_definitions": {
            "keyword03_dense07": "French Physics page/file score = 0.30 BM25 + 0.70 dense; other files keep current hybrid",
            "hybrid_file_dense_page": "current hybrid selects files; dense ranks French Physics pages inside selected files",
            "dense_candidate_hybrid_rerank": "current hybrid selects files; dense creates up-to-200 page candidates; current hybrid reranks those candidates",
        },
        "language_provenance_counts": dict(Counter(language_provenance.values())),
    }
    report = {"protocol": protocol, "arms": reports, "physics_only": physics_reports}
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Physics French quick combinations",
        "",
        "Canonical benchmark with native French queries on French Physics files; non-French files keep the current English hybrid route.",
        "",
        "| Arm | Recall@10 | nDCG@10 | Recall@20 | nDCG@20 | File recall@3 | Candidate recall@200 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in reports.items():
        m10, m20 = values["@10"], values["@20"]
        lines.append(f"| {name} | {m10['page_recall@10']:.2%} | {m10['ndcg@10']:.2f} | {m20['page_recall@20']:.2%} | {m20['ndcg@20']:.2f} | {m10['file_recall@3']:.2%} | {m10['candidate_pool_recall@200']:.2%} |")
    lines.extend(["", "## Physics-only result", "", "| Arm | Recall@10 | nDCG@10 | Recall@20 | nDCG@20 | File recall@3 | Candidate recall@200 |", "|---|---:|---:|---:|---:|---:|---:|"])
    for name, values in physics_reports.items():
        m10, m20 = values["@10"], values["@20"]
        lines.append(f"| {name} | {m10['page_recall@10']:.2%} | {m10['ndcg@10']:.2f} | {m20['page_recall@20']:.2%} | {m20['ndcg@20']:.2f} | {m10['file_recall@3']:.2%} | {m10['candidate_pool_recall@200']:.2%} |")
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"Output: {OUTPUT.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
