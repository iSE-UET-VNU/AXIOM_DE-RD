"""Evaluate four language-routed dense/hybrid scoring policies on the OHR benchmark.

The experiment reuses the canonical merged 220-query cache.  It builds scores
without qrels, then evaluates the resulting runs against the fixed OHR,
Physics and Industrial qrels.

The four policies are:

* ``dense_all``: dense score for every query/file pair;
* ``hybrid_all``: 0.70 BM25 + 0.30 dense for every pair;
* ``same_dense_cross_hybrid``: dense for same-language pairs, hybrid for
  cross-language pairs;
* ``same_hybrid_cross_dense``: hybrid for same-language pairs, dense for
  cross-language pairs.

The query language is English for the canonical merged benchmark.  File
language comes from the cached language detector for Physics/Industrial and
is explicitly English for OHR files.
"""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.experiments.run_ohr_replacement_all_arms import (  # noqa: E402
    OLD_LANGUAGE,
    file_id,
    load_inputs,
    metric_rows,
    read_jsonl,
    write_jsonl,
)


BASE = ROOT / "data/work/ohr_replacement_20260920/retrieval/combined_220_base"
OUTPUT = ROOT / "data/work/ohr_replacement_20260920/retrieval/language_routing_matrix_run1"

BM25_ALPHA = 0.70
FILE_DIRECT_WEIGHT = 0.50
FILE_PAGE_WEIGHT = 0.50
PAGE_BASE_WEIGHT = 0.85
PAGE_FILE_WEIGHT = 0.15
FILE_K = 3
CANDIDATE_DEPTH = 200
SAVED_DEPTH = 20
QUERY_LANGUAGE = "en"

ARM_POLICIES: dict[str, dict[str, str]] = {
    "dense_all": {"same": "dense", "cross": "dense"},
    "hybrid_all": {"same": "hybrid", "cross": "hybrid"},
    "same_dense_cross_hybrid": {"same": "dense", "cross": "hybrid"},
    "same_hybrid_cross_dense": {"same": "hybrid", "cross": "dense"},
}


def max_norm(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    maximum = float(values.max()) if values.size else 0.0
    return values / maximum if maximum > 0.0 else np.zeros_like(values)


def _source_from_id(value: str) -> str:
    return value.split("::", 1)[0]


def load_file_languages(file_ids: Sequence[str]) -> tuple[dict[str, str], dict[str, str]]:
    """Load file language and provenance, restricted to the canonical corpus."""
    language: dict[str, str] = {}
    provenance: dict[str, str] = {}
    canonical = set(file_ids)

    for row in read_jsonl(OLD_LANGUAGE):
        fid = str(row.get("file_id") or row.get("doc_id") or "")
        if fid in canonical:
            language[fid] = str(row.get("detected_language") or "unknown").lower()
            provenance[fid] = "cached_file_language_detection"

    for fid in file_ids:
        if fid.startswith("ohrbench::"):
            language[fid] = "en"
            provenance[fid] = "canonical_ohr_english_contract"

    missing = [fid for fid in file_ids if fid not in language]
    if missing:
        raise RuntimeError(f"Missing language metadata for {len(missing)} files: {missing[:5]}")
    return language, provenance


def build_arm(
    arm_name: str,
    policy: Mapping[str, str],
    *,
    page_ids: Sequence[str],
    file_ids: Sequence[str],
    query_rows: Sequence[Mapping[str, Any]],
    page_bm25: np.ndarray,
    file_bm25: np.ndarray,
    dense: np.ndarray,
    file_languages: Mapping[str, str],
) -> list[dict[str, Any]]:
    page_file_pos = np.asarray([file_ids.index(file_id(page)) for page in page_ids], dtype=np.int32)
    file_pos = {fid: index for index, fid in enumerate(file_ids)}
    rows: list[dict[str, Any]] = []

    for qi, query in enumerate(query_rows):
        qid = str(query["query_id"])
        query_text = str(query["query"])
        page_bm = max_norm(page_bm25[qi])
        direct_bm = max_norm(file_bm25[qi])
        dense_scores = np.asarray(dense[qi], dtype=np.float64)
        dense_norm = max_norm(dense_scores)

        page_score = np.zeros(len(page_ids), dtype=np.float64)
        direct_score = np.zeros(len(file_ids), dtype=np.float64)
        selected_modes: dict[str, str] = {}

        for fi, fid in enumerate(file_ids):
            same = file_languages[fid] == QUERY_LANGUAGE
            mode = policy["same"] if same else policy["cross"]
            selected_modes[fid] = mode
            mask = page_file_pos == fi
            dense_file_max = float(dense_norm[mask].max()) if np.any(mask) else 0.0
            if mode == "dense":
                page_score[mask] = dense_norm[mask]
                direct_score[fi] = dense_file_max
            elif mode == "hybrid":
                page_score[mask] = BM25_ALPHA * page_bm[mask] + (1.0 - BM25_ALPHA) * dense_norm[mask]
                direct_score[fi] = BM25_ALPHA * direct_bm[fi] + (1.0 - BM25_ALPHA) * dense_file_max
            else:
                raise ValueError(f"Unknown scoring mode: {mode}")

        page_max = np.zeros(len(file_ids), dtype=np.float64)
        np.maximum.at(page_max, page_file_pos, page_score)
        file_score = FILE_DIRECT_WEIGHT * direct_score + FILE_PAGE_WEIGHT * max_norm(page_max)
        file_order = np.argsort(-file_score, kind="stable")
        selected_positions = file_order[:FILE_K]
        selected_set = set(int(index) for index in selected_positions)
        final_score = PAGE_BASE_WEIGHT * page_score + PAGE_FILE_WEIGHT * max_norm(file_score)[page_file_pos]
        allowed = np.asarray([index for index, fi in enumerate(page_file_pos) if int(fi) in selected_set], dtype=np.int32)
        order = allowed[np.argsort(-final_score[allowed], kind="stable")]

        selected = [str(file_ids[int(index)]) for index in selected_positions]
        candidate = [str(page_ids[int(index)]) for index in order[:CANDIDATE_DEPTH]]
        returned = candidate[:SAVED_DEPTH]
        selected_relation = Counter(
            "same" if file_languages[fid] == QUERY_LANGUAGE else "cross" for fid in selected
        )
        returned_relation = Counter(
            "same" if file_languages[file_id(page)] == QUERY_LANGUAGE else "cross" for page in returned
        )
        mode_counts = Counter(selected_modes[fid] for fid in selected)

        rows.append(
            {
                "query_id": qid,
                "query": query_text,
                "query_language": QUERY_LANGUAGE,
                "arm": arm_name,
                "same_language_mode": policy["same"],
                "cross_language_mode": policy["cross"],
                "selected_files": selected,
                "candidate_pages": candidate,
                "returned_pages": returned,
                "returned_scores": [float(final_score[int(index)]) for index in order[:SAVED_DEPTH]],
                "selected_file_relation_counts": dict(selected_relation),
                "returned_page_relation_counts": dict(returned_relation),
                "selected_file_mode_counts": dict(mode_counts),
                "score_summary": {
                    "dense_max": float(dense_norm.max(initial=0.0)),
                    "bm25_max": float(page_bm.max(initial=0.0)),
                    "selected_file_score_max": float(file_score[selected_positions].max(initial=0.0)),
                },
            }
        )

    return rows


def relation_metrics(
    rows: Sequence[Mapping[str, Any]],
    qrels: Mapping[str, set[str]],
    file_languages: Mapping[str, str],
) -> dict[str, Any]:
    """Report metrics split by the language relation of gold evidence files."""
    groups: dict[str, list[Mapping[str, Any]]] = {"same_only": [], "cross_only": [], "mixed": []}
    for row in rows:
        gold_files = {file_id(page) for page in qrels[str(row["query_id"])]}
        relations = {"same" if file_languages[fid] == QUERY_LANGUAGE else "cross" for fid in gold_files}
        group = "mixed" if len(relations) > 1 else ("same_only" if "same" in relations else "cross_only")
        groups[group].append(row)
    result = {}
    for group, subset in groups.items():
        if subset:
            subset_qrels = {str(row["query_id"]): qrels[str(row["query_id"])] for row in subset}
            result[group] = {
                "queries": len(subset),
                "@10": metric_rows(subset, subset_qrels, 10),
                "@20": metric_rows(subset, subset_qrels, 20),
            }
        else:
            result[group] = {"queries": 0}
    return result


def source_metrics(
    rows: Sequence[Mapping[str, Any]],
    qrels: Mapping[str, set[str]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for source in ("vidore_physics", "vidore_industrial", "ohrbench"):
        subset = [row for row in rows if _source_from_id(str(row["query_id"])) == source]
        subset_qrels = {str(row["query_id"]): qrels[str(row["query_id"])] for row in subset}
        result[source] = {
            "queries": len(subset),
            "@10": metric_rows(subset, subset_qrels, 10),
            "@20": metric_rows(subset, subset_qrels, 20),
        }
    return result


def pairwise_diagnostics(
    dense_rows: Sequence[Mapping[str, Any]],
    hybrid_rows: Sequence[Mapping[str, Any]],
    qrels: Mapping[str, set[str]],
) -> dict[str, Any]:
    dense_by_qid = {str(row["query_id"]): row for row in dense_rows}
    hybrid_by_qid = {str(row["query_id"]): row for row in hybrid_rows}
    overlap10: list[float] = []
    union_hit10: list[float] = []
    dense_hit10: list[float] = []
    hybrid_hit10: list[float] = []
    for qid, gold in qrels.items():
        dense_top = set(dense_by_qid[qid]["returned_pages"][:10])
        hybrid_top = set(hybrid_by_qid[qid]["returned_pages"][:10])
        overlap10.append(len(dense_top & hybrid_top) / 10.0)
        union_hit10.append(float(bool((dense_top | hybrid_top) & gold)))
        dense_hit10.append(float(bool(dense_top & gold)))
        hybrid_hit10.append(float(bool(hybrid_top & gold)))
    return {
        "mean_top10_overlap": float(np.mean(overlap10)),
        "union_page_hit@10": float(np.mean(union_hit10)),
        "dense_page_hit@10": float(np.mean(dense_hit10)),
        "hybrid_page_hit@10": float(np.mean(hybrid_hit10)),
        "queries_dense_only_hit": int(sum(d > h for d, h in zip(dense_hit10, hybrid_hit10))),
        "queries_hybrid_only_hit": int(sum(h > d for d, h in zip(dense_hit10, hybrid_hit10))),
        "queries_both_hit": int(sum(d == 1.0 and h == 1.0 for d, h in zip(dense_hit10, hybrid_hit10))),
        "queries_both_missed": int(sum(d == 0.0 and h == 0.0 for d, h in zip(dense_hit10, hybrid_hit10))),
    }


def main() -> int:
    started = time.perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((BASE / "metadata.json").read_text(encoding="utf-8"))
    page_ids = [str(value) for value in metadata["page_ids"]]
    file_ids = [str(value) for value in metadata["file_ids"]]
    query_ids = [str(value) for value in metadata["query_ids"]]
    page_bm25 = np.load(BASE / "page_bm25_raw.npy", mmap_mode="r")
    file_bm25 = np.load(BASE / "file_bm25_raw.npy", mmap_mode="r")
    dense = np.load(BASE / "query_page_dense.npy", mmap_mode="r")
    queries, qrels, _ = load_inputs()

    if len(queries) != 220 or len(page_ids) != 5091 or len(file_ids) != 103:
        raise RuntimeError(f"Canonical shape mismatch: queries={len(queries)}, pages={len(page_ids)}, files={len(file_ids)}")
    if query_ids != [str(row["query_id"]) for row in queries]:
        raise RuntimeError("Cached query order does not match canonical merged inputs")
    if page_bm25.shape != (220, 5091) or dense.shape != (220, 5091) or file_bm25.shape != (220, 103):
        raise RuntimeError(f"Unexpected score shapes: page_bm25={page_bm25.shape}, dense={dense.shape}, file_bm25={file_bm25.shape}")

    file_languages, language_provenance = load_file_languages(file_ids)
    language_counts = Counter(file_languages.values())
    if set(language_counts) - {"en", "fr"}:
        raise RuntimeError(f"Unexpected file language labels: {language_counts}")

    runs: dict[str, list[dict[str, Any]]] = {}
    arm_reports: dict[str, Any] = {}
    for arm_name, policy in ARM_POLICIES.items():
        rows = build_arm(
            arm_name,
            policy,
            page_ids=page_ids,
            file_ids=file_ids,
            query_rows=queries,
            page_bm25=page_bm25,
            file_bm25=file_bm25,
            dense=dense,
            file_languages=file_languages,
        )
        runs[arm_name] = rows
        write_jsonl(OUTPUT / "runs" / f"{arm_name}.jsonl", rows)
        arm_reports[arm_name] = {
            "@10": metric_rows(rows, qrels, 10),
            "@20": metric_rows(rows, qrels, 20),
            "by_source": source_metrics(rows, qrels),
            "by_gold_language_relation": relation_metrics(rows, qrels, file_languages),
        }

    pairwise = pairwise_diagnostics(runs["dense_all"], runs["hybrid_all"], qrels)
    protocol = {
        "contract_version": "language-routing-matrix-run1-v1",
        "benchmark": {"queries": 220, "files": 103, "pages": 5091, "composition": "80 Physics + 75 Industrial + 65 OHR-Bench"},
        "query_language": QUERY_LANGUAGE,
        "file_language_counts": dict(language_counts),
        "file_language_provenance_counts": dict(Counter(language_provenance.values())),
        "file_k": FILE_K,
        "candidate_depth": CANDIDATE_DEPTH,
        "saved_depth": SAVED_DEPTH,
        "same_language_definition": "file_language == query_language",
        "cross_language_definition": "file_language != query_language",
        "hybrid_formula": f"{BM25_ALPHA:.2f} * normalized_BM25 + {1.0 - BM25_ALPHA:.2f} * normalized_dense",
        "file_score_formula": "0.50 * direct_file_score + 0.50 * max_page_score",
        "page_final_formula": "0.85 * page_score + 0.15 * normalized_file_score",
        "qrels_used_for_ranking": False,
        "arm_policies": ARM_POLICIES,
    }
    report = {"protocol": protocol, "arms": arm_reports, "dense_vs_hybrid_diagnostics": pairwise, "timing_seconds": {"total": time.perf_counter() - started}}
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Language routing matrix — Run 1",
        "",
        "Canonical benchmark: 80 Physics + 75 Industrial + 65 OHR-Bench; 220 queries, 103 files, 5,091 pages.",
        "",
        "| Arm | Recall@10 | nDCG@10 | Recall@20 | nDCG@20 | File recall@3 | Candidate recall@200 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm_name in ARM_POLICIES:
        m10 = arm_reports[arm_name]["@10"]
        m20 = arm_reports[arm_name]["@20"]
        lines.append(
            f"| {arm_name} | {m10['page_recall@10']:.2%} | {m10['ndcg@10']:.2f} | "
            f"{m20['page_recall@20']:.2%} | {m20['ndcg@20']:.2f} | "
            f"{m10['file_recall@3']:.2%} | {m10['candidate_pool_recall@200']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Arm definitions",
            "",
            "- `dense_all`: dense-only for same- and cross-language files.",
            "- `hybrid_all`: 0.70 normalized BM25 + 0.30 normalized dense for both relations.",
            "- `same_dense_cross_hybrid`: dense-only for same-language files; hybrid for cross-language files.",
            "- `same_hybrid_cross_dense`: hybrid for same-language files; dense-only for cross-language files.",
            "",
            "## Dense versus hybrid stream diagnostics",
            "",
            f"- Mean top-10 overlap: {pairwise['mean_top10_overlap']:.2%}.",
            f"- Dense/hybrid union page hit@10: {pairwise['union_page_hit@10']:.2%}.",
            f"- Dense-only hit queries: {pairwise['queries_dense_only_hit']}; hybrid-only: {pairwise['queries_hybrid_only_hit']}; both hit: {pairwise['queries_both_hit']}; both missed: {pairwise['queries_both_missed']}.",
            "",
            "## Source breakdown",
            "",
            "| Arm | Source | Queries | Recall@10 | Recall@20 | File recall@3 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for arm_name in ARM_POLICIES:
        for source in ("vidore_physics", "vidore_industrial", "ohrbench"):
            values = arm_reports[arm_name]["by_source"][source]
            lines.append(
                f"| {arm_name} | {source} | {values['queries']} | "
                f"{values['@10']['page_recall@10']:.2%} | "
                f"{values['@20']['page_recall@20']:.2%} | "
                f"{values['@10']['file_recall@3']:.2%} |"
            )
    lines.extend(
        [
            "",
            "Qrels are used only after ranking for evaluation and diagnostics.",
        ]
    )
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"Output: {OUTPUT.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
