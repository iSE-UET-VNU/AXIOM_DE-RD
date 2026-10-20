"""Ablate accent folding for the native-French Physics BM25 route.

The experiment reuses the cached native-French dense vectors and the merged
220-query corpus.  It changes only the French BM25 representation: NFC text
versus Unicode accent-folded text for both pages and queries.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys
import unicodedata
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.experiments.evaluate_physics_native_french_language_route import (  # noqa: E402
    ARM_CONFIGS,
    BASE,
    CANDIDATE_DEPTH,
    FILE_K,
    OUTPUT as NATIVE_OUTPUT,
    PAGE_EMBEDDINGS,
    SAVED_DEPTH,
    build_rows,
    cached_vector,
    file_id,
    load_file_languages,
    load_inputs,
    load_native_french_mapping,
    load_page_texts,
    metric_rows,
    read_jsonl,
    source_name,
    write_jsonl,
)
from src.retrieval.sparse import BM25Index  # noqa: E402


OUTPUT = NATIVE_OUTPUT.parent / "physics_french_accent_ablation"


def fold_accents(text: str) -> str:
    """Remove Unicode combining marks while preserving ordinary characters."""
    decomposed = unicodedata.normalize("NFD", str(text))
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")


def accent_stats(texts: Sequence[str]) -> dict[str, Any]:
    total_chars = sum(len(text) for text in texts)
    marked_chars = sum(
        1
        for text in texts
        for char in unicodedata.normalize("NFD", text)
        if unicodedata.category(char) == "Mn"
    )
    changed_texts = sum(fold_accents(text) != text for text in texts)
    return {
        "texts": len(texts),
        "total_characters": total_chars,
        "combining_mark_characters": marked_chars,
        "combining_mark_rate": marked_chars / total_chars if total_chars else 0.0,
        "texts_changed_by_folding": changed_texts,
        "changed_text_rate": changed_texts / len(texts) if texts else 0.0,
    }


def build_french_bm25_folded(
    page_ids: Sequence[str],
    page_texts: Mapping[str, str],
    file_ids: Sequence[str],
    french_queries: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    """Build page/file BM25 scores after applying the same accent fold."""
    page_index = BM25Index().build(
        [
            {
                "chunk_id": page,
                "doc_id": file_id(page),
                "text": fold_accents(page_texts[page]),
            }
            for page in page_ids
        ]
    )
    file_text: dict[str, list[str]] = {fid: [] for fid in file_ids}
    for page in page_ids:
        file_text[file_id(page)].append(fold_accents(page_texts[page]))
    file_index = BM25Index().build(
        [
            {"chunk_id": fid, "doc_id": fid, "text": "\n".join(file_text[fid])}
            for fid in file_ids
        ]
    )
    page_scores = np.zeros((len(french_queries), len(page_ids)), dtype=np.float32)
    file_scores = np.zeros((len(french_queries), len(file_ids)), dtype=np.float32)
    for qi, query in enumerate(french_queries):
        folded_query = fold_accents(str(query["query"]))
        for position, score in page_index.search(folded_query, len(page_ids)):
            page_scores[qi, int(position)] = float(score)
        for position, score in file_index.search(folded_query, len(file_ids)):
            file_scores[qi, int(position)] = float(score)
    return page_scores, file_scores


def build_report_markdown(
    arm_reports: Mapping[str, Mapping[str, Any]],
    physics_reports: Mapping[str, Mapping[str, Any]],
    stats: Mapping[str, Any],
) -> str:
    lines = [
        "# Physics French BM25 accent ablation",
        "",
        "Canonical benchmark: 80 Physics + 75 Industrial + 65 OHR-Bench; 220 queries, 103 files, 5,091 pages.",
        "",
        "Only the BM25 representation for French Physics files changes. Native French dense vectors, English routing for all other files, file/page protocol, and qrels are fixed.",
        "",
        "| Arm | Recall@10 | nDCG@10 | Recall@20 | nDCG@20 | File recall@3 | Candidate recall@200 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm_name, report in arm_reports.items():
        m10 = report["@10"]
        m20 = report["@20"]
        lines.append(
            f"| {arm_name} | {m10['page_recall@10']:.2%} | {m10['ndcg@10']:.2f} | "
            f"{m20['page_recall@20']:.2%} | {m20['ndcg@20']:.2f} | "
            f"{m10['file_recall@3']:.2%} | {m10['candidate_pool_recall@200']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Physics-only result",
            "",
            "| Arm | Recall@10 | nDCG@10 | Recall@20 | nDCG@20 | File recall@3 | Candidate recall@200 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for arm_name, report in physics_reports.items():
        m10 = report["@10"]
        m20 = report["@20"]
        lines.append(
            f"| {arm_name} | {m10['page_recall@10']:.2%} | {m10['ndcg@10']:.2f} | "
            f"{m20['page_recall@20']:.2%} | {m20['ndcg@20']:.2f} | "
            f"{m10['file_recall@3']:.2%} | {m10['candidate_pool_recall@200']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Accent coverage",
            "",
            f"- French Physics page texts: `{stats['page_texts']['texts']:,}`; changed by folding: `{stats['page_texts']['texts_changed_by_folding']:,}` ({stats['page_texts']['changed_text_rate']:.2%}).",
            f"- Native French queries: `{stats['queries']['texts']:,}`; changed by folding: `{stats['queries']['texts_changed_by_folding']:,}` ({stats['queries']['changed_text_rate']:.2%}).",
            f"- Combining-mark rate in French Physics page text: `{stats['page_texts']['combining_mark_rate']:.2%}` of characters.",
            "",
            "Qrels are used only after ranking. No embedding API call, rendering, or parsing was performed.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((BASE / "metadata.json").read_text(encoding="utf-8"))
    page_ids = [str(value) for value in metadata["page_ids"]]
    file_ids = [str(value) for value in metadata["file_ids"]]
    query_ids = [str(value) for value in metadata["query_ids"]]
    queries, qrels, _ = load_inputs()
    if (len(queries), len(file_ids), len(page_ids)) != (220, 103, 5091):
        raise RuntimeError("Canonical benchmark shape mismatch")
    if query_ids != [str(row["query_id"]) for row in queries]:
        raise RuntimeError("Canonical query order does not match the merged cache")

    page_bm25_en = np.load(BASE / "page_bm25_raw.npy", mmap_mode="r")
    file_bm25_en = np.load(BASE / "file_bm25_raw.npy", mmap_mode="r")
    query_page_dense_en = np.load(BASE / "query_page_dense.npy", mmap_mode="r")
    page_vectors = np.load(PAGE_EMBEDDINGS, mmap_mode="r")
    file_languages, language_provenance = load_file_languages(file_ids)
    page_texts = load_page_texts(page_ids)
    canonical_physics = [row for row in queries if str(row["query_id"]).startswith("vidore_physics::")]
    mapping, mapping_audit = load_native_french_mapping(canonical_physics)

    french_vectors = []
    for row in canonical_physics:
        vector, _ = cached_vector(mapping[str(row["query_id"])] ["french_query"])
        french_vectors.append(vector)
    french_matrix = np.asarray(french_vectors, dtype=np.float32)
    french_dense_matrix = np.asarray(french_matrix @ page_vectors.T, dtype=np.float32)
    french_query_index = {str(row["query_id"]): index for index, row in enumerate(canonical_physics)}
    query_page_dense_fr = {
        qid: french_dense_matrix[index]
        for qid, index in french_query_index.items()
    }

    french_query_rows = [{"query": mapping[str(row["query_id"])] ["french_query"]} for row in canonical_physics]
    from research.experiments.evaluate_physics_native_french_language_route import build_french_bm25

    page_bm25_fr_nfc, file_bm25_fr_nfc = build_french_bm25(
        page_ids, page_texts, file_ids, french_query_rows
    )
    page_bm25_fr_folded, file_bm25_fr_folded = build_french_bm25_folded(
        page_ids, page_texts, file_ids, french_query_rows
    )
    page_bm25_nfc = {
        qid: page_bm25_fr_nfc[index] for qid, index in french_query_index.items()
    }
    file_bm25_nfc = {
        qid: file_bm25_fr_nfc[index] for qid, index in french_query_index.items()
    }
    page_bm25_folded = {
        qid: page_bm25_fr_folded[index] for qid, index in french_query_index.items()
    }
    file_bm25_folded = {
        qid: file_bm25_fr_folded[index] for qid, index in french_query_index.items()
    }

    base_config = {
        "french_file_query": "french",
        "french_file_mode": "hybrid",
        "nonfrench_file_mode": "hybrid",
    }
    arm_configs = {
        "native_fr_hybrid_nfc": base_config,
        "native_fr_hybrid_accent_folded": base_config,
    }
    runs: dict[str, list[dict[str, Any]]] = {}
    arm_reports: dict[str, Any] = {}
    physics_reports: dict[str, Any] = {}
    for arm_name, config in arm_configs.items():
        page_bm25_fr = page_bm25_nfc if arm_name.endswith("nfc") else page_bm25_folded
        file_bm25_fr = file_bm25_nfc if arm_name.endswith("nfc") else file_bm25_folded
        rows = build_rows(
            arm_name=arm_name,
            config=config,
            queries=queries,
            page_ids=page_ids,
            file_ids=file_ids,
            file_languages=file_languages,
            query_page_dense_en=query_page_dense_en,
            page_bm25_en=page_bm25_en,
            file_bm25_en=file_bm25_en,
            query_page_dense_fr=query_page_dense_fr,
            page_bm25_fr=page_bm25_fr,
            file_bm25_fr=file_bm25_fr,
        )
        runs[arm_name] = rows
        write_jsonl(OUTPUT / "runs" / f"{arm_name}.jsonl", rows)
        arm_reports[arm_name] = {
            "@10": metric_rows(rows, qrels, 10),
            "@20": metric_rows(rows, qrels, 20),
        }
        physics_rows = [row for row in rows if source_name(str(row["query_id"])) == "vidore_physics"]
        physics_qrels = {str(row["query_id"]): qrels[str(row["query_id"])] for row in physics_rows}
        physics_reports[arm_name] = {
            "@10": metric_rows(physics_rows, physics_qrels, 10),
            "@20": metric_rows(physics_rows, physics_qrels, 20),
        }

    french_page_texts = [
        page_texts[page]
        for page in page_ids
        if file_languages[file_id(page)] == "fr" and file_id(page).startswith("vidore_physics::")
    ]
    french_queries = [mapping[str(row["query_id"])] ["french_query"] for row in canonical_physics]
    stats = {
        "page_texts": accent_stats(french_page_texts),
        "queries": accent_stats(french_queries),
        "file_language_counts": dict(Counter(file_languages.values())),
        "file_language_provenance_counts": dict(Counter(language_provenance.values())),
    }
    protocol = {
        "contract_version": "physics-french-accent-ablation-v1",
        "benchmark": {"queries": 220, "physics_queries": 80, "industrial_queries": 75, "ohr_queries": 65, "files": 103, "pages": 5091},
        "file_k": FILE_K,
        "candidate_depth": CANDIDATE_DEPTH,
        "saved_depth": SAVED_DEPTH,
        "qrels_used_for_ranking": False,
        "dense_source": str(NATIVE_OUTPUT / "native_french_query_embeddings.npy"),
        "bm25_variants": {"nfc": "existing Unicode NFC analyzer", "accent_folded": "NFD then remove combining marks"},
        "arms": list(arm_configs),
    }
    report = {
        "protocol": protocol,
        "arms": arm_reports,
        "physics_only": physics_reports,
        "accent_stats": stats,
        "mapping_rows": len(mapping_audit),
    }
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_jsonl(OUTPUT / "query_mapping_audit.jsonl", mapping_audit)
    report_md = build_report_markdown(arm_reports, physics_reports, stats)
    (OUTPUT / "report.md").write_text(report_md, encoding="utf-8")
    print(report_md, flush=True)
    print(f"Output: {OUTPUT.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
