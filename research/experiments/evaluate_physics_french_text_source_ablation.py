"""Compare French Physics BM25 over baseline text and embedded-image OCR text."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.experiments.evaluate_physics_french_accent_ablation import (  # noqa: E402
    build_french_bm25_folded,
    fold_accents,
)
from research.experiments.evaluate_physics_native_french_language_route import (  # noqa: E402
    BASE,
    BUNDLE,
    CANDIDATE_DEPTH,
    FILE_K,
    OLD_PAGES,
    OHR_PAGES,
    OUTPUT as NATIVE_OUTPUT,
    PAGE_EMBEDDINGS,
    SAVED_DEPTH,
    build_french_bm25,
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


BASELINE_PAGES = BUNDLE / "preparation/baseline_pdf_inspector_tesseract/pages.jsonl"
OUTPUT = NATIVE_OUTPUT.parent / "physics_french_text_source_ablation"


def text_map(path: Path) -> dict[str, str]:
    return {str(row["page_id"]): str(row.get("text") or "") for row in read_jsonl(path)}


def build_variant_scores(
    page_ids: Sequence[str],
    file_ids: Sequence[str],
    page_texts: Mapping[str, str],
    french_queries: Sequence[Mapping[str, Any]],
    folded: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if folded:
        return build_french_bm25_folded(page_ids, page_texts, file_ids, french_queries)
    return build_french_bm25(page_ids, page_texts, file_ids, french_queries)


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
    embedded_texts = load_page_texts(page_ids)
    baseline_old = text_map(BASELINE_PAGES)
    baseline_texts = {}
    for page in page_ids:
        if page.startswith("ohrbench::"):
            baseline_texts[page] = embedded_texts[page]
        elif page in baseline_old:
            baseline_texts[page] = baseline_old[page]
        else:
            raise RuntimeError(f"Missing baseline page text: {page}")

    canonical_physics = [row for row in queries if str(row["query_id"]).startswith("vidore_physics::")]
    mapping, mapping_audit = load_native_french_mapping(canonical_physics)
    french_queries = [{"query": mapping[str(row["query_id"])] ["french_query"]} for row in canonical_physics]
    french_vectors = [cached_vector(row["query"])[0] for row in french_queries]
    french_matrix = np.asarray(french_vectors, dtype=np.float32)
    french_dense_matrix = np.asarray(french_matrix @ page_vectors.T, dtype=np.float32)
    physics_index = {str(row["query_id"]): index for index, row in enumerate(canonical_physics)}
    query_page_dense_fr = {qid: french_dense_matrix[index] for qid, index in physics_index.items()}

    variants = {
        "nfc_baseline_pdf_inspector_tesseract": (baseline_texts, False),
        "nfc_embedded_ocr_eng": (embedded_texts, False),
        "accent_folded_baseline_pdf_inspector_tesseract": (baseline_texts, True),
        "accent_folded_embedded_ocr_eng": (embedded_texts, True),
    }
    runs: dict[str, list[dict[str, Any]]] = {}
    reports: dict[str, Any] = {}
    physics_reports: dict[str, Any] = {}
    for name, (texts, folded) in variants.items():
        page_raw, file_raw = build_variant_scores(
            page_ids, file_ids, texts, french_queries, folded
        )
        page_scores = {qid: page_raw[index] for qid, index in physics_index.items()}
        file_scores = {qid: file_raw[index] for qid, index in physics_index.items()}
        rows = build_rows(
            arm_name=name,
            config={
                "french_file_query": "french",
                "french_file_mode": "hybrid",
                "nonfrench_file_mode": "hybrid",
            },
            queries=queries,
            page_ids=page_ids,
            file_ids=file_ids,
            file_languages=file_languages,
            query_page_dense_en=query_page_dense_en,
            page_bm25_en=page_bm25_en,
            file_bm25_en=file_bm25_en,
            query_page_dense_fr=query_page_dense_fr,
            page_bm25_fr=page_scores,
            file_bm25_fr=file_scores,
        )
        runs[name] = rows
        write_jsonl(OUTPUT / "runs" / f"{name}.jsonl", rows)
        reports[name] = {"@10": metric_rows(rows, qrels, 10), "@20": metric_rows(rows, qrels, 20)}
        physics_rows = [row for row in rows if source_name(str(row["query_id"])) == "vidore_physics"]
        physics_qrels = {str(row["query_id"]): qrels[str(row["query_id"])] for row in physics_rows}
        physics_reports[name] = {
            "@10": metric_rows(physics_rows, physics_qrels, 10),
            "@20": metric_rows(physics_rows, physics_qrels, 20),
        }

    def text_stats(texts: Mapping[str, str]) -> dict[str, Any]:
        selected = [texts[page] for page in page_ids if page.startswith("vidore_physics::")]
        return {
            "pages": len(selected),
            "empty": sum(not text.strip() for text in selected),
            "mean_characters": sum(len(text) for text in selected) / max(len(selected), 1),
            "accented_characters": sum(sum(ord(char) > 127 for char in text) for text in selected),
            "fold_changed_pages": sum(fold_accents(text) != text for text in selected),
        }

    protocol = {
        "contract_version": "physics-french-text-source-ablation-v1",
        "benchmark": {"queries": 220, "physics_queries": 80, "industrial_queries": 75, "ohr_queries": 65, "files": 103, "pages": 5091},
        "file_k": FILE_K,
        "candidate_depth": CANDIDATE_DEPTH,
        "saved_depth": SAVED_DEPTH,
        "qrels_used_for_ranking": False,
        "french_dense_source": str(NATIVE_OUTPUT / "native_french_query_embeddings.npy"),
        "embedded_ocr_config": str(BUNDLE / "preparation/embedded_image_ocr/config.json"),
        "baseline_text_config": str(BUNDLE / "preparation/baseline_pdf_inspector_tesseract/summary.json"),
        "variants": list(variants),
        "language_provenance_counts": dict(Counter(language_provenance.values())),
    }
    report = {
        "protocol": protocol,
        "arms": reports,
        "physics_only": physics_reports,
        "text_stats": {name: text_stats(texts) for name, (texts, _) in variants.items()},
        "mapping_rows": len(mapping_audit),
    }
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_jsonl(OUTPUT / "query_mapping_audit.jsonl", mapping_audit)

    lines = [
        "# Physics French text-source ablation",
        "",
        "Canonical benchmark: 80 Physics + 75 Industrial + 65 OHR-Bench; 220 queries, 103 files, 5,091 pages.",
        "",
        "Only the French Physics BM25 text source and accent normalization vary. Native French dense vectors, English routing for non-French files, and the hierarchical protocol are fixed.",
        "",
        "| Arm | Recall@10 | nDCG@10 | Recall@20 | nDCG@20 | File recall@3 | Candidate recall@200 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in reports.items():
        m10, m20 = values["@10"], values["@20"]
        lines.append(
            f"| {name} | {m10['page_recall@10']:.2%} | {m10['ndcg@10']:.2f} | {m20['page_recall@20']:.2%} | {m20['ndcg@20']:.2f} | {m10['file_recall@3']:.2%} | {m10['candidate_pool_recall@200']:.2%} |"
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
    for name, values in physics_reports.items():
        m10, m20 = values["@10"], values["@20"]
        lines.append(
            f"| {name} | {m10['page_recall@10']:.2%} | {m10['ndcg@10']:.2f} | {m20['page_recall@20']:.2%} | {m20['ndcg@20']:.2f} | {m10['file_recall@3']:.2%} | {m10['candidate_pool_recall@200']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## French Physics text statistics",
            "",
            "| Variant | Mean chars/page | Empty pages | Pages changed by folding |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, stats in report["text_stats"].items():
        lines.append(
            f"| {name} | {stats['mean_characters']:.1f} | {stats['empty']} | {stats['fold_changed_pages']} |"
        )
    lines.extend(
        [
            "",
            "Qrels are used only after ranking. No embedding API call, rendering, or parsing was performed.",
        ]
    )
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"Output: {OUTPUT.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
