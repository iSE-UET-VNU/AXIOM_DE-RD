"""Evaluate native French query routing for French Physics files.

The experiment keeps the canonical 220-query OHR benchmark and the cached
merged page embeddings.  For the 80 Physics queries, French-language files
can use the cached native French query vector and French BM25 scores.  All
non-French files keep the current English-query hybrid route so the two
requested French-file modes are isolated.

No qrels are used to build scores, select languages, or rank pages.
"""

from __future__ import annotations

from collections import Counter
import hashlib
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


BUNDLE = ROOT / "data/work/light_debug_bundle_benchmark_default_bm25_tesseract_kf3_top20/bundle_debug_light-retrieval"
OHR_ROOT = ROOT / "data/work/ohr_replacement_20260920"
BASE = OHR_ROOT / "retrieval/combined_220_base"
PAGE_EMBEDDINGS = OHR_ROOT / "retrieval/embeddings/page_embeddings.npy"
OLD_DOCS = BUNDLE / "input/documents.jsonl"
OLD_PAGES = BUNDLE / "preparation/embedded_image_ocr/pages_with_embedded_ocr.jsonl"
OHR_PAGES = OHR_ROOT / "embedded_ocr_filtered/size_filter_only/pages.jsonl"
FRENCH_QUERIES = ROOT / "data/output/vsplade/vidore_v3_physics_french_302q/retrieval_french.jsonl"
ENGLISH_QUERIES = ROOT / "data/output/vsplade/vidore_v3_physics_english_302q/retrieval_english.jsonl"
FRENCH_EMBED_CACHE = ROOT / "data/work/vidore_physics_emb"
OUTPUT = OHR_ROOT / "retrieval/physics_native_french_language_route"

EMBEDDING_MODEL = "openai/text-embedding-3-small"
BM25_ALPHA = 0.70
FILE_K = 3
CANDIDATE_DEPTH = 200
SAVED_DEPTH = 20

ARM_CONFIGS: dict[str, dict[str, str]] = {
    "control_official_english_route": {
        "french_file_query": "english",
        "french_file_mode": "dense",
        "nonfrench_file_mode": "hybrid",
    },
    "native_fr_dense_fr_files": {
        "french_file_query": "french",
        "french_file_mode": "dense",
        "nonfrench_file_mode": "hybrid",
    },
    "native_fr_hybrid_fr_files": {
        "french_file_query": "french",
        "french_file_mode": "hybrid",
        "nonfrench_file_mode": "hybrid",
    },
    "control_english_dense_backbone": {
        "french_file_query": "english",
        "french_file_mode": "dense",
        "nonfrench_file_mode": "dense",
    },
    "native_fr_dense_pure_dense_backbone": {
        "french_file_query": "french",
        "french_file_mode": "dense",
        "nonfrench_file_mode": "dense",
    },
    "native_fr_hybrid_pure_dense_backbone": {
        "french_file_query": "french",
        "french_file_mode": "hybrid",
        "nonfrench_file_mode": "dense",
    },
}

PURE_DENSE_ARMS = (
    "control_english_dense_backbone",
    "native_fr_dense_pure_dense_backbone",
    "native_fr_hybrid_pure_dense_backbone",
)


def max_norm(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    maximum = float(values.max()) if values.size else 0.0
    return values / maximum if maximum > 0.0 else np.zeros_like(values)


def ndcg(ranked: Sequence[str], gold: set[str], depth: int) -> float:
    dcg = sum(
        1.0 / math.log2(rank + 2.0)
        for rank, page in enumerate(ranked[:depth])
        if page in gold
    )
    ideal = sum(1.0 / math.log2(rank + 2.0) for rank in range(min(len(gold), depth)))
    return dcg / ideal if ideal else 0.0


def source_name(qid: str) -> str:
    return qid.split("::", 1)[0]


def load_file_languages(file_ids: Sequence[str]) -> tuple[dict[str, str], dict[str, str]]:
    languages: dict[str, str] = {}
    provenance: dict[str, str] = {}
    canonical = set(file_ids)
    for row in read_jsonl(OLD_LANGUAGE):
        fid = str(row.get("file_id") or row.get("doc_id") or "")
        if fid in canonical:
            languages[fid] = str(row.get("detected_language") or "unknown").lower()
            provenance[fid] = "cached_file_language_detection"
    for fid in file_ids:
        if fid.startswith("ohrbench::"):
            languages[fid] = "en"
            provenance[fid] = "canonical_ohr_english_contract"
    missing = [fid for fid in file_ids if fid not in languages]
    if missing:
        raise RuntimeError(f"Missing language metadata for {len(missing)} files: {missing[:5]}")
    return languages, provenance


def load_page_texts(page_ids: Sequence[str]) -> dict[str, str]:
    documents = {
        str(row["doc_id"]): str(row.get("source") or "")
        for row in read_jsonl(OLD_DOCS)
    }
    old_pages = [
        row
        for row in read_jsonl(OLD_PAGES)
        if documents.get(str(row["doc_id"])) in {"vidore_physics", "vidore_industrial"}
    ]
    ohr_pages = read_jsonl(OHR_PAGES)
    merged = old_pages + ohr_pages
    merged_ids = [str(row["page_id"]) for row in merged]
    expected = [str(page) for page in page_ids]
    if merged_ids != expected:
        mismatch = next(
            (index for index, (actual, wanted) in enumerate(zip(merged_ids, expected)) if actual != wanted),
            min(len(merged_ids), len(expected)),
        )
        raise RuntimeError(f"Page corpus order mismatch at {mismatch}: {merged_ids[mismatch:mismatch + 1]} vs {expected[mismatch:mismatch + 1]}")
    return {str(row["page_id"]): str(row.get("text") or "") for row in merged}


def build_french_bm25(
    page_ids: Sequence[str],
    page_texts: Mapping[str, str],
    file_ids: Sequence[str],
    french_queries: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    from src.retrieval.sparse import BM25Index

    page_index = BM25Index().build(
        [
            {"chunk_id": page, "doc_id": file_id(page), "text": page_texts[page]}
            for page in page_ids
        ]
    )
    file_text: dict[str, list[str]] = {fid: [] for fid in file_ids}
    for page in page_ids:
        file_text[file_id(page)].append(page_texts[page])
    file_index = BM25Index().build(
        [
            {"chunk_id": fid, "doc_id": fid, "text": "\n".join(file_text[fid])}
            for fid in file_ids
        ]
    )
    page_scores = np.zeros((len(french_queries), len(page_ids)), dtype=np.float32)
    file_scores = np.zeros((len(french_queries), len(file_ids)), dtype=np.float32)
    for qi, query in enumerate(french_queries):
        text = str(query["query"])
        for position, score in page_index.search(text, len(page_ids)):
            page_scores[qi, int(position)] = float(score)
        for position, score in file_index.search(text, len(file_ids)):
            file_scores[qi, int(position)] = float(score)
    return page_scores, file_scores


def load_native_french_mapping(
    canonical_physics: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    french_rows = read_jsonl(FRENCH_QUERIES)
    english_rows = read_jsonl(ENGLISH_QUERIES)
    french_by_qid = {str(row["qid"]): row for row in french_rows}
    english_by_text: dict[str, list[dict[str, Any]]] = {}
    for row in english_rows:
        english_by_text.setdefault(str(row["query"]), []).append(row)

    mapping: dict[str, dict[str, Any]] = {}
    audit: list[dict[str, Any]] = []
    for canonical in canonical_physics:
        canonical_qid = str(canonical["query_id"])
        matches = english_by_text.get(str(canonical["query"]), [])
        if len(matches) != 1:
            raise RuntimeError(f"Expected one English artifact match for {canonical_qid}, got {len(matches)}")
        english = matches[0]
        english_number = int(str(english["qid"]).split("::", 1)[1])
        ordinal = english_number - 302
        french_qid = f"physics::{ordinal}"
        french = french_by_qid.get(french_qid)
        if french is None:
            raise RuntimeError(f"Missing native French query {french_qid} for {canonical_qid}")
        if int(str(canonical.get("source_query_id") or english_number)) != english_number:
            raise RuntimeError(
                f"Source query mismatch for {canonical_qid}: "
                f"canonical={canonical.get('source_query_id')} artifact={english_number}"
            )
        item = {
            "canonical_query_id": canonical_qid,
            "source_query_id": str(canonical.get("source_query_id") or ""),
            "english_artifact_qid": str(english["qid"]),
            "french_artifact_qid": french_qid,
            "english_query": str(english["query"]),
            "french_query": str(french["query"]),
            "french_ordinal": ordinal,
        }
        mapping[canonical_qid] = item
        audit.append(item)
    if len(mapping) != 80:
        raise RuntimeError(f"Expected 80 Physics mappings, got {len(mapping)}")
    return mapping, audit


def cached_vector(text: str) -> tuple[np.ndarray, str]:
    key = hashlib.sha1(f"{EMBEDDING_MODEL}|{text}".encode("utf-8", "ignore")).hexdigest()
    path = FRENCH_EMBED_CACHE / f"emb_{key}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing cached French query embedding: {path}")
    vector = np.asarray(json.loads(path.read_text(encoding="utf-8")), dtype=np.float32)
    if vector.shape != (1536,):
        raise RuntimeError(f"Unexpected vector shape for {path}: {vector.shape}")
    return vector, str(path)


def build_rows(
    *,
    arm_name: str,
    config: Mapping[str, str],
    queries: Sequence[Mapping[str, Any]],
    page_ids: Sequence[str],
    file_ids: Sequence[str],
    file_languages: Mapping[str, str],
    query_page_dense_en: np.ndarray,
    page_bm25_en: np.ndarray,
    file_bm25_en: np.ndarray,
    query_page_dense_fr: Mapping[str, np.ndarray],
    page_bm25_fr: Mapping[str, np.ndarray],
    file_bm25_fr: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    file_position = {fid: index for index, fid in enumerate(file_ids)}
    page_file_pos = np.asarray([file_position[file_id(page)] for page in page_ids], dtype=np.int32)
    physics_ids = set(query_page_dense_fr)
    rows: list[dict[str, Any]] = []

    for qi, query in enumerate(queries):
        qid = str(query["query_id"])
        is_physics = qid.startswith("vidore_physics::")
        use_native = is_physics and config["french_file_query"] == "french"

        dense_en = max_norm(query_page_dense_en[qi])
        bm25_en = max_norm(page_bm25_en[qi])
        direct_en = max_norm(file_bm25_en[qi])
        dense_fr = max_norm(query_page_dense_fr[qid]) if use_native else dense_en
        bm25_fr = max_norm(page_bm25_fr[qid]) if use_native else bm25_en
        direct_fr = max_norm(file_bm25_fr[qid]) if use_native else direct_en

        page_score = np.zeros(len(page_ids), dtype=np.float64)
        direct_score = np.zeros(len(file_ids), dtype=np.float64)
        selected_modes: dict[str, str] = {}
        selected_query_sources: dict[str, str] = {}
        for fi, fid in enumerate(file_ids):
            is_french_file = file_languages[fid] == "fr"
            if is_french_file:
                mode = config["french_file_mode"]
                source = "native_french" if use_native else "canonical_english"
                dense = dense_fr if use_native else dense_en
                bm25 = bm25_fr if use_native else bm25_en
                direct_bm25 = direct_fr if use_native else direct_en
            else:
                mode = config["nonfrench_file_mode"]
                source = "canonical_english"
                dense = dense_en
                bm25 = bm25_en
                direct_bm25 = direct_en
            mask = page_file_pos == fi
            dense_file_max = float(dense[mask].max()) if np.any(mask) else 0.0
            if mode == "dense":
                page_score[mask] = dense[mask]
                direct_score[fi] = dense_file_max
            elif mode == "hybrid":
                page_score[mask] = BM25_ALPHA * bm25[mask] + (1.0 - BM25_ALPHA) * dense[mask]
                direct_score[fi] = BM25_ALPHA * float(direct_bm25[fi]) + (1.0 - BM25_ALPHA) * dense_file_max
            else:
                raise ValueError(f"Unknown mode {mode!r}")
            selected_modes[fid] = mode
            selected_query_sources[fid] = source

        page_max = np.zeros(len(file_ids), dtype=np.float64)
        np.maximum.at(page_max, page_file_pos, page_score)
        file_score = 0.50 * direct_score + 0.50 * max_norm(page_max)
        selected_positions = np.argsort(-file_score, kind="stable")[:FILE_K]
        selected_set = {int(position) for position in selected_positions}
        final_score = 0.85 * page_score + 0.15 * max_norm(file_score)[page_file_pos]
        allowed = np.flatnonzero(np.isin(page_file_pos, list(selected_set)))
        order = allowed[np.argsort(-final_score[allowed], kind="stable")]
        selected = [file_ids[int(position)] for position in selected_positions]
        candidate = [page_ids[int(position)] for position in order[:CANDIDATE_DEPTH]]
        returned = candidate[:SAVED_DEPTH]
        rows.append(
            {
                "query_id": qid,
                "query": str(query["query"]),
                "query_language": "en",
                "arm": arm_name,
                "french_file_query": config["french_file_query"],
                "french_file_mode": config["french_file_mode"],
                "nonfrench_file_mode": config["nonfrench_file_mode"],
                "selected_files": selected,
                "candidate_pages": candidate,
                "returned_pages": returned,
                "returned_scores": [float(final_score[int(position)]) for position in order[:SAVED_DEPTH]],
                "selected_file_languages": dict(Counter(file_languages[fid] for fid in selected)),
                "selected_file_modes": dict(Counter(selected_modes[fid] for fid in selected)),
                "selected_file_query_sources": dict(Counter(selected_query_sources[fid] for fid in selected)),
            }
        )
    if len(rows) != len(queries) or len({row["query_id"] for row in rows}) != len(queries):
        raise RuntimeError(f"Invalid run shape for {arm_name}")
    return rows


def source_metrics(rows: Sequence[Mapping[str, Any]], qrels: Mapping[str, set[str]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for source in ("vidore_physics", "vidore_industrial", "ohrbench"):
        subset = [row for row in rows if source_name(str(row["query_id"])) == source]
        subset_qrels = {str(row["query_id"]): qrels[str(row["query_id"])] for row in subset}
        result[source] = {
            "queries": len(subset),
            "@10": metric_rows(subset, subset_qrels, 10),
            "@20": metric_rows(subset, subset_qrels, 20),
        }
    return result


def pairwise_diagnostics(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    qrels: Mapping[str, set[str]],
) -> dict[str, Any]:
    left_by_qid = {str(row["query_id"]): row for row in left}
    right_by_qid = {str(row["query_id"]): row for row in right}
    overlap: list[float] = []
    left_hits: list[float] = []
    right_hits: list[float] = []
    for qid, gold in qrels.items():
        left_top = set(left_by_qid[qid]["returned_pages"][:10])
        right_top = set(right_by_qid[qid]["returned_pages"][:10])
        overlap.append(len(left_top & right_top) / 10.0)
        left_hits.append(float(bool(left_top & gold)))
        right_hits.append(float(bool(right_top & gold)))
    return {
        "mean_top10_overlap": float(np.mean(overlap)),
        "left_page_hit@10": float(np.mean(left_hits)),
        "right_page_hit@10": float(np.mean(right_hits)),
        "right_only_hit_queries": int(sum(right_hit > left_hit for left_hit, right_hit in zip(left_hits, right_hits))),
        "left_only_hit_queries": int(sum(left_hit > right_hit for left_hit, right_hit in zip(left_hits, right_hits))),
        "both_hit_queries": int(sum(left_hit == 1.0 and right_hit == 1.0 for left_hit, right_hit in zip(left_hits, right_hits))),
        "both_missed_queries": int(sum(left_hit == 0.0 and right_hit == 0.0 for left_hit, right_hit in zip(left_hits, right_hits))),
    }


def main() -> int:
    started = time.perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((BASE / "metadata.json").read_text(encoding="utf-8"))
    page_ids = [str(value) for value in metadata["page_ids"]]
    file_ids = [str(value) for value in metadata["file_ids"]]
    query_ids = [str(value) for value in metadata["query_ids"]]
    queries, qrels, _ = load_inputs()
    if len(queries) != 220 or len(page_ids) != 5091 or len(file_ids) != 103:
        raise RuntimeError(f"Canonical shape mismatch: {len(queries)} queries, {len(file_ids)} files, {len(page_ids)} pages")
    if query_ids != [str(row["query_id"]) for row in queries]:
        raise RuntimeError("Canonical query order does not match the merged cache")

    page_bm25_en = np.load(BASE / "page_bm25_raw.npy", mmap_mode="r")
    file_bm25_en = np.load(BASE / "file_bm25_raw.npy", mmap_mode="r")
    query_page_dense_en = np.load(BASE / "query_page_dense.npy", mmap_mode="r")
    page_vectors = np.load(PAGE_EMBEDDINGS, mmap_mode="r")
    if page_bm25_en.shape != (220, 5091) or query_page_dense_en.shape != (220, 5091) or file_bm25_en.shape != (220, 103):
        raise RuntimeError(f"Unexpected canonical score shapes: {page_bm25_en.shape}, {query_page_dense_en.shape}, {file_bm25_en.shape}")
    if page_vectors.shape != (5091, 1536):
        raise RuntimeError(f"Unexpected page embedding shape: {page_vectors.shape}")

    file_languages, language_provenance = load_file_languages(file_ids)
    page_texts = load_page_texts(page_ids)
    canonical_physics = [row for row in queries if str(row["query_id"]).startswith("vidore_physics::")]
    mapping, mapping_audit = load_native_french_mapping(canonical_physics)
    french_vectors: list[np.ndarray] = []
    vector_paths: dict[str, str] = {}
    for row in canonical_physics:
        item = mapping[str(row["query_id"])]
        vector, path = cached_vector(item["french_query"])
        french_vectors.append(vector)
        vector_paths[str(row["query_id"])] = path
    french_matrix = np.asarray(french_vectors, dtype=np.float32)
    if french_matrix.shape != (80, 1536):
        raise RuntimeError(f"Unexpected native French query matrix shape: {french_matrix.shape}")
    french_dense_matrix = np.asarray(french_matrix @ page_vectors.T, dtype=np.float32)
    french_query_index = {str(row["query_id"]): index for index, row in enumerate(canonical_physics)}
    query_page_dense_fr = {
        qid: french_dense_matrix[index]
        for qid, index in french_query_index.items()
    }
    page_bm25_fr_matrix, file_bm25_fr_matrix = build_french_bm25(
        page_ids, page_texts, file_ids, [
            {"query": mapping[str(row["query_id"])] ["french_query"]}
            for row in canonical_physics
        ],
    )
    page_bm25_fr = {
        qid: page_bm25_fr_matrix[index]
        for qid, index in french_query_index.items()
    }
    file_bm25_fr = {
        qid: file_bm25_fr_matrix[index]
        for qid, index in french_query_index.items()
    }

    np.save(OUTPUT / "native_french_query_embeddings.npy", french_matrix)
    np.save(OUTPUT / "native_french_page_dense_raw.npy", french_dense_matrix)
    np.save(OUTPUT / "native_french_page_bm25_raw.npy", page_bm25_fr_matrix)
    np.save(OUTPUT / "native_french_file_bm25_raw.npy", file_bm25_fr_matrix)
    write_jsonl(OUTPUT / "query_mapping_audit.jsonl", mapping_audit)

    runs: dict[str, list[dict[str, Any]]] = {}
    arm_reports: dict[str, Any] = {}
    for arm_name, config in ARM_CONFIGS.items():
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
            "by_source": source_metrics(rows, qrels),
            "physics_only": {
                "queries": len(canonical_physics),
                "@10": metric_rows(
                    [row for row in rows if source_name(str(row["query_id"])) == "vidore_physics"],
                    {str(row["query_id"]): qrels[str(row["query_id"])] for row in rows if source_name(str(row["query_id"])) == "vidore_physics"},
                    10,
                ),
                "@20": metric_rows(
                    [row for row in rows if source_name(str(row["query_id"])) == "vidore_physics"],
                    {str(row["query_id"]): qrels[str(row["query_id"])] for row in rows if source_name(str(row["query_id"])) == "vidore_physics"},
                    20,
                ),
            },
        }

    physics_qrels = {
        qid: gold
        for qid, gold in qrels.items()
        if source_name(qid) == "vidore_physics"
    }
    comparison = pairwise_diagnostics(
        [row for row in runs["native_fr_dense_fr_files"] if source_name(str(row["query_id"])) == "vidore_physics"],
        [row for row in runs["native_fr_hybrid_fr_files"] if source_name(str(row["query_id"])) == "vidore_physics"],
        physics_qrels,
    )
    pure_dense_comparison = pairwise_diagnostics(
        [row for row in runs["native_fr_dense_pure_dense_backbone"] if source_name(str(row["query_id"])) == "vidore_physics"],
        [row for row in runs["native_fr_hybrid_pure_dense_backbone"] if source_name(str(row["query_id"])) == "vidore_physics"],
        physics_qrels,
    )
    protocol = {
        "contract_version": "physics-native-french-language-route-v2",
        "benchmark": {
            "queries": 220,
            "physics_queries": 80,
            "industrial_queries": 75,
            "ohr_queries": 65,
            "files": 103,
            "pages": 5091,
            "composition": "80 Physics + 75 Industrial + 65 OHR-Bench",
        },
        "file_language_counts": dict(Counter(file_languages.values())),
        "file_language_provenance_counts": dict(Counter(language_provenance.values())),
        "french_physics_file_count": sum(
            1 for fid in file_ids if fid.startswith("vidore_physics::") and file_languages[fid] == "fr"
        ),
        "file_k": FILE_K,
        "candidate_depth": CANDIDATE_DEPTH,
        "saved_depth": SAVED_DEPTH,
        "hybrid_formula": f"{BM25_ALPHA:.2f} * normalized BM25 + {1.0 - BM25_ALPHA:.2f} * normalized dense",
        "file_score_formula": "0.50 * direct file score + 0.50 * max page score",
        "page_final_formula": "0.85 * page score + 0.15 * normalized file score",
        "french_bm25_text_source": {
            "physics_industrial": str(OLD_PAGES),
            "ohr": str(OHR_PAGES),
        },
        "page_embedding_model": EMBEDDING_MODEL,
        "native_french_embedding_cache": str(FRENCH_EMBED_CACHE),
        "native_french_embedding_cache_hits": len(vector_paths),
        "qrels_used_for_ranking": False,
        "routing_note": "The first three arms preserve the canonical English hybrid route for non-French files. The pure-dense-backbone arms use English dense-only everywhere except the requested French-file mode, isolating query-language and French-BM25 effects.",
        "arms": ARM_CONFIGS,
        "pure_dense_backbone_arms": list(PURE_DENSE_ARMS),
    }
    report = {
        "protocol": protocol,
        "arms": arm_reports,
        "dense_vs_hybrid_native_french_diagnostics": comparison,
        "pure_dense_backbone_native_french_diagnostics": pure_dense_comparison,
        "timing_seconds": {"total": time.perf_counter() - started},
    }
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Physics native French language route",
        "",
        "Canonical benchmark: 80 Physics + 75 Industrial + 65 OHR-Bench; 220 queries, 103 files, 5,091 pages.",
        "",
        "The first three arms preserve the canonical English hybrid route for non-French files. The pure-dense-backbone arms use English dense-only everywhere, then change only the Physics French-file query/mode.",
        "",
        "| Arm | Recall@10 | nDCG@10 | Recall@20 | nDCG@20 | File recall@3 | Candidate recall@200 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm_name in ARM_CONFIGS:
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
            "## Physics-only result",
            "",
            "| Arm | Queries | Recall@10 | nDCG@10 | Recall@20 | nDCG@20 | File recall@3 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for arm_name in ARM_CONFIGS:
        values = arm_reports[arm_name]["physics_only"]
        lines.append(
            f"| {arm_name} | {values['queries']} | {values['@10']['page_recall@10']:.2%} | "
            f"{values['@10']['ndcg@10']:.2f} | {values['@20']['page_recall@20']:.2%} | "
            f"{values['@20']['ndcg@20']:.2f} | {values['@10']['file_recall@3']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Native French dense versus hybrid",
            "",
            f"- Mean top-10 overlap: {comparison['mean_top10_overlap']:.2%}.",
            f"- Dense-only hit@10: {comparison['left_page_hit@10']:.2%}; hybrid hit@10: {comparison['right_page_hit@10']:.2%}.",
            f"- Hybrid-only hit queries: {comparison['right_only_hit_queries']}; dense-only hit queries: {comparison['left_only_hit_queries']}.",
            f"- Both hit: {comparison['both_hit_queries']}; both missed: {comparison['both_missed_queries']}.",
            "",
            "## Pure dense backbone isolation",
            "",
            "| Arm | Recall@10 | nDCG@10 | Recall@20 | nDCG@20 | File recall@3 | Candidate recall@200 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for arm_name in PURE_DENSE_ARMS:
        values10 = arm_reports[arm_name]["physics_only"]["@10"]
        values20 = arm_reports[arm_name]["physics_only"]["@20"]
        lines.append(
            f"| {arm_name} | {values10['page_recall@10']:.2%} | {values10['ndcg@10']:.2f} | "
            f"{values20['page_recall@20']:.2%} | {values20['ndcg@20']:.2f} | "
            f"{values10['file_recall@3']:.2%} | {values10['candidate_pool_recall@200']:.2%} |"
        )
    lines.extend(
        [
            "",
            "Pure dense-backbone French dense versus hybrid:",
            "",
            f"- Mean top-10 overlap: {pure_dense_comparison['mean_top10_overlap']:.2%}.",
            f"- French dense-only hit@10: {pure_dense_comparison['left_page_hit@10']:.2%}; French hybrid hit@10: {pure_dense_comparison['right_page_hit@10']:.2%}.",
            f"- Hybrid-only hit queries: {pure_dense_comparison['right_only_hit_queries']}; dense-only hit queries: {pure_dense_comparison['left_only_hit_queries']}.",
            f"- Both hit: {pure_dense_comparison['both_hit_queries']}; both missed: {pure_dense_comparison['both_missed_queries']}.",
            "",
            "Qrels are used only after ranking for evaluation. Native French vectors were loaded from the existing text-embedding-3-small cache; no embedding API call was made.",
        ]
    )
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"Output: {OUTPUT.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
