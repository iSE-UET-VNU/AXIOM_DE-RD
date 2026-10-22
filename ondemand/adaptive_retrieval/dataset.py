"""Recreate the adaptive-router's frozen top-100 ranking and model rows."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from ondemand.bench import fingerprint, load_jsonl
from ondemand.light_retrieval import (
    DENSE_MODEL,
    Hierarchy,
    cache_misses,
    chunk_vectors,
    page_scores,
)
from ondemand.openrouter import embed

from .features import FEATURE_NAMES, extract_score_features


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _target_from_gold_ranks(gold_ranks: dict[str, int | None], cap: int) -> tuple[int, bool]:
    if not gold_ranks:
        raise ValueError("a query has no gold page labels")
    reachable = all(rank is not None and int(rank) <= cap for rank in gold_ranks.values())
    if not reachable:
        return cap, False
    return max(int(rank) for rank in gold_ranks.values() if rank is not None), True


def build_dataset(
    artifact_root: Path,
    benchmark_root: Path,
    output_dir: Path,
    cap: int = 100,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Recompute final scores with the router settings, using cache only.

    The frozen gate contributes its qrel-derived gold ranks only as labels and
    as an exact ranking-reproduction check. Gold data are never features.
    """
    artifact_root = Path(artifact_root)
    benchmark_root = Path(benchmark_root)
    output_dir = Path(output_dir)
    pages_path = artifact_root / "light_prep" / "pages_ppocrv5.jsonl"
    gate_path = artifact_root / "light_prep" / "gate_ppocrv5_ungated.json"
    cache_dir = artifact_root / "embedding_cache" / "te3s_ppocr"
    if not pages_path.is_file() or not gate_path.is_file() or not cache_dir.is_dir():
        raise FileNotFoundError("handoff pages, gate, or TE3-small embedding cache is missing")

    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    actual_fingerprint = fingerprint(benchmark_root)
    if actual_fingerprint != gate.get("bundle_fingerprint"):
        raise ValueError(
            f"benchmark fingerprint {actual_fingerprint} does not match gate artifact "
            f"{gate.get('bundle_fingerprint')}"
        )
    settings = {
        "dense_model": gate["dense_model"],
        "signals": gate["signals"],
        "analyzer": gate["analyzer"],
        "file_mode": gate["file_mode"],
        "file_k": int(gate["file_k"]),
        "w_dense": float(gate["w_dense"]),
        "parent_weight": float(gate["parent_weight"]),
        "cap": int(cap),
        "ranking_variant": gate["variant"],
    }
    if settings["dense_model"] != DENSE_MODEL:
        raise ValueError(f"gate uses {settings['dense_model']}, code expects {DENSE_MODEL}")
    if int(gate.get("n_corpus_pages", -1)) != 5091 or int(gate.get("n_queries", -1)) != 220:
        raise ValueError("the frozen adaptive-router artifact is not the expected 220-query / 5,091-page bundle")
    if int(gate.get("gate_k", -1)) != 20 or len(next(iter(gate["light_run_top100"].values()))) != cap:
        raise ValueError("the frozen gate does not contain the expected top-100 ranking")

    rows = load_jsonl(pages_path)
    queries = {row["query_id"]: row for row in load_jsonl(benchmark_root / "queries.jsonl")}
    qids = sorted(gate["light_run_top100"])
    if set(qids) != set(queries):
        raise ValueError(f"query ID mismatch: gate={len(qids)}, benchmark={len(queries)}")
    texts = [queries[qid]["query"] for qid in qids]

    # page_chunks() can split a page into more than one cached text; perform
    # the authoritative cache check inside chunk_vectors() before any embed.
    if cache_misses(texts, cache_dir):
        raise RuntimeError("some query embeddings are absent from the frozen cache; no API calls were made")

    hierarchy = Hierarchy(
        rows,
        file_k=settings["file_k"],
        page_k=cap,
        w_dense=settings["w_dense"],
        parent=settings["parent_weight"],
        signals=settings["signals"],
        analyzer=settings["analyzer"],
        file_mode=settings["file_mode"],
    )
    packed_vectors = output_dir / "router_embedding_vectors.npz"
    if packed_vectors.exists():
        packed = np.load(packed_vectors)
        owners = packed["owners"]
        page_vectors = packed["page_vectors"]
        query_vectors = packed["query_vectors"]
        if page_vectors.shape[0] != owners.shape[0] or query_vectors.shape != (len(qids), 1536):
            raise ValueError("packed embedding vectors do not match this benchmark; use a new output directory")
        print("[adaptive-retrieval] reusing packed local embedding vectors")
    else:
        owners, page_vectors = chunk_vectors(rows, cache_dir, allow_api=False)
        query_vectors = embed(texts, DENSE_MODEL, cache_dir)
        query_vectors /= np.linalg.norm(query_vectors, axis=1, keepdims=True) + 1e-9
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            packed_vectors,
            owners=owners,
            page_vectors=page_vectors,
            query_vectors=query_vectors,
        )

    all_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    fatal_mismatches: list[dict[str, Any]] = []
    gold_ranks_by_qid = gate["gold_rank"]
    for i, qid in enumerate(qids):
        dense = page_scores(owners, page_vectors, query_vectors[i], len(rows))
        recomputed_ranked, score_map, _ = hierarchy.rank(queries[qid]["query"], dense)
        recomputed_ranked = recomputed_ranked[:cap]
        expected = gate["light_run_top100"][qid][:cap]
        if len(recomputed_ranked) != cap or len(expected) != cap:
            raise ValueError(f"{qid} does not have {cap} candidate pages")
        expected_set = set(expected)
        score_curve = [float(score_map[page_id]) for page_id in expected if page_id in score_map]
        if len(score_curve) != cap:
            fatal_mismatches.append({"qid": qid, "reason": "frozen top-100 contains pages without recomputed scores"})
            continue
        if recomputed_ranked != expected:
            first = _matching_prefix(recomputed_ranked, expected)
            inversion = max(
                (score_curve[i + 1] - score_curve[i] for i in range(cap - 1)),
                default=0.0,
            )
            detail = {
                "qid": qid,
                "matching_prefix": first,
                "expected_at_first_difference": expected[first] if first < cap else None,
                "recomputed_at_first_difference": recomputed_ranked[first] if first < cap else None,
                "max_score_inversion_in_frozen_order": float(max(inversion, 0.0)),
                "same_top100_page_set": set(recomputed_ranked) == expected_set,
            }
            mismatches.append(detail)
            if set(recomputed_ranked) != expected_set or inversion > 1e-6:
                fatal_mismatches.append(detail)
        # Preserve the router's frozen ordering; in the one accepted numerical
        # tie case, it is the authoritative tie-break order for these scores.
        ranked = expected
        features = extract_score_features(score_curve)
        gold_ranks = gold_ranks_by_qid[qid]
        target_k, reachable = _target_from_gold_ranks(gold_ranks, cap)
        source = queries[qid].get("source") or qid.split("::", 1)[0]
        all_rows.append({
            "qid": qid,
            "source": source,
            "features": features,
            "target_k": target_k,
            "reachable_at_cap": reachable,
            "gold_ranks": gold_ranks,
            "ranked_pages": ranked,
            "gold_count": len(gold_ranks),
        })
        score_rows.append({"qid": qid, "ranked_pages": ranked, "scores": score_curve})
        if (i + 1) % 25 == 0 or i + 1 == len(qids):
            print(f"[adaptive-retrieval] rebuilt exact router scores {i + 1}/{len(qids)}")

    if fatal_mismatches:
        _write_json(output_dir / "ranking_reproduction.json", {
            "exact_matches": len(qids) - len(mismatches),
            "queries": len(qids),
            "mismatches": mismatches,
            "fatal_mismatches": fatal_mismatches,
            "settings": settings,
        })
        raise RuntimeError(f"router top-{cap} differs materially for {len(fatal_mismatches)} queries")

    feature_csv = output_dir / "feature_matrix.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    with feature_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["qid", "source", "target_k", "reachable_at_cap", "gold_count", *FEATURE_NAMES],
        )
        writer.writeheader()
        for row in all_rows:
            writer.writerow({
                "qid": row["qid"],
                "source": row["source"],
                "target_k": row["target_k"],
                "reachable_at_cap": row["reachable_at_cap"],
                "gold_count": row["gold_count"],
                **row["features"],
            })
    _write_jsonl(output_dir / "retrieval_scores.jsonl", score_rows)
    _write_jsonl(output_dir / "feature_labels.jsonl", [
        {key: row[key] for key in ("qid", "source", "target_k", "reachable_at_cap", "gold_count", "features")}
        for row in all_rows
    ])
    reproduction = {
        "exact_matches": len(qids) - len(mismatches),
        "queries": len(qids),
        "tie_only_mismatches": mismatches,
        "accepted": True,
        "acceptance_rule": "same top-100 page set and any frozen-order score inversion <= 1e-6",
        "corpus_pages": len(rows),
        "embedding_cache_files": sum(1 for _ in cache_dir.glob("*.json")),
        "settings": settings,
        "ranking_reproduced_under_tie_tolerance": True,
        "feature_count": len(FEATURE_NAMES),
    }
    _write_json(output_dir / "ranking_reproduction.json", reproduction)
    manifest = {
        "dataset": "merged 220-query benchmark",
        "benchmark_root": str(benchmark_root),
        "artifact_root": str(artifact_root),
        "gate_artifact": str(gate_path),
        "pages_artifact": str(pages_path),
        "features": list(FEATURE_NAMES),
        "label": "smallest rank containing every qrel gold page; set to cap if any gold is outside top-cap",
        "feature_target_separation": "gold ranks are labels/evaluation only and are never model inputs",
        "n_queries": len(all_rows),
        "n_corpus_pages": len(rows),
        "settings": settings,
        "ranking_reproduction": reproduction,
    }
    _write_json(output_dir / "dataset_manifest.json", manifest)
    return all_rows, manifest


def _matching_prefix(left: list[str], right: list[str]) -> int:
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count
