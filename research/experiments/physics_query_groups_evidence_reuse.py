"""Screen query grouping and shared-evidence baselines on Physics.

This is a cheap, retrieval-only experiment for the light-preparation question:
can related queries be batched so that one page/evidence preparation is reused?

The grouping decision never reads qrels.  It uses either query TF-IDF cosine,
overlap of the cached weighted-retrieval top pages, or their mean.  Qrels are
used only after groups and shared bundles have been constructed, for evaluation
and for post-hoc diagnostics of whether grouped queries really share gold
pages/files.

The important distinction in the output is:

* ``independent``: each query keeps its own cached candidate list;
* ``shared_union``: all group members contribute candidates.  This is a
  reference showing that global page deduplication already removes much of the
  obvious preparation duplication;
* ``shared_representative``: one medoid query supplies the group bundle;
* ``shared_centroid``: an offline prototype that builds one bundle from the
  average of member score profiles.  It is useful for measuring headroom, but
  it is not yet a deployable single-query retriever.

For shared methods, each query is reranked inside the shared page bundle by
its cached query-specific scores.  This models cheap query-specific ranking
after the expensive page preparation has been shared.

No parser, OCR, VLM, LLM, network call or new embedding is used.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import itertools
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.chunking_embedding.lexical import analyze  # noqa: E402
from src.evaluation.benchmarks.vidore_v3 import ViDoreV3  # noqa: E402
from src.retrieval.sparse import BM25Index  # noqa: E402


DEFAULT_BENCHMARK_ROOT = ROOT / "data/benchmark/vidore_v3"
DEFAULT_RUN = (
    ROOT
    / "data/benchmark/vidore_v3/results/physics_vsplade_bm25_fusion/"
    / "weighted_french_bm25-french_vs-english.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "data/benchmark/vidore_v3/results/physics_query_groups_evidence_reuse"
)

# These are query-form words, not corpus stopwords.  Keeping scientific terms,
# numbers and units is intentional because they are often the useful anchors.
FRENCH_QUERY_STOPWORDS = frozenset(
    "a à au aux avec ce ceci cela cette ces dans de des du elle en et est étaient "
    "être il ils la le les leur leurs lui mais me même ne nos notre nous on par pas "
    "pour que quel quelle quelles quels qui se ses son sont sur ta te tes ton tu un "
    "une vos votre vous y d l qu c s".split()
)


@dataclass(frozen=True)
class QueryRecord:
    qid: str
    query: str
    index: int


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_run(path: Path) -> dict[str, list[dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Retrieval run not found: {path}")
    output: dict[str, list[dict[str, Any]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        qid = str(row["qid"])
        chunks = []
        seen: set[str] = set()
        for chunk in row.get("chunks", []):
            page = str(chunk["chunk_id"])
            if page in seen:
                continue
            seen.add(page)
            chunks.append(
                {
                    "page_id": page,
                    "score": float(chunk.get("score", 0.0)),
                    "rank": len(chunks) + 1,
                    "text": str(chunk.get("text") or ""),
                }
            )
        output[qid] = chunks
    return output


def _ordered_questions(benchmark: ViDoreV3) -> list[QueryRecord]:
    questions = sorted(
        benchmark.questions(), key=lambda question: int(question.qid.rsplit("::", 1)[1])
    )
    return [QueryRecord(qid=question.qid, query=question.query, index=i) for i, question in enumerate(questions)]


def _validate_inputs(
    questions: Sequence[QueryRecord],
    run: Mapping[str, Sequence[Mapping[str, Any]]],
    qrels: Mapping[str, Mapping[str, int]],
    required_depth: int,
) -> None:
    expected = {question.qid for question in questions}
    actual = set(run)
    if actual != expected:
        raise RuntimeError(
            f"Run qids mismatch: missing={sorted(expected - actual)[:5]}, "
            f"extra={sorted(actual - expected)[:5]}"
        )
    short = [qid for qid in sorted(expected) if len(run[qid]) < required_depth]
    if short:
        raise RuntimeError(
            f"Run has fewer than {required_depth} candidates for {len(short)} qids; "
            f"examples={short[:5]}"
        )
    missing_qrels = [qid for qid in expected if not qrels.get(qid)]
    if missing_qrels:
        raise RuntimeError(f"Missing qrels for {len(missing_qrels)} queries")


def _query_tokens(query: str) -> list[str]:
    tokens = [token for token in analyze(query) if token not in FRENCH_QUERY_STOPWORDS]
    content = [token for token in tokens if len(token) > 2 or any(char.isdigit() for char in token)]
    return content or tokens


def _tfidf_vectors(questions: Sequence[QueryRecord]) -> list[dict[str, float]]:
    token_counts = [Counter(_query_tokens(question.query)) for question in questions]
    document_frequency: Counter[str] = Counter(
        token for counts in token_counts for token in counts
    )
    total = len(questions)
    vectors: list[dict[str, float]] = []
    for counts in token_counts:
        weights = {
            token: (1.0 + math.log(float(count)))
            * (math.log((total + 1.0) / (document_frequency[token] + 1.0)) + 1.0)
            for token, count in counts.items()
        }
        norm = math.sqrt(sum(value * value for value in weights.values())) or 1.0
        vectors.append({token: value / norm for token, value in weights.items()})
    return vectors


def _cosine_matrix(vectors: Sequence[Mapping[str, float]]) -> list[list[float]]:
    size = len(vectors)
    matrix = [[0.0] * size for _ in range(size)]
    for left in range(size):
        left_vector = vectors[left]
        for right in range(left):
            if len(left_vector) < len(vectors[right]):
                smaller, larger = left_vector, vectors[right]
            else:
                smaller, larger = vectors[right], left_vector
            score = sum(value * larger.get(token, 0.0) for token, value in smaller.items())
            matrix[left][right] = score
            matrix[right][left] = score
    return matrix


def _run_score_maps(
    questions: Sequence[QueryRecord],
    run: Mapping[str, Sequence[Mapping[str, Any]]],
    depth: int,
) -> tuple[dict[str, dict[str, float]], dict[str, list[str]]]:
    scores: dict[str, dict[str, float]] = {}
    pages: dict[str, list[str]] = {}
    for question in questions:
        hits = list(run[question.qid][:depth])
        pages[question.qid] = [str(hit["page_id"]) for hit in hits]
        scores[question.qid] = {
            str(hit["page_id"]): float(hit.get("score", 0.0)) for hit in hits
        }
    return scores, pages


def _build_page_bm25(
    run: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[BM25Index, dict[str, int]]:
    page_texts: dict[str, str] = {}
    for hits in run.values():
        for hit in hits:
            page_id = str(hit["page_id"])
            if page_id not in page_texts:
                page_texts[page_id] = str(hit.get("text") or "")
    records = [
        {"chunk_id": page_id, "doc_id": page_id, "text": text}
        for page_id, text in sorted(page_texts.items())
    ]
    index = BM25Index(analyzer_name="plain").build(records)
    return index, {page_id: position for position, page_id in enumerate(index.chunk_ids)}


def _overlap_matrix(
    questions: Sequence[QueryRecord],
    run: Mapping[str, Sequence[Mapping[str, Any]]],
    depth: int,
) -> list[list[float]]:
    signatures = [
        {str(hit["page_id"]) for hit in run[question.qid][:depth]}
        for question in questions
    ]
    size = len(signatures)
    matrix = [[0.0] * size for _ in range(size)]
    denominator = float(depth or 1)
    for left in range(size):
        for right in range(left):
            score = len(signatures[left] & signatures[right]) / denominator
            matrix[left][right] = score
            matrix[right][left] = score
    return matrix


def _mean_matrices(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> list[list[float]]:
    return [
        [(left[row][column] + right[row][column]) / 2.0 for column in range(len(left))]
        for row in range(len(left))
    ]


def _group_queries(
    questions: Sequence[QueryRecord],
    similarity: Sequence[Sequence[float]],
    *,
    max_group_size: int,
    min_similarity: float,
) -> list[dict[str, Any]]:
    """Partition queries around deterministic similarity medoids.

    A threshold prevents forced grouping of unrelated queries; the size cap
    keeps one broad topic from becoming a single expensive mega-batch.
    """
    if max_group_size < 1:
        raise ValueError("max_group_size must be positive")
    remaining = set(range(len(questions)))
    groups: list[dict[str, Any]] = []
    while remaining:
        seed = max(
            remaining,
            key=lambda index: (
                sum(similarity[index][other] for other in remaining if other != index)
                / max(1, len(remaining) - 1),
                -index,
            ),
        )
        neighbors = sorted(
            (
                other
                for other in remaining
                if other != seed and similarity[seed][other] >= min_similarity
            ),
            key=lambda other: (-similarity[seed][other], other),
        )
        members = [seed, *neighbors[: max_group_size - 1]]
        remaining.difference_update(members)
        groups.append(
            {
                "representative_index": seed,
                "member_indices": sorted(members),
            }
        )

    groups.sort(key=lambda group: group["representative_index"])
    output: list[dict[str, Any]] = []
    for group_index, group in enumerate(groups):
        members = group["member_indices"]
        pair_values = [
            similarity[left][right]
            for left, right in itertools.combinations(members, 2)
        ]
        output.append(
            {
                "group_id": f"g{group_index:03d}",
                "representative_index": group["representative_index"],
                "member_indices": members,
                "size": len(members),
                "mean_pair_similarity": (
                    statistics.mean(pair_values) if pair_values else 0.0
                ),
                "min_pair_similarity": min(pair_values) if pair_values else 0.0,
                "mean_seed_similarity": statistics.mean(
                    [similarity[group["representative_index"]][member] for member in members]
                ),
            }
        )
    return output


def _file_id(page_id: str) -> str:
    return page_id.split("#page=", 1)[0]


def _gold_sets(
    questions: Sequence[QueryRecord], qrels: Mapping[str, Mapping[str, int]]
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    gold_pages = {question.qid: set(qrels[question.qid]) for question in questions}
    gold_files = {
        qid: {_file_id(page_id) for page_id in pages}
        for qid, pages in gold_pages.items()
    }
    return gold_pages, gold_files


def _group_diagnostics(
    groups: Sequence[Mapping[str, Any]],
    questions: Sequence[QueryRecord],
    gold_pages: Mapping[str, set[str]],
    gold_files: Mapping[str, set[str]],
) -> dict[str, Any]:
    page_pairs = file_pairs = total_pairs = 0
    queries_with_page_peer: set[str] = set()
    queries_with_file_peer: set[str] = set()
    page_recall_sum = file_recall_sum = 0.0
    for group in groups:
        qids = [questions[index].qid for index in group["member_indices"]]
        for left, right in itertools.combinations(qids, 2):
            total_pairs += 1
            page_shared = bool(gold_pages[left] & gold_pages[right])
            file_shared = bool(gold_files[left] & gold_files[right])
            page_pairs += int(page_shared)
            file_pairs += int(file_shared)
            if page_shared:
                queries_with_page_peer.update((left, right))
            if file_shared:
                queries_with_file_peer.update((left, right))
        for qid in qids:
            page_peers = set().union(
                *(gold_pages[other] for other in qids if other != qid)
            ) if len(qids) > 1 else set()
            file_peers = set().union(
                *(gold_files[other] for other in qids if other != qid)
            ) if len(qids) > 1 else set()
            page_recall_sum += len(gold_pages[qid] & page_peers) / max(1, len(gold_pages[qid]))
            file_recall_sum += len(gold_files[qid] & file_peers) / max(1, len(gold_files[qid]))

    query_count = len(questions)
    return {
        "intra_group_pairs": total_pairs,
        "intra_group_shared_gold_page_pairs": page_pairs,
        "intra_group_shared_gold_file_pairs": file_pairs,
        "intra_group_pair_precision_shared_page": page_pairs / max(1, total_pairs),
        "intra_group_pair_precision_shared_file": file_pairs / max(1, total_pairs),
        "queries_with_shared_gold_page_peer": len(queries_with_page_peer),
        "queries_with_shared_gold_file_peer": len(queries_with_file_peer),
        "mean_fraction_of_gold_pages_within_group": page_recall_sum / max(1, query_count),
        "mean_fraction_of_gold_files_within_group": file_recall_sum / max(1, query_count),
    }


def _global_pair_diagnostics(
    questions: Sequence[QueryRecord],
    gold_pages: Mapping[str, set[str]],
    gold_files: Mapping[str, set[str]],
) -> dict[str, Any]:
    total = shared_page = shared_file = 0
    for left, right in itertools.combinations((question.qid for question in questions), 2):
        total += 1
        shared_page += int(bool(gold_pages[left] & gold_pages[right]))
        shared_file += int(bool(gold_files[left] & gold_files[right]))
    return {
        "all_query_pairs": total,
        "shared_gold_page_pair_rate": shared_page / max(1, total),
        "shared_gold_file_pair_rate": shared_file / max(1, total),
    }


def _bundle_pages(
    group: Mapping[str, Any],
    questions: Sequence[QueryRecord],
    run: Mapping[str, Sequence[Mapping[str, Any]]],
    score_maps: Mapping[str, Mapping[str, float]],
    *,
    depth: int,
    mode: str,
) -> list[str]:
    member_qids = [questions[index].qid for index in group["member_indices"]]
    if mode == "representative":
        representative_qid = questions[int(group["representative_index"])].qid
        return [str(hit["page_id"]) for hit in run[representative_qid][:depth]]

    page_ids = {
        str(hit["page_id"])
        for qid in member_qids
        for hit in run[qid][:depth]
    }
    if mode == "union":
        # A stable score makes this inventory readable; per-query reranking is
        # done later and is what each member actually consumes.
        aggregate = {
            page_id: max(score_maps[qid].get(page_id, 0.0) for qid in member_qids)
            for page_id in page_ids
        }
    elif mode == "centroid":
        aggregate: dict[str, float] = {}
        for qid in member_qids:
            values = score_maps[qid]
            maximum = max(values.values(), default=0.0) or 1.0
            for page_id in page_ids:
                aggregate[page_id] = aggregate.get(page_id, 0.0) + (
                    values.get(page_id, 0.0) / maximum / len(member_qids)
                )
    else:
        raise ValueError(f"Unknown bundle mode: {mode}")
    return sorted(page_ids, key=lambda page_id: (-aggregate[page_id], page_id))


def _rank_query_in_bundle(
    qid: str,
    query: str,
    bundle: Sequence[str],
    score_maps: Mapping[str, Mapping[str, float]],
    *,
    reranker: str,
    bm25_index: BM25Index | None,
    page_positions: Mapping[str, int],
) -> list[str]:
    if reranker == "source":
        scores = score_maps[qid]
    elif reranker == "bm25":
        if bm25_index is None:
            raise RuntimeError("BM25 reranker was requested without an index")
        allowed = {page_positions[page_id] for page_id in bundle if page_id in page_positions}
        scores = {
            bm25_index.chunk_ids[position]: float(score)
            for position, score in bm25_index.search(query, len(bundle), allowed)
        }
    else:
        raise ValueError(f"Unknown reranker: {reranker}")
    return sorted(bundle, key=lambda page_id: (-scores.get(page_id, 0.0), page_id))


def _dcg(relevances: Sequence[int]) -> float:
    return sum((2.0**relevance - 1.0) / math.log2(rank + 2.0) for rank, relevance in enumerate(relevances))


def _evaluate_rankings(
    rankings: Mapping[str, Sequence[str]],
    questions: Sequence[QueryRecord],
    qrels: Mapping[str, Mapping[str, int]],
    *,
    page_k: int = 10,
    file_k: int = 3,
) -> dict[str, Any]:
    per_query: list[dict[str, Any]] = []
    for question in questions:
        qid = question.qid
        ranked = list(rankings[qid])
        graded = qrels[qid]
        gold_pages = set(graded)
        top_pages = ranked[:page_k]
        retrieved_pages = set(top_pages)
        files: list[str] = []
        for page_id in ranked[:100]:
            file_id = _file_id(page_id)
            if file_id not in files:
                files.append(file_id)
            if len(files) >= file_k:
                break
        gold_files = {_file_id(page_id) for page_id in gold_pages}
        relevances = [int(graded.get(page_id, 0)) for page_id in top_pages]
        ideal = sorted((int(value) for value in graded.values()), reverse=True)[:page_k]
        dcg = _dcg(relevances)
        ideal_dcg = _dcg(ideal)
        per_query.append(
            {
                "qid": qid,
                "page_hit@10": bool(retrieved_pages & gold_pages),
                "page_recall@10": len(retrieved_pages & gold_pages) / max(1, len(gold_pages)),
                "page_precision@10": len(retrieved_pages & gold_pages) / page_k,
                "ndcg@10": dcg / ideal_dcg if ideal_dcg else 0.0,
                "file_hit@3": bool(set(files) & gold_files),
                "file_recall@3": len(set(files) & gold_files) / max(1, len(gold_files)),
                "gold_pages": sorted(gold_pages),
                "ranked_pages": ranked[:page_k],
            }
        )

    def mean(name: str) -> float:
        return statistics.mean(float(row[name]) for row in per_query)

    return {
        "ndcg@10": mean("ndcg@10"),
        "page_hit@10": mean("page_hit@10"),
        "page_recall@10": mean("page_recall@10"),
        "page_precision@10": mean("page_precision@10"),
        "file_hit@3": mean("file_hit@3"),
        "file_recall@3": mean("file_recall@3"),
        "per_query": per_query,
    }


def _cost_summary(
    independent_pages: set[str],
    bundle_pages: set[str],
    *,
    query_count: int,
    group_count: int,
    depth: int,
) -> dict[str, Any]:
    naive_slots = query_count * depth
    return {
        "retrieval_calls": group_count,
        "retrieval_call_reduction": 1.0 - group_count / max(1, query_count),
        "query_candidate_slots": naive_slots,
        "prepared_unique_pages": len(bundle_pages),
        "prepared_unique_files": len({_file_id(page_id) for page_id in bundle_pages}),
        "page_cache_savings_vs_naive_slots": 1.0 - len(bundle_pages) / max(1, naive_slots),
        "page_cache_savings_vs_independent_global_dedupe": 1.0 - len(bundle_pages) / max(1, len(independent_pages)),
        "independent_global_unique_pages": len(independent_pages),
    }


def _screen_config(
    *,
    view: str,
    group_size: int,
    min_similarity: float,
    groups: Sequence[Mapping[str, Any]],
    questions: Sequence[QueryRecord],
    run: Mapping[str, Sequence[Mapping[str, Any]]],
    qrels: Mapping[str, Mapping[str, int]],
    depths: Sequence[int],
    bm25_index: BM25Index,
    page_positions: Mapping[str, int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gold_pages, gold_files = _gold_sets(questions, qrels)
    diagnostics = _group_diagnostics(groups, questions, gold_pages, gold_files)
    config_name = f"{view}_g{group_size}"
    metrics: dict[str, Any] = {}
    bundle_rows: list[dict[str, Any]] = []
    for depth in depths:
        scores, pages_by_qid = _run_score_maps(questions, run, depth)
        independent_pages = set(page_id for pages in pages_by_qid.values() for page_id in pages)
        independent_rankings = {qid: list(pages) for qid, pages in pages_by_qid.items()}
        independent_eval = _evaluate_rankings(independent_rankings, questions, qrels)
        metrics[f"independent@{depth}"] = {
            "evaluation": independent_eval,
            "cost": _cost_summary(
                independent_pages,
                independent_pages,
                query_count=len(questions),
                group_count=len(questions),
                depth=depth,
            ),
        }

        mode_rerankers = (
            ("union", "source"),
            ("representative", "source"),
            ("representative", "bm25"),
            ("centroid", "source"),
        )
        for mode, reranker in mode_rerankers:
            bundles: dict[str, list[str]] = {}
            for group in groups:
                bundle = _bundle_pages(
                    group,
                    questions,
                    run,
                    scores,
                    depth=depth,
                    mode=mode,
                )
                bundles[str(group["group_id"])] = bundle
                bundle_rows.append(
                    {
                        "config": config_name,
                        "mode": mode,
                        "reranker": reranker,
                        "depth": depth,
                        "group_id": group["group_id"],
                        "representative_qid": questions[int(group["representative_index"])].qid,
                        "member_qids": [questions[index].qid for index in group["member_indices"]],
                        "bundle_size": len(bundle),
                        "bundle_pages": bundle,
                    }
                )
            rankings: dict[str, list[str]] = {}
            for group in groups:
                bundle = bundles[str(group["group_id"])]
                for index in group["member_indices"]:
                    qid = questions[index].qid
                    rankings[qid] = _rank_query_in_bundle(
                        qid,
                        questions[index].query,
                        bundle,
                        scores,
                        reranker=reranker,
                        bm25_index=bm25_index,
                        page_positions=page_positions,
                    )
            bundle_pages = set(page_id for bundle in bundles.values() for page_id in bundle)
            method_name = f"shared_{mode}" if reranker == "source" else f"shared_{mode}_{reranker}"
            call_count = len(questions) if mode == "union" else len(groups)
            metrics[f"{method_name}@{depth}"] = {
                "evaluation": _evaluate_rankings(rankings, questions, qrels),
                "cost": _cost_summary(
                    independent_pages,
                    bundle_pages,
                    query_count=len(questions),
                    group_count=call_count,
                    depth=depth,
                ),
            }
    return (
        {
            "config": config_name,
            "view": view,
            "group_size": group_size,
            "min_similarity": min_similarity,
            "groups": len(groups),
            "group_size_mean": statistics.mean(group["size"] for group in groups),
            "group_size_median": statistics.median(group["size"] for group in groups),
            "group_size_max": max(group["size"] for group in groups),
            "singleton_groups": sum(group["size"] == 1 for group in groups),
            "gold_overlap_diagnostics": diagnostics,
            "metrics": metrics,
        },
        bundle_rows,
    )


def _config_row(report: Mapping[str, Any], method: str, depth: int) -> dict[str, Any]:
    payload = report["metrics"][f"{method}@{depth}"]
    evaluation = payload["evaluation"]
    cost = payload["cost"]
    return {
        "config": report["config"],
        "view": report["view"],
        "group_size": report["group_size"],
        "groups": report["groups"],
        "method": method,
        "depth": depth,
        "ndcg@10": evaluation["ndcg@10"],
        "page_hit@10": evaluation["page_hit@10"],
        "page_recall@10": evaluation["page_recall@10"],
        "file_recall@3": evaluation["file_recall@3"],
        "retrieval_calls": cost["retrieval_calls"],
        "prepared_unique_pages": cost["prepared_unique_pages"],
        "prepared_unique_files": cost["prepared_unique_files"],
        "page_cache_savings_vs_independent_global_dedupe": cost[
            "page_cache_savings_vs_independent_global_dedupe"
        ],
    }


def _compact_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Remove repeated per-query payloads from the human-facing report."""
    compact = {key: value for key, value in config.items() if key != "metrics"}
    compact_metrics: dict[str, Any] = {}
    for method_key, payload in config["metrics"].items():
        compact_metrics[method_key] = {
            "evaluation": {
                key: value
                for key, value in payload["evaluation"].items()
                if key != "per_query"
            },
            "cost": payload["cost"],
        }
    compact["metrics"] = compact_metrics
    return compact


def _format_pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _write_report_markdown(
    path: Path,
    report: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    baseline = report["baseline"]
    lines = [
        "# Physics query grouping and shared evidence",
        "",
        "This is an offline exploratory baseline for light preparation. Query grouping uses only French query text and cached weighted-retrieval page overlap; qrels are used only for evaluation after grouping.",
        "",
        "## Protocol",
        "",
        f"- Dataset: `vidore_v3/physics`; queries: {report['queries']}; pages: {report['pages']}",
        f"- Source run: `{report['source_run']}`; query language: French; cached V-SPLADE query language caveat: `{report['validation']['vsplade_query_language_caveat']}`",
        f"- Group signature: overlap of the source run's top-{report['signature_depth']} pages; hybrid is the mean of this overlap and query TF-IDF cosine.",
        f"- Grouping: deterministic medoid seed, maximum group size in `{report['group_sizes']}`, minimum similarity `{report['min_similarity']}`.",
        "- Shared bundles are reranked per query by cached query-specific scores. `shared_union` is a reference because it still needs every member query's candidate list; representative/centroid are the retrieval-call-saving variants.",
        "",
        "## Current weighted baseline",
        "",
        f"At depth 100, independent weighted retrieval gives page recall@10 **{_format_pct(baseline['page_recall@10'])}**, nDCG@10 **{100.0 * baseline['ndcg@10']:.2f}**, and file recall@3 **{_format_pct(baseline['file_recall@3'])}**.",
        "",
        "## Screening at depth 100",
        "",
        "The rows are exploratory comparisons on the fixed benchmark; they are not a held-out estimate because the grouping configurations are screened on these qrels.",
        "",
        "| Config | Method | Groups | Page recall@10 | nDCG@10 | File recall@3 | Retrieval calls | Prepared pages | Savings vs global dedupe |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["depth"] != 100:
            continue
        lines.append(
            f"| {row['config']} | {row['method']} | {row['groups']} | "
            f"{_format_pct(row['page_recall@10'])} | {100.0 * row['ndcg@10']:.2f} | "
            f"{_format_pct(row['file_recall@3'])} | {row['retrieval_calls']} | "
            f"{row['prepared_unique_pages']} | "
            f"{_format_pct(row['page_cache_savings_vs_independent_global_dedupe'])} |"
        )
    lines += [
        "",
        "## What counts as related",
        "",
        "The group-level gold-overlap fields below are post-hoc diagnostics only. They answer whether the cheap grouping signal actually put queries sharing a gold page or gold file together.",
        "",
        f"For reference, among all query pairs the shared-gold-page rate is **{_format_pct(report['global_pair_diagnostics']['shared_gold_page_pair_rate'])}** and shared-gold-file rate is **{_format_pct(report['global_pair_diagnostics']['shared_gold_file_pair_rate'])}**.",
        "",
        "| Config | Intra-group pairs | Shared gold-page pair precision | Shared gold-file pair precision | Mean gold-file fraction available from group peers |",
        "|---|---:|---:|---:|---:|",
    ]
    for config in report["configs"]:
        diagnostics = config["gold_overlap_diagnostics"]
        lines.append(
            f"| {config['config']} | {diagnostics['intra_group_pairs']} | "
            f"{_format_pct(diagnostics['intra_group_pair_precision_shared_page'])} | "
            f"{_format_pct(diagnostics['intra_group_pair_precision_shared_file'])} | "
            f"{_format_pct(diagnostics['mean_fraction_of_gold_files_within_group'])} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- A large drop from `independent` to `shared_representative` means one query is not a safe proxy for its group at that bundle depth.",
        "- `shared_union` should preserve the independent ranking while exposing the cost of merely sharing prepared pages. If it does not, inspect the candidate-depth or tie behavior.",
        "- The useful production path is likely: conservative query grouping -> one shared preparation/cache bundle -> cheap query-specific reranking. Global page deduplication should be measured separately because it may already capture most page-level reuse.",
        "- The cached visual arm uses English query vectors while evaluation is French, so the evidence-overlap signal inherits that language confound.",
        "",
        "## Artifacts",
        "",
        "- `report.json`: complete metrics and configuration.",
        "- `screening.csv`-compatible JSONL: `screening_rows.jsonl`.",
        "- `groups.jsonl`: query membership and post-hoc overlap diagnostics.",
        "- `bundle_inventory.jsonl`: page bundles for each config/method/depth.",
        "- `per_query.jsonl`: detailed baseline and shared results for every screened row.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_int_list(value: str) -> list[int]:
    values = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--signature-depth", type=int, default=20)
    parser.add_argument("--min-similarity", type=float, default=0.05)
    parser.add_argument("--group-sizes", type=_parse_int_list, default=[4, 8, 12])
    parser.add_argument("--depths", type=_parse_int_list, default=[10, 25, 50, 100])
    args = parser.parse_args()

    started = time.perf_counter()
    if args.signature_depth <= 0:
        raise ValueError("--signature-depth must be positive")
    if not 0.0 <= args.min_similarity <= 1.0:
        raise ValueError("--min-similarity must be in [0, 1]")
    required_depth = max(max(args.depths), args.signature_depth)

    benchmark = ViDoreV3(root=args.benchmark_root, subset="physics", language="french")
    questions = _ordered_questions(benchmark)
    qrels = benchmark.qrels()
    run = _load_run(args.run)
    _validate_inputs(questions, run, qrels, required_depth)
    bm25_index, page_positions = _build_page_bm25(run)
    gold_pages, gold_files = _gold_sets(questions, qrels)

    lexical_similarity = _cosine_matrix(_tfidf_vectors(questions))
    evidence_similarity = _overlap_matrix(questions, run, args.signature_depth)
    similarities = {
        "lexical": lexical_similarity,
        "evidence": evidence_similarity,
        "hybrid": _mean_matrices(lexical_similarity, evidence_similarity),
    }

    configs: list[dict[str, Any]] = []
    all_group_rows: list[dict[str, Any]] = []
    all_bundle_rows: list[dict[str, Any]] = []
    all_screening_rows: list[dict[str, Any]] = []
    all_per_query_rows: list[dict[str, Any]] = []

    for view, similarity in similarities.items():
        for group_size in args.group_sizes:
            groups = _group_queries(
                questions,
                similarity,
                max_group_size=group_size,
                min_similarity=args.min_similarity,
            )
            config, bundle_rows = _screen_config(
                view=view,
                group_size=group_size,
                min_similarity=args.min_similarity,
                groups=groups,
                questions=questions,
                run=run,
                qrels=qrels,
                depths=args.depths,
                bm25_index=bm25_index,
                page_positions=page_positions,
            )
            configs.append(config)
            gold_pages, gold_files = _gold_sets(questions, qrels)
            for group in groups:
                member_qids = [questions[index].qid for index in group["member_indices"]]
                page_pair_count = file_pair_count = total_pair_count = 0
                for left, right in itertools.combinations(member_qids, 2):
                    total_pair_count += 1
                    page_pair_count += int(bool(gold_pages[left] & gold_pages[right]))
                    file_pair_count += int(bool(gold_files[left] & gold_files[right]))
                all_group_rows.append(
                    {
                        "config": config["config"],
                        "view": view,
                        "group_size": group_size,
                        "group_id": group["group_id"],
                        "representative_qid": questions[int(group["representative_index"])].qid,
                        "member_qids": member_qids,
                        "queries": [questions[index].query for index in group["member_indices"]],
                        "mean_pair_similarity": group["mean_pair_similarity"],
                        "min_pair_similarity": group["min_pair_similarity"],
                        "posthoc_pair_count": total_pair_count,
                        "posthoc_shared_gold_page_pairs": page_pair_count,
                        "posthoc_shared_gold_file_pairs": file_pair_count,
                    }
                )
            all_bundle_rows.extend(bundle_rows)
            for method_key, method_payload in config["metrics"].items():
                method, depth_text = method_key.rsplit("@", 1)
                row = _config_row(config, method, int(depth_text))
                all_screening_rows.append(row)
                for detail in method_payload["evaluation"]["per_query"]:
                    all_per_query_rows.append(
                        {
                            "config": config["config"],
                            "view": view,
                            "group_size": group_size,
                            "method": method,
                            "depth": int(depth_text),
                            **detail,
                        }
                    )

    baseline_metrics = next(
        config["metrics"]["independent@100"]["evaluation"]
        for config in configs
        if config["config"] == "hybrid_g8"
    )
    report = {
        "experiment": "physics_query_groups_evidence_reuse",
        "dataset": "vidore_v3/physics",
        "evaluation_language": "french",
        "queries": len(questions),
        "pages": sum(1 for _ in benchmark.corpus()),
        "source_candidate_pages": len({page_id for hits in run.values() for page_id in (hit["page_id"] for hit in hits)}),
        "source_run": str(args.run),
        "signature_depth": args.signature_depth,
        "min_similarity": args.min_similarity,
        "group_sizes": args.group_sizes,
        "depths": args.depths,
        "baseline": {
            key: baseline_metrics[key]
            for key in ("ndcg@10", "page_hit@10", "page_recall@10", "page_precision@10", "file_hit@3", "file_recall@3")
        },
        "similarity_views": {
            "lexical": "query TF-IDF cosine over content terms",
            "evidence": f"top-{args.signature_depth} page overlap of cached weighted retrieval",
            "hybrid": "mean(lexical, evidence)",
        },
        "global_pair_diagnostics": _global_pair_diagnostics(questions, gold_pages, gold_files),
        "configs": [_compact_config(config) for config in configs],
        "validation": {
            "qids": len(questions),
            "required_candidate_depth": required_depth,
            "all_qids_present": True,
            "all_qrels_present": True,
            "qrels_used_for_grouping": False,
            "qrels_used_for_evaluation_and_posthoc_diagnostics": True,
            "no_render_encode_api_or_new_model": True,
            "vsplade_query_language_caveat": "cached V-SPLADE query vectors are English; qrels/evaluation are French",
        },
        "timing_seconds": {"total": time.perf_counter() - started},
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "report.json", report)
    _write_jsonl(args.output_dir / "groups.jsonl", all_group_rows)
    _write_jsonl(args.output_dir / "bundle_inventory.jsonl", all_bundle_rows)
    _write_jsonl(args.output_dir / "screening_rows.jsonl", all_screening_rows)
    _write_jsonl(args.output_dir / "per_query.jsonl", all_per_query_rows)
    _write_report_markdown(args.output_dir / "report.md", report, all_screening_rows)

    best_page = sorted(
        [row for row in all_screening_rows if row["depth"] == 100 and row["method"] in {"shared_representative", "shared_centroid"}],
        key=lambda row: (-row["page_recall@10"], row["prepared_unique_pages"], row["config"], row["method"]),
    )[:5]
    print(
        json.dumps(
            {
                "status": "ok",
                "output_dir": str(args.output_dir),
                "queries": len(questions),
                "configs": len(configs),
                "baseline_page_recall@10": baseline_metrics["page_recall@10"],
                "baseline_file_recall@3": baseline_metrics["file_recall@3"],
                "top_shared_methods_at_depth100": best_page,
                "total_seconds": report["timing_seconds"]["total"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
