"""Run the research-only hierarchical Physics retrieval experiment.

The default run is offline: it reuses the KDL + pdf-inspector output and the
cached V-SPLADE vectors.  It emits page-level runs compatible with the existing
Physics evaluator plus stage traces, coverage metrics and a deterministic
5-fold out-of-fold configuration selection.

Example::

    python research/experiments/physics_hierarchical_retrieval.py

No parser, OCR, VLM, LLM or network call is made by this experiment.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.data_discovery.hierarchical import (  # noqa: E402
    CascadeConfig,
    HierarchyCorpus,
    HierarchyNode,
    aggregate_file_scores,
    build_bm25,
    diversify_pages,
    node_index_records,
    normalise_scores,
    sort_scores,
)
from src.evaluation.benchmarks.vidore_v3 import ViDoreV3  # noqa: E402
from src.retrieval.sparse import BM25Index  # noqa: E402


DEFAULT_PARSED_RUN = ROOT / "data/output/vidore-v3-physics-kdl-pdf-inspector/32c32a45a92c45bb"
DEFAULT_BENCHMARK_ROOT = ROOT / "data/benchmark/vidore_v3"
DEFAULT_PAGE_VECTOR_DIR = ROOT / "data/output/vsplade/vidore_v3_physics_48q"
DEFAULT_QUERY_VECTOR_DIR = ROOT / "data/output/vsplade/vidore_v3_physics_english_302q"
DEFAULT_PAIR_DIR = ROOT / "data/benchmark/vidore_v3/results/physics_vsplade_bm25_fusion"
DEFAULT_OUTPUT_DIR = ROOT / "data/benchmark/vidore_v3/results/physics_hierarchical_retrieval"


def _load_csr(path: Path) -> sparse.csr_matrix:
    if not path.is_file():
        raise FileNotFoundError(f"Sparse vector artifact not found: {path}")
    payload = np.load(path, allow_pickle=True)
    shape = tuple(int(value) for value in payload["shape"])
    return sparse.csr_matrix(
        (
            payload["data"].astype(np.float32, copy=False),
            payload["indices"].astype(np.int32, copy=False),
            payload["indptr"].astype(np.int32, copy=False),
        ),
        shape=shape,
    )


def _load_run(path: Path) -> dict[str, list[dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Retrieval run not found: {path}")
    output: dict[str, list[dict[str, Any]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            output[str(row["qid"])] = list(row["chunks"])
    return output


def _page_vector_units(path: Path) -> list[str]:
    payload = json.loads((path / "page_metadata.json").read_text(encoding="utf-8"))
    return [str(row["unit_id"]) for row in payload]


def _path_size_bytes(path: Path) -> int:
    """Return a deterministic on-disk size for a file or artifact directory."""
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    return 0


def _page_number(page_id: str) -> int:
    try:
        return int(page_id.rsplit("#page=", 1)[-1])
    except ValueError:
        return -1


def _file_id(page_id: str) -> str:
    return page_id.split("#page=", 1)[0]


def _index_scores(
    index: BM25Index,
    query: str,
    *,
    allowed_ids: set[str] | None = None,
    top_k: int | None = None,
) -> dict[str, float]:
    allowed_positions = None
    if allowed_ids is not None:
        allowed_positions = {
            position for position, node_id in enumerate(index.chunk_ids) if node_id in allowed_ids
        }
    hits = index.search(query, top_k or len(index.chunk_ids), allowed_positions)
    return {index.chunk_ids[position]: float(score) for position, score in hits}


def _top_hits(
    index: BM25Index,
    query: str,
    *,
    allowed_ids: set[str] | None = None,
    depth: int = 100,
) -> list[tuple[str, float]]:
    scores = _index_scores(index, query, allowed_ids=allowed_ids, top_k=depth)
    return sort_scores(scores)[:depth]


def _aggregate_node_scores(
    scores: Mapping[str, float],
    nodes: Mapping[str, HierarchyNode],
    *,
    pool: str,
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for node_id, score in scores.items():
        node = nodes.get(node_id)
        if node is not None and node.page_id is not None:
            grouped[node.page_id].append(max(0.0, float(score)))

    output: dict[str, float] = {}
    for page_id, values in grouped.items():
        values.sort(reverse=True)
        if pool == "max":
            output[page_id] = values[0]
        elif pool == "sum_top2":
            output[page_id] = sum(values[:2])
        elif pool == "coverage":
            output[page_id] = sum(
                1.0 / (10.0 + rank) for rank in range(1, min(10, len(values)) + 1)
            ) + values[0] * 1e-3
        else:
            raise ValueError(f"Unknown fine pool: {pool!r}")
    return output


def _fill_ranked(
    ranked: Sequence[tuple[str, float]],
    universe: Sequence[str],
    depth: int,
) -> list[tuple[str, float]]:
    output = list(ranked[:depth])
    seen = {node_id for node_id, _ in output}
    for node_id in universe:
        if len(output) >= depth:
            break
        if node_id not in seen:
            output.append((node_id, 0.0))
            seen.add(node_id)
    return output


def _neighbor_pages(
    corpus: HierarchyCorpus,
    page_ids: Iterable[str],
    window: int,
) -> set[str]:
    output = set(page_ids)
    if window <= 0:
        return output
    for page_id in list(page_ids):
        file_id = _file_id(page_id)
        pages = corpus.file_to_pages.get(file_id, [])
        try:
            index = pages.index(page_id)
        except ValueError:
            continue
        start = max(0, index - window)
        end = min(len(pages), index + window + 1)
        output.update(pages[start:end])
    return output


def _component_row(
    node_id: str,
    *,
    score: float,
    components: Mapping[str, float],
    rank: int,
) -> dict[str, Any]:
    return {
        "node_id": node_id,
        "rank": rank,
        "score": round(float(score), 8),
        "components": {key: round(float(value), 8) for key, value in components.items()},
    }


class HierarchicalRetriever:
    """One reusable scorer over a materialised hierarchy."""

    def __init__(
        self,
        corpus: HierarchyCorpus,
        *,
        page_index: BM25Index,
        file_indexes: Mapping[str, BM25Index],
        fine_indexes: Mapping[str, tuple[BM25Index, dict[str, HierarchyNode]]],
        visual_scores: Mapping[str, Mapping[str, float]],
    ) -> None:
        self.corpus = corpus
        self.page_index = page_index
        self.file_indexes = dict(file_indexes)
        self.fine_indexes = dict(fine_indexes)
        self.visual_scores = visual_scores
        self.page_to_file = {page_id: _file_id(page_id) for page_id in corpus.page_order}

    def retrieve(
        self,
        qid: str,
        query: str,
        config: CascadeConfig,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        started = time.perf_counter()

        page_bm25 = _index_scores(self.page_index, query, top_k=len(self.corpus.page_order))
        page_bm25_norm = normalise_scores(page_bm25)
        visual = {
            page_id: float(score)
            for page_id, score in self.visual_scores.get(qid, {}).items()
            if page_id in self.corpus.pages
        }
        visual_norm = normalise_scores(visual)
        page_base = {
            page_id: config.bm25_weight * page_bm25_norm.get(page_id, 0.0)
            + (1.0 - config.bm25_weight) * visual_norm.get(page_id, 0.0)
            for page_id in self.corpus.page_order
        }

        file_index = self.file_indexes[config.file_representation]
        file_direct = _index_scores(file_index, query, top_k=len(self.corpus.file_order))
        file_direct_norm = normalise_scores(file_direct)
        if config.file_pool_source == "page_bm25":
            file_pool_input = page_bm25_norm
        elif config.file_pool_source == "page_base":
            file_pool_input = page_base
        else:
            raise ValueError(f"Unknown file pool source: {config.file_pool_source!r}")
        file_pool = aggregate_file_scores(
            file_pool_input,
            self.page_to_file,
            self.corpus.file_order,
            config.file_pool,
        )
        file_pool_norm = normalise_scores(file_pool)
        file_scores = {
            file_id: config.file_direct_weight * file_direct_norm.get(file_id, 0.0)
            + (1.0 - config.file_direct_weight) * file_pool_norm.get(file_id, 0.0)
            for file_id in self.corpus.file_order
        }
        file_ranked_all = sort_scores(file_scores)
        file_ranked = _fill_ranked(file_ranked_all, self.corpus.file_order, config.k_files)
        selected_files = {file_id for file_id, _ in file_ranked[: config.k_files]}

        parent_norm = normalise_scores(file_scores)
        page_scores: dict[str, float] = {}
        page_components: dict[str, dict[str, float]] = {}
        for page_id in self.corpus.page_order:
            file_id = self.page_to_file[page_id]
            if file_id not in selected_files:
                continue
            parent_score = parent_norm.get(file_id, 0.0)
            score = (1.0 - config.parent_weight) * page_base[page_id] + config.parent_weight * parent_score
            page_scores[page_id] = score
            page_components[page_id] = {
                "page_bm25": page_bm25_norm.get(page_id, 0.0),
                "vsplade_page": visual_norm.get(page_id, 0.0),
                "parent_file_score": parent_score,
                "page_base": page_base[page_id],
            }
        page_ranked = _fill_ranked(
            sort_scores(page_scores),
            [page_id for page_id in self.corpus.page_order if self.page_to_file[page_id] in selected_files],
            config.page_depth,
        )
        page_candidate_ids = {page_id for page_id, _ in page_ranked[: config.page_depth]}

        fine_scores: dict[str, float] = {}
        fine_hits: list[tuple[str, float]] = []
        fine_page_scores: dict[str, float] = {}
        fine_started = time.perf_counter()
        fine_allowed_pages = _neighbor_pages(
            self.corpus, page_candidate_ids, config.neighbor_window
        )
        if config.fine_weight > 0 and config.fine_unit not in {"", "none"}:
            fine_index, fine_nodes = self.fine_indexes[config.fine_unit]
            allowed_nodes = {
                node_id for node_id, node in fine_nodes.items() if node.page_id in fine_allowed_pages
            }
            fine_hits = _top_hits(
                fine_index,
                query,
                allowed_ids=allowed_nodes,
                depth=max(config.final_depth, 100),
            )
            fine_scores = dict(fine_hits)
            fine_page_scores = _aggregate_node_scores(fine_scores, fine_nodes, pool=config.fine_pool)
        fine_page_norm = normalise_scores(fine_page_scores)

        final_scores: dict[str, float] = {}
        for page_id in self.corpus.page_order:
            if self.page_to_file[page_id] not in selected_files:
                continue
            page_score = page_scores.get(page_id, 0.0)
            fine_score = fine_page_norm.get(page_id, 0.0)
            final_scores[page_id] = (
                (1.0 - config.fine_weight) * page_score + config.fine_weight * fine_score
                if config.fine_weight > 0 and config.fine_unit not in {"", "none"}
                else page_score
            )
            page_components.setdefault(page_id, {})["fine_segment_score"] = fine_score
            page_components.setdefault(page_id, {})["final_score"] = final_scores[page_id]

        final_ranked = sort_scores(final_scores)
        final_ranked = diversify_pages(
            final_ranked,
            self.page_to_file,
            max_pages_per_file=config.file_quota,
            depth=config.final_depth,
        )
        final_ranked = _fill_ranked(final_ranked, list(final_scores), config.final_depth)

        run: list[dict[str, Any]] = []
        for rank, (page_id, score) in enumerate(final_ranked, 1):
            components = page_components.get(page_id, {})
            run.append(
                {
                    "chunk_id": page_id,
                    "doc_id": page_id,
                    "text": self.corpus.pages[page_id].text,
                    "score": round(float(score), 8),
                    "rank": rank,
                    "scores": {key: round(float(value), 8) for key, value in components.items()},
                }
            )

        file_trace = [
            _component_row(
                file_id,
                score=score,
                components={
                    "file_direct_bm25": file_direct_norm.get(file_id, 0.0),
                    "page_pool": file_pool_norm.get(file_id, 0.0),
                    "final_file": file_scores.get(file_id, 0.0),
                },
                rank=rank,
            )
            for rank, (file_id, score) in enumerate(file_ranked_all, 1)
        ]
        page_trace = [
            _component_row(
                page_id,
                score=page_scores.get(page_id, 0.0),
                components=page_components.get(page_id, {}),
                rank=rank,
            )
            for rank, (page_id, _) in enumerate(page_ranked, 1)
        ]
        fine_trace: list[dict[str, Any]] = []
        if config.fine_unit not in {"", "none"} and config.fine_unit in self.fine_indexes:
            _, fine_nodes = self.fine_indexes[config.fine_unit]
            fine_trace = [
                {
                    "node_id": node_id,
                    "page_id": fine_nodes[node_id].page_id,
                    "file_id": fine_nodes[node_id].file_id,
                    "level": fine_nodes[node_id].level,
                    "score": round(float(score), 8),
                    "text": fine_nodes[node_id].text[:500],
                }
                for node_id, score in fine_hits
            ]

        elapsed = time.perf_counter() - started
        trace = {
            "qid": qid,
            "config": asdict(config),
            "file_candidates": file_trace,
            "page_candidates": page_trace,
            "fine_candidates": fine_trace,
            "fine_candidate_pages": sorted(fine_allowed_pages),
            "final_pages": [page_id for page_id, _ in final_ranked],
            "timing_seconds": {
                "total": round(elapsed, 6),
                "fine_stage": round(time.perf_counter() - fine_started, 6),
            },
            "counts": {
                "files_indexed": len(self.corpus.files),
                "pages_indexed": len(self.corpus.pages),
                "fine_nodes_indexed": (
                    len(self.fine_indexes.get(config.fine_unit, (None, {}))[1])
                    if config.fine_unit in self.fine_indexes
                    else 0
                ),
                "selected_files": len(selected_files),
                "page_candidates": len(page_candidate_ids),
                "fine_candidates": len(fine_hits),
            },
        }
        return run, trace


def _load_visual_scores(
    page_vector_dir: Path,
    query_vector_dir: Path,
    page_ids: Sequence[str],
    qids: Sequence[str],
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    page_matrix = _load_csr(page_vector_dir / "page_vectors.npz")
    query_matrix = _load_csr(query_vector_dir / "query_vectors.npz")
    metadata_units = _page_vector_units(page_vector_dir)
    if page_matrix.shape[0] != len(metadata_units):
        raise RuntimeError("V-SPLADE page vector and metadata counts differ")
    if set(metadata_units) != set(page_ids):
        missing = sorted(set(page_ids) - set(metadata_units))[:5]
        extra = sorted(set(metadata_units) - set(page_ids))[:5]
        raise RuntimeError(f"V-SPLADE page inventory mismatch; missing={missing}, extra={extra}")
    if query_matrix.shape[0] != len(qids):
        raise RuntimeError(
            f"V-SPLADE query count {query_matrix.shape[0]} != Physics qids {len(qids)}"
        )
    page_position = {unit: position for position, unit in enumerate(metadata_units)}
    page_order_positions = [page_position[page_id] for page_id in page_ids]
    ordered_page_matrix = page_matrix[page_order_positions]
    scores = (query_matrix @ ordered_page_matrix.T).toarray()
    output = {
        qid: {page_id: float(scores[row, position]) for position, page_id in enumerate(page_ids)}
        for row, qid in enumerate(qids)
    }
    return output, {
        "page_vectors": str(page_vector_dir / "page_vectors.npz"),
        "query_vectors": str(query_vector_dir / "query_vectors.npz"),
        "page_vector_artifact_bytes": _path_size_bytes(page_vector_dir),
        "query_vector_artifact_bytes": _path_size_bytes(query_vector_dir),
        "query_language": query_vector_dir.name.removeprefix("vidore_v3_physics_").removesuffix("_302q"),
    }


def _build_corpus_and_indexes(
    parsed_run: Path,
    page_ids: Sequence[str],
    *,
    fine_units: Sequence[str],
) -> tuple[HierarchyCorpus, BM25Index, dict[str, BM25Index], dict[str, tuple[BM25Index, dict[str, HierarchyNode]]], dict[str, int]]:
    started = time.perf_counter()
    corpus = HierarchyCorpus.from_parsed_run(parsed_run, subset="physics", page_ids=page_ids)
    page_index = build_bm25(
        (page_id, corpus.pages[page_id].text) for page_id in corpus.page_order
    )
    file_texts = {
        representation: corpus.file_texts(representation)
        for representation in ("all_text", "headings_stubs")
    }
    file_indexes = {
        representation: build_bm25(texts.items())
        for representation, texts in file_texts.items()
    }
    fine_indexes: dict[str, tuple[BM25Index, dict[str, HierarchyNode]]] = {}
    fine_counts: dict[str, int] = {}
    fine_text_bytes: dict[str, int] = {}
    for unit in fine_units:
        if unit in {"", "none"}:
            continue
        nodes = {node.node_id: node for node in corpus.nodes_for_unit(unit)}
        records = node_index_records(nodes.values(), corpus, include_context=True)
        fine_indexes[unit] = (build_bm25(records), nodes)
        fine_counts[unit] = len(nodes)
        fine_text_bytes[unit] = sum(len(text.encode("utf-8")) for _, text in records)
    return corpus, page_index, file_indexes, fine_indexes, {
        "build_seconds": round(time.perf_counter() - started, 6),
        "files": len(corpus.files),
        "pages": len(corpus.pages),
        "pages_with_parser_text": sum(
            bool(node.metadata.get("has_parser_text")) for node in corpus.pages.values()
        ),
        "blocks": len(corpus.blocks),
        "paragraphs": len(corpus.paragraphs),
        "sentences": len(corpus.sentences),
        "evidence_atoms": len(corpus.evidence_atoms),
        "fine_nodes": fine_counts,
        "index_size_bytes": {
            "page_bm25_logical_text": sum(
                len(corpus.pages[page_id].text.encode("utf-8"))
                for page_id in corpus.page_order
            ),
            "file_bm25_logical_text": {
                representation: sum(
                    len(text.encode("utf-8")) for text in texts.values()
                )
                for representation, texts in file_texts.items()
            },
            "fine_bm25_logical_text": fine_text_bytes,
        },
    }


def _configurations() -> list[CascadeConfig]:
    """Small fixed screening set; all weights are selected before evaluation."""
    configs: list[CascadeConfig] = []
    for representation, pool, k_files in (
        ("all_text", "max", 10),
        ("all_text", "sum_top2", 10),
        ("all_text", "coverage", 10),
        ("headings_stubs", "max", 10),
        ("headings_stubs", "sum_top2", 10),
        ("all_text", "max", 5),
        ("all_text", "max", 3),
    ):
        configs.append(
            CascadeConfig(
                name=f"cascade-{representation}-{pool}-kf{k_files}-fusion",
                file_representation=representation,
                file_pool=pool,
                k_files=k_files,
                page_depth=100,
                bm25_weight=0.70,
                parent_weight=0.15,
            )
        )

    for page_depth in (20, 50):
        configs.append(
            CascadeConfig(
                name=f"cascade-all_text-max-kf10-p{page_depth}-fusion",
                file_representation="all_text",
                file_pool="max",
                k_files=10,
                page_depth=page_depth,
                bm25_weight=0.70,
                parent_weight=0.15,
            )
        )

    fine_base = dict(
        file_representation="all_text",
        file_pool="max",
        k_files=10,
        page_depth=100,
        bm25_weight=0.70,
        parent_weight=0.15,
        fine_weight=0.25,
    )
    for unit in ("block", "paragraph", "sentence_group3", "sentence_group5", "atom"):
        configs.append(CascadeConfig(name=f"fine-{unit}", fine_unit=unit, **fine_base))
    for page_depth in (20, 50):
        configs.append(
            CascadeConfig(
                name=f"fine-block-p{page_depth}",
                fine_unit="block",
                page_depth=page_depth,
                **{key: value for key, value in fine_base.items() if key != "page_depth"},
            )
        )
    for unit in ("paragraph", "sentence_group5"):
        configs.append(
            CascadeConfig(
                name=f"fine-{unit}-sum2-neighbor",
                fine_unit=unit,
                fine_pool="sum_top2",
                neighbor_window=1,
                **fine_base,
            )
        )
    return configs


def _unique_files(chunks: Sequence[dict[str, Any]], depth: int = 100) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for chunk in chunks[:depth]:
        file_id = _file_id(str(chunk["chunk_id"]))
        if file_id not in seen:
            seen.add(file_id)
            output.append(file_id)
        if len(output) >= 10:
            break
    return output


def _stage_metrics(
    traces: Mapping[str, dict[str, Any]],
    qids: Sequence[str],
    qrels: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    file_recalls: dict[int, list[float]] = {1: [], 2: [], 3: [], 5: [], 10: []}
    page_candidate_recall: list[float] = []
    fine_page_recall: list[float] = []
    conditional_fine_recall: list[float] = []
    conditional_count = 0
    for qid in qids:
        trace = traces[qid]
        gold_pages = set(qrels.get(qid, {}))
        gold_files = {_file_id(page) for page in gold_pages}
        file_candidates = [str(row["node_id"]) for row in trace["file_candidates"]]
        for k in file_recalls:
            found = set(file_candidates[:k]) & gold_files
            file_recalls[k].append(len(found) / len(gold_files) if gold_files else 0.0)

        page_candidates = {
            str(row["node_id"]) for row in trace["page_candidates"]
        }
        page_candidate_recall.append(
            len(page_candidates & gold_pages) / len(gold_pages) if gold_pages else 0.0
        )
        fine_pages = {
            str(row["page_id"])
            for row in trace.get("fine_candidates", [])
            if row.get("page_id")
        }
        fine_value = len(fine_pages & gold_pages) / len(gold_pages) if gold_pages else 0.0
        fine_page_recall.append(fine_value)
        if page_candidates & gold_pages:
            conditional_count += 1
            conditional_fine_recall.append(fine_value)

    mean = lambda values: round(sum(values) / len(values), 6) if values else 0.0
    return {
        "file_candidate_recall": {f"@{k}": mean(values) for k, values in file_recalls.items()},
        "page_candidate_recall": mean(page_candidate_recall),
        "fine_page_recall_proxy": mean(fine_page_recall),
        "fine_page_recall_proxy_conditional_on_page_candidate": mean(conditional_fine_recall),
        "conditional_page_candidate_queries": conditional_count,
        "queries": len(qids),
    }


def _stratified_folds(
    qids: Sequence[str],
    questions: Mapping[str, Any],
    qrels: Mapping[str, Mapping[str, int]],
    *,
    n_folds: int = 5,
    seed: int = 20260729,
) -> list[list[str]]:
    groups: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    for qid in qids:
        question = questions[qid]
        query_types = tuple(sorted(getattr(question, "query_types", ()) or ()))
        query_format = str(getattr(question, "query_format", "") or "")
        gold_file_count = len({_file_id(page) for page in qrels.get(qid, {})})
        groups[(query_types, query_format, gold_file_count)].append(qid)

    folds: list[list[str]] = [[] for _ in range(n_folds)]
    fold_sizes = [0] * n_folds
    for group_key in sorted(groups, key=lambda value: repr(value)):
        members = sorted(
            groups[group_key],
            key=lambda qid: hashlib.sha256(f"{seed}:{qid}".encode()).hexdigest(),
        )
        for qid in members:
            target = min(range(n_folds), key=lambda index: (fold_sizes[index], index))
            folds[target].append(qid)
            fold_sizes[target] += 1
    return [sorted(fold, key=lambda qid: int(qid.rsplit("::", 1)[1])) for fold in folds]


def _metric_for_qids(
    method_metrics: Mapping[str, Any],
    qids: set[str],
) -> tuple[float, float]:
    rows = [row for row in method_metrics["per_query"] if row["qid"] in qids]
    if not rows:
        return 0.0, 0.0
    page = sum(float(row["page_recall@10"]) for row in rows) / len(rows)
    file_values: list[float] = []
    for row in rows:
        candidates = row["top10_files_from_top100_pages"][:3]
        gold = set(row["gold_files"])
        file_values.append(len(set(candidates) & gold) / len(gold) if gold else 0.0)
    return page, sum(file_values) / len(file_values)


def _cross_validated_selection(
    qids: Sequence[str],
    questions: Mapping[str, Any],
    qrels: Mapping[str, Mapping[str, int]],
    config_metrics: Mapping[str, Mapping[str, Any]],
    config_runs: Mapping[str, Mapping[str, list[dict[str, Any]]]],
) -> dict[str, Any]:
    folds = _stratified_folds(qids, questions, qrels)
    selected: list[dict[str, Any]] = []
    oof_run: dict[str, list[dict[str, Any]]] = {}
    qid_set = set(qids)
    for fold_index, heldout in enumerate(folds):
        heldout_set = set(heldout)
        train_set = qid_set - heldout_set
        ranking: list[tuple[float, float, str]] = []
        for name, metrics in config_metrics.items():
            page, file = _metric_for_qids(metrics, train_set)
            ranking.append((page, file, name))
        _, _, winner = max(ranking, key=lambda item: (item[0], item[1], item[2]))
        for qid in heldout:
            oof_run[qid] = config_runs[winner][qid]
        selected.append(
            {
                "fold": fold_index,
                "heldout_qids": heldout,
                "selected_config": winner,
                "train_page_recall@10": _metric_for_qids(config_metrics[winner], train_set)[0],
                "train_file_recall@3": _metric_for_qids(config_metrics[winner], train_set)[1],
            }
        )
    return {
        "folds": selected,
        "out_of_fold_run": oof_run,
        "selected_config_counts": {
            name: sum(row["selected_config"] == name for row in selected)
            for name in config_metrics
        },
    }


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Physics hierarchical retrieval",
        "",
        "Offline file → page → fine-evidence experiment using KDL + pdf-inspector and cached V-SPLADE vectors.",
        "",
        "## Corpus and protocol",
        "",
        f"- Queries: **{report['queries']}** French Physics queries.",
        f"- Pages: **{report['pages']}**; files: **{report['index_counts']['files']}**; pages with parser text: **{report['index_counts']['pages_with_parser_text']}**.",
        "- Page metrics use top-10 pages; file metrics use the first 3 unique files in the first 100 page results.",
        "- Sentence-level results are page-backed proxies because the benchmark qrels remain page-level.",
        f"- Index build: **{report['timing_seconds']['index_build']:.2f}s**; cached page/query vector artifacts: **{report['sources']['page_vector_artifact_bytes']:,} / {report['sources']['query_vector_artifact_bytes']:,} bytes**.",
        "",
        "## Screening results",
        "",
        "| Method | nDCG@10 | Page hit@10 | Page recall@10 | Δ page pp | File hit@3 | File recall@3 | Δ file pp | Page candidate recall | Fine proxy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, method in report["methods"].items():
        metrics = method["retrieval_metrics"]
        stage = method.get("stage_metrics", {})
        file_metrics = metrics["file_metrics_by_k"]["3"]
        comparison = method.get("comparison_to_baseline") or {}
        lines.append(
            f"| {name} | {metrics['ndcg@10']:.2f} | {metrics['page_hit@10']:.2%} | "
            f"{metrics['page_recall@10']:.2%} | {comparison.get('page_recall_delta_pp', 0.0):+.2f} | "
            f"{file_metrics['file_hit']:.2%} | {file_metrics['file_recall']:.2%} | "
            f"{comparison.get('file_recall_delta_pp', 0.0):+.2f} | "
            f"{stage.get('page_candidate_recall', 0):.2%} | "
            f"{stage.get('fine_page_recall_proxy', 0):.2%} |"
        )
    lines += [
        "",
        "## Out-of-fold selection",
        "",
        "The winner for each fold is selected on the other four folds; the aggregate is evaluated once on the held-out queries.",
        "",
        *(
            [
                f"- OOF nDCG@10: **{report['cv']['retrieval_metrics']['ndcg@10']:.2f}**.",
                f"- OOF page recall@10: **{report['cv']['retrieval_metrics']['page_recall@10']:.2%}**.",
                f"- OOF page hit@10: **{report['cv']['retrieval_metrics']['page_hit@10']:.2%}**; file hit@3: **{report['cv']['retrieval_metrics']['file_metrics_by_k']['3']['file_hit']:.2%}**.",
                f"- OOF file recall@3: **{report['cv']['retrieval_metrics']['file_metrics_by_k']['3']['file_recall']:.2%}**.",
                f"- OOF file candidate recall: @3 **{report['cv']['stage_metrics']['file_candidate_recall']['@3']:.2%}**, @5 **{report['cv']['stage_metrics']['file_candidate_recall']['@5']:.2%}**, @10 **{report['cv']['stage_metrics']['file_candidate_recall']['@10']:.2%}**; page candidate recall **{report['cv']['stage_metrics']['page_candidate_recall']:.2%}**.",
                f"- OOF gain: **{report['cv']['comparison_to_baseline']['page_recall_delta_pp']:+.2f}pp** page recall, **{report['cv']['comparison_to_baseline']['file_recall_delta_pp']:+.2f}pp** file recall.",
                f"- OOF page bootstrap CI: `{report['cv']['comparison_to_baseline']['page_recall_ci95_pp']}` pp; p=`{report['cv']['comparison_to_baseline']['page_recall_p_two_sided']:.4f}`.",
                f"- Selected configuration counts: `{report['cv']['selected_config_counts']}`.",
            ]
            if not report["cv"].get("skipped")
            else ["- Cross-validation was skipped."]
        ),
        "",
        "## Caveats",
        "",
        "- The V-SPLADE cache may contain English query vectors evaluated against French qrels; this is recorded in the JSON metadata.",
        "- No qrel content modality is used as a router feature.",
        "- This is retrieval-only; no end-to-end QA claim is made.",
        "",
        "Full stage traces are in `stage_traces.jsonl`; page runs are in `runs/`.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parsed-run", type=Path, default=DEFAULT_PARSED_RUN)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--page-vector-dir", type=Path, default=DEFAULT_PAGE_VECTOR_DIR)
    parser.add_argument("--query-vector-dir", type=Path, default=DEFAULT_QUERY_VECTOR_DIR)
    parser.add_argument("--pair-dir", type=Path, default=DEFAULT_PAIR_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip-cv", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    benchmark = ViDoreV3(root=args.benchmark_root, subset="physics", language="french")
    questions_list = sorted(
        list(benchmark.questions()), key=lambda question: int(question.qid.rsplit("::", 1)[1])
    )
    questions = {question.qid: question for question in questions_list}
    qids = [question.qid for question in questions_list]
    qrels = benchmark.qrels()

    page_ids = _page_vector_units(args.page_vector_dir)
    if len(page_ids) != 1674:
        raise RuntimeError(f"Expected the canonical 1,674 Physics pages, got {len(page_ids)}")
    if len(qids) != 302:
        raise RuntimeError(f"Expected 302 French Physics queries, got {len(qids)}")

    configurations = _configurations()
    fine_units = sorted({config.fine_unit for config in configurations if config.fine_unit not in {"", "none"}})
    corpus, page_index, file_indexes, fine_indexes, index_counts = _build_corpus_and_indexes(
        args.parsed_run, page_ids, fine_units=fine_units
    )
    visual_scores, visual_meta = _load_visual_scores(
        args.page_vector_dir, args.query_vector_dir, page_ids, qids
    )
    retriever = HierarchicalRetriever(
        corpus,
        page_index=page_index,
        file_indexes=file_indexes,
        fine_indexes=fine_indexes,
        visual_scores=visual_scores,
    )

    baseline_path = args.pair_dir / "bm25_french_bm25-french_vs-english.jsonl"
    baseline_run = _load_run(baseline_path)
    if set(baseline_run) != set(qids):
        raise RuntimeError("Cached BM25 baseline does not contain exactly the 302 Physics qids")
    methods: dict[str, dict[str, Any]] = {
        "PDF-inspector + BM25 baseline": {
            "config": None,
            "retrieval_metrics": _derived_metrics(baseline_run, qids, qrels),
            "stage_metrics": {},
        }
    }
    config_runs: dict[str, dict[str, list[dict[str, Any]]]] = {}
    config_metrics: dict[str, dict[str, Any]] = {}
    all_traces: dict[str, dict[str, Any]] = {}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = args.output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    for config in configurations:
        config_started = time.perf_counter()
        run_by_qid: dict[str, list[dict[str, Any]]] = {}
        traces: dict[str, dict[str, Any]] = {}
        for question in questions_list:
            run, trace = retriever.retrieve(question.qid, question.query, config)
            run_by_qid[question.qid] = run
            traces[question.qid] = trace
        run_metrics = _derived_metrics(run_by_qid, qids, qrels)
        stage = _stage_metrics(traces, qids, qrels)
        name = config.name
        config_runs[name] = run_by_qid
        config_metrics[name] = run_metrics
        all_traces[name] = traces
        methods[name] = {
            "config": asdict(config),
            "retrieval_metrics": run_metrics,
            "stage_metrics": stage,
            "timing_seconds": round(time.perf_counter() - config_started, 6),
        }
        _write_run(
            runs_dir / f"{_safe_name(name)}.jsonl",
            run_by_qid,
            qids,
            queries={qid: questions[qid].query for qid in qids},
        )

    if args.skip_cv:
        cv_payload = {
            "skipped": True,
            "reason": "--skip-cv",
            "retrieval_metrics": {},
            "selected_config_counts": {},
            "folds": [],
        }
    else:
        cv = _cross_validated_selection(qids, questions, qrels, config_metrics, config_runs)
        oof_run = cv.pop("out_of_fold_run")
        cv_metrics = _derived_metrics(oof_run, qids, qrels)
        cv_payload = {
            **cv,
            "retrieval_metrics": cv_metrics,
            "stage_metrics": _stage_metrics(
                {qid: all_traces[cv_row["selected_config"]][qid] for cv_row in cv["folds"] for qid in cv_row["heldout_qids"]},
                qids,
                qrels,
            ),
            "comparison_to_baseline": _paired_comparison(
                methods["PDF-inspector + BM25 baseline"]["retrieval_metrics"],
                cv_metrics,
            ),
        }
        _write_run(
            args.output_dir / "oof_run.jsonl",
            oof_run,
            qids,
            queries={qid: questions[qid].query for qid in qids},
        )

    trace_path = args.output_dir / "stage_traces.jsonl"
    with trace_path.open("w", encoding="utf-8") as handle:
        for name, traces in all_traces.items():
            for qid in qids:
                handle.write(json.dumps({"method": name, **traces[qid]}, ensure_ascii=False) + "\n")

    report: dict[str, Any] = {
        "dataset": "vidore_v3/physics",
        "evaluation_language": "french",
        "queries": len(qids),
        "pages": len(page_ids),
        "index_counts": index_counts,
        "sources": {
            "parsed_run": str(args.parsed_run),
            "baseline_run": str(baseline_path),
            **visual_meta,
        },
        "methods": methods,
        "cv": cv_payload,
        "timing_seconds": {
            "total": round(time.perf_counter() - started, 6),
            "index_build": index_counts["build_seconds"],
        },
        "notes": [
            "The hierarchy is research-only; production retrieval service code is unchanged.",
            "Qrels remain page-level. Fine-stage recall is a page-backed proxy, not sentence-level gold recall.",
            "The cached V-SPLADE query language is recorded in sources.query_language and may be English.",
            "The cached BM25 baseline is used for the headline comparison; hierarchy page text is rebuilt from the parsed run.",
        ],
    }
    baseline_method = methods["PDF-inspector + BM25 baseline"]["retrieval_metrics"]
    for name, method in methods.items():
        if name != "PDF-inspector + BM25 baseline":
            method["comparison_to_baseline"] = _paired_comparison(
                baseline_method, method["retrieval_metrics"]
            )
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "report.md").write_text(_markdown(report), encoding="utf-8")

    summary = {
        name: {
            "page_recall@10": round(value["retrieval_metrics"]["page_recall@10"] * 100, 2),
            "file_recall@3": round(value["retrieval_metrics"]["file_metrics_by_k"]["3"]["file_recall"] * 100, 2),
            **(
                {
                    "page_delta_pp": round(value["comparison_to_baseline"]["page_recall_delta_pp"], 2),
                    "file_delta_pp": round(value["comparison_to_baseline"]["file_recall_delta_pp"], 2),
                }
                if "comparison_to_baseline" in value
                else {}
            ),
        }
        for name, value in methods.items()
    }
    print(json.dumps({"summary": summary, "cv": cv_payload, "output": str(args.output_dir)}, ensure_ascii=False, indent=2))


def _derived_metrics(
    run: Mapping[str, list[dict[str, Any]]],
    qids: Sequence[str],
    qrels: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    """Evaluate page/file metrics without rebuilding RunRecord objects."""
    page_hit: list[float] = []
    page_recall: list[float] = []
    page_precision: list[float] = []
    ndcg: list[float] = []
    file_by_k: dict[int, list[float]] = {1: [], 3: [], 5: [], 10: []}
    file_hit_by_k: dict[int, list[float]] = {1: [], 3: [], 5: [], 10: []}
    per_query: list[dict[str, Any]] = []
    for qid in qids:
        gold_pages = set(qrels.get(qid, {}))
        gold_files = {_file_id(page) for page in gold_pages}
        pages = [str(row["chunk_id"]) for row in run[qid][:10]]
        found_pages = set(pages) & gold_pages
        page_hit.append(float(bool(found_pages)))
        page_recall.append(len(found_pages) / len(gold_pages) if gold_pages else 0.0)
        page_precision.append(len(found_pages) / 10.0)
        ndcg.append(_ndcg_at_k(pages, qrels.get(qid, {}), 10))
        files = _unique_files(run[qid], 100)
        for k in file_by_k:
            found_files = set(files[:k]) & gold_files
            file_by_k[k].append(len(found_files) / len(gold_files) if gold_files else 0.0)
            file_hit_by_k[k].append(float(bool(found_files)))
        per_query.append(
            {
                "qid": qid,
                "gold_pages": sorted(gold_pages),
                "gold_files": sorted(gold_files),
                "top10_pages": pages,
                "top10_files_from_top100_pages": files,
                "page_hit@10": bool(found_pages),
                "page_recall@10": page_recall[-1],
                "page_precision@10": page_precision[-1],
            }
        )
    mean = lambda values: sum(values) / len(values) if values else 0.0
    return {
        "queries": len(qids),
        "ndcg@10": 100.0 * mean(ndcg),
        "page_hit@10": mean(page_hit),
        "page_recall@10": mean(page_recall),
        "page_precision@10": mean(page_precision),
        "file_metrics_by_k": {
            str(k): {
                "file_recall": mean(file_by_k[k]),
                "file_hit": mean(file_hit_by_k[k]),
            }
            for k in file_by_k
        },
        "per_query": per_query,
    }


def _paired_comparison(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    samples: int = 2000,
    seed: int = 20260729,
) -> dict[str, Any]:
    """Paired query bootstrap for page recall and derived file recall@3."""
    base_rows = {str(row["qid"]): row for row in baseline["per_query"]}
    candidate_rows = {str(row["qid"]): row for row in candidate["per_query"]}
    qids = sorted(set(base_rows) & set(candidate_rows))
    page_diffs: list[float] = []
    file_diffs: list[float] = []
    for qid in qids:
        left = base_rows[qid]
        right = candidate_rows[qid]
        page_diffs.append(float(right["page_recall@10"]) - float(left["page_recall@10"]))
        left_files = set(left["top10_files_from_top100_pages"][:3])
        right_files = set(right["top10_files_from_top100_pages"][:3])
        gold = set(left["gold_files"])
        left_recall = len(left_files & gold) / len(gold) if gold else 0.0
        right_recall = len(right_files & gold) / len(gold) if gold else 0.0
        file_diffs.append(right_recall - left_recall)

    def summarize(diffs: list[float]) -> dict[str, Any]:
        if not diffs:
            return {"delta": 0.0, "delta_pp": 0.0, "ci95": [0.0, 0.0], "p_two_sided": 1.0, "n": 0}
        rng = random.Random(seed)
        size = len(diffs)
        sampled: list[float] = []
        for _ in range(samples):
            sampled.append(sum(diffs[rng.randrange(size)] for _ in range(size)) / size)
        sampled.sort()
        low = sampled[int(0.025 * samples)]
        high = sampled[min(samples - 1, int(0.975 * samples))]
        observed = sum(diffs) / size
        if observed >= 0:
            extreme = sum(value <= 0 for value in sampled)
        else:
            extreme = sum(value >= 0 for value in sampled)
        return {
            "delta": observed,
            "delta_pp": observed * 100.0,
            "ci95": [low * 100.0, high * 100.0],
            "p_two_sided": min(1.0, 2.0 * extreme / samples),
            "n": size,
        }

    page = summarize(page_diffs)
    file = summarize(file_diffs)
    return {
        "page_recall_delta_pp": page["delta_pp"],
        "page_recall_ci95_pp": page["ci95"],
        "page_recall_p_two_sided": page["p_two_sided"],
        "file_recall_delta_pp": file["delta_pp"],
        "file_recall_ci95_pp": file["ci95"],
        "file_recall_p_two_sided": file["p_two_sided"],
        "queries": len(qids),
        "bootstrap_samples": samples,
    }


def _ndcg_at_k(ranked_pages: Sequence[str], qrel: Mapping[str, int], k: int) -> float:
    gains = {str(page): int(score) for page, score in qrel.items()}
    actual = sum(
        (2.0 ** gains[page] - 1.0) / np.log2(rank + 1)
        for rank, page in enumerate(ranked_pages[:k], 1)
        if page in gains
    )
    ideal_scores = sorted(gains.values(), reverse=True)[:k]
    ideal = sum(
        (2.0 ** score - 1.0) / np.log2(rank + 1)
        for rank, score in enumerate(ideal_scores, 1)
    )
    return actual / ideal if ideal else 0.0


def _write_run(
    path: Path,
    run: Mapping[str, list[dict[str, Any]]],
    qids: Sequence[str],
    *,
    queries: Mapping[str, str] | None = None,
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for qid in qids:
            handle.write(
                json.dumps(
                    {"qid": qid, "query": (queries or {}).get(qid, ""), "chunks": run[qid]},
                    ensure_ascii=False,
                )
                + "\n"
            )


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)


if __name__ == "__main__":
    main()
