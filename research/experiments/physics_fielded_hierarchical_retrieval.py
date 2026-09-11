"""Field-aware indexing plus a conservative hierarchical Physics experiment.

This runner transfers data-lakehouse ideas without importing a lakehouse
engine: typed block fields act as zones, several indexes propose candidates,
and a compact file stage is followed by exact page scoring.  It is deliberately
offline and deterministic.  It reuses the cached KDL + pdf-inspector output
and cached V-SPLADE vectors; no parser, OCR, VLM, LLM or cross-encoder call is
made.

The experiment has three separable layers:

* E1: flat BM25 versus a fixed field-aware page index;
* E2/E3: a file synopsis index and a top-30 lexical/visual candidate union;
* E4: compact file ranking followed by page retrieval at Kf=3, 5 and 10.

Example::

    python research/experiments/physics_fielded_hierarchical_retrieval.py
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.data_discovery.hierarchical import (  # noqa: E402
    HierarchyCorpus,
    build_bm25,
    normalise_scores,
    sort_scores,
)
from research.experiments.physics_hierarchical_retrieval import (  # noqa: E402
    _derived_metrics,
    _file_id,
    _index_scores,
    _load_run,
    _load_visual_scores,
    _paired_comparison,
    _page_vector_units,
    _path_size_bytes,
    _safe_name,
    _stratified_folds,
    _write_run,
)
from src.evaluation.pipeline_pages import canonical_doc, documents, page_blocks  # noqa: E402
from src.evaluation.benchmarks.vidore_v3 import ViDoreV3  # noqa: E402
from src.retrieval.sparse import BM25Index  # noqa: E402


DEFAULT_PARSED_RUN = ROOT / "data/output/vidore-v3-physics-kdl-pdf-inspector/32c32a45a92c45bb"
DEFAULT_BENCHMARK_ROOT = ROOT / "data/benchmark/vidore_v3"
DEFAULT_PAGE_VECTOR_DIR = ROOT / "data/output/vsplade/vidore_v3_physics_48q"
DEFAULT_QUERY_VECTOR_DIR = ROOT / "data/output/vsplade/vidore_v3_physics_english_302q"
DEFAULT_PAIR_DIR = ROOT / "data/benchmark/vidore_v3/results/physics_vsplade_bm25_fusion"
DEFAULT_HIERARCHICAL_RUN = ROOT / "data/benchmark/vidore_v3/results/physics_hierarchical_retrieval/oof_run.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "data/benchmark/vidore_v3/results/physics_fielded_hierarchical_retrieval"

EXPECTED_PAGES = 1674
EXPECTED_QUERIES = 302
PAGE_PROPOSAL_DEPTH = 30
FILE_SYNOPSIS_DEPTH = 10
PAGE_DEPTH = 100
FINAL_DEPTH = 100
VISUAL_WEIGHT = 0.30
PARENT_WEIGHT = 0.15
LEGACY_TARGET_PAGE_RECALL = 47.47

FIELD_ORDER = (
    "title",
    "heading",
    "body",
    "caption",
    "table",
    "formula",
    "figure",
    "boilerplate",
    "section_context",
)

# These are fixed semantic priors, selected before reading Physics qrels.
FIELD_WEIGHTS = {
    "title": 3.0,
    "heading": 3.0,
    "body": 1.0,
    "caption": 2.0,
    "table": 2.0,
    "formula": 2.0,
    "figure": 2.0,
    "boilerplate": 0.0,
    "section_context": 1.0,
}


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def _join(parts: Sequence[str]) -> str:
    return " ".join(part for part in (_clean(value) for value in parts) if part).strip()


def _weighted_text(record: Mapping[str, str], weights: Mapping[str, float]) -> str:
    """Build a deterministic BM25F-like text without resetting field IDF."""
    parts: list[str] = []
    for field_name in FIELD_ORDER:
        text = _clean(record.get(field_name) or "")
        repetitions = int(round(float(weights.get(field_name, 0.0))))
        if text and repetitions > 0:
            parts.extend([text] * repetitions)
    return " ".join(parts)


def _block_fields(node: Any) -> tuple[str, ...]:
    """Map normalized KDL block metadata to one or more typed fields."""
    kind = str(node.metadata.get("kind") or "text")
    block_type = str(node.metadata.get("block_type") or "").casefold()
    if kind == "heading":
        return ("title", "heading") if block_type == "title" else ("heading",)
    if kind == "caption":
        return ("caption",)
    if kind == "table":
        return ("table",)
    if kind == "formula":
        return ("formula",)
    if kind == "figure":
        return ("figure",)
    if kind in {"page_header", "page_footer", "page_number", "boilerplate"}:
        return ("boilerplate",)
    return ("body",)


def _raw_block_fields(block: Mapping[str, Any]) -> tuple[str, ...]:
    """Map raw KDL labels before the hierarchy drops boilerplate nodes."""
    raw_type = str(block.get("type") or "").casefold().replace("_", "")
    raw_label = str(block.get("raw_label") or block.get("label") or "").casefold()
    if raw_label == "title" or raw_type == "title":
        return ("title", "heading")
    if raw_type in {"sectionheader", "heading"}:
        return ("heading",)
    if raw_type in {"caption"} or raw_label == "caption":
        return ("caption",)
    if raw_type in {"table", "tablegroup", "tablerow", "tableheader", "chart"} or raw_label in {"table", "chart"}:
        return ("table",)
    if raw_type in {"equationblock", "formula", "equation"} or raw_label == "formula":
        return ("formula",)
    if raw_type in {"figure", "image", "diagram", "picture", "flowchart"} or raw_label in {"picture", "flowchart"}:
        return ("figure",)
    if raw_type in {"pageheader", "pagefooter", "pagenumber"} or raw_label in {"page-header", "page-footer", "page-number"}:
        return ("boilerplate",)
    return ("body",)


@dataclass
class FieldRecordSet:
    """Field values and inventory statistics for page and file units."""

    page_records: dict[str, dict[str, str]]
    file_records: dict[str, dict[str, str]]
    stats: dict[str, Any]


@dataclass
class FieldedBM25:
    """Small explainable BM25F-like index with per-field diagnostics."""

    unit_ids: list[str]
    weights: dict[str, float]
    indexes: dict[str, BM25Index]
    combined_index: BM25Index
    logical_text_bytes: dict[str, int]
    combined_logical_text_bytes: int

    @classmethod
    def build(
        cls,
        records: Mapping[str, Mapping[str, str]],
        *,
        fields: Sequence[str],
        weights: Mapping[str, float],
    ) -> "FieldedBM25":
        unit_ids = list(records)
        indexes: dict[str, BM25Index] = {}
        logical_text_bytes: dict[str, int] = {}
        for field_name in fields:
            values = {
                unit_id: str(records[unit_id].get(field_name) or "")
                for unit_id in unit_ids
            }
            indexes[field_name] = build_bm25(values.items())
            logical_text_bytes[field_name] = sum(
                len(value.encode("utf-8")) for value in values.values()
            )
        effective_weights = {
            field_name: float(weights.get(field_name, 0.0)) for field_name in fields
        }
        combined_values = {
            unit_id: _weighted_text(records[unit_id], effective_weights)
            for unit_id in unit_ids
        }
        return cls(
            unit_ids=unit_ids,
            weights=effective_weights,
            indexes=indexes,
            combined_index=build_bm25(combined_values.items()),
            logical_text_bytes=logical_text_bytes,
            combined_logical_text_bytes=sum(
                len(value.encode("utf-8")) for value in combined_values.values()
            ),
        )

    def score(self, query: str) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
        """Return combined BM25F-like scores and field-support explanations."""
        raw_combined = _index_scores(
            self.combined_index, query, top_k=len(self.unit_ids)
        )
        combined = {unit_id: float(raw_combined.get(unit_id, 0.0)) for unit_id in self.unit_ids}
        components: dict[str, dict[str, float]] = {
            unit_id: {} for unit_id in self.unit_ids
        }
        for field_name, index in self.indexes.items():
            raw = _index_scores(index, query, top_k=len(self.unit_ids))
            normalized = normalise_scores(raw)
            weight = self.weights.get(field_name, 0.0)
            if weight <= 0:
                continue
            for unit_id, value in normalized.items():
                contribution = weight * float(value)
                if contribution > 0:
                    components[unit_id][field_name] = round(contribution, 8)
        return combined, components


def _build_field_records(
    corpus: HierarchyCorpus,
    parsed_run: Path | str | None = None,
) -> FieldRecordSet:
    """Materialize typed page fields and bounded deterministic file synopses."""
    page_records: dict[str, dict[str, str]] = {
        page_id: {field_name: "" for field_name in FIELD_ORDER}
        for page_id in corpus.page_order
    }
    block_counts: Counter[str] = Counter()
    raw_pages: dict[str, list[dict[str, Any]]] = {}
    raw_titles: dict[str, str] = {}
    raw_block_count = 0
    if parsed_run is not None:
        for document in documents(parsed_run):
            document_meta = document.get("document") or {}
            file_id = f"physics::{canonical_doc(str(document_meta.get('file_name') or ''))}"
            raw_titles[file_id] = str(document_meta.get("title") or "")
            for page_number, blocks in page_blocks(document).items():
                page_id = f"{file_id}#page={int(page_number)}"
                raw_pages[page_id] = list(blocks)
                raw_block_count += len(blocks)
    page_field_parts: dict[str, dict[str, list[str]]] = {
        page_id: {field_name: [] for field_name in FIELD_ORDER}
        for page_id in corpus.page_order
    }
    active_sections: dict[str, str] = {}

    for page_id in corpus.page_order:
        file_id = _file_id(page_id)
        active_section = active_sections.get(file_id, "")
        raw_blocks = raw_pages.get(page_id)
        if raw_blocks is None:
            nodes = sorted(
                (node for node in corpus.blocks.values() if node.page_id == page_id),
                key=lambda node: (node.block_index if node.block_index is not None else 10**9, node.node_id),
            )
            raw_blocks = [
                {
                    "type": node.metadata.get("block_type"),
                    "text": node.text,
                    "kind": node.metadata.get("kind"),
                }
                for node in nodes
            ]
        for block in raw_blocks:
            raw_type = str(block.get("type") or block.get("kind") or "text")
            block_counts[raw_type] += 1
            text = _clean(block.get("text"))
            if not text:
                continue
            fields = _raw_block_fields(block)
            for field_name in fields:
                page_field_parts[page_id][field_name].append(text)
            if "heading" in fields:
                active_section = text[:240]
        active_sections[file_id] = active_section
        if active_section:
            page_field_parts[page_id]["section_context"].append(active_section)
        for field_name in FIELD_ORDER:
            page_records[page_id][field_name] = _join(page_field_parts[page_id][field_name])

    file_records: dict[str, dict[str, str]] = {
        file_id: {field_name: "" for field_name in FIELD_ORDER}
        for file_id in corpus.file_order
    }
    file_stats: dict[str, dict[str, Any]] = {}
    for file_id in corpus.file_order:
        page_ids = corpus.file_to_pages.get(file_id, [])
        # A KDL document.title can be a publication-level title shared by many
        # PDFs. Keep it once at the file level; repeating it on every page made
        # a common metadata phrase dominate the page index.
        file_records[file_id]["title"] = _join([
            file_id.rsplit("::", 1)[-1],
            raw_titles.get(file_id) or corpus.file_titles.get(file_id, file_id),
        ])
        for page_id in page_ids:
            fields = page_field_parts[page_id]
            for field_name in ("heading", "caption", "table", "formula", "figure", "section_context"):
                file_records[file_id][field_name] = _join(
                    [file_records[file_id][field_name], _join(fields[field_name])]
                )
            body_tokens = _join(fields["body"]).split()[:64]
            file_records[file_id]["body"] = _join(
                [file_records[file_id]["body"], " ".join(body_tokens)]
            )
            file_records[file_id]["boilerplate"] = _join(
                [file_records[file_id]["boilerplate"], _join(fields["boilerplate"])]
            )
        file_stats[file_id] = {
            "page_count": len(page_ids),
            "nonempty_fields": [
                field_name
                for field_name, value in file_records[file_id].items()
                if value
            ],
        }

    page_nonempty = {
        field_name: sum(bool(records[field_name]) for records in page_records.values())
        for field_name in FIELD_ORDER
    }
    file_nonempty = {
        field_name: sum(bool(records[field_name]) for records in file_records.values())
        for field_name in FIELD_ORDER
    }
    stats = {
        "field_order": list(FIELD_ORDER),
        "field_weights": FIELD_WEIGHTS,
        "page_count": len(page_records),
        "file_count": len(file_records),
        "block_count": raw_block_count or sum(block_counts.values()),
        "block_kind_counts": dict(sorted(block_counts.items())),
        "page_nonempty_fields": page_nonempty,
        "file_nonempty_fields": file_nonempty,
        "files": file_stats,
    }
    return FieldRecordSet(page_records=page_records, file_records=file_records, stats=stats)


@dataclass
class QuerySignals:
    flat_scores: dict[str, float]
    field_scores: dict[str, float]
    field_components: dict[str, dict[str, float]]
    field_section_scores: dict[str, float]
    field_section_components: dict[str, dict[str, float]]
    file_scores: dict[str, float]
    file_components: dict[str, dict[str, float]]
    visual_scores: dict[str, float]


def _row(node_id: str, score: float, rank: int, components: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "node_id": node_id,
        "rank": rank,
        "score": round(float(score), 8),
        "components": {
            str(key): round(float(value), 8)
            for key, value in (components or {}).items()
            if float(value) != 0.0
        },
    }


def _run_from_ranked(
    corpus: HierarchyCorpus,
    ranked: Sequence[tuple[str, float]],
) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": page_id,
            "doc_id": page_id,
            "text": corpus.pages[page_id].text,
            "score": round(float(score), 8),
            "rank": rank,
        }
        for rank, (page_id, score) in enumerate(ranked[:FINAL_DEPTH], 1)
    ]


def _file_pool(values: Mapping[str, list[float]], mode: str) -> dict[str, float]:
    output: dict[str, float] = {}
    for file_id, scores in values.items():
        ordered = sorted((max(0.0, float(value)) for value in scores), reverse=True)
        top = ordered[:2]
        if mode == "max":
            output[file_id] = ordered[0] if ordered else 0.0
        elif mode == "sum_top2":
            output[file_id] = sum(top)
        elif mode == "mean_top2":
            output[file_id] = sum(top) / len(top) if top else 0.0
        else:
            raise ValueError(f"Unknown file pool: {mode!r}")
    return output


def _ranked_file_rows(scores: Mapping[str, float]) -> list[dict[str, Any]]:
    return [_row(file_id, score, rank) for rank, (file_id, score) in enumerate(sort_scores(scores), 1)]


def _retrieve_flat(
    corpus: HierarchyCorpus,
    scores: Mapping[str, float],
    components: Mapping[str, Mapping[str, float]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    normalized = normalise_scores(scores)
    ranked = sort_scores(normalized)
    page_candidates = ranked[:PAGE_DEPTH]
    run = _run_from_ranked(corpus, ranked)
    trace = {
        "file_proposal_ranked": [],
        "file_candidates": [],
        "selected_files": [],
        "page_candidates": [
            _row(page_id, score, rank, components.get(page_id, {}))
            for rank, (page_id, score) in enumerate(page_candidates, 1)
        ],
        "final_pages": [page_id for page_id, _ in ranked[:FINAL_DEPTH]],
        "top10_pages": [page_id for page_id, _ in ranked[:10]],
        "proposal_file_ids": [],
        "scanned_page_ids": list(corpus.page_order),
        "counts": {
            "proposal_files": 0,
            "selected_files": len(corpus.file_order),
            "pages_scanned": len(corpus.page_order),
            "page_candidates": len(page_candidates),
        },
    }
    return run, trace


def _retrieve_hierarchical(
    corpus: HierarchyCorpus,
    signals: QuerySignals,
    *,
    strategy: str,
    k_files: int,
    visual_enabled: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run proposal -> compact file ranking -> page retrieval."""
    started = time.perf_counter()
    # The typed index is a coarse proposal signal.  Keep the original
    # full-text BM25 for exact page verification: data skipping should narrow
    # the scan, not replace the strongest page-level evidence signal.
    proposal_lexical_raw = signals.field_section_scores
    proposal_lexical_norm = normalise_scores(proposal_lexical_raw)
    verification_lexical_norm = normalise_scores(signals.flat_scores)
    visual_raw = signals.visual_scores if visual_enabled else {}
    visual_norm = normalise_scores(visual_raw)
    page_base = {
        page_id: (1.0 - VISUAL_WEIGHT) * verification_lexical_norm.get(page_id, 0.0)
        + (VISUAL_WEIGHT * visual_norm.get(page_id, 0.0) if visual_enabled else 0.0)
        for page_id in corpus.page_order
    }

    lexical_ranked = sort_scores(proposal_lexical_raw)[:PAGE_PROPOSAL_DEPTH]
    flat_ranked = sort_scores(signals.flat_scores)[:PAGE_PROPOSAL_DEPTH]
    visual_ranked = sort_scores(visual_raw)[:PAGE_PROPOSAL_DEPTH]
    synopsis_ranked = sort_scores(signals.file_scores)[:FILE_SYNOPSIS_DEPTH]
    page_proposal_ids = {
        page_id for page_id, _ in lexical_ranked
    } | {page_id for page_id, _ in flat_ranked} | {
        page_id for page_id, _ in visual_ranked
    }
    candidate_files = {
        _file_id(page_id) for page_id in page_proposal_ids
    }
    candidate_files.update(file_id for file_id, _ in synopsis_ranked)
    if not candidate_files:
        candidate_files = set(corpus.file_order[: max(k_files, FILE_SYNOPSIS_DEPTH)])

    lexical_file_max: dict[str, list[float]] = defaultdict(list)
    flat_file_max: dict[str, list[float]] = defaultdict(list)
    visual_file_max: dict[str, list[float]] = defaultdict(list)
    evidence_values: dict[str, list[float]] = defaultdict(list)
    for page_id in page_proposal_ids:
        file_id = _file_id(page_id)
        evidence_values[file_id].append(page_base.get(page_id, 0.0))
    for page_id, _ in lexical_ranked:
        lexical_file_max[_file_id(page_id)].append(proposal_lexical_norm.get(page_id, 0.0))
    for page_id, _ in flat_ranked:
        flat_file_max[_file_id(page_id)].append(verification_lexical_norm.get(page_id, 0.0))
    for page_id, _ in visual_ranked:
        visual_file_max[_file_id(page_id)].append(visual_norm.get(page_id, 0.0))

    evidence_sum = _file_pool(evidence_values, "sum_top2")
    evidence_mean = _file_pool(evidence_values, "mean_top2")
    evidence_max = _file_pool(evidence_values, "max")
    synopsis_norm = normalise_scores(signals.file_scores)
    evidence_sum_norm = normalise_scores(evidence_sum)
    proposal_scores = {
        file_id: max(
            synopsis_norm.get(file_id, 0.0),
            max(lexical_file_max.get(file_id, [0.0])),
            max(flat_file_max.get(file_id, [0.0])),
            max(visual_file_max.get(file_id, [0.0])) if visual_enabled else 0.0,
        )
        for file_id in candidate_files
    }
    if strategy == "synopsis":
        compact_raw = {file_id: synopsis_norm.get(file_id, 0.0) for file_id in candidate_files}
    elif strategy == "sum_top2":
        compact_raw = {file_id: evidence_sum.get(file_id, 0.0) for file_id in candidate_files}
    elif strategy == "mean_synopsis_sum_top2":
        # The plan's third strategy is the fixed mean of the first two
        # strategies: file synopsis score and the sum of the two strongest
        # page evidence scores.  Normalize each source independently before
        # averaging because they are on different BM25/pooled-score scales.
        # This is deliberately 0.50/0.50, with no qrel-fitted tuning.
        compact_raw = {
            file_id: 0.50 * synopsis_norm.get(file_id, 0.0)
            + 0.50 * evidence_sum_norm.get(file_id, 0.0)
            for file_id in candidate_files
        }
    else:
        raise ValueError(f"Unknown file selection strategy: {strategy!r}")

    compact_ranked = sort_scores(compact_raw)
    selected_files = {file_id for file_id, _ in compact_ranked[:k_files]}
    compact_norm = normalise_scores(compact_raw)
    selected_page_ids = [
        page_id for page_id in corpus.page_order if _file_id(page_id) in selected_files
    ]
    parent_scores = {
        file_id: compact_norm.get(file_id, 0.0) for file_id in selected_files
    }
    parent_norm = normalise_scores(parent_scores)
    page_scores = {
        page_id: (1.0 - PARENT_WEIGHT) * page_base.get(page_id, 0.0)
        + PARENT_WEIGHT * parent_norm.get(_file_id(page_id), 0.0)
        for page_id in selected_page_ids
    }
    page_ranked = sort_scores(page_scores)
    page_candidates = page_ranked[:PAGE_DEPTH]
    run = _run_from_ranked(corpus, page_ranked)
    page_components = {
        page_id: {
            "flat_bm25_verification": verification_lexical_norm.get(page_id, 0.0),
            "field_lexical_proposal": proposal_lexical_norm.get(page_id, 0.0),
            "vsplade_page": visual_norm.get(page_id, 0.0),
            "parent_file": parent_norm.get(_file_id(page_id), 0.0),
            **signals.field_section_components.get(page_id, {}),
        }
        for page_id, _ in page_candidates
    }
    trace = {
        "file_proposal_ranked": _ranked_file_rows(proposal_scores),
        "file_candidates": _ranked_file_rows(compact_raw),
        "selected_files": [file_id for file_id, _ in compact_ranked[:k_files]],
        "page_candidates": [
            _row(page_id, score, rank, page_components.get(page_id, {}))
            for rank, (page_id, score) in enumerate(page_candidates, 1)
        ],
        "final_pages": [page_id for page_id, _ in page_ranked[:FINAL_DEPTH]],
        "top10_pages": [page_id for page_id, _ in page_ranked[:10]],
        "proposal_file_ids": sorted(candidate_files),
        "lexical_page_ids": [page_id for page_id, _ in lexical_ranked],
        "flat_page_ids": [page_id for page_id, _ in flat_ranked],
        "visual_page_ids": [page_id for page_id, _ in visual_ranked],
        "synopsis_file_ids": [file_id for file_id, _ in synopsis_ranked],
        "file_pool_components": {
            file_id: {
                "file_synopsis": synopsis_norm.get(file_id, 0.0),
                "evidence_max": evidence_max.get(file_id, 0.0),
                "evidence_sum_top2": evidence_sum.get(file_id, 0.0),
                "evidence_mean_top2": evidence_mean.get(file_id, 0.0),
                "proposal_score": proposal_scores.get(file_id, 0.0),
            }
            for file_id in sorted(candidate_files)
        },
        "scanned_page_ids": selected_page_ids,
        "counts": {
            "proposal_files": len(candidate_files),
            "selected_files": len(selected_files),
            "pages_scanned": len(selected_page_ids),
            "page_candidates": len(page_candidates),
            "lexical_page_proposal": len(lexical_ranked),
            "flat_page_proposal": len(flat_ranked),
            "visual_page_proposal": len(visual_ranked),
            "synopsis_file_proposal": len(synopsis_ranked),
        },
        "timing_seconds": {"total": round(time.perf_counter() - started, 6)},
    }
    return run, trace


def _metric_for_qids(metric: Mapping[str, Any], qids: set[str], file_k: int) -> tuple[float, float]:
    rows = [row for row in metric["per_query"] if row["qid"] in qids]
    if not rows:
        return 0.0, 0.0
    page = sum(float(row["page_recall@10"]) for row in rows) / len(rows)
    file_values = []
    for row in rows:
        gold = set(row["gold_files"])
        found = set(row["top10_files_from_top100_pages"][:file_k])
        file_values.append(len(found & gold) / len(gold) if gold else 0.0)
    return page, sum(file_values) / len(file_values)


def _oof_select(
    qids: Sequence[str],
    questions: Mapping[str, Any],
    qrels: Mapping[str, Mapping[str, int]],
    metrics: Mapping[str, Mapping[str, Any]],
    runs: Mapping[str, Mapping[str, list[dict[str, Any]]]],
    traces: Mapping[str, Mapping[str, dict[str, Any]]],
    *,
    file_k: int,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    folds = _stratified_folds(qids, questions, qrels)
    all_qids = set(qids)
    oof_run: dict[str, list[dict[str, Any]]] = {}
    oof_trace: dict[str, dict[str, Any]] = {}
    selected: list[dict[str, Any]] = []
    for fold_index, heldout in enumerate(folds):
        heldout_set = set(heldout)
        train_set = all_qids - heldout_set
        ranking = []
        for name, metric in metrics.items():
            page, file = _metric_for_qids(metric, train_set, file_k)
            ranking.append((page, file, name))
        _, _, winner = max(ranking, key=lambda item: (item[0], item[1], item[2]))
        for qid in heldout:
            oof_run[qid] = runs[winner][qid]
            oof_trace[qid] = traces[winner][qid]
        selected.append(
            {
                "fold": fold_index,
                "heldout_qids": heldout,
                "selected_method": winner,
                "train_page_recall@10": _metric_for_qids(metrics[winner], train_set, file_k)[0],
                "train_file_recall": _metric_for_qids(metrics[winner], train_set, file_k)[1],
            }
        )
    return (
        {
            "folds": selected,
            "selected_method_counts": {
                name: sum(row["selected_method"] == name for row in selected)
                for name in metrics
            },
        },
        {"run": oof_run, "traces": oof_trace},
        {"metrics": _derived_metrics(oof_run, qids, qrels)},
    )


def _stage_metrics(
    traces: Mapping[str, dict[str, Any]],
    qids: Sequence[str],
    qrels: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    proposal_ks = (1, 3, 5, 10)
    selected_ks = (1, 3, 5, 10)
    proposal_ranked: dict[int, list[float]] = {k: [] for k in proposal_ks}
    proposal_coverage: list[float] = []
    selected_ranked: dict[int, list[float]] = {k: [] for k in selected_ks}
    page_candidate: list[float] = []
    scanned_files: list[float] = []
    scanned_pages: list[float] = []
    for qid in qids:
        trace = traces[qid]
        gold_pages = set(qrels.get(qid, {}))
        gold_files = {_file_id(page_id) for page_id in gold_pages}
        proposal_ids = set(trace.get("proposal_file_ids", []))
        proposal_rank = [str(row["node_id"]) for row in trace.get("file_proposal_ranked", [])]
        selected_rank = [str(row["node_id"]) for row in trace.get("file_candidates", [])]
        proposal_coverage.append(
            len(proposal_ids & gold_files) / len(gold_files) if gold_files else 0.0
        )
        for k in proposal_ks:
            proposal_ranked[k].append(
                len(set(proposal_rank[:k]) & gold_files) / len(gold_files) if gold_files else 0.0
            )
        for k in selected_ks:
            selected_ranked[k].append(
                len(set(selected_rank[:k]) & gold_files) / len(gold_files) if gold_files else 0.0
            )
        page_ids = {
            str(row["node_id"]) for row in trace.get("page_candidates", [])
        }
        page_candidate.append(
            len(page_ids & gold_pages) / len(gold_pages) if gold_pages else 0.0
        )
        scanned_files.append(float(trace.get("counts", {}).get("selected_files", 0)))
        scanned_pages.append(float(trace.get("counts", {}).get("pages_scanned", 0)))

    mean = lambda values: round(sum(values) / len(values), 6) if values else 0.0
    return {
        "proposal_file_recall": {f"@{k}": mean(values) for k, values in proposal_ranked.items()},
        "proposal_file_coverage": mean(proposal_coverage),
        "selected_file_recall": {f"@{k}": mean(values) for k, values in selected_ranked.items()},
        "page_candidate_recall": mean(page_candidate),
        "mean_selected_files": mean(scanned_files),
        "mean_pages_scanned": mean(scanned_pages),
        "queries": len(qids),
    }


def _file_synopsis_metrics(
    qids: Sequence[str],
    qrels: Mapping[str, Mapping[str, int]],
    states: Mapping[str, QuerySignals],
) -> dict[str, Any]:
    values: dict[int, list[float]] = {1: [], 3: [], 5: [], 10: []}
    per_query: list[dict[str, Any]] = []
    for qid in qids:
        gold = {_file_id(page_id) for page_id in qrels.get(qid, {})}
        ranked = [file_id for file_id, _ in sort_scores(states[qid].file_scores)]
        for k in values:
            values[k].append(len(set(ranked[:k]) & gold) / len(gold) if gold else 0.0)
        per_query.append(
            {
                "qid": qid,
                "gold_files": sorted(gold),
                "ranked_files": ranked,
                "file_recall": {
                    f"@{k}": values[k][-1] for k in values
                },
            }
        )
    return {
        "file_recall": {
            f"@{k}": sum(rows) / len(rows) if rows else 0.0
            for k, rows in values.items()
        },
        "per_query": per_query,
    }


def _direct_file_index_summary(
    qids: Sequence[str],
    qrels: Mapping[str, Mapping[str, int]],
    scores_by_qid: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """Summarize a direct file index without deriving files from page runs."""
    values: dict[int, list[float]] = {1: [], 3: [], 5: [], 10: []}
    ranked_by_qid: dict[str, list[str]] = {}
    for qid in qids:
        gold = {_file_id(page_id) for page_id in qrels.get(qid, {})}
        ranked = [file_id for file_id, _ in sort_scores(scores_by_qid.get(qid, {}))]
        ranked_by_qid[qid] = ranked
        for k in values:
            values[k].append(
                len(set(ranked[:k]) & gold) / len(gold) if gold else 0.0
            )
    return {
        "file_recall": {
            f"@{k}": sum(rows) / len(rows) if rows else 0.0
            for k, rows in values.items()
        },
        "ranked_files_by_qid": ranked_by_qid,
    }


def _error_rows(
    qids: Sequence[str],
    qrels: Mapping[str, Mapping[str, int]],
    traces: Mapping[str, dict[str, Any]],
    states: Mapping[str, QuerySignals],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for qid in qids:
        trace = traces[qid]
        state = states[qid]
        proposal = set(trace.get("proposal_file_ids", []))
        selected = set(trace.get("selected_files", []))
        page_candidates = {
            str(row["node_id"]) for row in trace.get("page_candidates", [])
        }
        top10 = set(trace.get("top10_pages", []))
        lexical_ids = set(trace.get("lexical_page_ids", []))
        flat_ids = set(trace.get("flat_page_ids", []))
        visual_ids = set(trace.get("visual_page_ids", []))
        synopsis_ids = set(trace.get("synopsis_file_ids", []))
        gold_by_file: dict[str, set[str]] = defaultdict(set)
        for page_id in qrels.get(qid, {}):
            gold_by_file[_file_id(page_id)].add(page_id)
        for file_id, gold_pages in sorted(gold_by_file.items()):
            if file_id not in proposal:
                category = "absent_from_proposal"
            elif file_id not in selected:
                category = "proposal_present_but_compact_ranked_out"
            elif not (gold_pages & page_candidates):
                category = "selected_file_but_page_candidate_miss"
            elif not (gold_pages & top10):
                category = "page_candidate_but_page_rank_miss"
            else:
                category = "recovered"
            matched_fields = set()
            for page_id in gold_pages:
                matched_fields.update(state.field_section_components.get(page_id, {}))
            source = []
            if any(page_id in lexical_ids for page_id in gold_pages):
                source.append("field_page_top30")
            if any(page_id in flat_ids for page_id in gold_pages):
                source.append("flat_bm25_page_top30")
            if any(page_id in visual_ids for page_id in gold_pages):
                source.append("visual_page_top30")
            if file_id in synopsis_ids:
                source.append("file_synopsis_top10")
            row = {
                "qid": qid,
                "gold_file": file_id,
                "gold_pages": sorted(gold_pages),
                "category": category,
                "matched_fields": sorted(matched_fields),
                "proposal_sources": source,
                "file_rank": next(
                    (int(item["rank"]) for item in trace.get("file_candidates", []) if item["node_id"] == file_id),
                    None,
                ),
                "proposal_file_rank": next(
                    (int(item["rank"]) for item in trace.get("file_proposal_ranked", []) if item["node_id"] == file_id),
                    None,
                ),
                "selected": file_id in selected,
                "page_candidate_count": len(page_candidates),
                "selected_file_count": len(selected),
            }
            rows.append(row)
            counts[category] += 1
    return rows, dict(sorted(counts.items()))


def _stage_for_flat() -> dict[str, Any]:
    return {
        "proposal_file_recall": {},
        "proposal_file_coverage": None,
        "selected_file_recall": {},
        "page_candidate_recall": None,
        "mean_selected_files": None,
        "mean_pages_scanned": EXPECTED_PAGES,
        "queries": EXPECTED_QUERIES,
    }


def _compact_trace(trace: Mapping[str, Any]) -> dict[str, Any]:
    """Keep per-query recall diagnostics small enough for routine inspection."""
    return {
        "proposal_file_ids": list(trace.get("proposal_file_ids", [])),
        "selected_files": list(trace.get("selected_files", [])),
        "file_proposal_ranked": list(trace.get("file_proposal_ranked", [])[:20]),
        "file_candidates": list(trace.get("file_candidates", [])[:20]),
        "page_candidate_ids": [
            str(row["node_id"]) for row in trace.get("page_candidates", [])
        ],
        "top10_pages": list(trace.get("top10_pages", [])),
        "counts": dict(trace.get("counts", {})),
        "timing_seconds": dict(trace.get("timing_seconds", {})),
    }


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Physics field-aware hierarchical retrieval",
        "",
        "Offline KDL/PDF-inspector field indexing plus file proposal and page verification.",
        "",
        "## Protocol",
        "",
        f"- Corpus: **{report['queries']}** French queries, **{report['pages']}** pages, **{report['files']}** files.",
        f"- Field records: **{report['index']['block_count']}** blocks; no parser/OCR/model call is made.",
        "- Proposal: top-30 field-aware lexical pages + top-30 flat BM25 pages + top-30 cached V-SPLADE pages + top-10 file synopsis files.",
        "- OOF selection: existing deterministic 5-fold query-level split; selection is separate for Kf=3, 5 and 10.",
        f"- Strict target: page recall@10 > **{LEGACY_TARGET_PAGE_RECALL:.2f}%** at Kf=3; current hierarchy reference is **44.28%**.",
        "",
        "## Full-set screening",
        "",
        "| Method | Kf | nDCG@10 | Page recall@10 | File recall@3 | Proposal coverage | Page candidate recall | Mean files | Mean pages |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, method in report["methods"].items():
        metric = method["retrieval_metrics"]
        stage = method.get("stage_metrics") or {}
        file_recall = metric["file_metrics_by_k"]["3"]["file_recall"]
        proposal_coverage = stage.get("proposal_file_coverage")
        lines.append(
            f"| {name} | {method.get('k_files', '-')} | {metric['ndcg@10']:.2f} | "
            f"{metric['page_recall@10']:.2%} | {file_recall:.2%} | "
            f"{proposal_coverage:.2%} | " if proposal_coverage is not None else
            f"| {name} | {method.get('k_files', '-')} | {metric['ndcg@10']:.2f} | "
            f"{metric['page_recall@10']:.2%} | {file_recall:.2%} | - | "
        )
        # The conditional expression above keeps the table readable while the
        # remaining columns are appended uniformly below.
        lines[-1] += (
            f"{stage.get('page_candidate_recall', '-') if stage.get('page_candidate_recall') is not None else '-'} | "
            f"{stage.get('mean_selected_files', '-') if stage.get('mean_selected_files') is not None else '-'} | "
            f"{stage.get('mean_pages_scanned', '-') if stage.get('mean_pages_scanned') is not None else '-'} |"
        )
    lines += ["", "## Out-of-fold results", ""]
    for key, payload in report["cv"].items():
        metric = payload["retrieval_metrics"]
        stage = payload["stage_metrics"]
        lines.extend([
            f"### {key}",
            "",
            f"- Selected methods: `{payload['selected_method_counts']}`.",
            f"- Page recall@10: **{metric['page_recall@10']:.2%}**; nDCG@10: **{metric['ndcg@10']:.2f}**.",
            f"- File recall@3: **{metric['file_metrics_by_k']['3']['file_recall']:.2%}**.",
        ])
        if stage.get("selected_file_recall"):
            lines.extend([
                f"- Selected-file recall@10: **{stage['selected_file_recall']['@10']:.2%}**.",
                f"- Proposal coverage: **{stage['proposal_file_coverage']:.2%}**; page candidate recall: **{stage['page_candidate_recall']:.2%}**.",
                f"- Mean selected files/pages: **{stage['mean_selected_files']:.2f} / {stage['mean_pages_scanned']:.2f}**.",
            ])
        lines.extend([
            f"- Delta versus legacy target: **{metric['page_recall@10'] * 100.0 - LEGACY_TARGET_PAGE_RECALL:+.2f} pp**.",
            "",
        ])
    lines += [
        "## Direct file-index label ablation",
        "",
        "This is measured before page-derived file metrics: the same French query is scored against a labeled field synopsis and a flat all-text file index.",
        "",
        "| File index | Recall@1 | Recall@3 | Recall@5 | Recall@10 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, payload in (
        ("field-labelled synopsis", report["file_index_ablation"]["field_labelled"]),
        ("all-text flat", report["file_index_ablation"]["all_text_flat"]),
    ):
        recall = payload["file_recall"]
        lines.append(
            f"| {name} | {recall['@1']:.2%} | {recall['@3']:.2%} | "
            f"{recall['@5']:.2%} | {recall['@10']:.2%} |"
        )
    ablation = report["file_index_ablation"]
    lines += [
        "",
        f"- Top-3 file ranking changed for **{ablation['top3_rank_changed_queries']} / {report['queries']}** queries; mean top-3 overlap was **{ablation['mean_top3_file_overlap']:.2f} files**.",
        "",
        "## File miss taxonomy for primary Kf=3 OOF",
        "",
        f"`{report['primary_error_counts']}`",
        "",
        "## Interpretation notes",
        "",
        "- The field weights are fixed semantic priors, not qrel-fitted parameters.",
        "- KDL labels are preserved; document-wide section context is flat because the cached section_hierarchy field is empty.",
        "- V-SPLADE query vectors are cached English translations evaluated against French qrels; this language asymmetry remains a caveat.",
        "- Full per-query traces are in `per_query.jsonl`; primary error rows are in `error_analysis.jsonl`.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parsed-run", type=Path, default=DEFAULT_PARSED_RUN)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--page-vector-dir", type=Path, default=DEFAULT_PAGE_VECTOR_DIR)
    parser.add_argument("--query-vector-dir", type=Path, default=DEFAULT_QUERY_VECTOR_DIR)
    parser.add_argument("--pair-dir", type=Path, default=DEFAULT_PAIR_DIR)
    parser.add_argument("--hierarchical-run", type=Path, default=DEFAULT_HIERARCHICAL_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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
    if len(page_ids) != EXPECTED_PAGES:
        raise RuntimeError(f"Expected {EXPECTED_PAGES} Physics pages, got {len(page_ids)}")
    if len(qids) != EXPECTED_QUERIES:
        raise RuntimeError(f"Expected {EXPECTED_QUERIES} French Physics queries, got {len(qids)}")

    index_started = time.perf_counter()
    corpus = HierarchyCorpus.from_parsed_run(args.parsed_run, subset="physics", page_ids=page_ids)
    if len(corpus.file_order) != 42 or len(corpus.page_order) != EXPECTED_PAGES:
        raise RuntimeError(
            f"Parsed inventory mismatch: files={len(corpus.file_order)}, pages={len(corpus.page_order)}"
        )
    field_records = _build_field_records(corpus, args.parsed_run)
    page_fields = ("title", "heading", "body", "caption", "table", "formula", "figure", "boilerplate")
    page_fields_with_context = (*page_fields, "section_context")
    page_field_index = FieldedBM25.build(
        field_records.page_records,
        fields=page_fields,
        weights=FIELD_WEIGHTS,
    )
    page_field_context_index = FieldedBM25.build(
        field_records.page_records,
        fields=page_fields_with_context,
        weights=FIELD_WEIGHTS,
    )
    file_index = FieldedBM25.build(
        field_records.file_records,
        fields=page_fields_with_context,
        weights=FIELD_WEIGHTS,
    )
    all_text_file_index = build_bm25(corpus.file_texts("all_text").items())
    flat_index = build_bm25(
        (page_id, corpus.pages[page_id].text) for page_id in corpus.page_order
    )
    index_seconds = time.perf_counter() - index_started

    visual_by_qid, visual_meta = _load_visual_scores(
        args.page_vector_dir, args.query_vector_dir, page_ids, qids
    )
    query_states: dict[str, QuerySignals] = {}
    all_text_file_scores_by_qid: dict[str, dict[str, float]] = {}
    signal_started = time.perf_counter()
    for question in questions_list:
        flat_scores = _index_scores(flat_index, question.query, top_k=len(page_ids))
        field_scores, field_components = page_field_index.score(question.query)
        field_section_scores, field_section_components = page_field_context_index.score(question.query)
        file_scores, file_components = file_index.score(question.query)
        all_text_file_scores_by_qid[question.qid] = _index_scores(
            all_text_file_index,
            question.query,
            top_k=len(corpus.file_order),
        )
        query_states[question.qid] = QuerySignals(
            flat_scores=flat_scores,
            field_scores=field_scores,
            field_components=field_components,
            field_section_scores=field_section_scores,
            field_section_components=field_section_components,
            file_scores=file_scores,
            file_components=file_components,
            visual_scores=visual_by_qid[question.qid],
        )

    methods: dict[str, dict[str, Any]] = {}
    runs: dict[str, dict[str, list[dict[str, Any]]]] = {}
    traces: dict[str, dict[str, dict[str, Any]]] = {}

    def add_method(name: str, run_by_qid: dict[str, list[dict[str, Any]]], trace_by_qid: dict[str, dict[str, Any]], *, kind: str, k_files: int | None) -> None:
        metric = _derived_metrics(run_by_qid, qids, qrels)
        stage = _stage_metrics(trace_by_qid, qids, qrels) if kind == "hierarchical" else _stage_for_flat()
        methods[name] = {
            "kind": kind,
            "k_files": k_files,
            "retrieval_metrics": metric,
            "stage_metrics": stage,
            "query_latency_seconds": round(
                sum(float(trace.get("timing_seconds", {}).get("total", 0.0)) for trace in trace_by_qid.values()) / len(qids),
                6,
            ),
        }
        runs[name] = run_by_qid
        traces[name] = trace_by_qid

    for name, signal_key in (
        ("flat-bm25", "flat_scores"),
        ("field-bm25", "field_scores"),
        ("field-section-bm25", "field_section_scores"),
    ):
        run_by_qid: dict[str, list[dict[str, Any]]] = {}
        trace_by_qid: dict[str, dict[str, Any]] = {}
        for question in questions_list:
            state = query_states[question.qid]
            scores = getattr(state, signal_key)
            components = (
                state.field_components if signal_key == "field_scores"
                else state.field_section_components if signal_key == "field_section_scores"
                else {}
            )
            run_by_qid[question.qid], trace_by_qid[question.qid] = _retrieve_flat(corpus, scores, components)
        add_method(name, run_by_qid, trace_by_qid, kind="flat", k_files=None)

    hierarchy_names_by_k: dict[int, list[str]] = {3: [], 5: [], 10: []}
    for strategy in ("synopsis", "sum_top2", "mean_synopsis_sum_top2"):
        for k_files in (3, 5, 10):
            name = f"hier-field-section-{strategy}-kf{k_files}-fusion"
            hierarchy_names_by_k[k_files].append(name)
            run_by_qid = {}
            trace_by_qid = {}
            for question in questions_list:
                run_by_qid[question.qid], trace_by_qid[question.qid] = _retrieve_hierarchical(
                    corpus,
                    query_states[question.qid],
                    strategy=strategy,
                    k_files=k_files,
                )
            add_method(name, run_by_qid, trace_by_qid, kind="hierarchical", k_files=k_files)

    # One fixed visual ablation isolates whether visual proposal/ranking adds
    # anything to the new indexing and hierarchy.
    ablation_name = "hier-field-section-sum_top2-kf3-lexical-only"
    ablation_runs: dict[str, list[dict[str, Any]]] = {}
    ablation_traces: dict[str, dict[str, Any]] = {}
    for question in questions_list:
        ablation_runs[question.qid], ablation_traces[question.qid] = _retrieve_hierarchical(
            corpus,
            query_states[question.qid],
            strategy="sum_top2",
            k_files=3,
            visual_enabled=False,
        )
    add_method(ablation_name, ablation_runs, ablation_traces, kind="hierarchical", k_files=3)

    baseline_path = args.pair_dir / "bm25_french_bm25-french_vs-english.jsonl"
    baseline_run = _load_run(baseline_path)
    if set(baseline_run) != set(qids):
        raise RuntimeError("Cached BM25 baseline does not contain exactly the 302 Physics qids")
    baseline_metric = _derived_metrics(baseline_run, qids, qrels)
    if not args.hierarchical_run.is_file():
        raise FileNotFoundError(f"Current hierarchical reference not found: {args.hierarchical_run}")
    hierarchical_metric = _derived_metrics(_load_run(args.hierarchical_run), qids, qrels)
    references = {
        "cached_pdf_inspector_bm25": baseline_metric,
        "current_hierarchical_oof": hierarchical_metric,
        "legacy_target_page_recall@10_percent": LEGACY_TARGET_PAGE_RECALL,
    }
    for name, method in methods.items():
        method["comparison_to_cached_bm25"] = _paired_comparison(baseline_metric, method["retrieval_metrics"])
        method["comparison_to_current_hierarchical"] = _paired_comparison(hierarchical_metric, method["retrieval_metrics"])
        method["page_recall_delta_to_legacy_target_pp"] = method["retrieval_metrics"]["page_recall@10"] * 100.0 - LEGACY_TARGET_PAGE_RECALL

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = args.output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    for name, run in runs.items():
        _write_run(
            runs_dir / f"{_safe_name(name)}.jsonl",
            run,
            qids,
            queries={question.qid: question.query for question in questions_list},
        )

    flat_oof_meta, flat_oof_bundle, flat_oof_result = _oof_select(
        qids,
        questions,
        qrels,
        {name: methods[name]["retrieval_metrics"] for name in ("flat-bm25", "field-bm25", "field-section-bm25")},
        runs,
        traces,
        file_k=3,
    )
    cv: dict[str, Any] = {
        "flat_e1": {
            **flat_oof_meta,
            "retrieval_metrics": flat_oof_result["metrics"],
            "stage_metrics": _stage_for_flat(),
            "comparison_to_cached_bm25": _paired_comparison(baseline_metric, flat_oof_result["metrics"]),
            "comparison_to_current_hierarchical": _paired_comparison(hierarchical_metric, flat_oof_result["metrics"]),
        }
    }
    oof_bundles: dict[str, tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]] = {
        "flat_e1": (flat_oof_bundle["run"], flat_oof_bundle["traces"])
    }
    for k_files, names in hierarchy_names_by_k.items():
        meta, bundle, result = _oof_select(
            qids,
            questions,
            qrels,
            {name: methods[name]["retrieval_metrics"] for name in names},
            runs,
            traces,
            file_k=k_files,
        )
        metric = result["metrics"]
        stage = _stage_metrics(bundle["traces"], qids, qrels)
        cv[f"hierarchical_kf{k_files}"] = {
            **meta,
            "retrieval_metrics": metric,
            "stage_metrics": stage,
            "comparison_to_cached_bm25": _paired_comparison(baseline_metric, metric),
            "comparison_to_current_hierarchical": _paired_comparison(hierarchical_metric, metric),
            "page_recall_delta_to_legacy_target_pp": metric["page_recall@10"] * 100.0 - LEGACY_TARGET_PAGE_RECALL,
        }
        oof_bundles[f"hierarchical_kf{k_files}"] = (bundle["run"], bundle["traces"])

    primary_run, primary_traces = oof_bundles["hierarchical_kf3"]
    error_rows, error_counts = _error_rows(qids, qrels, primary_traces, query_states)

    index_manifest = {
        **field_records.stats,
        "page_index_fields": page_field_index.logical_text_bytes,
        "page_index_fields_with_context": page_field_context_index.logical_text_bytes,
        "file_index_fields": file_index.logical_text_bytes,
        "flat_page_logical_text_bytes": sum(
            len(corpus.pages[page_id].text.encode("utf-8")) for page_id in corpus.page_order
        ),
        "index_build_seconds": round(index_seconds, 6),
        "query_signal_build_seconds": round(time.perf_counter() - signal_started, 6),
        "cached_visual_artifact_bytes": _path_size_bytes(args.page_vector_dir),
    }
    synopsis_metrics = _file_synopsis_metrics(qids, qrels, query_states)
    labelled_file_summary = _direct_file_index_summary(
        qids,
        qrels,
        {qid: state.file_scores for qid, state in query_states.items()},
    )
    all_text_file_summary = _direct_file_index_summary(
        qids,
        qrels,
        all_text_file_scores_by_qid,
    )
    labelled_ranked = labelled_file_summary.pop("ranked_files_by_qid")
    all_text_ranked = all_text_file_summary.pop("ranked_files_by_qid")
    top3_changed = sum(
        labelled_ranked.get(qid, [])[:3] != all_text_ranked.get(qid, [])[:3]
        for qid in qids
    )
    top3_overlap = sum(
        len(
            set(labelled_ranked.get(qid, [])[:3])
            & set(all_text_ranked.get(qid, [])[:3])
        )
        for qid in qids
    ) / len(qids)
    report: dict[str, Any] = {
        "dataset": "vidore_v3/physics",
        "evaluation_language": "french",
        "queries": len(qids),
        "pages": len(page_ids),
        "files": len(corpus.file_order),
        "references": references,
        "index": index_manifest,
        "file_synopsis": synopsis_metrics,
        "file_index_ablation": {
            "field_labelled": labelled_file_summary,
            "all_text_flat": all_text_file_summary,
            "top3_rank_changed_queries": top3_changed,
            "mean_top3_file_overlap": round(top3_overlap, 6),
        },
        "methods": methods,
        "cv": cv,
        "primary_error_counts": error_counts,
        "sources": {
            "parsed_run": str(args.parsed_run),
            "baseline_run": str(baseline_path),
            "hierarchical_reference_run": str(args.hierarchical_run),
            **visual_meta,
        },
        "fixed_policy": {
            "field_weights": FIELD_WEIGHTS,
            "page_proposal_depth": PAGE_PROPOSAL_DEPTH,
            "file_synopsis_depth": FILE_SYNOPSIS_DEPTH,
            "visual_weight": VISUAL_WEIGHT,
            "parent_weight": PARENT_WEIGHT,
            "no_qrels_used_for_index_build": True,
            "no_new_model_or_parser": True,
        },
        "timing_seconds": {
            "total": round(time.perf_counter() - started, 6),
            "index_build": round(index_seconds, 6),
            "signal_build": round(time.perf_counter() - signal_started, 6),
        },
        "notes": [
            "The section_hierarchy cache is empty; section_context is a document-order flat nearest-heading field.",
            "Boilerplate is retained in the schema but has fixed weight zero in the scored fields.",
            "File proposal coverage and compact file selection are reported separately.",
            "V-SPLADE query vectors are cached English translations evaluated against French qrels.",
            "All full-set rows are screening diagnostics; headline choices come from the five-fold OOF rows.",
        ],
    }

    for key, (run, _) in oof_bundles.items():
        _write_run(
            args.output_dir / ("oof_run.jsonl" if key == "hierarchical_kf3" else f"{key}_oof_run.jsonl"),
            run,
            qids,
            queries={question.qid: question.query for question in questions_list},
        )

    with (args.output_dir / "per_query.jsonl").open("w", encoding="utf-8") as handle:
        for method_name, method_traces in traces.items():
            metric_rows = {
                row["qid"]: row for row in methods[method_name]["retrieval_metrics"]["per_query"]
            }
            for qid in qids:
                handle.write(
                    json.dumps(
                        {
                            "method": method_name,
                            "qid": qid,
                            "metrics": metric_rows[qid],
                            "trace": _compact_trace(method_traces[qid]),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    with (args.output_dir / "error_analysis.jsonl").open("w", encoding="utf-8") as handle:
        for row in error_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_dir / "index_manifest.json").write_text(
        json.dumps(index_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "report.md").write_text(_markdown(report) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "output": str(args.output_dir),
                "references": {
                    "cached_bm25_page_recall@10": round(baseline_metric["page_recall@10"] * 100, 2),
                    "current_hierarchical_page_recall@10": round(hierarchical_metric["page_recall@10"] * 100, 2),
                },
                "oof": {
                    key: {
                        "page_recall@10": round(value["retrieval_metrics"]["page_recall@10"] * 100, 2),
                        "file_recall@3": round(value["retrieval_metrics"]["file_metrics_by_k"]["3"]["file_recall"] * 100, 2),
                        "proposal_coverage": value["stage_metrics"].get("proposal_file_coverage"),
                        "mean_pages_scanned": value["stage_metrics"].get("mean_pages_scanned"),
                    }
                    for key, value in cv.items()
                },
                "primary_error_counts": error_counts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
