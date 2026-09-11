"""Quick page retrieval test using PDF-inspector layout fields.

The experiment compares cached multilingual-E5 full-page scores with newly
encoded page representations made from PDF-inspector/KDL layout fields.  It
then fuses each semantic page stream with the existing hierarchical OOF page
run.  File selection is unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.data_discovery.hierarchical import HierarchyCorpus  # noqa: E402
from research.experiments.physics_fielded_hierarchical_retrieval import (  # noqa: E402
    _build_field_records,
)
from research.experiments.physics_hierarchical_retrieval import (  # noqa: E402
    _derived_metrics,
    _load_run,
    _page_vector_units,
    _stratified_folds,
    _write_run,
)
from research.experiments.physics_quick_page_semantic_hierarchy import (  # noqa: E402
    _candidate_recall,
    _fuse,
    _metric_for_qids,
    _normalise,
    _run_from_scores,
)
from src.evaluation.benchmarks.vidore_v3 import ViDoreV3  # noqa: E402


DEFAULT_ROOT = ROOT / "data/benchmark/vidore_v3"
DEFAULT_PARSED_RUN = ROOT / "data/output/vidore-v3-physics-kdl-pdf-inspector/32c32a45a92c45bb"
DEFAULT_PAGE_VECTOR_DIR = ROOT / "data/output/vsplade/vidore_v3_physics_48q"
DEFAULT_QUERY_EMBEDDINGS = DEFAULT_ROOT / "results/physics_multilingual_e5_baseline/query_embeddings.npy"
DEFAULT_ALL_TEXT_SCORES = DEFAULT_ROOT / "results/physics_multilingual_e5_short_segments/page_dense_scores.npy"
DEFAULT_HIERARCHICAL_RUN = DEFAULT_ROOT / "results/physics_hierarchical_retrieval/oof_run.jsonl"
DEFAULT_OUTPUT = DEFAULT_ROOT / "results/physics_quick_page_layout_semantic"
EXPECTED_PAGES = 1674
EXPECTED_QUERIES = 302
DEPTH = 100
FIELD_TOKEN_CAP = 64

LAYOUT_VARIANTS: dict[str, tuple[str, ...]] = {
    "structured": (
        "title",
        "heading",
        "caption",
        "table",
        "formula",
        "figure",
        "section_context",
    ),
    "structured_body": (
        "title",
        "heading",
        "caption",
        "table",
        "formula",
        "figure",
        "section_context",
        "body",
    ),
}
FIELD_LABELS = {
    "title": "Title",
    "heading": "Heading",
    "caption": "Caption",
    "table": "Table",
    "formula": "Formula",
    "figure": "Figure",
    "section_context": "Section heading",
    "body": "Body",
}


def _excerpt(text: str, cap: int = FIELD_TOKEN_CAP) -> str:
    return " ".join(str(text or "").split()[:cap])


def _layout_passage(record: Mapping[str, str], fields: Sequence[str]) -> str:
    parts = []
    for field_name in fields:
        value = _excerpt(record.get(field_name, ""))
        if value:
            parts.append(f"{FIELD_LABELS[field_name]}: {value}")
    return "passage: " + " ".join(parts)


def _encode_layout_pages(
    records: Mapping[str, Mapping[str, str]],
    page_order: Sequence[str],
    *,
    model_name: str,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    vectors: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {
        "model": model_name,
        "prefix": "passage: ",
        "field_token_cap": FIELD_TOKEN_CAP,
        "variants": {name: list(fields) for name, fields in LAYOUT_VARIANTS.items()},
    }
    for variant, fields in LAYOUT_VARIANTS.items():
        passages = [_layout_passage(records[page_id], fields) for page_id in page_order]
        encoded = model.encode(
            passages,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype(np.float32, copy=False)
        if encoded.shape[0] != len(page_order):
            raise RuntimeError(f"Unexpected {variant} embedding shape: {encoded.shape}")
        vectors[variant] = encoded
        metadata.setdefault("dimensions", {})[variant] = int(encoded.shape[1])
        metadata.setdefault("nonempty_pages", {})[variant] = sum(bool(text.strip()) for text in passages)
    return vectors, metadata


def _dense_run_metrics(
    dense: np.ndarray,
    pages: list[str],
    qids: list[str],
    qrels: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    return _derived_metrics(_run_from_scores(pages, qids, dense), qids, qrels)


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Physics quick page layout semantic retrieval",
        "",
        "Page-stage fusion of cached all-text multilingual-E5 scores and newly encoded PDF-inspector layout-field representations.",
        "",
        "## Protocol",
        "",
        f"- Corpus: **{report['queries']}** French queries, **{report['pages']}** pages, **{report['files']}** files.",
        f"- New page representations: field values capped at **{FIELD_TOKEN_CAP}** tokens each and prefixed with their layout role.",
        "- Existing file selection is unchanged; only page candidate/ranking streams are fused.",
        "- All fixed score-fusion weights are evaluated with five-fold OOF selection.",
        "",
        "## Full-set screening",
        "",
        "| Method | Page recall@10 | nDCG@10 | File recall@3 |",
        "|---|---:|---:|---:|",
    ]
    for name, value in sorted(
        report["methods"].items(),
        key=lambda item: item[1]["page_recall@10"],
        reverse=True,
    ):
        lines.append(
            f"| {name} | {value['page_recall@10']:.2%} | {value['ndcg@10']:.2f} | "
            f"{value['file_recall@3']:.2%} |"
        )
    lines += ["", "## Candidate union", "", "| Semantic stream | Union page recall before ranking |", "|---|---:|"]
    for name, value in report["candidate_union_recall"].items():
        lines.append(f"| {name} | {value:.2%} |")
    lines += ["", "## Out-of-fold selection", ""]
    for key, value in report["oof"].items():
        metric = value["metrics"]
        lines += [
            f"### {key}",
            "",
            f"- Selected methods: `{value['selected_method_counts']}`.",
            f"- Page recall@10: **{metric['page_recall@10']:.2%}**; nDCG@10: **{metric['ndcg@10']:.2f}**.",
            f"- File recall@3: **{metric['file_metrics_by_k']['3']['file_recall']:.2%}**.",
            "",
        ]
    lines += [
        "## Notes",
        "",
        "- The cached all-text control is the existing short-segment multilingual-E5 page score.",
        "- No new image encoding, OCR, LLM or qrel-derived feature is used.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--parsed-run", type=Path, default=DEFAULT_PARSED_RUN)
    parser.add_argument("--page-vector-dir", type=Path, default=DEFAULT_PAGE_VECTOR_DIR)
    parser.add_argument("--query-embeddings", type=Path, default=DEFAULT_QUERY_EMBEDDINGS)
    parser.add_argument("--all-text-scores", type=Path, default=DEFAULT_ALL_TEXT_SCORES)
    parser.add_argument("--hierarchical-run", type=Path, default=DEFAULT_HIERARCHICAL_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default="intfloat/multilingual-e5-small")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    benchmark = ViDoreV3(root=args.benchmark_root, subset="physics", language="french")
    questions = sorted(list(benchmark.questions()), key=lambda q: int(q.qid.rsplit("::", 1)[1]))
    qids = [q.qid for q in questions]
    qrels = benchmark.qrels()
    pages = _page_vector_units(args.page_vector_dir)
    if len(qids) != EXPECTED_QUERIES or len(pages) != EXPECTED_PAGES:
        raise RuntimeError(f"Inventory mismatch: queries={len(qids)}, pages={len(pages)}")
    query_vectors = np.load(args.query_embeddings, mmap_mode="r").astype(np.float32, copy=False)
    all_text_dense = np.load(args.all_text_scores, mmap_mode="r").astype(np.float32, copy=False)
    if query_vectors.shape != (EXPECTED_QUERIES, 384):
        raise RuntimeError(f"Unexpected query embedding shape: {query_vectors.shape}")
    if all_text_dense.shape != (EXPECTED_QUERIES, EXPECTED_PAGES):
        raise RuntimeError(f"Unexpected all-text score shape: {all_text_dense.shape}")
    current = _load_run(args.hierarchical_run)

    corpus = HierarchyCorpus.from_parsed_run(args.parsed_run, subset="physics", page_ids=pages)
    field_records = _build_field_records(corpus, args.parsed_run)
    layout_vectors, semantic_meta = _encode_layout_pages(
        field_records.page_records,
        corpus.page_order,
        model_name=args.model,
        batch_size=args.batch_size,
    )
    position = {page: index for index, page in enumerate(pages)}
    dense_scores: dict[str, np.ndarray] = {
        "all_text_e5_cached": all_text_dense,
    }
    for variant, page_vectors in layout_vectors.items():
        dense_scores[f"layout_{variant}"] = np.asarray(
            [query_vectors[index] @ page_vectors.T for index in range(len(qids))],
            dtype=np.float32,
        )
    runs: dict[str, dict[str, list[dict[str, Any]]]] = {
        "current_hierarchy": current,
        "all_text_e5_cached": _run_from_scores(pages, qids, all_text_dense),
    }
    for name, dense in dense_scores.items():
        if name == "all_text_e5_cached":
            continue
        runs[name] = _run_from_scores(pages, qids, dense)
    for stream_name, dense in dense_scores.items():
        for current_weight in (0.30, 0.50, 0.70):
            name = f"current_plus_{stream_name}_currentw{current_weight:.2f}"
            runs[name] = _fuse(current, dense, pages, qids, current_weight=current_weight)

    metrics = {name: _derived_metrics(run, qids, qrels) for name, run in runs.items()}
    candidate_union_recall = {
        stream_name: _candidate_recall(
            current,
            _run_from_scores(pages, qids, dense),
            qids,
            qrels,
        )
        for stream_name, dense in dense_scores.items()
    }
    folds = _stratified_folds(qids, {q.qid: q for q in questions}, qrels)
    oof: dict[str, Any] = {}
    for k in (3,):
        # Selection is limited to page-stage arms, with the current hierarchy
        # retained as a control.  The same fixed arm set is tested on every fold.
        selected: list[dict[str, Any]] = []
        oof_run: dict[str, list[dict[str, Any]]] = {}
        candidates = list(runs)
        for fold, heldout in enumerate(folds):
            train = set(qids) - set(heldout)
            winner = max(candidates, key=lambda name: (_metric_for_qids(metrics[name], train), name))
            for qid in heldout:
                oof_run[qid] = runs[winner][qid]
            selected.append({
                "fold": fold,
                "selected": winner,
                "train_page_recall@10": _metric_for_qids(metrics[winner], train),
                "test_page_recall@10": _metric_for_qids(metrics[winner], set(heldout)),
            })
        oof_metric = _derived_metrics(oof_run, qids, qrels)
        oof[f"kf{k}"] = {
            "metrics": oof_metric,
            "folds": selected,
            "selected_method_counts": {
                name: sum(row["selected"] == name for row in selected)
                for name in candidates
            },
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = args.output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    for name, run in runs.items():
        _write_run(runs_dir / f"{name}.jsonl", run, qids, queries={q.qid: q.query for q in questions})
    for key, value in oof.items():
        _write_run(args.output_dir / f"oof_{key}.jsonl", oof_run, qids, queries={q.qid: q.query for q in questions})
    for variant, vectors in layout_vectors.items():
        np.save(args.output_dir / f"{variant}_page_embeddings.npy", vectors)
        np.save(args.output_dir / f"{variant}_dense_scores.npy", dense_scores[f"layout_{variant}"])
    report = {
        "dataset": "vidore_v3/physics",
        "queries": len(qids),
        "pages": len(pages),
        "files": len(corpus.file_order),
        "semantic": semantic_meta,
        "references": {
            "current_hierarchical": metrics["current_hierarchy"],
            "all_text_e5_cached": metrics["all_text_e5_cached"],
        },
        "methods": {
            name: {
                "page_recall@10": value["page_recall@10"],
                "ndcg@10": value["ndcg@10"],
                "file_recall@3": value["file_metrics_by_k"]["3"]["file_recall"],
            }
            for name, value in metrics.items()
        },
        "candidate_union_recall": candidate_union_recall,
        "oof": oof,
        "fixed_policy": {
            "fusion_current_weights": [0.30, 0.50, 0.70],
            "semantic_weight_is_1_minus_current_weight": True,
            "field_token_cap": FIELD_TOKEN_CAP,
            "no_qrels_used_for_page_encoding": True,
        },
        "sources": {
            "parsed_run": str(args.parsed_run),
            "query_embeddings": str(args.query_embeddings),
            "all_text_scores": str(args.all_text_scores),
            "hierarchical_run": str(args.hierarchical_run),
        },
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "report.md").write_text(_markdown(report) + "\n", encoding="utf-8")
    best = max(metrics, key=lambda name: (metrics[name]["page_recall@10"], metrics[name]["ndcg@10"]))
    print(json.dumps({
        "output": str(args.output_dir),
        "best_full_set": best,
        "best_full_page_recall@10": round(metrics[best]["page_recall@10"] * 100, 2),
        "oof_page_recall@10": round(oof["kf3"]["metrics"]["page_recall@10"] * 100, 2),
        "candidate_union_recall": {k: round(v * 100, 2) for k, v in candidate_union_recall.items()},
        "selected_methods": oof["kf3"]["selected_method_counts"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
