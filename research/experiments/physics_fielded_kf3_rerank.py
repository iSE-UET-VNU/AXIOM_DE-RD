"""Reranking after the field-aware Physics Kf=3 cascade.

The file proposal and compact stage are kept fixed at the current field-aware
``sum_top2`` policy.  Only the page stage changes:

    selected files -> base page rank -> top-R candidates -> cross-encoder
    rerank -> final top-10

The experiment is deliberately candidate-only.  It does not change file
selection or use qrels to build an index.  By default it uses the cached
multilingual E5 page scores; an optional cached cross-encoder mode is also
available for a GPU-backed run.  The ``openrouter`` mode calls a
provider-hosted reranker via OpenRouter; it does not imply that arbitrary
Hugging Face checkpoints are available through that API.
"""

from __future__ import annotations

import argparse
import json
import os
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
)
from research.experiments.physics_fielded_hierarchical_retrieval import (  # noqa: E402
    EXPECTED_PAGES,
    FIELD_WEIGHTS,
    FILE_SYNOPSIS_DEPTH,
    PAGE_PROPOSAL_DEPTH,
    PARENT_WEIGHT,
    VISUAL_WEIGHT,
    FieldedBM25,
    QuerySignals,
    _build_field_records,
    _index_scores,
    _load_visual_scores,
    _retrieve_hierarchical,
)
from research.experiments.physics_hierarchical_retrieval import (  # noqa: E402
    _derived_metrics,
    _safe_name,
    _stratified_folds,
    _write_run,
)
from src.evaluation.benchmarks.vidore_v3 import ViDoreV3  # noqa: E402


DEFAULT_BENCHMARK_ROOT = ROOT / "data/benchmark/vidore_v3"
DEFAULT_PARSED_RUN = ROOT / "data/output/vidore-v3-physics-kdl-pdf-inspector/32c32a45a92c45bb"
DEFAULT_PAGE_VECTOR_DIR = ROOT / "data/output/vsplade/vidore_v3_physics_48q"
DEFAULT_QUERY_VECTOR_DIR = ROOT / "data/output/vsplade/vidore_v3_physics_english_302q"
DEFAULT_OUTPUT_DIR = ROOT / "data/benchmark/vidore_v3/results/physics_fielded_kf3_rerank"
DEFAULT_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
DEFAULT_OPENROUTER_MODEL = "cohere/rerank-v3.5"
DEFAULT_OPENROUTER_URL = "https://openrouter.ai/api/v1/rerank"
DEFAULT_E5_SCORES = ROOT / "data/benchmark/vidore_v3/results/physics_multilingual_e5_baseline/page_dense_scores.npy"

FILE_BUDGET = 3
FINAL_DEPTH = 10
CANDIDATE_DEPTHS = (20, 50, 100)
CE_WEIGHT = 0.80


def _dotenv_value(name: str) -> str | None:
    """Read one secret from the process or the local .env without logging it."""
    value = os.environ.get(name)
    if value:
        return value
    dotenv = ROOT / ".env"
    if not dotenv.exists():
        return None
    prefix = f"{name}="
    for line in dotenv.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith(prefix):
            continue
        value = stripped[len(prefix) :].strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        return value or None
    return None


def _openrouter_rerank_scores(
    *,
    corpus: HierarchyCorpus,
    qids: Sequence[str],
    queries: Mapping[str, str],
    candidate_pages_by_qid: Mapping[str, Sequence[str]],
    model_name: str,
    endpoint: str,
    workers: int,
    timeout_seconds: float,
    max_retries: int,
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    """Call OpenRouter's rerank endpoint once per query over the R=100 band."""
    import requests
    from concurrent.futures import ThreadPoolExecutor, as_completed

    api_key = _dotenv_value("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set in the environment or repository .env"
        )

    session_headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": "AXIOM Physics fielded retrieval rerank",
    }

    def request_one(qid: str) -> tuple[str, dict[str, float]]:
        page_ids = list(candidate_pages_by_qid[qid])
        documents = [corpus.pages[page_id].text for page_id in page_ids]
        payload = {
            "model": model_name,
            "query": queries[qid],
            "documents": documents,
            "top_n": len(documents),
            "return_documents": False,
        }
        last_error: str | None = None
        for attempt in range(max_retries + 1):
            try:
                response = requests.post(
                    endpoint,
                    headers=session_headers,
                    json=payload,
                    timeout=timeout_seconds,
                )
                if response.status_code == 429 or response.status_code >= 500:
                    last_error = f"HTTP {response.status_code}"
                    if attempt < max_retries:
                        time.sleep(min(2.0**attempt, 8.0))
                        continue
                response.raise_for_status()
                data = response.json()
                results = data.get("results")
                if not isinstance(results, list) or len(results) != len(page_ids):
                    raise RuntimeError(
                        f"OpenRouter returned {len(results) if isinstance(results, list) else 'no'} "
                        f"results for {len(page_ids)} documents"
                    )
                scores: dict[str, float] = {}
                for row in results:
                    index = int(row["index"])
                    scores[page_ids[index]] = float(row["relevance_score"])
                if len(scores) != len(page_ids):
                    raise RuntimeError("OpenRouter returned duplicate/missing document indexes")
                return qid, scores
            except Exception as exc:  # retry transient and fail clearly after the final attempt
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < max_retries:
                    time.sleep(min(2.0**attempt, 8.0))
                    continue
        raise RuntimeError(f"OpenRouter rerank failed for {qid}: {last_error}")

    scores: dict[str, dict[str, float]] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(request_one, qid): qid for qid in qids}
        for completed, future in enumerate(as_completed(futures), 1):
            qid, values = future.result()
            scores[qid] = values
            if completed % 25 == 0 or completed == len(qids):
                print(f"OpenRouter rerank requests: {completed}/{len(qids)}", flush=True)
    return scores, {
        "endpoint": endpoint,
        "model": model_name,
        "requests": len(qids),
        "documents_per_request": max(len(candidate_pages_by_qid[qid]) for qid in qids),
        "workers": max(1, workers),
        "credential_source": "environment_or_dotenv",
    }


def _normalise_array(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if not len(array):
        return array
    minimum = float(array.min())
    shifted = array - minimum if minimum < 0.0 else array.copy()
    maximum = float(shifted.max())
    return shifted / maximum if maximum > 0.0 else np.zeros_like(shifted)


def _build_states(
    parsed_run: Path,
    page_ids: Sequence[str],
    qids: Sequence[str],
    questions: Sequence[Any],
    page_vector_dir: Path,
    query_vector_dir: Path,
) -> tuple[HierarchyCorpus, dict[str, QuerySignals], dict[str, Any], float]:
    started = time.perf_counter()
    corpus = HierarchyCorpus.from_parsed_run(
        parsed_run, subset="physics", page_ids=page_ids
    )
    if len(corpus.file_order) != 42 or len(corpus.page_order) != EXPECTED_PAGES:
        raise RuntimeError(
            f"Parsed inventory mismatch: files={len(corpus.file_order)}, "
            f"pages={len(corpus.page_order)}"
        )

    field_records = _build_field_records(corpus, parsed_run)
    page_fields = (
        "title",
        "heading",
        "body",
        "caption",
        "table",
        "formula",
        "figure",
        "boilerplate",
    )
    page_fields_with_context = (*page_fields, "section_context")
    page_field_index = FieldedBM25.build(
        field_records.page_records, fields=page_fields, weights=FIELD_WEIGHTS
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
    flat_index = build_bm25(
        (page_id, corpus.pages[page_id].text) for page_id in corpus.page_order
    )
    visual_by_qid, visual_meta = _load_visual_scores(
        page_vector_dir, query_vector_dir, page_ids, qids
    )

    states: dict[str, QuerySignals] = {}
    for question in questions:
        flat_scores = _index_scores(
            flat_index, question.query, top_k=len(page_ids)
        )
        field_scores, field_components = page_field_index.score(question.query)
        field_section_scores, field_section_components = page_field_context_index.score(
            question.query
        )
        file_scores, file_components = file_index.score(question.query)
        states[question.qid] = QuerySignals(
            flat_scores=flat_scores,
            field_scores=field_scores,
            field_components=field_components,
            field_section_scores=field_section_scores,
            field_section_components=field_section_components,
            file_scores=file_scores,
            file_components=file_components,
            visual_scores=visual_by_qid[question.qid],
        )
    return corpus, states, visual_meta, time.perf_counter() - started


def _base_runs(
    corpus: HierarchyCorpus,
    states: Mapping[str, QuerySignals],
    qids: Sequence[str],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    runs: dict[str, list[dict[str, Any]]] = {}
    traces: dict[str, dict[str, Any]] = {}
    for qid in qids:
        runs[qid], traces[qid] = _retrieve_hierarchical(
            corpus,
            states[qid],
            strategy="sum_top2",
            k_files=FILE_BUDGET,
        )
    return runs, traces


def _rerank_run(
    base_runs: Mapping[str, list[dict[str, Any]]],
    ce_scores: Mapping[str, Mapping[str, float]],
    *,
    candidate_depth: int,
    mode: str,
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for qid, base_items in base_runs.items():
        candidates = list(base_items[:candidate_depth])
        candidate_ids = {str(item["chunk_id"]) for item in candidates}
        ce_values = [float(ce_scores[qid][str(item["chunk_id"])]) for item in candidates]
        base_values = [float(item.get("score", 0.0)) for item in candidates]
        ce_norm = _normalise_array(ce_values)
        base_norm = _normalise_array(base_values)
        if mode == "ce_only":
            combined = ce_norm
        elif mode == "ce_base80":
            combined = CE_WEIGHT * ce_norm + (1.0 - CE_WEIGHT) * base_norm
        else:
            raise ValueError(f"Unknown rerank mode: {mode!r}")

        order = sorted(
            range(len(candidates)),
            key=lambda index: (-float(combined[index]), str(candidates[index]["chunk_id"])),
        )
        ranked: list[dict[str, Any]] = []
        for rank, index in enumerate(order, 1):
            item = dict(candidates[index])
            item["rank"] = rank
            item["score"] = round(float(combined[index]), 8)
            item["rerank_score"] = round(float(ce_values[index]), 8)
            item["base_score"] = round(float(base_values[index]), 8)
            item["rerank_mode"] = mode
            ranked.append(item)

        # A reranker may only reorder its candidate band.  Preserve the
        # original tail so canonical file metrics still see 100 page records.
        tail = [dict(item) for item in base_items if str(item["chunk_id"]) not in candidate_ids]
        for offset, item in enumerate(tail, len(ranked) + 1):
            item["rank"] = offset
            item["rerank_mode"] = mode
        output[qid] = ranked + tail
    return output


def _candidate_recall(
    base_runs: Mapping[str, list[dict[str, Any]]],
    qids: Sequence[str],
    qrels: Mapping[str, Mapping[str, int]],
    depth: int,
) -> float:
    values: list[float] = []
    for qid in qids:
        gold = set(qrels.get(qid, {}))
        candidate = {str(item["chunk_id"]) for item in base_runs[qid][:depth]}
        values.append(len(candidate & gold) / len(gold) if gold else 0.0)
    return float(np.mean(values)) if values else 0.0


def _selected_page_coverage(
    traces: Mapping[str, Mapping[str, Any]],
    corpus: HierarchyCorpus,
    qids: Sequence[str],
    qrels: Mapping[str, Mapping[str, int]],
) -> float:
    values: list[float] = []
    for qid in qids:
        gold = set(qrels.get(qid, {}))
        selected = set(traces[qid].get("selected_files", []))
        available = {
            page_id
            for page_id in corpus.page_order
            if page_id.split("#page=", 1)[0] in selected
        }
        values.append(len(available & gold) / len(gold) if gold else 0.0)
    return float(np.mean(values)) if values else 0.0


def _metric_for_qids(metric: Mapping[str, Any], qids: set[str]) -> float:
    rows = [row for row in metric["per_query"] if row["qid"] in qids]
    if not rows:
        return 0.0
    return float(np.mean([float(row["page_recall@10"]) for row in rows]))


def _write_report(
    output_dir: Path,
    report: Mapping[str, Any],
    methods: Mapping[str, Mapping[str, Any]],
    oof: Mapping[str, Any],
) -> None:
    lines = [
        "# Physics field-aware Kf=3 page reranking",
        "",
        "Fixed file stage: field-aware proposal union + `sum_top2` compact selection, Kf=3.",
        "Only the page candidate band and reranker vary.",
        "",
        "| Method | Candidate depth | Page recall@10 | nDCG@10 | File recall@3 | Candidate page recall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, value in sorted(
        methods.items(), key=lambda item: item[1]["page_recall@10"], reverse=True
    ):
        lines.append(
            f"| {name} | {value['candidate_depth']} | {value['page_recall@10']:.2%} "
            f"| {value['ndcg@10']:.2f} | {value['file_recall@3']:.2%} "
            f"| {value['candidate_page_recall']:.2%} |"
        )
    lines += [
        "",
        f"OOF selected method: `{oof['selected_method']}`.",
        f"OOF page recall@10: **{oof['metrics']['page_recall@10']:.2%}**.",
        f"OOF nDCG@10: **{oof['metrics']['ndcg@10']:.2f}**.",
        "",
        "| Fold | Selected method | Test page recall@10 |",
        "|---:|---|---:|",
    ]
    for row in oof["folds"]:
        lines.append(
            f"| {row['fold']} | `{row['selected_method']}` "
            f"| {row['test_page_recall@10']:.2%} |"
        )
    lines += [
        "",
        "## Protocol notes",
        "",
        "- The reranker sees only the first R pages from the fixed Kf=3 base run.",
        "- Reranking changes order only; it cannot recover a page outside the candidate band.",
        "- The `*_base80` arm uses a predeclared 0.80 reranker / 0.20 base-score blend.",
        "- No qrels are used for field indexes, file selection or reranker scoring.",
        "- V-SPLADE query vectors are cached English translations evaluated against French qrels.",
        f"- Model inference device: `{report['model']['device']}`.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--parsed-run", type=Path, default=DEFAULT_PARSED_RUN)
    parser.add_argument("--page-vector-dir", type=Path, default=DEFAULT_PAGE_VECTOR_DIR)
    parser.add_argument("--query-vector-dir", type=Path, default=DEFAULT_QUERY_VECTOR_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--reranker",
        choices=("cached_e5", "cross_encoder", "openrouter"),
        default="cached_e5",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--openrouter-model", default=DEFAULT_OPENROUTER_MODEL)
    parser.add_argument("--openrouter-url", default=DEFAULT_OPENROUTER_URL)
    parser.add_argument("--openrouter-workers", type=int, default=4)
    parser.add_argument("--openrouter-timeout", type=float, default=120.0)
    parser.add_argument("--openrouter-retries", type=int, default=2)
    parser.add_argument("--e5-scores", type=Path, default=DEFAULT_E5_SCORES)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    started = time.perf_counter()
    benchmark = ViDoreV3(root=args.benchmark_root, subset="physics", language="french")
    questions = sorted(
        list(benchmark.questions()), key=lambda question: int(question.qid.rsplit("::", 1)[1])
    )
    qids = [question.qid for question in questions]
    if len(qids) != 302:
        raise RuntimeError(f"Expected 302 Physics qids, got {len(qids)}")
    qrels = benchmark.qrels()
    page_ids = json.loads((args.page_vector_dir / "page_metadata.json").read_text(encoding="utf-8"))
    page_ids = [str(row["unit_id"]) for row in page_ids]
    if len(page_ids) != EXPECTED_PAGES or len(set(page_ids)) != EXPECTED_PAGES:
        raise RuntimeError(f"Expected {EXPECTED_PAGES} unique page IDs, got {len(set(page_ids))}")

    corpus, states, visual_meta, setup_seconds = _build_states(
        args.parsed_run,
        page_ids,
        qids,
        questions,
        args.page_vector_dir,
        args.query_vector_dir,
    )
    base_runs, base_traces = _base_runs(corpus, states, qids)

    max_depth = max(CANDIDATE_DEPTHS)
    pairs: list[tuple[str, str]] = []
    pair_keys: list[tuple[str, str]] = []
    candidate_pages_by_qid: dict[str, list[str]] = {}
    queries = {question.qid: question.query for question in questions}
    for qid in qids:
        pages = [str(item["chunk_id"]) for item in base_runs[qid][:max_depth]]
        candidate_pages_by_qid[qid] = pages
        for page_id in pages:
            pair_keys.append((qid, page_id))
            if args.reranker == "cross_encoder":
                pairs.append((queries[qid], corpus.pages[page_id].text))

    model_started = time.perf_counter()
    reranker_meta: dict[str, Any] = {}
    if args.reranker == "cached_e5":
        dense = np.load(args.e5_scores, mmap_mode="r")
        if dense.shape != (len(qids), len(page_ids)):
            raise RuntimeError(
                f"Unexpected cached E5 score shape {dense.shape}; "
                f"expected {(len(qids), len(page_ids))}"
            )
        page_positions = {page_id: index for index, page_id in enumerate(page_ids)}
        ce_scores = {
            qid: {
                page_id: float(dense[row, page_positions[page_id]])
                for page_id in candidate_pages_by_qid[qid]
            }
            for row, qid in enumerate(qids)
        }
        reranker_name = "intfloat/multilingual-e5-small (cached page scores)"
    elif args.reranker == "cross_encoder":
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(args.model, max_length=args.max_length, device=args.device)
        values = model.predict(
            pairs,
            batch_size=args.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        ce_scores = {qid: {} for qid in qids}
        for (qid, page_id), value in zip(pair_keys, values):
            ce_scores[qid][page_id] = float(np.asarray(value).reshape(-1)[0])
        reranker_name = args.model
    else:
        ce_scores, reranker_meta = _openrouter_rerank_scores(
            corpus=corpus,
            qids=qids,
            queries=queries,
            candidate_pages_by_qid=candidate_pages_by_qid,
            model_name=args.openrouter_model,
            endpoint=args.openrouter_url,
            workers=args.openrouter_workers,
            timeout_seconds=args.openrouter_timeout,
            max_retries=args.openrouter_retries,
        )
        reranker_name = args.openrouter_model
    model_seconds = time.perf_counter() - model_started
    runtime_device = (
        args.device
        if args.reranker == "cross_encoder"
        else "api" if args.reranker == "openrouter" else "cache"
    )

    runs: dict[str, dict[str, list[dict[str, Any]]]] = {
        "base_fielded_kf3": base_runs,
    }
    method_specs: dict[str, tuple[int, str]] = {}
    rerank_prefix = (
        "e5"
        if args.reranker == "cached_e5"
        else "ce"
        if args.reranker == "cross_encoder"
        else "or"
    )
    for depth in CANDIDATE_DEPTHS:
        for mode in ("ce_only", "ce_base80"):
            name = f"{rerank_prefix}_{mode.removeprefix('ce_')}_r{depth}"
            method_specs[name] = (depth, mode)
            runs[name] = _rerank_run(
                base_runs,
                ce_scores,
                candidate_depth=depth,
                mode=mode,
            )

    method_metrics: dict[str, dict[str, Any]] = {}
    for name, run in runs.items():
        metric = _derived_metrics(run, qids, qrels)
        if name == "base_fielded_kf3":
            depth = max_depth
        else:
            depth = method_specs[name][0]
        method_metrics[name] = {
            "candidate_depth": depth,
            "page_recall@10": float(metric["page_recall@10"]),
            "ndcg@10": float(metric["ndcg@10"]),
            "page_hit@10": float(metric["page_hit@10"]),
            "file_recall@3": float(metric["file_metrics_by_k"]["3"]["file_recall"]),
            "candidate_page_recall": (
                _candidate_recall(base_runs, qids, qrels, depth)
            ),
            "metrics": metric,
        }

    # Fixed candidate arms are selected only within the training folds.  The
    # full-set rows remain useful screening diagnostics but are not promoted.
    folds = _stratified_folds(
        qids,
        {question.qid: question for question in questions},
        qrels,
    )
    candidate_names = list(method_metrics)
    oof_run: dict[str, list[dict[str, Any]]] = {}
    fold_rows: list[dict[str, Any]] = []
    for fold, heldout in enumerate(folds):
        train = set(qids) - set(heldout)
        winner = max(
            candidate_names,
            key=lambda name: (
                _metric_for_qids(method_metrics[name]["metrics"], train),
                name,
            ),
        )
        oof_run.update({qid: runs[winner][qid] for qid in heldout})
        test_page = _metric_for_qids(method_metrics[winner]["metrics"], set(heldout))
        fold_rows.append(
            {
                "fold": fold,
                "selected_method": winner,
                "train_page_recall@10": _metric_for_qids(
                    method_metrics[winner]["metrics"], train
                ),
                "test_page_recall@10": test_page,
            }
        )
    oof_metrics = _derived_metrics(oof_run, qids, qrels)
    selected_method = max(
        candidate_names,
        key=lambda name: (
            method_metrics[name]["page_recall@10"],
            method_metrics[name]["ndcg@10"],
            name,
        ),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = args.output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    for name, run in runs.items():
        _write_run(runs_dir / f"{_safe_name(name)}.jsonl", run, qids, queries=queries)
    _write_run(args.output_dir / "oof_run.jsonl", oof_run, qids, queries=queries)

    per_query_path = args.output_dir / "per_query.jsonl"
    with per_query_path.open("w", encoding="utf-8") as handle:
        for name, run in runs.items():
            metric_rows = {
                row["qid"]: row for row in method_metrics[name]["metrics"]["per_query"]
            }
            for qid in qids:
                base_candidates = candidate_pages_by_qid[qid]
                handle.write(
                    json.dumps(
                        {
                            "method": name,
                            "qid": qid,
                            "candidate_depth": method_metrics[name]["candidate_depth"],
                            "candidate_pages": base_candidates[: method_metrics[name]["candidate_depth"]]
                            if method_metrics[name]["candidate_depth"]
                            else [],
                            "selected_files": base_traces[qid].get("selected_files", []),
                            "metrics": metric_rows[qid],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    report: dict[str, Any] = {
        "dataset": "vidore_v3/physics",
        "queries": len(qids),
        "pages": len(page_ids),
        "files": len(corpus.file_order),
        "file_stage": {
            "policy": "field-aware proposal union + sum_top2 compact ranking",
            "k_files": FILE_BUDGET,
            "proposal_page_depth": PAGE_PROPOSAL_DEPTH,
            "synopsis_depth": FILE_SYNOPSIS_DEPTH,
            "selected_page_coverage": _selected_page_coverage(
                base_traces, corpus, qids, qrels
            ),
            "note": "File selection is reused from the current field-aware Kf=3 policy.",
        },
        "page_stage": {
            "base_policy": "0.70 flat BM25 + 0.30 V-SPLADE, plus 0.15 parent file prior",
            "candidate_depths": list(CANDIDATE_DEPTHS),
            "final_depth": FINAL_DEPTH,
            "ce_weight_in_ce_base80": CE_WEIGHT,
        },
        "model": {
            "name": reranker_name,
            "type": args.reranker,
            "device": runtime_device,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "pairs": len(pair_keys),
            **reranker_meta,
        },
        "methods": {
            name: {
                key: value
                for key, value in metrics.items()
                if key != "metrics"
            }
            for name, metrics in method_metrics.items()
        },
        "best_full_set_method": selected_method,
        "oof": {
            "metrics": {
                "page_recall@10": float(oof_metrics["page_recall@10"]),
                "ndcg@10": float(oof_metrics["ndcg@10"]),
                "page_hit@10": float(oof_metrics["page_hit@10"]),
                "file_recall@3": float(
                    oof_metrics["file_metrics_by_k"]["3"]["file_recall"]
                ),
            },
            "folds": fold_rows,
            "selected_method": max(
                set(row["selected_method"] for row in fold_rows),
                key=lambda name: sum(row["selected_method"] == name for row in fold_rows),
            ),
        },
        "timing_seconds": {
            "setup_and_signal_build": setup_seconds,
            "reranker_score_loading_or_inference": model_seconds,
            "total": time.perf_counter() - started,
        },
        "sources": {
            "parsed_run": str(args.parsed_run),
            "page_vector_dir": str(args.page_vector_dir),
            "query_vector_dir": str(args.query_vector_dir),
            "query_language": "english V-SPLADE against French qrels",
            "visual_cache": visual_meta,
        },
        "notes": [
            "This is a fixed full-set screening plus five-fold OOF selection among predeclared rerank arms.",
            "The reranker can only reorder the base top-R candidate band.",
            "No qrels are used to build indexes or score model pairs.",
            "The Kf=3 file-stage policy is field-aware sum_top2, not the older all_text-max cascade.",
            f"Reranker mode: {args.reranker}.",
        ],
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_report(args.output_dir, report, method_metrics, report["oof"])
    print(
        json.dumps(
            {
                "output": str(args.output_dir),
                "pairs": len(pair_keys),
                "best_full_set_method": selected_method,
                "best_full_page_recall@10": round(
                    method_metrics[selected_method]["page_recall@10"] * 100, 2
                ),
                "oof_page_recall@10": round(oof_metrics["page_recall@10"] * 100, 2),
                "reranker_seconds": round(model_seconds, 3),
                "device": runtime_device,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
