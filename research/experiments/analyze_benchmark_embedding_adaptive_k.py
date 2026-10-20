"""Dense text-embedding retrieval plus score-gap adaptive page cutoffs.

This experiment uses the text already produced by PDF-inspector/Tesseract.  It
does not render pages or call V-SPLADE.  A multilingual E5 encoder is used with
the standard ``query:`` / ``passage:`` prefixes.  The page vectors are cached
so reruns only recompute missing artifacts.

Two retrieval views are reported:

* ``global``: rank all pages by query/page cosine similarity;
* ``hierarchical``: select Kf files by the maximum dense page score, then rank
  pages only inside those files.

The adaptive policies are fixed-10, largest adjacent gap, and largest gap plus
the fixed buffer of five pages used by the Adaptive-k paper.  Qrels are used
only after ranking for evaluation.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence
import argparse
import json
import math
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    pos = (len(ordered) - 1) * q
    low, high = math.floor(pos), math.ceil(pos)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def _find_local_model(model_name: str) -> str:
    if Path(model_name).exists():
        return model_name
    safe = "models--" + model_name.replace("/", "--")
    roots = [
        Path.home() / ".cache" / "huggingface" / "hub" / safe / "snapshots",
        Path("C:/Users/admin/.cache/huggingface/hub") / safe / "snapshots",
    ]
    for root in roots:
        snapshots = sorted(path for path in root.glob("*") if path.is_dir()) if root.exists() else []
        if snapshots:
            return str(snapshots[-1])
    return model_name


def _encode_cached(
    *,
    model_name: str,
    page_texts: Sequence[str],
    queries: Sequence[str],
    output_dir: Path,
    batch_size: int,
    device: str,
    chunk_chars: int,
    provider: str,
    force: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    page_path = output_dir / "page_embeddings.npy"
    query_path = output_dir / "query_embeddings.npy"
    meta_path = output_dir / "embedding_metadata.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    if page_path.is_file() and query_path.is_file() and meta_path.is_file() and not force:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        pages = np.load(page_path, mmap_mode="r")
        query_vectors = np.load(query_path, mmap_mode="r")
        if pages.shape[0] == len(page_texts) and query_vectors.shape[0] == len(queries):
            return np.asarray(pages), np.asarray(query_vectors), meta

    model_started = time.perf_counter()
    page_chunks: list[list[str]] = []
    chunk_inputs: list[str] = []
    for text in page_texts:
        if not text.strip():
            chunks = []
        else:
            chunks = [text[start : start + chunk_chars] for start in range(0, len(text), chunk_chars)]
        page_chunks.append(chunks)
        chunk_inputs.extend(chunks)
    if provider == "openrouter":
        from src.chunking_embedding.embedders.openrouter import OpenRouterEmbedder
        from src.utils.env import load_dotenv_file

        load_dotenv_file(ROOT)
        embedder = OpenRouterEmbedder(
            model=model_name,
            batch_size=batch_size,
            cache_dir=output_dir / "api_cache",
        )
        chunk_vectors = np.asarray(embedder.embed(chunk_inputs), dtype=np.float32)
        query_vectors = np.asarray(embedder.embed(list(queries)), dtype=np.float32)
        provider_meta = {
            "provider": "openrouter",
            "local_model": None,
            "dimension": int(chunk_vectors.shape[1]),
            "api_calls": int(embedder.stats["api_calls"]),
            "cache_hits": int(embedder.stats["cache_hits"]),
            "tokens": int(embedder.stats["tokens"]),
        }
    else:
        from sentence_transformers import SentenceTransformer

        local_model = _find_local_model(model_name)
        model = SentenceTransformer(local_model, device=device)
        # E5 has a 512-token input limit.  SentenceTransformers truncates
        # chunks consistently; the cache stores vectors only, never text.
        model.max_seq_length = min(int(getattr(model, "max_seq_length", 512) or 512), 512)
        chunk_vectors = model.encode(
            ["passage: " + chunk for chunk in chunk_inputs],
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        query_vectors = model.encode(
            ["query: " + text for text in queries],
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        provider_meta = {
            "provider": "local",
            "local_model": local_model,
            "dimension": int(chunk_vectors.shape[1]),
            "api_calls": 0,
            "cache_hits": 0,
            "tokens": 0,
        }
    page_vectors_list: list[np.ndarray] = []
    cursor = 0
    for chunks in page_chunks:
        if not chunks:
            page_vectors_list.append(np.zeros(chunk_vectors.shape[1], dtype=np.float32))
            continue
        pooled = chunk_vectors[cursor : cursor + len(chunks)].mean(axis=0)
        norm = float(np.linalg.norm(pooled))
        page_vectors_list.append(pooled / norm if norm else pooled)
        cursor += len(chunks)
    page_vectors = np.vstack(page_vectors_list).astype(np.float32)
    np.save(page_path, page_vectors)
    np.save(query_path, query_vectors)
    meta = {
        "model": model_name,
        **provider_meta,
        "device": device,
        "page_count": len(page_texts),
        "page_chunk_count": len(chunk_inputs),
        "chunk_chars": chunk_chars,
        "query_count": len(queries),
        "batch_size": batch_size,
        "normalized": True,
        "prefixes": {"page": "passage: " if provider == "local" else "", "query": "query: " if provider == "local" else ""},
        "seconds": time.perf_counter() - model_started,
    }
    _write_json(meta_path, meta)
    return page_vectors, query_vectors, meta


def _largest_gap_cut(scores: Sequence[float], minimum: int, maximum: int, buffer: int) -> tuple[int, float]:
    last = min(maximum, len(scores))
    if last <= minimum:
        return last, 0.0
    candidates = [(rank, max(0.0, scores[rank - 1] - scores[rank])) for rank in range(minimum, last)]
    rank, gap = max(candidates, key=lambda item: (item[1], -item[0]))
    return min(last, rank + buffer), gap


def _metric_rows(rows: Sequence[Mapping[str, Any]], qrels: Mapping[str, set[str]], policy: str) -> dict[str, Any]:
    coverage, recall10, hit10, file_recall, pages, files = [], [], [], [], [], []
    for row in rows:
        gold = qrels.get(str(row["query_id"]), set())
        returned = list(row["returned_pages"])
        found = set(returned) & gold
        top10 = set(returned[:10]) & gold
        gold_files = {page.split("#page=", 1)[0] for page in gold}
        selected_files = set(row["selected_files"])
        returned_files = {page.split("#page=", 1)[0] for page in returned}
        coverage.append(len(found) / len(gold) if gold else 0.0)
        recall10.append(len(top10) / len(gold) if gold else 0.0)
        hit10.append(float(bool(top10)))
        file_recall.append(len(selected_files & gold_files) / len(gold_files) if gold_files else 0.0)
        pages.append(len(returned))
        files.append(len(returned_files))
    mean = lambda values: sum(values) / len(values) if values else 0.0
    return {
        "policy": policy,
        "queries": len(rows),
        "page_recall_at_returned_k": mean(coverage),
        "page_recall_at_10": mean(recall10),
        "page_hit_at_10": mean(hit10),
        "file_recall_at_selected_files": mean(file_recall),
        "mean_returned_pages": mean(pages),
        "median_returned_pages": statistics.median(pages) if pages else 0.0,
        "mean_returned_files": mean(files),
        "p95_returned_pages": _quantile(pages, 0.95),
    }


def _arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="openai/text-embedding-3-small")
    parser.add_argument("--provider", choices=("local", "openrouter"), default="openrouter")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--chunk-chars", type=int, default=1800)
    parser.add_argument("--file-k", type=int, default=3)
    parser.add_argument("--max-page-k", type=int, default=100)
    parser.add_argument("--minimum-page-k", type=int, default=10)
    parser.add_argument("--force-encode", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments().parse_args(argv)
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pages = _read_jsonl(args.cache_dir / "parsing/pages.jsonl")
    queries = _read_jsonl(args.dataset_root / "queries.jsonl")
    qrel_rows = _read_jsonl(args.cache_dir / "qrels_page_level.jsonl")
    qrels: dict[str, set[str]] = defaultdict(set)
    for row in qrel_rows:
        qrels[str(row["query_id"])].add(str(row["page_id"]))
    page_ids = [str(row["page_id"]) for row in pages]
    page_docs = [str(row["doc_id"]) for row in pages]
    page_texts = [str(row.get("text") or "") for row in pages]
    query_ids = [str(row["query_id"]) for row in queries]
    query_texts = [str(row["query"]) for row in queries]
    empty_pages = sum(not text.strip() for text in page_texts)

    encode_started = time.perf_counter()
    page_vectors, query_vectors, embedding_meta = _encode_cached(
        model_name=args.model,
        page_texts=page_texts,
        queries=query_texts,
        output_dir=args.output_dir / "embeddings",
        batch_size=args.batch_size,
        device=args.device,
        chunk_chars=args.chunk_chars,
        provider=args.provider,
        force=args.force_encode,
    )
    encode_seconds = time.perf_counter() - encode_started

    files = sorted(set(page_docs))
    page_positions_by_file: dict[str, list[int]] = defaultdict(list)
    for position, doc_id in enumerate(page_docs):
        page_positions_by_file[doc_id].append(position)
    policy_rows: dict[str, list[dict[str, Any]]] = {}
    policy_names = ("global_fixed10", "global_largest_gap", "global_largest_gap_buffer5", "hier_fixed10", "hier_largest_gap", "hier_largest_gap_buffer5")
    for policy in policy_names:
        policy_rows[policy] = []
    retrieval_started = time.perf_counter()
    for query_id, query_vector in zip(query_ids, query_vectors):
        scores = np.asarray(page_vectors @ query_vector, dtype=np.float32)
        if empty_pages:
            scores = scores.copy()
            for position, text in enumerate(page_texts):
                if not text.strip():
                    scores[position] = -1.0
        global_order = np.argsort(-scores, kind="stable")[: args.max_page_k]
        global_scores = [float(scores[position]) for position in global_order]
        file_scores = {doc_id: max(float(scores[position]) for position in positions) for doc_id, positions in page_positions_by_file.items()}
        ranked_files = sorted(file_scores.items(), key=lambda item: (-item[1], item[0]))
        selected_files = [doc_id for doc_id, _score in ranked_files[: args.file_k]]
        selected_positions = [position for doc_id in selected_files for position in page_positions_by_file[doc_id]]
        selected_positions.sort(key=lambda position: (-float(scores[position]), position))
        hier_order = selected_positions[: args.max_page_k]
        hier_scores = [float(scores[position]) for position in hier_order]

        ranking_specs = {
            "global": (global_order, global_scores, [doc_id for doc_id, _score in sorted(file_scores.items(), key=lambda item: (-item[1], item[0]))]),
            "hier": (hier_order, hier_scores, selected_files),
        }
        for prefix, (order, ranked_scores, chosen_files) in ranking_specs.items():
            if not ranked_scores:
                continue
            last = len(ranked_scores)
            largest, gap = _largest_gap_cut(ranked_scores, args.minimum_page_k, last, buffer=0)
            largest_b5, gap_b5 = _largest_gap_cut(ranked_scores, args.minimum_page_k, last, buffer=5)
            decisions = {
                f"{prefix}_fixed10": (min(10, last), 0.0, "fixed-10"),
                f"{prefix}_largest_gap": (largest, gap, "largest-adjacent-gap"),
                f"{prefix}_largest_gap_buffer5": (largest_b5, gap_b5, "largest-adjacent-gap-plus-buffer5"),
            }
            for policy, (cut, gap_value, reason) in decisions.items():
                returned = [page_ids[int(position)] for position in order[:cut]]
                policy_rows[policy].append({
                    "query_id": query_id,
                    "selected_files": chosen_files[: args.file_k],
                    "returned_pages": returned,
                    "stop_rank": cut,
                    "stop_gap": gap_value,
                    "stop_reason": reason,
                })
    retrieval_seconds = time.perf_counter() - retrieval_started
    metrics = {policy: _metric_rows(rows, qrels, policy) for policy, rows in policy_rows.items()}
    report = {
        "contract_version": "benchmark-embedding-adaptive-k-v1",
        "cache_dir": str(args.cache_dir),
        "dataset_root": str(args.dataset_root),
        "model": embedding_meta,
        "files": len(files),
        "pages": len(pages),
        "queries": len(queries),
        "empty_pages": empty_pages,
        "query_sources": {source: sum(row.get("source") == source for row in queries) for source in sorted({row.get("source") for row in queries})},
        "query_language_input": "queries.jsonl as-is; benchmark queries are English",
        "page_text_sources": {source: sum(row.get("text_source") == source for row in pages) for source in sorted({row.get("text_source") for row in pages})},
        "metrics": metrics,
        "timing_seconds": {"embedding": encode_seconds, "retrieval_and_cutoff": retrieval_seconds, "total": time.perf_counter() - started},
        "policy": {
            "file_stage": "max dense page score per file; top file-k",
            "page_stage": "dense cosine page ranking",
            "largest_gap": "largest adjacent score drop after minimum page-k",
            "buffer": 5,
            "qrels_used_for_ranking": False,
        },
    }
    _write_json(args.output_dir / "report.json", report)
    _write_jsonl(args.output_dir / "per_query_global_fixed10.jsonl", policy_rows["global_fixed10"])
    _write_jsonl(args.output_dir / "per_query_global_largest_gap.jsonl", policy_rows["global_largest_gap"])
    _write_jsonl(args.output_dir / "per_query_global_largest_gap_buffer5.jsonl", policy_rows["global_largest_gap_buffer5"])
    _write_jsonl(args.output_dir / "per_query_hier_fixed10.jsonl", policy_rows["hier_fixed10"])
    _write_jsonl(args.output_dir / "per_query_hier_largest_gap.jsonl", policy_rows["hier_largest_gap"])
    _write_jsonl(args.output_dir / "per_query_hier_largest_gap_buffer5.jsonl", policy_rows["hier_largest_gap_buffer5"])

    lines = [
        "# Dense text embedding + adaptive-k benchmark",
        "",
        f"- Corpus: {len(files)} files / {len(pages)} pages; queries: {len(queries)}.",
        f"- Model: `{args.model}`; input: cached PDF-inspector/Tesseract text; empty pages: {empty_pages}.",
        "- Query language: English text read from `queries.jsonl` as-is; no translation or qrel signal used.",
        "- `global`: all pages ranked by dense cosine; `hier`: top-3 files selected by max dense page score, then pages ranked inside them.",
        "",
        "| Policy | Page recall at returned K | Page recall@10 | Page hit@10 | File recall | Mean pages |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for policy in policy_names:
        item = metrics[policy]
        lines.append(f"| {policy} | {item['page_recall_at_returned_k']:.2%} | {item['page_recall_at_10']:.2%} | {item['page_hit_at_10']:.2%} | {item['file_recall_at_selected_files']:.2%} | {item['mean_returned_pages']:.2f} |")
    lines.extend([
        "",
        f"Embedding seconds: `{encode_seconds:.2f}`; retrieval/cutoff seconds: `{retrieval_seconds:.2f}`.",
        "",
        "The adaptive policies change pool size, not the ordering of the first ten pages. Their effect should therefore be judged by downstream second-retrieval recall and cost as well as page recall@10.",
    ])
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
