"""Compare PyMuPDF native text with the cached pdf-inspector preparation.

This is a retrieval-only ablation over the existing 220-query debug bundle.
It keeps the cached OCR text as a fallback, rebuilds the same plain-token
BM25 indexes, and applies the same Kf=3 hierarchical scoring contract.
"""

from __future__ import annotations

from collections import defaultdict
import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pymupdf

from src.retrieval.sparse import BM25Index


DEFAULT_BUNDLE = ROOT / "data/work/light_debug_bundle_benchmark_default_bm25_tesseract_kf3_top20"
DEFAULT_DATASET = ROOT / "data/raw/BENCHMARK"
DEFAULT_OUTPUT = DEFAULT_BUNDLE / "pymupdf_ablation"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def normalise(scores: Mapping[str, float]) -> dict[str, float]:
    maximum = max((float(value) for value in scores.values() if float(value) > 0), default=0.0)
    if maximum <= 0:
        return {key: 0.0 for key in scores}
    return {key: max(0.0, float(value)) / maximum for key, value in scores.items()}


def sort_scores(scores: Mapping[str, float]) -> list[tuple[str, float]]:
    return sorted(
        ((key, float(value)) for key, value in scores.items()),
        key=lambda item: (-item[1], item[0]),
    )


def page_search(index: BM25Index, query: str) -> dict[str, float]:
    return {
        index.chunk_ids[position]: float(score)
        for position, score in index.search(query, len(index.chunk_ids))
    }


def file_pool_max(
    page_scores: Mapping[str, float],
    page_to_file: Mapping[str, str],
    file_ids: Sequence[str],
) -> dict[str, float]:
    output = {file_id: 0.0 for file_id in file_ids}
    for page_id, score in page_scores.items():
        file_id = page_to_file[page_id]
        output[file_id] = max(output[file_id], max(0.0, float(score)))
    return output


def ndcg(ranked: Sequence[str], qrels: Mapping[str, int], depth: int = 10) -> float:
    actual = sum(
        (2.0 ** int(qrels[page_id]) - 1.0) / __import__("math").log2(rank + 1)
        for rank, page_id in enumerate(ranked[:depth], start=1)
        if page_id in qrels
    )
    ideal_values = sorted((int(value) for value in qrels.values()), reverse=True)[:depth]
    ideal = sum(
        (2.0 ** value - 1.0) / __import__("math").log2(rank + 1)
        for rank, value in enumerate(ideal_values, start=1)
    )
    return actual / ideal if ideal else 0.0


def evaluate(
    rows: Sequence[Mapping[str, Any]],
    qrels: Mapping[str, Mapping[str, int]],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    page_recall10: list[float] = []
    page_recall20: list[float] = []
    page_hit10: list[float] = []
    ndcg10: list[float] = []
    selected_file_recall: list[float] = []
    derived_file_recall: list[float] = []
    per_query: list[dict[str, Any]] = []

    for row in rows:
        query_id = str(row["query_id"])
        gold_pages = set(qrels.get(query_id, {}))
        gold_files = {page_id.split("#page=", 1)[0] for page_id in gold_pages}
        returned = [str(page_id) for page_id in row["returned_pages"]]
        selected = [str(file_id) for file_id in row["selected_files"]]
        derived_files: list[str] = []
        for page_id in row["candidate_pages_top100"]:
            file_id = page_id.split("#page=", 1)[0]
            if file_id not in derived_files:
                derived_files.append(file_id)
            if len(derived_files) >= 3:
                break
        denominator = len(gold_pages) or 1
        file_denominator = len(gold_files) or 1
        result = {
            "query_id": query_id,
            "page_recall@10": len(set(returned[:10]) & gold_pages) / denominator,
            "page_recall@20": len(set(returned[:20]) & gold_pages) / denominator,
            "page_hit@10": bool(set(returned[:10]) & gold_pages),
            "ndcg@10": 100.0 * ndcg(returned, qrels.get(query_id, {})),
            "candidate_file_recall@3": len(set(selected[:3]) & gold_files) / file_denominator,
            "derived_file_recall@3": len(set(derived_files[:3]) & gold_files) / file_denominator,
            "selected_files": selected,
            "gold_files": sorted(gold_files),
            "gold_pages": sorted(gold_pages),
            "top20_pages": returned[:20],
        }
        page_recall10.append(result["page_recall@10"])
        page_recall20.append(result["page_recall@20"])
        page_hit10.append(float(result["page_hit@10"]))
        ndcg10.append(result["ndcg@10"])
        selected_file_recall.append(result["candidate_file_recall@3"])
        derived_file_recall.append(result["derived_file_recall@3"])
        per_query.append(result)

    mean = lambda values: sum(values) / len(values) if values else 0.0
    return {
        "queries": len(rows),
        "page_recall@10": mean(page_recall10),
        "page_recall@20": mean(page_recall20),
        "page_hit@10": mean(page_hit10),
        "ndcg@10": mean(ndcg10),
        "candidate_file_recall@3": mean(selected_file_recall),
        "derived_file_recall@3": mean(derived_file_recall),
    }, per_query


def load_native_text(
    bundle: Path,
    dataset_root: Path,
) -> tuple[dict[str, str], dict[str, Any]]:
    rows = read_jsonl(bundle / "preparation/pages.jsonl")
    by_path: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_path[str(row["relative_path"])].append(row)

    native: dict[str, str] = {}
    errors: list[dict[str, str]] = []
    nonempty = 0
    started = time.perf_counter()
    for relative_path, page_rows in sorted(by_path.items()):
        path = dataset_root / relative_path
        try:
            with pymupdf.open(str(path)) as document:
                for row in page_rows:
                    page_index = int(row["page_index"])
                    text = document[page_index].get_text("text") or ""
                    native[str(row["page_id"])] = text
                    nonempty += bool(text.strip())
        except Exception as exc:
            errors.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})

    return native, {
        "documents": len(by_path),
        "pages": len(rows),
        "native_nonempty_pages": int(nonempty),
        "errors": errors,
        "seconds": round(time.perf_counter() - started, 3),
    }


def build_variant(
    name: str,
    page_rows: Sequence[Mapping[str, Any]],
    native_text: Mapping[str, str],
    bundle: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    page_order = [str(row["page_id"]) for row in page_rows]
    file_order = list(dict.fromkeys(str(row["doc_id"]) for row in page_rows))
    baseline_text = {str(row["page_id"]): str(row.get("text") or "") for row in page_rows}
    source = {str(row["page_id"]): str(row.get("text_source") or "") for row in page_rows}

    if name == "pymupdf_first":
        text = {
            page_id: native_text.get(page_id, "").strip() or baseline_text[page_id]
            for page_id in page_order
        }
    elif name == "pymupdf_replace_pdf_inspector":
        text = {
            page_id: (
                native_text.get(page_id, "").strip()
                if source[page_id] == "pdf_inspector" and native_text.get(page_id, "").strip()
                else baseline_text[page_id]
            )
            for page_id in page_order
        }
    else:
        raise ValueError(f"Unknown variant: {name}")

    page_to_file = {page_id: str(row["doc_id"]) for page_id, row in zip(page_order, page_rows, strict=True)}
    file_text = {file_id: "\n".join(text[page_id] for page_id in page_order if page_to_file[page_id] == file_id) for file_id in file_order}
    page_index = BM25Index(analyzer_name="plain").build(
        {"chunk_id": page_id, "doc_id": page_to_file[page_id], "text": text[page_id]}
        for page_id in page_order
    )
    file_index = BM25Index(analyzer_name="plain").build(
        {"chunk_id": file_id, "doc_id": file_id, "text": file_text[file_id]}
        for file_id in file_order
    )

    query_rows = read_jsonl(bundle / "input/queries.jsonl")
    qrel_rows = read_jsonl(bundle / "input/qrels_page_level.jsonl")
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for row in qrel_rows:
        qrels[str(row["query_id"])][str(row["page_id"])] = int(row.get("relevance", 1))

    rows: list[dict[str, Any]] = []
    for query in query_rows:
        query_id = str(query["query_id"])
        query_text = str(query["query"])
        page_raw = page_search(page_index, query_text)
        page_norm = normalise(page_raw)
        page_base = {page_id: 0.70 * page_norm.get(page_id, 0.0) for page_id in page_order}
        direct_raw = page_search(file_index, query_text)
        direct_norm = normalise(direct_raw)
        pooled = normalise(file_pool_max(page_base, page_to_file, file_order))
        file_scores = {
            file_id: 0.50 * direct_norm.get(file_id, 0.0) + 0.50 * pooled.get(file_id, 0.0)
            for file_id in file_order
        }
        selected = [file_id for file_id, _score in sort_scores(file_scores)[:3]]
        parent_norm = normalise(file_scores)
        selected_set = set(selected)
        final_scores = {
            page_id: 0.85 * page_base[page_id] + 0.15 * parent_norm.get(page_to_file[page_id], 0.0)
            for page_id in page_order
            if page_to_file[page_id] in selected_set
        }
        ranked = sort_scores(final_scores)
        top100 = [page_id for page_id, _score in ranked[:100]]
        chunks = [
            {"page_id": page_id, "doc_id": page_to_file[page_id], "rank": rank, "score": round(score, 8)}
            for rank, (page_id, score) in enumerate(ranked[:20], start=1)
        ]
        rows.append({
            "query_id": query_id,
            "query": query_text,
            "policy": name,
            "kf": 3,
            "kp": min(20, len(ranked)),
            "selected_files": selected,
            "candidate_page_count": sum(page_to_file[page_id] in selected_set for page_id in page_order),
            "candidate_pages_top100": top100,
            "returned_pages": [page_id for page_id, _score in ranked[:20]],
            "chunks": chunks,
        })

    metrics, per_query = evaluate(rows, qrels)
    return {"metrics": metrics, "per_query": per_query, "rows": rows, "page_text": text}, metrics


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    bundle = args.bundle.resolve()
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    page_rows = read_jsonl(bundle / "preparation/pages.jsonl")
    native_text, native_stats = load_native_text(bundle, dataset_root)

    reports: dict[str, Any] = {}
    for variant in ("pymupdf_first", "pymupdf_replace_pdf_inspector"):
        result, metrics = build_variant(variant, page_rows, native_text, bundle)
        variant_dir = output_dir / variant
        write_jsonl(variant_dir / "run_top20.jsonl", result["rows"])
        write_jsonl(variant_dir / "per_query.jsonl", result["per_query"])
        write_jsonl(
            variant_dir / "pages.jsonl",
            [
                {"page_id": page_id, "text": text}
                for page_id, text in result["page_text"].items()
            ],
        )
        reports[variant] = metrics

    baseline_report = json.loads((bundle / "retrieval/report.json").read_text(encoding="utf-8"))
    report = {
        "contract_version": "benchmark-pymupdf-ablation-v1",
        "bundle": str(bundle),
        "dataset_root": str(dataset_root),
        "native_extraction": native_stats,
        "baseline": {
            "page_recall@10": baseline_report["metrics"]["page_recall@10"],
            "page_recall@20": baseline_report["metrics"]["page_recall@20"],
            "page_hit@10": baseline_report["metrics"]["page_hit@10"],
            "ndcg@10": baseline_report["metrics"]["ndcg@10"],
            "candidate_file_recall@3": baseline_report["metrics"]["file_candidate_recall@3"],
            "derived_file_recall@3": baseline_report["metrics"]["derived_file_recall@3"],
        },
        "variants": reports,
        "retrieval_contract": {
            "page_bm25_weight": 0.70,
            "file_direct_weight": 0.50,
            "file_pool": "max",
            "parent_weight": 0.15,
            "file_k": 3,
            "saved_page_k": 20,
            "analyzer": "plain",
            "ocr": "cached baseline fallback only",
            "qrels_used_for_ranking": False,
        },
    }
    write_json(output_dir / "report.json", report)
    lines = [
        "# PyMuPDF native-text ablation",
        "",
        "Same cached OCR fallback, queries, qrels and hierarchical BM25 contract as the debug baseline.",
        "",
        "| Arm | Recall@10 | Recall@20 | Hit@10 | nDCG@10 | File recall@3 | Derived file recall@3 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    def row(label: str, item: Mapping[str, Any]) -> str:
        return f"| {label} | {item['page_recall@10']:.2%} | {item['page_recall@20']:.2%} | {item['page_hit@10']:.2%} | {item['ndcg@10']:.2f} | {item['candidate_file_recall@3']:.2%} | {item['derived_file_recall@3']:.2%} |"
    lines.append(row("Baseline pdf-inspector + cached OCR", report["baseline"]))
    for variant, metrics in reports.items():
        lines.append(row(variant, metrics))
    lines += [
        "",
        f"Native PyMuPDF extraction: {native_stats['native_nonempty_pages']}/{native_stats['pages']} pages non-empty in {native_stats['seconds']:.3f}s.",
        "",
        "The two variants differ only in whether PyMuPDF is allowed to replace cached Tesseract text on pages where native text is available.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
