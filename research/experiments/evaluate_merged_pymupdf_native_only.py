"""Evaluate strict native PyMuPDF text on the merged 220-query benchmark.

The official merged PyMuPDF arm is intentionally a fallback arm: it uses
native PyMuPDF text when available and cached preparation text otherwise.
This experiment removes that fallback and keeps empty native pages empty. It
is meant to explain the official arm, not to replace it in the main table.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time
from typing import Any

import pymupdf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.experiments.run_ohr_replacement_all_arms import (  # noqa: E402
    OLD_BASE_PAGES,
    OLD_DOCS,
    OHR_BASE_PAGES,
    build_bm25_arm,
    filter_old,
    load_inputs,
    load_page_corpora,
    metric_rows,
    read_jsonl,
    write_jsonl,
)


DEFAULT_OUTPUT = ROOT / "data/work/ohr_replacement_20260920/retrieval/pymupdf_native_only"
DATASET_ROOT = ROOT / "data/raw/0.1. BENCHMARK"


def page_path(row: dict[str, Any]) -> Path:
    relative = str(row.get("relative_path") or "")
    path = DATASET_ROOT / relative
    if path.is_file():
        return path
    source_path = Path(str(row.get("source_path") or ""))
    if source_path.is_file():
        return source_path
    raise FileNotFoundError(f"Cannot resolve source PDF for {row['page_id']}: {relative}")


def load_current_rows() -> list[dict[str, Any]]:
    source_by_doc = {
        str(row["doc_id"]): str(row.get("source") or "")
        for row in read_jsonl(OLD_DOCS)
    }
    old_rows = filter_old(read_jsonl(OLD_BASE_PAGES), source_by_doc)
    ohr_rows = read_jsonl(OHR_BASE_PAGES)
    rows = old_rows + ohr_rows
    if len(rows) != 5091:
        raise RuntimeError(f"Expected 5,091 merged pages, got {len(rows)}")
    return rows


def extract_native(rows: list[dict[str, Any]]) -> tuple[dict[str, str], list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[page_path(row)].append(row)

    started = time.perf_counter()
    native: dict[str, str] = {}
    page_meta: list[dict[str, Any]] = []
    for path, page_rows in sorted(grouped.items(), key=lambda item: str(item[0])):
        with pymupdf.open(path) as document:
            for row in page_rows:
                text = (document[int(row["page_index"])].get_text("text") or "").strip()
                page_id = str(row["page_id"])
                native[page_id] = text
                page_meta.append(
                    {
                        "page_id": page_id,
                        "source": str(row.get("source") or ""),
                        "relative_path": str(row.get("relative_path") or ""),
                        "page_index": int(row["page_index"]),
                        "parser_text_source": str(row.get("text_source") or ""),
                        "native_nonempty": bool(text),
                        "native_chars": len(text),
                        "native_words": len(text.split()),
                    }
                )

    native_nonempty = sum(bool(value) for value in native.values())
    meta = {
        "documents": len(grouped),
        "pages": len(rows),
        "native_nonempty_pages": native_nonempty,
        "native_empty_pages": len(rows) - native_nonempty,
        "seconds": time.perf_counter() - started,
        "native_empty_by_source": dict(
            Counter(row["source"] for row in page_meta if not row["native_nonempty"])
        ),
        "native_nonempty_by_source": dict(
            Counter(row["source"] for row in page_meta if row["native_nonempty"])
        ),
    }
    return native, page_meta, meta


def main() -> int:
    output = DEFAULT_OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    rows = load_current_rows()
    native, page_meta, native_stats = extract_native(rows)
    write_jsonl(output / "page_native_status.jsonl", page_meta)

    queries, qrels, languages = load_inputs()
    page_ids = [str(row["page_id"]) for row in rows]
    native_text = {page_id: native.get(page_id, "") for page_id in page_ids}
    started = time.perf_counter()
    native_rows = build_bm25_arm(native_text, queries, languages)
    retrieval_seconds = time.perf_counter() - started
    write_jsonl(output / "run_top20.jsonl", native_rows)

    corpora, _ = load_page_corpora()
    baseline_rows = read_jsonl(
        ROOT / "data/work/ohr_replacement_20260920/retrieval/merged_220_all_arms/runs/baseline_bm25.jsonl"
    )
    fallback_rows = read_jsonl(
        ROOT / "data/work/ohr_replacement_20260920/retrieval/merged_220_all_arms/runs/pymupdf.jsonl"
    )
    report = {
        "contract_version": "merged-pymupdf-native-only-v1",
        "protocol": {
            "queries": 220,
            "files": 103,
            "pages": 5091,
            "file_k": 3,
            "candidate_depth": 200,
            "saved_depth": 20,
            "qrels_used_for_ranking": False,
            "native_call": "Page.get_text('text')",
            "fallback": "disabled; native-empty pages have empty text",
        },
        "native_extraction": native_stats,
        "retrieval_seconds": retrieval_seconds,
        "metrics": {
            "baseline_bm25": {
                "@10": metric_rows(baseline_rows, qrels, 10),
                "@20": metric_rows(baseline_rows, qrels, 20),
            },
            "pymupdf_fallback_official": {
                "@10": metric_rows(fallback_rows, qrels, 10),
                "@20": metric_rows(fallback_rows, qrels, 20),
            },
            "pymupdf_native_only": {
                "@10": metric_rows(native_rows, qrels, 10),
                "@20": metric_rows(native_rows, qrels, 20),
            },
        },
        "artifacts": {
            "page_native_status": str((output / "page_native_status.jsonl").resolve()),
            "run_top20": str((output / "run_top20.jsonl").resolve()),
        },
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Merged PyMuPDF native-only ablation",
        "",
        "| Arm | Recall@10 | Recall@20 | nDCG@10 | File recall@3 | Candidate recall@200 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "baseline_bm25": "Baseline BM25",
        "pymupdf_fallback_official": "PyMuPDF + fallback (official)",
        "pymupdf_native_only": "PyMuPDF native-only",
    }
    for key, label in labels.items():
        m10 = report["metrics"][key]["@10"]
        m20 = report["metrics"][key]["@20"]
        lines.append(
            f"| {label} | {m10['page_recall@10']:.2%} | "
            f"{m20['page_recall@20']:.2%} | {m10['ndcg@10']:.2f} | "
            f"{m10['file_recall@3']:.2%} | {m10['candidate_pool_recall@200']:.2%} |"
        )
    lines += [
        "",
        f"Native extraction: {native_stats['native_nonempty_pages']}/"
        f"{native_stats['pages']} non-empty pages in "
        f"{native_stats['seconds']:.3f}s.",
        f"Native-only BM25 retrieval: {retrieval_seconds:.3f}s.",
        "",
        "This is an explanatory ablation. The official PyMuPDF arm keeps the "
        "cached page text when native extraction is empty.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Output: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
