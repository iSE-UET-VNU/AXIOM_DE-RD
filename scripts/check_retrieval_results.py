#!/usr/bin/env python3
"""Recompute page-level retrieval metrics from one on-demand benchmark run.

The checker intentionally does not import the pipeline metric implementation.
It independently scores raw page rankings and keeps file-level scoring only for
``file_recall``. This makes it useful for detecting accidental deduplication
before truncation.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from typing import Any
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _unique_top(items: list[str], k: int) -> list[str]:
    """Truncate the raw ranking first, then deduplicate identifiers."""
    output: list[str] = []
    for item in items[:k]:
        if item and item not in output:
            output.append(item)
    return output


def _short_id(value: str, head: int = 8, tail: int = 4) -> str:
    """Keep identifiers readable in spreadsheets while retaining audit clues."""
    text = str(value or "")
    page = ""
    if "#page=" in text:
        text, raw_page = text.split("#page=", 1)
        page = f"#p{raw_page}"
    namespace = ""
    if "::" in text:
        namespace, text = text.split("::", 1)
        namespace += ":"
    if len(text) > head + tail + 4:
        text = f"{text[:head]}....{text[-tail:]}"
    return namespace + text + page


def _score(ranked: list[str], grades: dict[str, float], k: int) -> tuple[float, float]:
    unique = _unique_top(ranked, k)
    recall = sum(doc_id in unique for doc_id in grades) / len(grades)
    gains = [grades.get(doc_id, 0.0) for doc_id in unique]
    ideal_gains = sorted(grades.values(), reverse=True)[:k]

    def dcg(values: list[float]) -> float:
        return sum(
            (2.0**gain - 1.0)
            / (1.0 if rank == 1 else math.log2(rank))
            for rank, gain in enumerate(values, start=1)
        )

    ideal = dcg(ideal_gains)
    ndcg = dcg(gains) / ideal if ideal else 0.0
    return recall, ndcg


def _grades(row: dict[str, Any], question: dict[str, Any] | None = None) -> tuple[dict[str, float], dict[str, float]]:
    """Return file-level and page-level qrels for one retrieval row."""
    file_grades: dict[str, float] = {}
    for qrel in row.get("gold_qrels") or []:
        doc_id = str(qrel.get("doc_id") or "")
        if doc_id:
            file_grades[doc_id] = max(
                file_grades.get(doc_id, 0.0), float(qrel.get("relevance") or 0.0)
            )
    page_source = row.get("gold_page_qrels")
    if page_source is None and question is not None:
        page_source = question.get("gold_page_qrels")
    page_grades = {
        str(qrel.get("page_id") or ""): float(qrel.get("relevance") or 0.0)
        for qrel in page_source or []
        if qrel.get("page_id")
    }
    return file_grades, page_grades


def _latest_report(run_dir: Path, prefix: str) -> Path | None:
    reports = sorted((run_dir / "reports").glob(f"lake_{prefix}_ver*.json"))
    return reports[-1] if reports else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("data/benchmark/0_benchmark_on_demand_basic_fixed"),
    )
    parser.add_argument("--top-k-pages", type=int, default=None)
    parser.add_argument("--top-k-chunks", type=int, default=None)
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Per-query CSV path (default: <run-dir>/retrieval_queries.csv)",
    )
    parser.add_argument(
        "--html-output",
        type=Path,
        default=None,
        help="Color-coded HTML path (default: <run-dir>/retrieval_queries.html)",
    )
    parser.add_argument(
        "--docbench-root",
        type=Path,
        default=None,
        help="Benchmark bundle root, needed when old rows lack gold_page_qrels.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    retrieval_path = run_dir / "retrieval" / "lake_baseline_legacy.jsonl"
    if not retrieval_path.is_file():
        raise FileNotFoundError(f"retrieval file not found: {retrieval_path}")

    report_path = _latest_report(run_dir, "baseline_legacy")
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path else {}
    question_by_qid: dict[str, dict[str, Any]] = {}
    manifest_paths = sorted(run_dir.glob("manifest_lake_ver*.json"))
    manifest_root = None
    if manifest_paths:
        manifest_root = json.loads(manifest_paths[-1].read_text(encoding="utf-8")).get(
            "docbench_root"
        )
    docbench_root = args.docbench_root or manifest_root
    if docbench_root:
        from research.data_discovery.run_docbench_e2e import _load_docbench

        _documents, questions = _load_docbench(Path(docbench_root))
        question_by_qid = {str(question["qid"]): question for question in questions}
    top_pages = int(args.top_k_pages or report.get("top_k_pages") or 20)
    top_chunks = int(args.top_k_chunks or report.get("top_k_chunks") or 10)
    if top_pages <= 0 or top_chunks <= 0:
        raise ValueError("top-k values must be positive")

    light: dict[str, list[float]] = {"recall@10": [], "ndcg@10": [], "recall@20": [], "ndcg@20": []}
    accurate: dict[str, list[float]] = {"recall@10": [], "ndcg@10": [], "recall@20": [], "ndcg@20": []}
    file_recall: list[float] = []
    missing_qrels = 0
    missing_page_qrels = 0
    malformed = 0
    accurate_scope_misses = 0
    query_rows: list[dict[str, Any]] = []
    for row in _read_jsonl(retrieval_path):
        grades, page_grades = _grades(row, question_by_qid.get(str(row.get("qid"))))
        if not grades:
            missing_qrels += 1
            continue
        if not page_grades:
            missing_page_qrels += 1
            continue

        hits = row.get("hits") or []
        chunks = row.get("chunks") or []
        if len(hits) < min(top_pages, 20) or len(chunks) < min(top_chunks, 10):
            malformed += 1

        light_pages = [str(hit.get("page_id") or "") for hit in hits]
        light_files = [str(hit.get("source_uri") or "") for hit in hits]
        page_to_file = {
            str(hit.get("page_id") or ""): str(hit.get("source_uri") or "")
            for hit in hits
            if hit.get("page_id") and hit.get("source_uri")
        }
        accurate_pages: list[str] = []
        accurate_files: list[str] = []
        for chunk in chunks:
            page_id = str(chunk.get("doc_id") or "")
            file_id = page_to_file.get(page_id)
            if file_id is None:
                accurate_scope_misses += 1
            accurate_pages.append(page_id)
            accurate_files.append(file_id or "")

        for k in (10, 20):
            recall, ndcg = _score(light_pages, page_grades, k)
            light[f"recall@{k}"].append(recall)
            light[f"ndcg@{k}"].append(ndcg)
        for k in (10, 20):
            recall, ndcg = _score(accurate_pages, page_grades, k)
            accurate[f"recall@{k}"].append(recall)
            accurate[f"ndcg@{k}"].append(ndcg)
        file_recall_20 = _score(light_files, grades, 20)[0]
        file_recall.append(file_recall_20)

        light_files_10 = _unique_top(light_files, 10)
        light_files_20 = _unique_top(light_files, 20)
        accurate_files_10 = _unique_top(accurate_files, 10)
        accurate_files_20 = _unique_top(accurate_files, 20)
        light_recall_10, _ = _score(light_pages, page_grades, 10)
        light_recall_20, _ = _score(light_pages, page_grades, 20)
        accurate_recall_10, _ = _score(accurate_pages, page_grades, 10)
        accurate_recall_20, _ = _score(accurate_pages, page_grades, 20)
        query_rows.append(
            {
                "qid": row.get("qid", ""),
                "question": row.get("question", ""),
                "gold_files": " | ".join(_short_id(item) for item in sorted(grades)),
                "gold_pages": " | ".join(_short_id(item) for item in sorted(page_grades)),
                "light_top20_pages": " | ".join(
                    f"{_short_id(str(hit.get('page_id') or ''))} -> "
                    f"{_short_id(str(hit.get('source_uri') or ''))}"
                    for hit in hits[:20]
                ),
                "light_unique_files_top10": " | ".join(_short_id(item) for item in light_files_10),
                "light_unique_files_top20": " | ".join(_short_id(item) for item in light_files_20),
                "file_recall_at_20": round(file_recall_20, 4),
                "light_recall_at_10": round(light_recall_10, 4),
                "light_recall_at_20": round(light_recall_20, 4),
                "light_ndcg_at_10": round(_score(light_pages, page_grades, 10)[1], 4),
                "light_ndcg_at_20": round(_score(light_pages, page_grades, 20)[1], 4),
                "accurate_top10_chunks": " | ".join(
                    f"{_short_id(str(chunk.get('chunk_id') or ''))} / "
                    f"{_short_id(str(chunk.get('doc_id') or ''))} -> "
                    f"{_short_id(page_to_file.get(str(chunk.get('doc_id') or ''), ''))}"
                    for chunk in chunks[:10]
                ),
                "accurate_unique_files_top10": " | ".join(_short_id(item) for item in accurate_files_10),
                "accurate_recall_at_10": round(accurate_recall_10, 4),
                "accurate_recall_at_20": round(accurate_recall_20, 4),
                "accurate_ndcg_at_10": round(_score(accurate_pages, page_grades, 10)[1], 4),
                "_light_hit10": bool(set(light_files_10) & set(grades)),
                "_light_hit20": bool(set(light_files_20) & set(grades)),
                "_accurate_hit10": bool(set(accurate_files_10) & set(grades)),
                "_light_ndcg10": _score(light_pages, page_grades, 10)[1],
                "_light_ndcg20": _score(light_pages, page_grades, 20)[1],
                "_accurate_ndcg10": _score(accurate_pages, page_grades, 10)[1],
                "_file_recall20": file_recall_20,
                "_light_recall10": light_recall_10,
                "_light_recall20": light_recall_20,
                "_accurate_recall10": accurate_recall_10,
            }
        )

    computed = {
        "run_dir": str(run_dir),
        "source_report": str(report_path) if report_path else None,
        "query_count": len(light["recall@10"]),
        "missing_qrels": missing_qrels,
        "missing_page_qrels": missing_page_qrels,
        "malformed_rows": malformed,
        "accurate_scope_misses": accurate_scope_misses,
        "top_k_pages": top_pages,
        "top_k_chunks": top_chunks,
        "file_recall@20": _mean(file_recall),
        "light": {key: _mean(value) for key, value in light.items()},
        "accurate": {key: _mean(value) for key, value in accurate.items()},
    }

    expected = {
        "light": {
            "recall@20": report.get("light_page_recall_at_20"),
            "ndcg@10": report.get("light_ndcg_at_10"),
            "recall@10": report.get("light_recall_at_10"),
            "ndcg@20": report.get("light_ndcg_at_20"),
        },
        "accurate": {
            "ndcg@10": report.get("accurate_ndcg_at_10"),
            "recall@10": report.get("accurate_recall_at_10"),
        },
    }
    expected["file_recall@20"] = report.get("light_file_recall_at_20")
    comparisons: dict[str, dict[str, Any]] = {}
    for stage, values in expected.items():
        if stage == "file_recall@20":
            expected_value = values
            if expected_value is not None:
                actual = computed[stage]
                comparisons[stage] = {
                    "computed": actual,
                    "reported": expected_value,
                    "match": abs(float(actual) - float(expected_value)) <= 1e-4,
                }
            continue
        for metric, expected_value in values.items():
            if expected_value is None:
                continue
            actual = computed[stage][metric]
            comparisons[f"{stage}.{metric}"] = {
                "computed": actual,
                "reported": expected_value,
                "match": abs(float(actual) - float(expected_value)) <= 1e-4,
            }
    computed["report_comparison"] = comparisons

    csv_path = args.csv_output or (run_dir / "retrieval_queries.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "qid", "question", "gold_files", "gold_pages", "light_top20_pages",
        "light_unique_files_top10", "light_unique_files_top20",
        "file_recall_at_20",
        "light_recall_at_10", "light_recall_at_20", "light_ndcg_at_10",
        "light_ndcg_at_20", "accurate_top10_chunks", "accurate_unique_files_top10",
        "accurate_recall_at_10", "accurate_recall_at_20", "accurate_ndcg_at_10",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(query_rows)
        writer.writerow({})
        formula_writer = csv.writer(handle)
        formula_writer.writerow(["FORMULA", "NDCG per query", "DCG@k / IDCG@k", ""])
        formula_writer.writerow([
            "FORMULA",
            "DCG@k",
            "Σ((2^relevance_at_rank - 1) / log2(rank + 1)) for retrieved top-k pages",
            "",
        ])
        formula_writer.writerow([
            "FORMULA",
            "IDCG@k",
            "Σ((2^ideal_relevance_at_rank - 1) / log2(rank + 1)) for ideal top-k gold pages",
            "",
        ])
        formula_writer.writerow([
            "FORMULA",
            "File deduplication order",
            "file_recall only: raw top-k pages -> source_uri -> unique files -> score",
            "",
        ])
        summary_writer = csv.writer(handle)
        summary_writer.writerow(["SUMMARY", "Metric", "Formula", "Value"])
        def excel_column(number: int) -> str:
            letters = ""
            while number:
                number, remainder = divmod(number - 1, 26)
                letters = chr(65 + remainder) + letters
            return letters

        query_first_row = 2
        query_last_row = len(query_rows) + 1
        for label, key in (
            ("File Recall@20", "file_recall_at_20"),
            ("Light NDCG@10", "light_ndcg_at_10"),
            ("Light NDCG@20", "light_ndcg_at_20"),
            ("Light Recall@10", "light_recall_at_10"),
            ("Light Recall@20", "light_recall_at_20"),
            ("Accurate NDCG@10", "accurate_ndcg_at_10"),
            ("Accurate Recall@10", "accurate_recall_at_10"),
        ):
            column = excel_column(columns.index(key) + 1)
            formula = (
                f"=SUM({column}{query_first_row}:{column}{query_last_row})/"
                f"COUNT({column}{query_first_row}:{column}{query_last_row})"
            )
            summary_writer.writerow(["SUMMARY", label, formula, formula])
    computed["csv_output"] = str(csv_path)

    html_path = args.html_output or (run_dir / "retrieval_queries.html")
    html_path.parent.mkdir(parents=True, exist_ok=True)
    headers = columns
    table_rows = []
    for item in query_rows:
        cells = []
        for column in headers:
            value = html.escape(str(item.get(column, "")))
            hit = column == "light_unique_files_top10" and item["_light_hit10"]
            hit = hit or (column == "light_unique_files_top20" and item["_light_hit20"])
            hit = hit or (column == "accurate_unique_files_top10" and item["_accurate_hit10"])
            style = ' style="background:#ffb3b3"' if hit else ""
            cells.append(f"<td{style}>{value}</td>")
        table_rows.append("<tr>" + "".join(cells) + "</tr>")
    html_doc = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Retrieval query check</title>"
        "<style>body{font:14px sans-serif}table{border-collapse:collapse}"
        "th,td{border:1px solid #bbb;padding:6px;vertical-align:top;max-width:560px;"
        "white-space:pre-wrap}th{position:sticky;top:0;background:#eee}</style></head>"
        "<body><p><b>Red cell:</b> retrieved unique file contains at least one gold evidence file.</p>"
        "<table><thead><tr>"
        + "".join(f"<th>{html.escape(column)}</th>" for column in headers)
        + "</tr></thead><tbody>"
        + "".join(table_rows)
        + "</tbody></table>"
        + "<h2>Recomputed summary</h2>"
        + "<p>Each metric is the arithmetic mean over query-level values in the table. "
          "For each query, NDCG@k = DCG@k / IDCG@k; the reported aggregate is "
          "sum(query NDCG@k) / number of queries. Recall/NDCG rank pages; only file recall deduplicates by file.</p>"
        + "<ul>"
        + "".join(
            f"<li>{html.escape(label)} = "
            f"{sum(float(item[key]) for item in query_rows):.4f} / {len(query_rows)} "
            f"= {sum(float(item[key]) for item in query_rows) / len(query_rows):.4f}</li>"
            for label, key in (
                ("Light NDCG@10", "_light_ndcg10"),
                ("Light NDCG@20", "_light_ndcg20"),
                ("Light Recall@10", "_light_recall10"),
                ("Light Recall@20", "_light_recall20"),
                ("Accurate NDCG@10", "_accurate_ndcg10"),
                ("Accurate Recall@10", "_accurate_recall10"),
                ("File Recall@20", "_file_recall20"),
        )
        )
        + "</ul></body></html>"
    )
    html_path.write_text(html_doc, encoding="utf-8")
    computed["html_output"] = str(html_path)

    print(json.dumps(computed, indent=2, ensure_ascii=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(computed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0 if all(item["match"] for item in comparisons.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
