"""Compare PDF-inspector BM25 and English-query V-SPLADE per query.

The paired runs are evaluated against the French ViDoRe V3 Physics qrels:

* BM25 receives the French query over PDF-inspector page text.
* V-SPLADE receives the English translation over rendered page images.

The primary discovery metric is file recall@3. It is computed by taking the
first 100 ranked page candidates, deduplicating them by source file, and
keeping the first three unique files. Page-level top-10 diagnostics are also
kept so that file-level and page-level behaviour can be compared directly.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.benchmarks.vidore_v3 import ViDoreV3  # noqa: E402


DEFAULT_RUN_DIR = ROOT / "data/benchmark/vidore_v3/results/physics_vsplade_bm25_fusion"
DEFAULT_OUTPUT_DIR = DEFAULT_RUN_DIR / "per_query_file_recall_at3"


def _read_run(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[str(record["qid"])] = record
    return records


def _file_id(page_id: str) -> str:
    return page_id.split("#page=", 1)[0]


def _unique_files(chunks: list[dict[str, Any]], file_k: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        file_id = _file_id(str(chunk["chunk_id"]))
        if file_id not in seen:
            seen.add(file_id)
            out.append(file_id)
        if len(out) == file_k:
            break
    return out


def _rank_of_first(chunks: list[dict[str, Any]], gold: set[str], depth: int) -> int | None:
    for position, chunk in enumerate(chunks[:depth], 1):
        if str(chunk["chunk_id"]) in gold:
            return position
    return None


def _arm_stats(
    chunks: list[dict[str, Any]],
    gold_pages: set[str],
    page_k: int,
    file_k: int,
    file_candidate_depth: int,
) -> dict[str, Any]:
    retrieved_pages = [str(chunk["chunk_id"]) for chunk in chunks[:page_k]]
    page_hits = sorted(gold_pages & set(retrieved_pages))
    gold_files = {_file_id(page) for page in gold_pages}
    files = _unique_files(chunks[:file_candidate_depth], file_k)
    file_hits = sorted(gold_files & set(files))
    return {
        "page_hit": bool(page_hits),
        "page_recall": len(page_hits) / len(gold_pages) if gold_pages else 0.0,
        "first_gold_page_rank": _rank_of_first(chunks, gold_pages, page_k),
        "gold_pages_found": page_hits,
        "file_hit": bool(file_hits),
        "file_recall": len(file_hits) / len(gold_files) if gold_files else 0.0,
        "first_gold_file_rank": next(
            (rank for rank, file_id in enumerate(files, 1) if file_id in gold_files), None
        ),
        "gold_files_found": file_hits,
        "top_pages": retrieved_pages,
        "top_files": files,
    }


def _mean(items: list[float]) -> float:
    return sum(items) / len(items) if items else 0.0


def _group_value(value: Any) -> str:
    if isinstance(value, list):
        return " + ".join(str(item) for item in value) or "(none)"
    return str(value or "(none)")


def _group_summary(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = row["metadata"].get(field) if field != "gold_file_count" else row[field]
        groups[_group_value(value)].append(row)
    return {
        name: _aggregate(group)
        for name, group in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
    }


def _group_summary_components(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    """Aggregate list-valued metadata by each individual component."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = row["metadata"].get(field)
        values = value if isinstance(value, list) else [value]
        for component in values:
            groups[_group_value(component)].append(row)
    return {
        name: _aggregate(group)
        for name, group in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    file_classes = Counter(row["file_class"] for row in rows)
    page_classes = Counter(row["page_class"] for row in rows)
    winners = Counter(row["winner"] for row in rows)
    return {
        "queries": len(rows),
        "file_classes": dict(sorted(file_classes.items())),
        "page_classes": dict(sorted(page_classes.items())),
        "winners_by_file_recall@3": dict(sorted(winners.items())),
        "bm25_page_hit_rate": _mean([float(row["bm25"]["page_hit"]) for row in rows]),
        "vsplade_page_hit_rate": _mean([float(row["vsplade"]["page_hit"]) for row in rows]),
        "bm25_page_recall": _mean([row["bm25"]["page_recall"] for row in rows]),
        "vsplade_page_recall": _mean([row["vsplade"]["page_recall"] for row in rows]),
        "bm25_file_hit_rate": _mean([float(row["bm25"]["file_hit"]) for row in rows]),
        "vsplade_file_hit_rate": _mean([float(row["vsplade"]["file_hit"]) for row in rows]),
        "bm25_file_recall@3": _mean([row["bm25"]["file_recall"] for row in rows]),
        "vsplade_file_recall@3": _mean([row["vsplade"]["file_recall"] for row in rows]),
        "delta_file_recall@3_bm25_minus_vsplade": _mean(
            [row["bm25"]["file_recall"] - row["vsplade"]["file_recall"] for row in rows]
        ),
        "mean_top10_page_overlap": _mean([row["top10_page_jaccard"] for row in rows]),
        "mean_top3_file_overlap": _mean([row["top3_file_jaccard"] for row in rows]),
    }


def _compact_example(row: dict[str, Any]) -> dict[str, Any]:
    def compact(arm: str) -> dict[str, Any]:
        return {
            key: row[arm][key]
            for key in (
                "page_recall",
                "first_gold_page_rank",
                "gold_pages_found",
                "file_recall",
                "first_gold_file_rank",
                "gold_files_found",
                "top_pages",
                "top_files",
            )
        }

    return {
        "qid": row["qid"],
        "query_french": row["query_french"],
        "query_english": row["query_english"],
        "answer": row["answer"],
        "metadata": row["metadata"],
        "gold_pages": row["gold_pages"],
        "gold_files": row["gold_files"],
        "file_class": row["file_class"],
        "page_class": row["page_class"],
        "winner": row["winner"],
        "bm25": compact("bm25"),
        "vsplade": compact("vsplade"),
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    overview = report["overview"]
    setup = report["setup"]

    def pct(value: float) -> str:
        return f"{value:.1%}"

    def file_name(file_id: str) -> str:
        return file_id.split("::", 1)[-1]

    def md(value: Any, limit: int | None = None) -> str:
        text = str(value).replace("|", "\\|").replace("\n", " ")
        if limit is not None and len(text) > limit:
            text = text[: limit - 1].rstrip() + "..."
        return text

    def group_table(title: str, groups: dict[str, dict[str, Any]]) -> list[str]:
        lines = [
            f"### {title}",
            "",
            "| Group | Queries | BM25 file recall@3 | V-SPLADE file recall@3 | Delta BM25-V | BM25 hit@3 | V-SPLADE hit@3 | Preferred |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
        for name, stats in groups.items():
            delta = stats["delta_file_recall@3_bm25_minus_vsplade"]
            preferred = "BM25" if delta > 1e-12 else "V-SPLADE" if delta < -1e-12 else "tie"
            lines.append(
                f"| {md(name)} | {stats['queries']} | {pct(stats['bm25_file_recall@3'])} | "
                f"{pct(stats['vsplade_file_recall@3'])} | {delta:+.1%} | "
                f"{pct(stats['bm25_file_hit_rate'])} | {pct(stats['vsplade_file_hit_rate'])} | {preferred} |"
            )
        lines.append("")
        return lines

    lines = [
        "# Physics: PDF-inspector BM25 vs V-SPLADE retrieval correlation",
        "",
        "This report compares the two rankings query by query. BM25 uses French queries over PDF-inspector page text; V-SPLADE uses the paired English translation over rendered page images. Evaluation remains against the French Physics qrels.",
        "",
        "## Protocol",
        "",
        f"- Page diagnostic: first `{setup['page_k']}` ranked pages.",
        f"- Primary file diagnostic: first `{setup['file_k']}` unique files after deduplicating the first `{setup['file_candidate_depth']}` page candidates.",
        "- A file is relevant when it contains at least one gold evidence page.",
        "- `file recall@3` is the fraction of gold evidence files recovered in those three files; it is not page recall.",
        "- The full query-level comparison is in [per_query.md](per_query.md). Full untruncated values are in [report.json](report.json).",
        "",
        "## Overall file-level correlation",
        "",
        "| Metric | PDF-inspector + BM25 | V-SPLADE |",
        "|---|---:|---:|",
        f"| File recall@3 | {pct(overview['bm25_file_recall@3'])} | {pct(overview['vsplade_file_recall@3'])} |",
        f"| File hit@3 | {pct(overview['bm25_file_hit_rate'])} | {pct(overview['vsplade_file_hit_rate'])} |",
        f"| Page hit@10 (secondary) | {pct(overview['bm25_page_hit_rate'])} | {pct(overview['vsplade_page_hit_rate'])} |",
        f"| Page recall@10 (secondary) | {pct(overview['bm25_page_recall'])} | {pct(overview['vsplade_page_recall'])} |",
        f"| Mean top-3 file-set Jaccard | {pct(overview['mean_top3_file_overlap'])} | - |",
        "",
        "## File-level complementarity",
        "",
        "The classes below use file hit@3, so they answer the data-discovery question directly. The separate page class is retained in JSON for comparison with the earlier page-level report.",
        "",
        "| Class | Queries | Share | Meaning |",
        "|---|---:|---:|---|",
    ]
    meanings = {
        "both_hit": "Both retrieve at least one gold file in top-3 files.",
        "bm25_only": "Only PDF-inspector + BM25 retrieves a gold file in top-3.",
        "vsplade_only": "Only V-SPLADE retrieves a gold file in top-3.",
        "both_miss": "Neither retrieves a gold file in top-3.",
    }
    for name in ("both_hit", "bm25_only", "vsplade_only", "both_miss"):
        count = overview["file_classes"].get(name, 0)
        lines.append(f"| {name} | {count} | {count / overview['queries']:.1%} | {meanings[name]} |")

    lines += [
        "",
        "## Winner counts",
        "",
        "A winner means the arm has higher file recall@3 for that query; ties are expected for one-file queries where both either hit or miss.",
        "",
        "| Winner | Queries | Share |",
        "|---|---:|---:|",
        f"| BM25 | {overview['winners_by_file_recall@3'].get('bm25', 0)} | {overview['winners_by_file_recall@3'].get('bm25', 0) / overview['queries']:.1%} |",
        f"| V-SPLADE | {overview['winners_by_file_recall@3'].get('vsplade', 0)} | {overview['winners_by_file_recall@3'].get('vsplade', 0) / overview['queries']:.1%} |",
        f"| Tie | {overview['winners_by_file_recall@3'].get('tie', 0)} | {overview['winners_by_file_recall@3'].get('tie', 0) / overview['queries']:.1%} |",
        "",
        "## Which query groups favor which arm?",
        "",
    ]
    for field, title in (
        ("by_query_type", "Query-type combinations"),
        ("by_query_type_component", "Query-type components"),
        ("by_query_format", "Query format"),
        ("by_content_type", "Evidence content type"),
        ("by_source_type", "Evidence source type"),
        ("by_gold_file_count", "Number of gold files"),
    ):
        lines.extend(group_table(title, report[field]))

    lines += [
        "## Strongest directional differences",
        "",
        "Only groups with at least five queries are shown, to reduce overinterpretation of tiny groups.",
        "",
    ]
    candidates: list[tuple[float, str, str, dict[str, Any]]] = []
    for field in (
        "by_query_type_component",
        "by_query_format",
        "by_content_type",
        "by_source_type",
        "by_gold_file_count",
    ):
        for name, stats in report[field].items():
            if stats["queries"] >= 5:
                candidates.append((stats["delta_file_recall@3_bm25_minus_vsplade"], field, name, stats))
    for delta, field, name, stats in sorted(candidates, key=lambda item: item[0], reverse=True)[:5]:
        lines.append(f"- **BM25 advantage** - `{md(name)}` ({field}, n={stats['queries']}): {delta:+.1%} file recall@3.")
    for delta, field, name, stats in sorted(candidates, key=lambda item: item[0])[:5]:
        lines.append(f"- **V-SPLADE advantage** - `{md(name)}` ({field}, n={stats['queries']}): {delta:+.1%} BM25-V-SPLADE file recall@3.")

    lines += ["", "## Example queries", ""]
    for label in ("bm25_only", "vsplade_only", "both_miss"):
        examples = report["examples"].get(label, [])
        lines += [f"### {label}", ""]
        if not examples:
            lines.append("No examples.")
        for example in examples:
            lines += [
                f"- `{example['qid']}` - {md(example['query_french'], 220)}",
                f"  - Gold files: {', '.join(file_name(item) for item in example['gold_files'])}",
                f"  - BM25 top-3 files: {', '.join(file_name(item) for item in example['bm25']['top_files'])}; file recall@3 `{example['bm25']['file_recall']:.1%}`.",
                f"  - V-SPLADE top-3 files: {', '.join(file_name(item) for item in example['vsplade']['top_files'])}; file recall@3 `{example['vsplade']['file_recall']:.1%}`.",
                f"  - Query types: `{md(example['metadata'].get('query_types', '(none)'))}`; content: `{md(example['metadata'].get('content_type', '(none)'))}`.",
            ]
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_per_query_markdown(path: Path, report: dict[str, Any]) -> None:
    def file_name(file_id: str) -> str:
        return file_id.split("::", 1)[-1]

    def md(value: Any, limit: int | None = None) -> str:
        text = str(value).replace("|", "\\|").replace("\n", " ")
        if limit is not None and len(text) > limit:
            text = text[: limit - 1].rstrip() + "..."
        return text

    lines = [
        "# Physics: per-query BM25 vs V-SPLADE comparison",
        "",
        "The primary class and winner use file recall@3. The three files are the first unique files obtained from the first 100 ranked page candidates.",
        "",
        "| QID | French query | Gold files | BM25 top-3 files | BM25 file recall@3 | BM25 page recall@10 / rank | V-SPLADE top-3 files | V-SPLADE file recall@3 | V-SPLADE page recall@10 / rank | Winner | File class | Page class@10 |",
        "|---|---|---|---|---:|---:|---|---:|---:|---|---|---|",
    ]
    for row in report["per_query"]:
        lines.append(
            f"| `{row['qid']}` | {md(row['query_french'], 240)} | "
            f"{md('<br>'.join(file_name(item) for item in row['gold_files']), 180)} | "
            f"{md('<br>'.join(file_name(item) for item in row['bm25']['top_files']), 240)} | "
            f"{row['bm25']['file_recall']:.1%} | "
            f"{row['bm25']['page_recall']:.1%} / {row['bm25']['first_gold_page_rank'] or '-'} | "
            f"{md('<br>'.join(file_name(item) for item in row['vsplade']['top_files']), 240)} | "
            f"{row['vsplade']['file_recall']:.1%} | "
            f"{row['vsplade']['page_recall']:.1%} / {row['vsplade']['first_gold_page_rank'] or '-'} | "
            f"{row['winner']} | {row['file_class']} | {row['page_class']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument(
        "--bm25-run",
        type=Path,
        default=None,
        help="Optional lexical run JSONL. Defaults to the PDF-inspector BM25 run.",
    )
    parser.add_argument(
        "--vsplade-run",
        type=Path,
        default=None,
        help="Optional V-SPLADE run JSONL. Defaults to the English-query V-SPLADE run.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--depth", "--page-k", dest="page_k", type=int, default=10)
    parser.add_argument(
        "--file-k",
        type=int,
        default=3,
        help="Number of unique files used for the primary file-level metric.",
    )
    parser.add_argument(
        "--file-candidate-depth",
        type=int,
        default=100,
        help="Number of ranked page candidates from which unique files are formed.",
    )
    parser.add_argument("--examples", type=int, default=12)
    args = parser.parse_args()

    if args.page_k <= 0 or args.file_k <= 0 or args.file_candidate_depth <= 0:
        raise ValueError("page-k, file-k and file-candidate-depth must be positive")

    bm25_path = args.bm25_run or (args.run_dir / "bm25_french_bm25-french_vs-english.jsonl")
    vsplade_path = args.vsplade_run or (args.run_dir / "vsplade_french_bm25-french_vs-english.jsonl")
    bm25_runs = _read_run(bm25_path)
    vsplade_runs = _read_run(vsplade_path)

    benchmark = ViDoreV3(
        root=ROOT / "data/benchmark/vidore_v3", subset="physics", language="french"
    )
    french_questions = sorted(
        benchmark.questions(), key=lambda question: int(question.qid.rsplit("::", 1)[1])
    )
    english_questions = sorted(
        ViDoreV3(
            root=ROOT / "data/benchmark/vidore_v3", subset="physics", language="english"
        ).questions(),
        key=lambda question: int(question.qid.rsplit("::", 1)[1]),
    )
    if len(french_questions) != 302 or len(english_questions) != 302:
        raise RuntimeError(
            f"Expected 302 French and English Physics queries, found {len(french_questions)} and {len(english_questions)}"
        )
    qids = {question.qid for question in french_questions}
    if set(bm25_runs) != qids:
        raise RuntimeError("BM25 run does not contain exactly the 302 French query IDs.")
    if set(vsplade_runs) != qids:
        raise RuntimeError("V-SPLADE run does not contain exactly the 302 French query IDs.")
    import pandas as pd

    raw_queries = pd.read_parquet(
        ROOT / "data/benchmark/vidore_v3/physics/queries.parquet",
        engine="fastparquet",
    ).to_dict("records")
    metadata = {
        f"physics::{row['query_id']}": {
            key: row.get(key)
            for key in ("query_types", "query_format", "content_type", "source_type")
        }
        for row in raw_queries
        if str(row.get("language", "")).lower() == "french"
    }

    qrels = benchmark.qrels()
    rows: list[dict[str, Any]] = []
    for french, english in zip(french_questions, english_questions):
        bm25_chunks = bm25_runs[french.qid]["chunks"]
        vsplade_chunks = vsplade_runs[french.qid]["chunks"]
        if len(bm25_chunks) < args.file_candidate_depth:
            raise RuntimeError(f"BM25 has fewer than 100 candidates for {french.qid}")
        if len(vsplade_chunks) < args.file_candidate_depth:
            raise RuntimeError(f"V-SPLADE has fewer than 100 candidates for {french.qid}")

        gold_pages = set(qrels.get(french.qid, {}))
        bm25 = _arm_stats(
            bm25_chunks, gold_pages, args.page_k, args.file_k, args.file_candidate_depth
        )
        vsplade = _arm_stats(
            vsplade_chunks, gold_pages, args.page_k, args.file_k, args.file_candidate_depth
        )

        def classify(left: dict[str, Any], right: dict[str, Any], key: str) -> str:
            if left[key] and right[key]:
                return "both_hit"
            if left[key]:
                return "bm25_only"
            if right[key]:
                return "vsplade_only"
            return "both_miss"

        file_class = classify(bm25, vsplade, "file_hit")
        page_class = classify(bm25, vsplade, "page_hit")
        if bm25["file_recall"] > vsplade["file_recall"]:
            winner = "bm25"
        elif vsplade["file_recall"] > bm25["file_recall"]:
            winner = "vsplade"
        else:
            winner = "tie"

        bm25_top = set(bm25["top_pages"])
        vsplade_top = set(vsplade["top_pages"])
        bm25_files = set(bm25["top_files"])
        vsplade_files = set(vsplade["top_files"])
        gold_files = sorted({_file_id(page) for page in gold_pages})
        rows.append(
            {
                "qid": french.qid,
                "query_french": french.query,
                "query_english": english.query,
                "answer": french.answer,
                "metadata": metadata.get(french.qid, {}),
                "gold_pages": sorted(gold_pages),
                "gold_files": gold_files,
                "gold_file_count": len(gold_files),
                "class": file_class,
                "file_class": file_class,
                "page_class": page_class,
                "winner": winner,
                "bm25": bm25,
                "vsplade": vsplade,
                "top10_page_jaccard": len(bm25_top & vsplade_top) / len(bm25_top | vsplade_top),
                "top3_file_jaccard": len(bm25_files & vsplade_files) / len(bm25_files | vsplade_files),
            }
        )

    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_class[row["file_class"]].append(row)
    examples: dict[str, list[dict[str, Any]]] = {}
    for label, group in by_class.items():
        ranked = sorted(
            group,
            key=lambda row: (
                row["vsplade"]["first_gold_file_rank"] or 999
                if label == "vsplade_only"
                else row["bm25"]["first_gold_file_rank"] or 999,
                -max(row["bm25"]["file_recall"], row["vsplade"]["file_recall"]),
                row["qid"],
            ),
        )
        examples[label] = [_compact_example(row) for row in ranked[: args.examples]]

    report = {
        "setup": {
            "benchmark": "vidore_v3/physics",
            "query_count": len(rows),
            "evaluation_language": "french",
            "bm25_query_language": "french",
            "vsplade_query_language": "english",
            "page_k": args.page_k,
            "file_k": args.file_k,
            "file_candidate_depth": args.file_candidate_depth,
            "file_metric": "fraction of gold files found in first 3 unique files from first 100 page candidates",
            "bm25_run": str(bm25_path),
            "vsplade_run": str(vsplade_path),
        },
        "overview": _aggregate(rows),
        "by_query_type": _group_summary(rows, "query_types"),
        "by_query_type_component": _group_summary_components(rows, "query_types"),
        "by_query_format": _group_summary(rows, "query_format"),
        "by_content_type": _group_summary(rows, "content_type"),
        "by_source_type": _group_summary(rows, "source_type"),
        "by_gold_file_count": _group_summary(rows, "gold_file_count"),
        "examples": examples,
        "per_query": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_markdown(args.output_dir / "report.md", report)
    _write_per_query_markdown(args.output_dir / "per_query.md", report)
    print(json.dumps(report["overview"], ensure_ascii=True, indent=2))
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
