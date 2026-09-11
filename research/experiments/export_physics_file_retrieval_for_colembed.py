"""Export Physics file-retrieval scopes for an external second retriever.

The source experiment already evaluated the fixed proposal plus compact file
selection for Kf in {3, 5, 10, 15, 20}.  This exporter deliberately does not
rerun retrieval or use qrels to construct the per-query export.  Qrels are read
only to calculate the summary file-recall metadata.

Each Kf JSONL row contains the query, ordered selected file IDs, and every
page ID belonging to those files.  The latter is the intended input scope for
ColEmbed/ColVec second retrieval; it is not the light page ranker's top-Kp
output.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.benchmarks.vidore_v3 import ViDoreV3  # noqa: E402


DEFAULT_BENCHMARK_ROOT = ROOT / "data" / "benchmark" / "vidore_v3"
DEFAULT_SOURCE_DIR = (
    ROOT / "data" / "benchmark" / "vidore_v3" / "results"
    / "physics_topk_budget_ablation"
)
DEFAULT_PAGE_METADATA = (
    ROOT / "data" / "output" / "vsplade" / "vidore_v3_physics_48q"
    / "page_metadata.json"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "data" / "benchmark" / "vidore_v3" / "exports"
    / "physics_file_retrieval_for_colembed"
)

K_FILE_SETTINGS = (3, 5, 10, 15, 20)


def _file_id(page_id: str) -> str:
    return page_id.rsplit("#page=", 1)[0]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def _load_pages(path: Path) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON list in {path}")

    pages: list[dict[str, Any]] = []
    pages_by_file: dict[str, list[str]] = defaultdict(list)
    seen_pages: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError(f"Page metadata entry is not an object in {path}")
        page_id = str(item["unit_id"])
        file_id = f"physics::{item['doc_id']}"
        expected_page_id = f"{file_id}#page={item['page_number_in_doc']}"
        if page_id != expected_page_id:
            raise ValueError(
                f"Page ID mismatch: metadata has {page_id!r}, expected "
                f"{expected_page_id!r}"
            )
        if page_id in seen_pages:
            raise ValueError(f"Duplicate page ID in page metadata: {page_id}")
        seen_pages.add(page_id)
        record = {
            "page_id": page_id,
            "file_id": file_id,
            "file_name": str(item.get("file_name") or f"{item['doc_id']}.pdf"),
            "corpus_id": int(item["corpus_id"]),
            "page_number_in_doc": int(item["page_number_in_doc"]),
        }
        pages.append(record)
        pages_by_file[file_id].append(page_id)
    return pages, dict(pages_by_file)


def _questions(benchmark: ViDoreV3) -> dict[str, str]:
    result = {str(question.qid): str(question.query) for question in benchmark.questions()}
    if len(result) != 302:
        raise ValueError(f"Expected 302 French Physics queries, found {len(result)}")
    return result


def _validate_source_rows(
    rows: list[dict[str, Any]],
    *,
    pages_by_file: Mapping[str, list[str]],
    query_text: Mapping[str, str],
) -> dict[int, dict[str, dict[str, Any]]]:
    by_kf: dict[int, dict[str, dict[str, Any]]] = {
        kf: {} for kf in K_FILE_SETTINGS
    }
    for row in rows:
        arm = str(row.get("arm", ""))
        if not arm.startswith("full_kf"):
            continue
        try:
            kf = int(arm.removeprefix("full_kf"))
        except ValueError:
            continue
        if kf not in by_kf:
            continue
        qid = str(row["qid"])
        if qid in by_kf[kf]:
            raise ValueError(f"Duplicate qid {qid} in source arm {arm}")
        if qid not in query_text:
            raise ValueError(f"Source qid {qid} is not in the French query set")

        selected_files = [str(file_id) for file_id in row["selected_files"]]
        if not selected_files or len(selected_files) > kf or len(set(selected_files)) != len(selected_files):
            raise ValueError(f"{arm}/{qid} has invalid selected_files")
        if any(file_id not in pages_by_file for file_id in selected_files):
            missing = sorted(set(selected_files) - set(pages_by_file))
            raise ValueError(f"{arm}/{qid} references unknown files: {missing}")

        counts = row.get("proposal_counts", {})
        source_pages_scanned = int(counts.get("pages_scanned", -1))
        selected_page_ids = [
            page_id
            for file_id in selected_files
            for page_id in pages_by_file[file_id]
        ]
        if len(selected_page_ids) != source_pages_scanned:
            raise ValueError(
                f"{arm}/{qid}: selected page count {len(selected_page_ids)} does not "
                f"match source pages_scanned={source_pages_scanned}"
            )
        if len(set(selected_page_ids)) != len(selected_page_ids):
            raise ValueError(f"{arm}/{qid} has duplicate selected pages")
        by_kf[kf][qid] = row

    for kf, rows_by_qid in by_kf.items():
        if len(rows_by_qid) != len(query_text):
            raise ValueError(
                f"full_kf{kf} has {len(rows_by_qid)} queries, expected {len(query_text)}"
            )
    return by_kf


def _file_metrics(
    selected_files: list[str],
    qrels: Mapping[str, int],
) -> tuple[float, float]:
    gold_files = {_file_id(str(page_id)) for page_id in qrels}
    found = len(set(selected_files) & gold_files)
    recall = found / len(gold_files) if gold_files else 0.0
    return recall, float(bool(found))


def _portable_path(path: Path) -> str:
    """Prefer a repository-relative path so the manifest can be shared."""
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def export(
    *,
    benchmark_root: Path,
    source_dir: Path,
    page_metadata: Path,
    output_dir: Path,
) -> dict[str, Any]:
    benchmark = ViDoreV3(
        root=benchmark_root,
        subset="physics",
        language="french",
    )
    query_text = _questions(benchmark)
    qrels = benchmark.qrels()
    pages, pages_by_file = _load_pages(page_metadata)
    if len(pages) != 1674 or len(pages_by_file) != 42:
        raise ValueError(
            f"Corpus inventory mismatch: pages={len(pages)}, files={len(pages_by_file)}"
        )

    source_rows = _read_jsonl(source_dir / "per_query.jsonl")
    source_by_kf = _validate_source_rows(
        source_rows,
        pages_by_file=pages_by_file,
        query_text=query_text,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    page_manifest_path = output_dir / "page_manifest.jsonl"
    _write_jsonl(page_manifest_path, pages)

    file_manifest: list[dict[str, Any]] = []
    for file_id in sorted(pages_by_file):
        page_ids = pages_by_file[file_id]
        matching = next(page for page in pages if page["file_id"] == file_id)
        file_manifest.append({
            "file_id": file_id,
            "file_name": matching["file_name"],
            "page_count": len(page_ids),
            "page_ids": page_ids,
        })
    file_manifest_path = output_dir / "file_manifest.jsonl"
    _write_jsonl(file_manifest_path, file_manifest)

    settings: list[dict[str, Any]] = []
    for kf in K_FILE_SETTINGS:
        export_rows: list[dict[str, Any]] = []
        file_recall_values: list[float] = []
        file_hit_values: list[float] = []
        page_counts: list[int] = []
        selected_file_counts: list[int] = []
        for qid in sorted(source_by_kf[kf], key=lambda value: int(value.rsplit("::", 1)[1])):
            source = source_by_kf[kf][qid]
            selected_files = [str(file_id) for file_id in source["selected_files"]]
            selected_file_records = []
            candidate_page_ids: list[str] = []
            for rank, file_id in enumerate(selected_files, 1):
                file_pages = pages_by_file[file_id]
                matching = next(page for page in pages if page["file_id"] == file_id)
                selected_file_records.append({
                    "rank": rank,
                    "file_id": file_id,
                    "file_name": matching["file_name"],
                    "page_count": len(file_pages),
                })
                candidate_page_ids.extend(file_pages)

            recall, hit = _file_metrics(selected_files, qrels.get(qid, {}))
            file_recall_values.append(recall)
            file_hit_values.append(hit)
            page_counts.append(len(candidate_page_ids))
            selected_file_counts.append(len(selected_files))

            # Deliberately omit gold pages/files and all qrel-derived values.
            export_rows.append({
                "qid": qid,
                "query": query_text[qid],
                "k_files": kf,
                "selected_file_ids": selected_files,
                "selected_files": selected_file_records,
                "candidate_page_ids": candidate_page_ids,
                "candidate_page_count": len(candidate_page_ids),
            })

        export_path = output_dir / f"queries_kf{kf}.jsonl"
        _write_jsonl(export_path, export_rows)
        settings.append({
            "k_files": kf,
            "export": export_path.name,
            "queries": len(export_rows),
            "mean_file_recall": sum(file_recall_values) / len(file_recall_values),
            "file_recall_percent": 100.0 * sum(file_recall_values) / len(file_recall_values),
            "file_hit": sum(file_hit_values) / len(file_hit_values),
            "avg_candidate_pages_per_query": sum(page_counts) / len(page_counts),
            "min_candidate_pages_per_query": min(page_counts),
            "max_candidate_pages_per_query": max(page_counts),
            "avg_selected_files_per_query": sum(selected_file_counts) / len(selected_file_counts),
            "pages_are": "all pages belonging to selected files",
        })

    manifest = {
        "format": "axiom.physics.file_retrieval_for_second_retrieval.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "vidore_v3/physics",
        "language": "french",
        "purpose": (
            "Per-query file-retrieval scopes for an external ColVec/ColEmbed "
            "second retrieval; no light page top-Kp filter is applied."
        ),
        "corpus": {
            "queries": len(query_text),
            "files": len(pages_by_file),
            "pages": len(pages),
        },
        "source": {
            "experiment": "physics_topk_budget_ablation",
            "source_dir": _portable_path(source_dir),
            "source_per_query": _portable_path(source_dir / "per_query.jsonl"),
            "proposal": (
                "top-30 structural pages UNION top-30 flat BM25 pages UNION "
                "top-30 V-SPLADE pages UNION top-10 structural synopsis files"
            ),
            "compact_file_strategy": "sum_top2",
            "proposal_depth_is_fixed": True,
            "qrels_used_to_build_retrieval": False,
        },
        "interpretation": {
            "file_recall": (
                "Mean over queries of the fraction of gold evidence files present "
                "in the selected Kf files."
            ),
            "avg_candidate_pages_per_query": (
                "Mean number of pages sent as the ColVec candidate scope: all pages "
                "from the selected files. This is not Kp and is not top-10 page recall."
            ),
            "query_rows_contain_gold_labels": False,
            "second_retrieval_input": "candidate_page_ids",
        },
        "files": {
            "page_manifest": page_manifest_path.name,
            "file_manifest": file_manifest_path.name,
        },
        "settings": settings,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Physics file-retrieval exports for ColVec/ColEmbed",
        "",
        "This package contains per-query file scopes for an external second retrieval.",
        "The second retriever should score every `candidate_page_id` in the selected",
        "file scope. No light page `Kp` cutoff is applied here.",
        "",
        "## Summary",
        "",
        "| Setting | File recall | File hit | Avg files/query | Avg selected pages/query | Export |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for item in settings:
        lines.append(
            f"| Kf={item['k_files']} | {item['file_recall_percent']:.2f}% | "
            f"{100.0 * item['file_hit']:.2f}% | "
            f"{item['avg_selected_files_per_query']:.2f} | "
            f"{item['avg_candidate_pages_per_query']:.2f} | "
            f"`{item['export']}` |"
        )
    lines += [
        "",
        "## How to consume",
        "",
        "1. Join each `candidate_page_id` to the partner's page-image corpus using `page_manifest.jsonl`.",
        "2. For each query, run ColVec/ColEmbed only over that query's candidate page IDs.",
        "3. Keep the returned ColVec score/rank separate from the light file-retrieval metadata.",
        "",
        "The `file_recall` values are evaluation metadata calculated with the benchmark qrels;",
        "gold pages/files are intentionally not included in the per-query export rows.",
        "The source corpus has 42 files and 1,674 pages; `candidate_page_count` is the",
        "actual per-query second-retrieval scope.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--page-metadata", type=Path, default=DEFAULT_PAGE_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    manifest = export(
        benchmark_root=args.benchmark_root,
        source_dir=args.source_dir,
        page_metadata=args.page_metadata,
        output_dir=args.output_dir,
    )
    for item in manifest["settings"]:
        print(
            f"Kf={item['k_files']}: file_recall={item['file_recall_percent']:.2f}% "
            f"avg_pages/query={item['avg_candidate_pages_per_query']:.2f} "
            f"export={item['export']}"
        )


if __name__ == "__main__":
    main()
