"""Create the OHR-only input view for the replacement benchmark run.

The retrieval corpus remains the full benchmark and is handled separately.
This helper only materializes the 65 OHR queries/documents/qrels so the new
preparation stage never touches the already cached Physics/Industrial files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    documents = read_jsonl(args.dataset_root / "documents.jsonl")
    queries = read_jsonl(args.dataset_root / "queries.jsonl")
    qrels = read_jsonl(args.dataset_root / "qrels.jsonl")
    docs = [row for row in documents if row.get("source") == "ohrbench"]
    query_ids = {str(row["query_id"]) for row in queries if row.get("source") == "ohrbench"}
    query_rows = [row for row in queries if str(row.get("query_id")) in query_ids]
    qrel_rows = [row for row in qrels if str(row.get("query_id")) in query_ids]

    if len(docs) != 65 or len(query_rows) != 65 or len(qrel_rows) != 65:
        raise RuntimeError(
            f"Expected 65 OHR documents/queries/qrels, got "
            f"{len(docs)}/{len(query_rows)}/{len(qrel_rows)}"
        )
    doc_ids = {str(row["doc_id"]) for row in docs}
    if {str(row["doc_id"]) for row in qrel_rows} != doc_ids:
        raise RuntimeError("OHR qrels do not cover exactly the 65 OHR documents")
    for row in qrel_rows:
        evidence = row.get("evidence") or []
        if len(evidence) != 1 or "page_index" not in evidence[0]:
            raise RuntimeError(f"OHR qrel is not an explicit page-level record: {row}")
        page_index = int(evidence[0]["page_index"])
        doc = next(item for item in docs if item["doc_id"] == row["doc_id"])
        page_count = int(doc.get("metadata", {}).get("page_count", 0))
        if page_index < 0 or page_index >= page_count:
            raise RuntimeError(f"OHR qrel page {page_index} outside {row['doc_id']} ({page_count})")

    write_jsonl(args.output_dir / "documents.jsonl", docs)
    write_jsonl(args.output_dir / "queries.jsonl", query_rows)
    # The generic benchmark runner accepts explicit page_index qrels.
    normalized = []
    for row in qrel_rows:
        evidence = row["evidence"][0]
        normalized.append({
            "query_id": row["query_id"],
            "doc_id": row["doc_id"],
            "page_index": int(evidence["page_index"]),
            "relevance": int(row.get("relevance", 1)),
            "source_evidence_id": f"{row['doc_id']}#page={int(evidence['page_index'])}",
        })
    write_jsonl(args.output_dir / "qrels.jsonl", normalized)
    manifest = {
        "scope": "ohrbench-only-preparation",
        "documents": len(docs),
        "queries": len(query_rows),
        "qrels": len(normalized),
        "old_cache_reuse": ["vidore_physics", "vidore_industrial"],
        "retrieval_corpus": "full benchmark (103 documents) after merge",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
