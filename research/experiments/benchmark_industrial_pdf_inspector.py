"""Measure standalone pdf-inspector page-text extraction for ViDoRe Industrial.

This is a read-only timing probe: it creates no ingestion/output artifacts and
does not invoke KDL, OCR, chunking, embeddings, or retrieval.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from time import perf_counter
import argparse

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.data_discovery.pipeline import PdfInspectorPageParser


PDF_ROOTS = {
    "industrial": ROOT / "data/raw/benchmarks/vidore_v3/vidore_v3_industrial/pdfs",
    "physics": ROOT / "data/raw/benchmarks/vidore_v3/vidore_v3_physics/pdfs",
}
EXPECTED_DOCUMENTS = {"industrial": 27, "physics": 42}


def main() -> None:
    arguments = argparse.ArgumentParser(description=__doc__)
    arguments.add_argument("--subset", choices=sorted(PDF_ROOTS), default="industrial")
    args = arguments.parse_args()
    files = sorted(PDF_ROOTS[args.subset].rglob("*.pdf"))
    expected = EXPECTED_DOCUMENTS[args.subset]
    if len(files) != expected:
        raise RuntimeError(f"Expected {expected} {args.subset} PDFs, found {len(files)}")

    parser = PdfInspectorPageParser()
    started = perf_counter()
    documents: list[dict[str, object]] = []
    pages = 0
    text_bytes = 0
    ocr_pages = 0
    for path in files:
        file_started = perf_counter()
        evidence = parser.parse(path, source_uri=str(path))
        elapsed = perf_counter() - file_started
        pages += len(evidence)
        text_bytes += sum(len(page.text.encode("utf-8")) for page in evidence)
        ocr_pages += sum(page.needs_ocr for page in evidence)
        documents.append(
            {
                "file": path.name,
                "pages": len(evidence),
                "seconds": round(elapsed, 6),
                "text_bytes": sum(len(page.text.encode("utf-8")) for page in evidence),
                "pages_needing_ocr": sum(page.needs_ocr for page in evidence),
            }
        )

    total = perf_counter() - started
    print(
        json.dumps(
            {
                "corpus": f"vidore_v3/{args.subset}",
                "parser": "pdf-inspector",
                "documents": len(files),
                "pages": pages,
                "wall_seconds": round(total, 6),
                "seconds_per_page": round(total / pages, 6),
                "text_bytes": text_bytes,
                "pages_needing_ocr": ocr_pages,
                "per_document": documents,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
