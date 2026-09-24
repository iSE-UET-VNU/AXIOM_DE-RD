"""Build a Tesseract-free PP-OCRv5 input bundle from pdf-inspector pages.

The Colab notebook consumes one-page PDFs plus ``pages.jsonl``. This builder
selects pages using only pdf-inspector's ``needs_ocr`` signal and native text
quality, so the resulting experiment does not use Tesseract even as a weak
page selector.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import re
import sys
import zipfile

import pymupdf as fitz

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.data_discovery.pipeline import PdfInspectorPageParser  # noqa: E402
from research.data_discovery.run_docbench_e2e import _load_docbench  # noqa: E402


IMAGE_MARKER = re.compile(r"(?:\[image[^\]]*\]|!\[[^\]]*\]|\[picture\])", re.I)


def _selection_reasons(page: object) -> list[str]:
    text = str(getattr(page, "text", "") or "").strip()
    reasons: list[str] = []
    if bool(getattr(page, "needs_ocr", False)):
        reasons.append("pdf_inspector_needs_ocr")
    if not text:
        reasons.append("empty_native_text")
    markers = len(IMAGE_MARKER.findall(text))
    words = len(re.findall(r"\w+", text, flags=re.UNICODE))
    if markers and words <= max(8, markers * 3):
        reasons.append("mostly_image_placeholders")
    return reasons


def build(docbench_root: Path, output: Path) -> tuple[int, int]:
    documents, _questions = _load_docbench(docbench_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    parser = PdfInspectorPageParser()
    manifest: list[dict[str, object]] = []
    source_pages = 0

    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for document in documents:
            pdf_path = Path(str(document["pdf_path"])).resolve()
            pages = parser.parse(pdf_path, source_uri=str(document["doc_id"]))
            source_pages += len(pages)
            for page in pages:
                reasons = _selection_reasons(page)
                if not reasons:
                    continue
                member = (
                    f"pages/{str(document['doc_id']).replace('::', '__')}"
                    f"__p{page.page_index}.pdf"
                )
                with fitz.open(str(pdf_path)) as source_doc, fitz.open() as one_page:
                    one_page.insert_pdf(
                        source_doc,
                        from_page=page.page_index,
                        to_page=page.page_index,
                        links=False,
                        annots=False,
                        widgets=False,
                    )
                    archive.writestr(member, one_page.tobytes())
                manifest.append(
                    {
                        "unit": page.page_id,
                        "page_id": page.page_id,
                        "doc_id": page.source_uri,
                        "page_index": page.page_index,
                        "page_number": page.page_number,
                        "file": member,
                        "inspector_text": page.text,
                        "needs_ocr": page.needs_ocr,
                        "selection_reasons": reasons,
                        "evidence": [],
                    }
                )
        archive.writestr(
            "pages.jsonl",
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest),
        )

    return source_pages, len(manifest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docbench-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "benchmark" / "ppocrv5" / "light_ocr_bundle.zip",
    )
    args = parser.parse_args()
    root = args.docbench_root.resolve()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    total, selected = build(root, output.resolve())
    print(f"pdf-inspector pages: {total}; PP-OCRv5 candidates: {selected}")
    print(f"bundle: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
