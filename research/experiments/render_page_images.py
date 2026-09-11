"""Render ViDoRe V3 Physics PDFs into the filenames expected by ColVec notebooks.

The webAI-ColVec notebook uses ``physics__<doc_id>#page=<n>.png`` as its page
key and converts the double underscore back to ``physics::<doc_id>#page=<n>``.
This small renderer keeps that contract and uses zero-based ViDoRe page
numbers.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

try:
    import pymupdf
except ImportError as exc:  # pragma: no cover - exercised in the Colab runtime
    raise SystemExit("PyMuPDF is required: pip install pymupdf") from exc


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PDF_DIR = ROOT / "data/raw/benchmarks/vidore_v3_physics"
DEFAULT_OUTPUT_DIR = ROOT / "data/work/vidore_physics_page_images"


def render(pdf_dir: Path, output_dir: Path, dpi: int) -> int:
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No Physics PDFs found in {pdf_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rendered = 0
    for pdf_path in pdfs:
        with pymupdf.open(pdf_path) as document:
            page_count = document.page_count
            for page_number, page in enumerate(document):
                output_path = output_dir / (
                    f"physics__{pdf_path.stem}#page={page_number}.png"
                )
                if not output_path.exists():
                    pixmap = page.get_pixmap(dpi=dpi, alpha=False)
                    pixmap.save(output_path)
                rendered += 1
        print(f"{pdf_path.name}: {page_count} pages", flush=True)
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=144)
    args = parser.parse_args()
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    count = render(args.pdf_dir, args.output_dir, args.dpi)
    print(f"rendered/verified {count} page images in {args.output_dir}")
    if count != 1674:
        raise RuntimeError(f"Expected 1,674 Physics pages, found {count}")


if __name__ == "__main__":
    main()
