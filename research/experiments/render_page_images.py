"""Render ViDoRe page images locally from the source PDFs.

The ViDoRe corpus.parquet ships text only (371 KB, columns corpus_id / doc_id /
markdown / page_number_in_doc) -- the image column is never projected. But we
have the 42 physics PDFs, and PyMuPDF renders them with no API and no GPU, so a
visual arm does not actually need the 442 MB-2.2 GB image download.

Page keys match the retrieval unit_id (subset::<file>#page=N, 0-based) so image
and text arms are directly comparable.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import fitz

from src.evaluation.benchmarks.vidore_v3 import unit_id
from src.evaluation.pipeline_pages import canonical_doc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-dir", default="data/raw/benchmarks/vidore_v3_physics")
    parser.add_argument("--out", default="data/work/vidore_physics_page_images")
    parser.add_argument("--subset", default="physics")
    parser.add_argument("--dpi", type=int, default=144)
    args = parser.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    pdfs = sorted((ROOT / args.pdf_dir).glob("*.pdf"))
    total = 0
    for path in pdfs:
        doc = canonical_doc(path.name)
        with fitz.open(path) as pdf:
            for index, page in enumerate(pdf):
                key = unit_id(args.subset, doc, index)
                target = out / (key.replace("/", "_").replace("::", "__") + ".png")
                if target.exists():
                    total += 1
                    continue
                pix = page.get_pixmap(dpi=args.dpi)
                pix.save(target)
                total += 1
    print(f"{len(pdfs)} PDFs -> {total} page images at {args.dpi} dpi in {out}")


if __name__ == "__main__":
    main()
