"""Materialise a fixed two-dimensional embedded-image size-filter grid."""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.experiments.prepare_embedded_ocr_filter_ablations import (  # noqa: E402
    merge_policy,
    read_jsonl,
    write_jsonl,
)


AREA_THRESHOLDS = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.05)
PIXEL_THRESHOLDS = (0, 1024, 4096, 16384, 65536, 262144)


def arm_name(area: float, pixels: int) -> str:
    area_name = f"{area:.4f}".rstrip("0").rstrip(".").replace(".", "p") or "0"
    return f"area_{area_name}_pixels_{pixels}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-pages", type=Path, required=True)
    parser.add_argument("--ocr-results", type=Path, required=True)
    parser.add_argument("--size-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    pages = read_jsonl(args.native_pages)
    results = read_jsonl(args.ocr_results)
    audit = {str(row["key"]): row for row in read_jsonl(args.size_audit)}
    arms = []
    for area in AREA_THRESHOLDS:
        for pixels in PIXEL_THRESHOLDS:
            name = arm_name(area, pixels)
            def transform(row, area=area, pixels=pixels):
                item = audit.get(str(row["key"]))
                if item is None or item.get("area_ratio") is None:
                    return str(row.get("text") or "")
                keep = float(item["area_ratio"]) >= area and int(item["pixel_area"]) >= pixels
                return str(row.get("text") or "") if keep else ""
            prepared, summary = merge_policy(pages, results, transform, name)
            target = args.output_dir / "corpora" / name / "pages.jsonl"
            write_jsonl(target, prepared)
            arms.append({"name": name, "min_area_ratio": area, "min_area_percent": area * 100, "min_pixel_area": pixels, **summary, "pages": str(target.resolve())})
            print(
                f"{name}: accepted={summary.get('accepted_images', 0)} "
                f"rejected={summary.get('rejected_images', 0)}",
                flush=True,
            )
    manifest = {
        "protocol": "independent size filter; keep OCR iff bbox/page area >= threshold AND raster width*height >= threshold; missing geometry is kept",
        "area_thresholds": list(AREA_THRESHOLDS), "pixel_thresholds": list(PIXEL_THRESHOLDS),
        "qrels_used": False, "arms": arms,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"arms": len(arms), "manifest": str((args.output_dir / 'manifest.json').resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
