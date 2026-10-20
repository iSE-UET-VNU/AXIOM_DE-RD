"""Build an interactive HTML audit of size-filtered embedded-image OCR."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.experiments.ocr_benchmark_embedded_images import norm_label  # noqa: E402
from research.experiments.prepare_embedded_ocr_filter_ablations import (  # noqa: E402
    geometry_by_occurrence,
    read_jsonl,
    write_jsonl,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-pages", type=Path, required=True)
    parser.add_argument("--ocr-results", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--area-threshold", type=float, default=0.005)
    parser.add_argument("--pixel-threshold", type=int, default=1024)
    parser.add_argument("--side-threshold", type=int, default=16)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pages = read_jsonl(args.native_pages)
    results = read_jsonl(args.ocr_results)
    geometry, geometry_stats = geometry_by_occurrence(pages, args.dataset_root)
    rows: list[dict[str, Any]] = []
    for row in results:
        text = str(row.get("text") or "").strip()
        if row.get("status") != "ok" or not text:
            continue
        label = norm_label(str(row["image_label"]))
        geo = geometry.get((str(row["page_id"]), label))
        width, height = int(row.get("width") or 0), int(row.get("height") or 0)
        pixel_area = width * height
        area_ratio = float(geo["area_ratio"]) if geo else None
        small_display = area_ratio is not None and area_ratio < args.area_threshold
        low_pixels = pixel_area < args.pixel_threshold or min(width, height) < args.side_threshold
        reasons = (["small_display"] if small_display else []) + (["low_resolution"] if low_pixels else [])
        rows.append({
            "key": row["key"], "page_id": row["page_id"], "doc_id": row["doc_id"], "source": row["source"],
            "page_index": row["page_index"], "image_label": label, "occurrence_on_page": row["occurrence_on_page"],
            "image_path": str(Path(str(row["image_path"])).resolve()), "width": width, "height": height,
            "pixel_area": pixel_area, "area_ratio": area_ratio, "area_percent": area_ratio * 100 if area_ratio is not None else None,
            "current_rejected": bool(reasons), "reasons": reasons, "ocr_text": text, "ocr_words": len(text.split()),
        })
    rows.sort(key=lambda item: (not item["current_rejected"], item["area_ratio"] if item["area_ratio"] is not None else 999.0, item["pixel_area"]))
    write_jsonl(args.output_dir / "size_audit.jsonl", rows)

    cards = []
    for item in rows:
        image_src = Path(os.path.relpath(item["image_path"], args.output_dir)).as_posix()
        reason = ", ".join(item["reasons"]) or "accepted"
        area = "unknown" if item["area_percent"] is None else f"{item['area_percent']:.4f}%"
        cards.append(f"""
<article class="card" data-rejected="{str(item['current_rejected']).lower()}" data-source="{html.escape(item['source'])}" data-area="{item['area_ratio'] if item['area_ratio'] is not None else -1}" data-pixels="{item['pixel_area']}">
  <div class="image"><img loading="lazy" src="{html.escape(image_src)}" alt="{html.escape(item['image_label'])}"></div>
  <div class="body">
    <h3>{html.escape(item['image_label'])} <span>{html.escape(reason)}</span></h3>
    <p><b>{item['width']}×{item['height']}</b> ({item['pixel_area']:,} px) · page area <b>{area}</b> · {item['ocr_words']} OCR words</p>
    <p class="id">{html.escape(item['page_id'])}</p>
    <pre>{html.escape(item['ocr_text'])}</pre>
  </div>
</article>""")
    rejected = sum(item["current_rejected"] for item in rows)
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Embedded image size audit</title>
<style>
body{{font:14px system-ui;margin:0;background:#f3f5f7;color:#18212b}}header{{position:sticky;top:0;background:#fff;padding:14px 20px;border-bottom:1px solid #ccd5df;z-index:2}}h1{{font-size:20px;margin:0 0 8px}}.controls{{display:flex;gap:14px;flex-wrap:wrap;align-items:center}}main{{padding:16px;display:grid;grid-template-columns:repeat(auto-fill,minmax(520px,1fr));gap:12px}}.card{{background:white;border:1px solid #d5dde5;border-radius:8px;display:grid;grid-template-columns:220px 1fr;min-height:220px;overflow:hidden}}.image{{display:flex;align-items:center;justify-content:center;background:#e8edf2;padding:8px}}img{{max-width:100%;max-height:260px;image-rendering:auto}}.body{{padding:10px;min-width:0}}h3{{margin:0 0 8px}}h3 span{{font-size:12px;color:#a33}}p{{margin:5px 0}}.id{{font:11px monospace;overflow-wrap:anywhere;color:#52616f}}pre{{white-space:pre-wrap;max-height:180px;overflow:auto;background:#f7f8fa;padding:8px;border-radius:5px}}.hidden{{display:none}}
</style></head><body>
<header><h1>Embedded-image OCR size audit</h1><div>{len(rows):,} non-empty OCR images · {rejected:,} rejected at area &lt; {args.area_threshold*100:.2f}% OR pixels &lt; {args.pixel_threshold:,} OR min side &lt; {args.side_threshold}px</div>
<div class="controls"><label>Show <select id="status"><option value="rejected">rejected</option><option value="accepted">accepted</option><option value="all">all</option></select></label><label>Source <select id="source"><option value="all">all</option><option value="vidore_physics">physics</option><option value="vidore_industrial">industrial</option></select></label><label>Area ≤ % <input id="area" type="number" step="0.1" value="5"></label><label>Pixels ≤ <input id="pixels" type="number" value="65536"></label><span id="shown"></span></div></header>
<main>{''.join(cards)}</main>
<script>
const cards=[...document.querySelectorAll('.card')]; function apply(){{const s=document.querySelector('#status').value,src=document.querySelector('#source').value,a=Number(document.querySelector('#area').value)/100,p=Number(document.querySelector('#pixels').value);let n=0;for(const c of cards){{const okStatus=s==='all'||(s==='rejected')===(c.dataset.rejected==='true');const okSrc=src==='all'||c.dataset.source===src;const area=Number(c.dataset.area);const okArea=area<0||area<=a;const okPixels=Number(c.dataset.pixels)<=p;const show=okStatus&&okSrc&&okArea&&okPixels;c.classList.toggle('hidden',!show);if(show)n++;}}document.querySelector('#shown').textContent=`${{n}} shown`;}} document.querySelectorAll('select,input').forEach(e=>e.addEventListener('input',apply));apply();
</script></body></html>"""
    (args.output_dir / "index.html").write_text(document, encoding="utf-8")
    summary = {"images_with_nonempty_ocr": len(rows), "current_rejected": rejected, "current_accepted": len(rows) - rejected, "geometry": dict(geometry_stats), "html": str((args.output_dir / 'index.html').resolve())}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
