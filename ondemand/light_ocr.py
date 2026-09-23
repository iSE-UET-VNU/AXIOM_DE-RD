import json
import statistics
import zipfile
from pathlib import Path

from .bench import BENCH, load_jsonl, work, write_jsonl
from .text import real_text

ENGINE = "ppocrv5_doc"
RESULTS = "light_ocr_results_comparison_btw_ppocr.zip"
OUTPUT = "pages_ocr_ppocr.jsonl"


def read_results(path):
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
        texts = {r["unit"]: r for r in map(json.loads, zf.open(f"light_ocr_{ENGINE}.jsonl").read().decode().splitlines())}
        boxes = {}
        source = f"light_ocr_boxes_{ENGINE}.jsonl"
        if source in names:
            boxes = {r["unit"]: r["lines"] for r in map(json.loads, zf.open(source).read().decode().splitlines())}
    return texts, boxes


def merge(bench=BENCH, results=None):
    out = work("light_prep", bench=bench)
    results = Path(results) if results else Path(__file__).resolve().parent.parent / RESULTS
    texts, boxes = read_results(results)
    tesseract = {r["unit"]: r for r in load_jsonl(out / "ocr_results.jsonl")}
    rows, replaced, emptied = [], 0, 0
    for row in load_jsonl(out / "pages_ocr_sparse.jsonl"):
        row = dict(row)
        new = texts.get(row["page_id"])
        if new is not None and row["ocr_applied"]:
            old = (tesseract.get(row["page_id"]) or {}).get("text", "").strip()
            kept = row["text"]
            if old and kept.endswith(old):
                kept = kept[:-len(old)]
            kept = real_text(kept)
            text = (kept + "\n" + new["text"]).strip()
            lines = boxes.get(row["page_id"], [])
            row.update(text=text, visual_only=not text.strip(),
                       text_source=("pdf_inspector+" if kept else "") + ENGINE,
                       ocr_word_count=len(new["text"].split()),
                       ocr_mean_confidence=round(100 * statistics.mean([l["score"] for l in lines]), 2) if lines else 0.0,
                       ocr_seconds=new["seconds"], ocr_render_seconds=0.0, ocr_error=new.get("error"))
            replaced += 1
            emptied += not text.strip()
        rows.append(row)
    write_jsonl(out / OUTPUT, rows)
    print(f"{replaced} pages replaced with {ENGINE}, {emptied} of them empty -> {out / OUTPUT}")
    return out / OUTPUT


if __name__ == "__main__":
    merge()
