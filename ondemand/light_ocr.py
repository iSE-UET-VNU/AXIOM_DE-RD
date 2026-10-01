import json
import statistics
import zipfile
from pathlib import Path

from .bench import BENCH, load_jsonl, work, write_jsonl
from .light_prep import wants_ocr as is_target
from .text import real_text

ENGINE = "ppocrv5_doc"
RESULTS = "light_ocr_results_comparison_btw_ppocr.zip"
OUTPUT = "pages_ocr_ppocr.jsonl"
PURE_OUTPUT = "pages_ppocrv5.jsonl"


def find_engine(names, engine=None):
    found = sorted(n[len("light_ocr_"):-len(".jsonl")] for n in names
                   if n.startswith("light_ocr_") and n.endswith(".jsonl") and "boxes" not in n)
    if engine:
        if engine not in found:
            raise SystemExit(f"{engine!r} is not in the results; found {', '.join(found) or 'nothing'}")
        return engine
    if not found:
        raise SystemExit("the results hold no light_ocr_<engine>.jsonl file")
    if len(found) > 1:
        raise SystemExit(f"the results hold several engines ({', '.join(found)}); pick one with --engine")
    return found[0]


def read_results(path, engine=None):
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
        engine = find_engine(names, engine)
        texts = {r["unit"]: r for r in map(json.loads, zf.open(f"light_ocr_{engine}.jsonl").read().decode().splitlines())}
        boxes = {}
        source = f"light_ocr_boxes_{engine}.jsonl"
        if source in names:
            boxes = {r["unit"]: r["lines"] for r in map(json.loads, zf.open(source).read().decode().splitlines())}
        inspector = zf.read("pages_inspector.jsonl").decode() if "pages_inspector.jsonl" in names else None
    return texts, boxes, engine, inspector


def merge_pure(bench=BENCH, results=None, pages_name="pages_inspector.jsonl", name=None, engine=None):
    out = work("light_prep", bench=bench)
    results = Path(results) if results else Path(__file__).resolve().parent.parent / RESULTS
    texts, boxes, engine, inspector = read_results(results, engine)
    source = out / pages_name
    if inspector and not source.exists():
        source.write_text(inspector, encoding="utf-8")
        print(f"took {pages_name} from the results bundle -> {source}")
    if not source.exists():
        raise SystemExit(f"{source} is missing and the results bundle does not carry it; "
                         f"run `python -m ondemand light-prep --ocr none` first")
    rows, replaced, missing, blank = [], 0, [], 0
    for row in load_jsonl(source):
        row = dict(row)
        new = texts.get(row["page_id"])
        if new is None:
            if is_target(row):
                missing.append(row["page_id"])
        else:
            base = real_text(row["text"])
            text = (base + "\n" + new["text"]).strip()
            lines = boxes.get(row["page_id"], [])
            row.update(text=text, visual_only=not text.strip(), ocr_applied=True,
                       text_source=("pdf_inspector+" if base else "") + engine,
                       ocr_word_count=len(new["text"].split()),
                       ocr_mean_confidence=round(100 * statistics.mean([l["score"] for l in lines]), 2) if lines else 0.0,
                       ocr_seconds=new["seconds"], ocr_render_seconds=0.0, ocr_error=new.get("error"))
            replaced += 1
            blank += not text.strip()
        rows.append(row)
    if missing:
        raise SystemExit(f"{len(missing)} pages need OCR but are absent from {results.name}, e.g. {missing[:3]}")
    path = out / (name or PURE_OUTPUT)
    write_jsonl(path, rows)
    print(f"{replaced} pages carry {engine} text, {blank} of them still empty -> {path}")
    return path


def merge(bench=BENCH, results=None):
    out = work("light_prep", bench=bench)
    results = Path(results) if results else Path(__file__).resolve().parent.parent / RESULTS
    texts, boxes = read_results(results, ENGINE)[:2]
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
