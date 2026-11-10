import csv
import io
import json
import shutil
import statistics
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib import import_module
from time import perf_counter

import fitz

from .bench import BENCH, documents, load_jsonl, unit_id, work, write_jsonl
from .text import is_uninformative, mostly_image, real_text
from .timing import Timings

OCR_LANGUAGES, OCR_DPI, OCR_PSM = "fra+eng", 200, 3


def inspect_pdf(path):
    api = import_module("pdf_inspector")
    classification = api.classify_pdf(str(path))
    with fitz.open(str(path)) as pdf:
        dims = [(float(p.rect.width), float(p.rect.height)) for p in pdf]
    count = min(int(classification.page_count), len(dims))
    if count <= 0:
        return [], str(classification.pdf_type)
    requests = [(i, [[0.0, 0.0, w, h]]) for i, (w, h) in enumerate(dims[:count])]
    results = api.extract_text_in_regions(str(path), requests)
    if len(results) != len(requests):
        raise ValueError("pdf-inspector returned a different number of pages than requested")
    needing = {int(p) for p in classification.pages_needing_ocr}
    pages = []
    for i, result in enumerate(results):
        region = result.regions[0] if result.regions else None
        pages.append({"page_index": i, "text": str(region.text or "").strip() if region else "",
                      "needs_ocr": bool(region.needs_ocr) if region else i in needing})
    return pages, str(classification.pdf_type)


def ocr_page(path, page0, language=OCR_LANGUAGES, dpi=OCR_DPI):
    started = perf_counter()
    try:
        with fitz.open(str(path)) as pdf:
            image = pdf.load_page(int(page0)).get_pixmap(dpi=int(dpi), alpha=False).tobytes("png")
        render = perf_counter() - started
        binary = shutil.which("tesseract") or "tesseract"
        t = perf_counter()
        done = subprocess.run([binary, "stdin", "stdout", "-l", language, "--psm", str(OCR_PSM), "tsv"],
                              input=image, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=120)
        ocr = perf_counter() - t
        if done.returncode != 0:
            return {"text": "", "word_count": 0, "mean_confidence": 0.0, "render_seconds": round(render, 6),
                    "ocr_seconds": round(ocr, 6), "error": done.stderr.decode("utf-8", "replace").strip() or f"exit {done.returncode}"}
        words, confidences = [], []
        for row in csv.DictReader(io.StringIO(done.stdout.decode("utf-8", "replace")), delimiter="\t"):
            token = (row.get("text") or "").strip()
            if not token:
                continue
            words.append(token)
            try:
                conf = float(row.get("conf", "-1"))
            except (TypeError, ValueError):
                conf = -1.0
            if conf >= 0:
                confidences.append(conf)
        return {"text": " ".join(words), "word_count": len(words),
                "mean_confidence": round(float(statistics.mean(confidences)) if confidences else 0.0, 4),
                "render_seconds": round(render, 6), "ocr_seconds": round(ocr, 6), "error": None}
    except Exception as error:
        return {"text": "", "word_count": 0, "mean_confidence": 0.0, "render_seconds": round(perf_counter() - started, 6),
                "ocr_seconds": 0.0, "error": f"{type(error).__name__}: {error}"}


def _ocr_job(job):
    unit, path, page0 = job
    return unit, ocr_page(path, page0)


def page_row(doc, page0, text, **fields):
    row = {"page_id": unit_id(doc["doc_id"], page0), "doc_id": doc["doc_id"], "source": doc["source"],
           "source_path": doc["metadata"].get("source_path", ""), "relative_path": doc["path"],
           "page_index": page0, "page_number": page0 + 1, "text": text, "visual_only": is_uninformative(text),
           "needs_ocr": False, "parse_status": "ok", "parse_error": None, "pdf_type": None,
           "text_source": "empty" if is_uninformative(text) else "pdf_inspector", "ocr_applied": False,
           "ocr_word_count": 0, "ocr_mean_confidence": 0.0, "ocr_render_seconds": 0.0, "ocr_seconds": 0.0, "ocr_error": None}
    row.update(fields)
    return row


def wants_ocr(row):
    return is_uninformative(row["text"]) or row["needs_ocr"] or mostly_image(row["text"])


def run(bench=BENCH, workers=8):
    out = work("light_prep", bench=bench)
    timings = Timings(out / "timings.jsonl", stage_group="light_prep")
    rows = []
    for doc in documents(bench):
        started = perf_counter()
        expected = doc["metadata"]["page_count"]
        try:
            pages, pdf_type = inspect_pdf(bench / doc["path"])
        except Exception as error:
            rows.extend(page_row(doc, p, "", parse_status="error", parse_error=f"{type(error).__name__}: {error}")
                        for p in range(expected))
            timings.record("pdf_inspector_parse", "page", expected, perf_counter() - started, doc_id=doc["doc_id"], failed=True)
            continue
        if len(pages) != expected:
            raise ValueError(f"{doc['doc_id']}: parsed {len(pages)} pages, manifest says {expected}")
        rows.extend(page_row(doc, p["page_index"], p["text"], needs_ocr=p["needs_ocr"], pdf_type=pdf_type) for p in pages)
        timings.record("pdf_inspector_parse", "page", len(pages), perf_counter() - started, doc_id=doc["doc_id"])

    cache_path = out / "ocr_results.jsonl"
    cached = {r["unit"]: r for r in (load_jsonl(cache_path) if cache_path.exists() else [])
              if r["language"] == OCR_LANGUAGES and r["dpi"] == OCR_DPI}
    targets = [r for r in rows if wants_ocr(r)]
    todo = [r for r in targets if r["page_id"] not in cached]
    if todo:
        started = perf_counter()
        with ProcessPoolExecutor(max_workers=workers) as pool, cache_path.open("a", encoding="utf-8") as sink:
            futures = [pool.submit(_ocr_job, (r["page_id"], str(bench / r["relative_path"]), r["page_index"])) for r in todo]
            for future in as_completed(futures):
                unit, res = future.result()
                record = {"unit": unit, "language": OCR_LANGUAGES, "dpi": OCR_DPI, **res}
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                cached[unit] = record
                timings.record("tesseract_ocr", "page", 1, res["render_seconds"] + res["ocr_seconds"], unit_id=unit)
        timings.record("tesseract_ocr_wall", "corpus", len(todo), perf_counter() - started, workers=workers)

    final = []
    for r in rows:
        r = dict(r)
        res = cached.get(r["page_id"]) if wants_ocr(r) else None
        if res is not None:
            r.update(ocr_applied=True, ocr_word_count=res["word_count"], ocr_mean_confidence=res["mean_confidence"],
                     ocr_render_seconds=res["render_seconds"], ocr_seconds=res["ocr_seconds"], ocr_error=res["error"])
            if res["text"].strip():
                kept = real_text(r["text"])
                r.update(text=(kept + "\n" + res["text"]).strip(),
                         text_source="pdf_inspector+tesseract" if kept else "tesseract", visual_only=False)
        final.append(r)
    write_jsonl(out / "pages_ocr_sparse.jsonl", final)
    return out / "pages_ocr_sparse.jsonl"
