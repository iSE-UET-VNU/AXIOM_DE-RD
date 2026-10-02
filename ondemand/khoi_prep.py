import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from time import perf_counter

from .bench import BENCH, documents, load_jsonl, work, write_jsonl
from .light_prep import inspect_pdf, ocr_page, page_row

LANGUAGE, DPI = "eng", 144
CACHE = "ocr_results_eng144.jsonl"
OUTPUT = "pages_ocr_khoi.jsonl"


def is_placeholder(text):
    value = text.strip()
    return bool(value) and value.startswith("[Image:") and value.endswith("]")


def wants_ocr(row):
    return bool(row["needs_ocr"]) or not row["text"].strip() or is_placeholder(row["text"])


def _job(job):
    unit, path, page0 = job
    return unit, ocr_page(path, page0, language=LANGUAGE, dpi=DPI)


def run(bench=BENCH, workers=8):
    out = work("light_prep", bench=bench)
    rows = []
    for doc in documents(bench):
        expected = doc["metadata"]["page_count"]
        try:
            pages, pdf_type = inspect_pdf(bench / doc["path"])
        except Exception as error:
            rows.extend(page_row(doc, p, "", parse_status="error", parse_error=f"{type(error).__name__}: {error}")
                        for p in range(expected))
            continue
        rows.extend(page_row(doc, p["page_index"], p["text"], needs_ocr=p["needs_ocr"], pdf_type=pdf_type) for p in pages)

    cache_path = out / CACHE
    cached = {r["unit"]: r for r in (load_jsonl(cache_path) if cache_path.exists() else [])}
    targets = [r for r in rows if wants_ocr(r)]
    todo = [r for r in targets if r["page_id"] not in cached]
    print(f"{len(rows)} pages, {len(targets)} need OCR, {len(todo)} to run at {LANGUAGE} dpi={DPI}")
    if todo:
        started = perf_counter()
        with ProcessPoolExecutor(max_workers=workers) as pool, cache_path.open("a", encoding="utf-8") as sink:
            futures = [pool.submit(_job, (r["page_id"], str(bench / r["relative_path"]), r["page_index"])) for r in todo]
            for n, future in enumerate(as_completed(futures), 1):
                unit, res = future.result()
                record = {"unit": unit, "language": LANGUAGE, "dpi": DPI, **res}
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                cached[unit] = record
                if n % 100 == 0:
                    print(f"  {n}/{len(todo)}")
        print(f"OCR wall {perf_counter() - started:.1f}s")

    final, replaced = [], 0
    for r in rows:
        r = dict(r)
        res = cached.get(r["page_id"]) if wants_ocr(r) else None
        if res is not None:
            r.update(ocr_applied=True, ocr_word_count=res["word_count"], ocr_mean_confidence=res["mean_confidence"],
                     ocr_render_seconds=res["render_seconds"], ocr_seconds=res["ocr_seconds"], ocr_error=res["error"])
            if res["text"].strip():
                r.update(text=res["text"].strip(), text_source="tesseract", visual_only=False)
                replaced += 1
            else:
                r.update(text_source="empty" if not r["text"].strip() else r["text_source"])
        final.append(r)
    write_jsonl(out / OUTPUT, final)
    print(f"{replaced} pages replaced with tesseract text -> {out / OUTPUT}")
    return out / OUTPUT


if __name__ == "__main__":
    run()
