"""Replace selective OHR full-page Tesseract with PP-OCRv5-mobile OCR.

No qrels are read here.  The script only consumes the frozen OHR preparation
manifest, so the OCR choice cannot be affected by evidence labels.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
import os
from pathlib import Path
import statistics
import time
import traceback
from typing import Any

import numpy as np
import pypdfium2 as pdfium


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data/work/ohr_replacement_20260920/ohr_preparation/parsing/pages.jsonl"
DEFAULT_OUT = ROOT / "data/work/ohr_replacement_20260920/ppocrv5_mobile"
WORKER_MODEL: Any | None = None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def render_page(pdf_path: str, page_index: int) -> np.ndarray:
    document = pdfium.PdfDocument(pdf_path)
    page = document[page_index]
    # 2x PDF coordinate scale is 144 DPI, identical to the frozen Tesseract run.
    bitmap = page.render(scale=2.0)
    image = bitmap.to_pil().convert("RGB")
    page.close()
    document.close()
    return np.asarray(image)


def text_from_result(result: Any) -> tuple[str, list[float]]:
    data = result.json.get("res", {})
    texts = list(data.get("rec_texts") or [])
    scores = list(data.get("rec_scores") or [])
    polys = list(data.get("rec_polys") or [])
    indexed = []
    for index, value in enumerate(texts):
        text = str(value).strip()
        if not text:
            continue
        polygon = polys[index] if index < len(polys) else []
        if polygon:
            xs = [point[0] for point in polygon]
            ys = [point[1] for point in polygon]
            order = (min(ys), min(xs))
        else:
            order = (index, 0)
        indexed.append((order, text, float(scores[index]) if index < len(scores) else 0.0))
    indexed.sort(key=lambda item: item[0])
    return "\n".join(item[1] for item in indexed), [item[2] for item in indexed]


def init_ocr_worker(cache_dir: str) -> None:
    """Load exactly one OCR model per process and prevent BLAS oversubscription."""
    os.environ["PADDLE_PDX_CACHE_HOME"] = cache_dir
    os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "BOS")
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    from paddleocr import PaddleOCR
    global WORKER_MODEL
    WORKER_MODEL = PaddleOCR(
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="PP-OCRv5_mobile_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )


def ocr_one_page(row: dict[str, Any]) -> dict[str, Any]:
    if WORKER_MODEL is None:
        raise RuntimeError("OCR worker model was not initialized")
    render_started = time.perf_counter()
    try:
        image = render_page(str(row["source_path"]), int(row["page_index"]))
        render_seconds = time.perf_counter() - render_started
        ocr_started = time.perf_counter()
        results = list(WORKER_MODEL.predict(image))
        if not results:
            raise RuntimeError("PP-OCRv5 returned no result object")
        text, scores = text_from_result(results[0])
        return {"page_id": str(row["page_id"]), "text": text, "word_count": len(text.split()), "mean_confidence": float(statistics.mean(scores)) if scores else 0.0, "render_seconds": render_seconds, "ocr_seconds": time.perf_counter() - ocr_started, "error": None}
    except Exception as exc:
        return {"page_id": str(row["page_id"]), "text": "", "word_count": 0, "mean_confidence": 0.0, "render_seconds": time.perf_counter() - render_started, "ocr_seconds": 0.0, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, help="For smoke testing only; never use for final metrics.")
    parser.add_argument("--page-id", action="append", help="Restrict to explicit page IDs for debugging; never use for final metrics.")
    parser.add_argument("--ocr-workers", type=int, default=min(os.cpu_count() or 1, 12), help="One model actor per worker. Keep at or below logical CPUs.")
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / "ocr_results.jsonl"
    completed = {row["page_id"]: row for row in read_jsonl(results_path)} if results_path.exists() else {}
    pages = read_jsonl(INPUT)
    targets = [row for row in pages if bool(row.get("needs_ocr"))]
    if len(targets) != 262:
        raise RuntimeError(f"Expected the frozen 262 OHR OCR targets, got {len(targets)}")
    if args.limit:
        targets = targets[:args.limit]
    if args.page_id:
        requested = set(args.page_id)
        targets = [row for row in targets if str(row["page_id"]) in requested]
        if len(targets) != len(requested):
            raise RuntimeError("One or more requested --page-id values are not frozen OCR targets")
    if args.ocr_workers < 1 or args.ocr_workers > (os.cpu_count() or 1):
        raise ValueError(f"--ocr-workers must be in 1..{os.cpu_count() or 1}; model workers above CPU count oversubscribe this machine")

    cache_dir = out / "paddlex-cache"
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(cache_dir))
    os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "BOS")
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    started = time.perf_counter()
    pending = [row for row in targets if str(row["page_id"]) not in completed or completed[str(row["page_id"])].get("error") is not None]
    with ProcessPoolExecutor(max_workers=args.ocr_workers, mp_context=mp.get_context("spawn"), initializer=init_ocr_worker, initargs=(str(cache_dir),)) as executor:
        futures = {executor.submit(ocr_one_page, row): str(row["page_id"]) for row in pending}
        for position, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            append_jsonl(results_path, record)
            completed[record["page_id"]] = record
            print(f"[{position}/{len(pending)}] {record['page_id']}: {record['word_count']} words", flush=True)

    if args.limit or args.page_id:
        return
    if len(completed) != 262:
        raise RuntimeError(f"Expected 262 completed OCR rows, got {len(completed)}")
    output_pages = []
    for row in pages:
        result = completed.get(str(row["page_id"]))
        updated = dict(row)
        if result is not None:
            updated.update({"text": result["text"], "text_source": "ppocrv5_mobile" if result["text"] else "empty", "ocr_applied": True, "ocr_word_count": result["word_count"], "ocr_mean_confidence": result["mean_confidence"], "ocr_render_seconds": result["render_seconds"], "ocr_seconds": result["ocr_seconds"], "ocr_error": result["error"], "ocr_engine": "PP-OCRv5_mobile_det+rec"})
        output_pages.append(updated)
    (out / "pages.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_pages), encoding="utf-8")
    errors = [row for row in completed.values() if row["error"]]
    summary = {"model": "PP-OCRv5_mobile_det + PP-OCRv5_mobile_rec", "render_dpi": 144, "targets": 262, "nonempty": sum(bool(row["text"]) for row in completed.values()), "errors": len(errors), "mean_words": float(statistics.mean(row["word_count"] for row in completed.values())), "mean_confidence": float(statistics.mean(row["mean_confidence"] for row in completed.values())), "wall_seconds_including_model_load": time.perf_counter() - started}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
