"""OCR every PDF embedded-image occurrence in the unified benchmark.

This is a light-preparation experiment.  It never changes the canonical
PDF-inspector/Tesseract cache: it creates a second pages JSONL whose ``text``
is the original page text with the OCR text inserted immediately after the
corresponding ``[Image: <label>]`` marker.  This keeps the PDF reading order
instead of creating a detached image-OCR field at the end of a page.

The run is resumable.  ``ocr_results.jsonl`` is appended per image occurrence;
on rerun completed occurrences are not sent to Tesseract again.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from subprocess import PIPE, run
from time import perf_counter
from typing import Any, Iterable

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[2]
IMAGE_MARKER = __import__("re").compile(r"\[Image:\s*([^\]]+?)\]")


def now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def norm_label(value: str) -> str:
    return Path(value.lstrip("/")).stem


def safe_piece(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


@dataclass(frozen=True)
class ImageTask:
    key: str
    page_id: str
    doc_id: str
    source: str
    page_index: int
    image_label: str
    occurrence_on_page: int
    image_path: Path
    width: int | None
    height: int | None


def result_key(page_id: str, label: str, occurrence: int) -> str:
    return f"{page_id}\t{label}\t{occurrence}"


def call_tesseract(task: ImageTask, executable: Path, language: str, psm: int) -> dict[str, Any]:
    started = perf_counter()
    completed = run(
        [str(executable), str(task.image_path), "stdout", "-l", language, "--psm", str(psm)],
        stdin=PIPE,
        stdout=PIPE,
        stderr=PIPE,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    seconds = perf_counter() - started
    return {
        "key": task.key,
        "page_id": task.page_id,
        "doc_id": task.doc_id,
        "source": task.source,
        "page_index": task.page_index,
        "image_label": task.image_label,
        "occurrence_on_page": task.occurrence_on_page,
        "image_path": str(task.image_path),
        "width": task.width,
        "height": task.height,
        "status": "ok" if completed.returncode == 0 else "error",
        "returncode": completed.returncode,
        "seconds": seconds,
        "text": completed.stdout.strip() if completed.returncode == 0 else "",
        "stderr": completed.stderr.strip()[-2000:],
    }


def completed_keys(results_path: Path) -> set[str]:
    if not results_path.is_file():
        return set()
    return {
        row["key"]
        for row in jsonl(results_path)
        if row.get("status") in {"ok", "error"} and row.get("key")
    }


def save_image(image: Any, path: Path) -> tuple[int | None, int | None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    pil_image = image.image
    width, height = pil_image.size
    # PNG is universally readable by local Windows Tesseract, independent of
    # the original PDF's DCT/JPX/CCITT encoding.
    pil_image.save(path, format="PNG")
    return int(width), int(height)


def extract_and_schedule(
    pages: list[dict[str, Any]],
    dataset_root: Path,
    output_dir: Path,
    done: set[str],
    executor: concurrent.futures.ThreadPoolExecutor,
    futures: set[concurrent.futures.Future[dict[str, Any]]],
    result_stream: Any,
    executable: Path,
    language: str,
    psm: int,
    max_pending: int,
    counters: Counter[str],
) -> None:
    """Extract images sequentially per PDF while keeping a bounded OCR queue."""

    def drain(wait_for_one: bool) -> None:
        if not futures:
            return
        completed, _ = concurrent.futures.wait(
            futures,
            return_when=(concurrent.futures.FIRST_COMPLETED if wait_for_one else concurrent.futures.ALL_COMPLETED),
        )
        for future in completed:
            futures.remove(future)
            row = future.result()
            result_stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            result_stream.flush()
            counters["ocr_completed"] += 1
            counters[f"ocr_{row['status']}"] += 1

    pages_by_pdf: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for page in pages:
        source = dataset_root / page["relative_path"]
        pages_by_pdf[source].append(page)

    for pdf_path, pdf_pages in pages_by_pdf.items():
        counters["pdfs_started"] += 1
        try:
            reader = PdfReader(str(pdf_path))
            for page_row in pdf_pages:
                marker_labels = [match.group(1).strip() for match in IMAGE_MARKER.finditer(page_row.get("text", ""))]
                if not marker_labels:
                    continue
                page_index = int(page_row["page_index"])
                if page_index >= len(reader.pages):
                    counters["invalid_page_index"] += 1
                    continue
                pdf_page = reader.pages[page_index]
                images_by_label: dict[str, Any] = {}
                for image in pdf_page.images:
                    images_by_label.setdefault(norm_label(image.name), image)
                occurrences = Counter()
                for raw_label in marker_labels:
                    label = norm_label(raw_label)
                    occurrence = occurrences[label]
                    occurrences[label] += 1
                    key = result_key(page_row["page_id"], label, occurrence)
                    counters["markers_seen"] += 1
                    if key in done:
                        counters["resume_skipped"] += 1
                        continue
                    image = images_by_label.get(label)
                    if image is None:
                        row = {
                            "key": key, "page_id": page_row["page_id"], "doc_id": page_row["doc_id"],
                            "source": page_row.get("source"), "page_index": page_index,
                            "image_label": label, "occurrence_on_page": occurrence,
                            "status": "missing_image_object", "returncode": None, "seconds": 0.0,
                            "text": "", "stderr": f"No pypdf image named {label}",
                        }
                        result_stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                        result_stream.flush()
                        counters["missing_image_object"] += 1
                        continue
                    image_path = output_dir / "images" / safe_piece(page_row["doc_id"]) / f"p{page_index:04d}_{safe_piece(label)}_{occurrence}.png"
                    try:
                        width, height = save_image(image, image_path)
                    except Exception as exc:  # image decoding is input-dependent
                        row = {
                            "key": key, "page_id": page_row["page_id"], "doc_id": page_row["doc_id"],
                            "source": page_row.get("source"), "page_index": page_index,
                            "image_label": label, "occurrence_on_page": occurrence,
                            "status": "extract_error", "returncode": None, "seconds": 0.0,
                            "text": "", "stderr": repr(exc),
                        }
                        result_stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                        result_stream.flush()
                        counters["extract_error"] += 1
                        continue
                    task = ImageTask(key, page_row["page_id"], page_row["doc_id"], page_row.get("source", ""), page_index, label, occurrence, image_path, width, height)
                    futures.add(executor.submit(call_tesseract, task, executable, language, psm))
                    counters["ocr_scheduled"] += 1
                    while len(futures) >= max_pending:
                        drain(wait_for_one=True)
            counters["pdfs_complete"] += 1
        except Exception as exc:  # a malformed PDF must not discard other documents
            counters["pdf_errors"] += 1
            print(f"[pdf-error] {pdf_path.name}: {exc}", flush=True)
    drain(wait_for_one=False)


def merged_pages(pages: list[dict[str, Any]], results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    by_page_label: dict[str, dict[str, deque[str]]] = defaultdict(lambda: defaultdict(deque))
    for row in sorted(results, key=lambda r: (r.get("page_id", ""), r.get("image_label", ""), int(r.get("occurrence_on_page", 0)))):
        if row.get("status") == "ok" and row.get("text", "").strip():
            by_page_label[row["page_id"]][row["image_label"]].append(row["text"].strip())
    stats = Counter()
    output: list[dict[str, Any]] = []
    for page in pages:
        label_queues = by_page_label.get(page["page_id"], {})
        def replace(match: Any) -> str:
            label = norm_label(match.group(1).strip())
            queue = label_queues.get(label)
            if not queue:
                return match.group(0)
            text = queue.popleft()
            stats["inserted_images"] += 1
            stats["inserted_words"] += len(text.split())
            # Keep the original marker (baseline text remains a prefix/superset)
            # and append OCR right at its reading-order location.
            return f"{match.group(0)}\n{text}"
        new_page = dict(page)
        new_page["text"] = IMAGE_MARKER.sub(replace, page.get("text", ""))
        new_page["embedded_image_ocr_inserted"] = True
        output.append(new_page)
    return output, dict(stats)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data/raw/BENCHMARK")
    parser.add_argument("--pages", type=Path, required=True, help="Existing PDF-inspector/Tesseract pages.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tesseract", type=Path, default=Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"))
    parser.add_argument("--language", default="eng")
    parser.add_argument("--psm", type=int, default=6)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--max-pending", type=int, default=192, help="Bound queued Tesseract jobs; defaults to 4x workers.")
    args = parser.parse_args()
    if not args.pages.is_file():
        raise FileNotFoundError(args.pages)
    if not args.tesseract.is_file():
        raise FileNotFoundError(args.tesseract)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pages = list(jsonl(args.pages))
    results_path = args.output_dir / "ocr_results.jsonl"
    events_path = args.output_dir / "events.jsonl"
    done = completed_keys(results_path)
    config = {
        "created_at": now(), "dataset_root": str(args.dataset_root.resolve()), "pages": str(args.pages.resolve()),
        "pages_sha256": hashlib.sha256(args.pages.read_bytes()).hexdigest(), "tesseract": str(args.tesseract),
        "language": args.language, "psm": args.psm, "workers": args.workers, "max_pending": args.max_pending,
        "protocol": "OCR each embedded image occurrence; retain native marker and insert raw OCR immediately after it.",
    }
    write_json(args.output_dir / "config.json", config)
    started = perf_counter()
    counters: Counter[str] = Counter()
    with events_path.open("a", encoding="utf-8") as events, results_path.open("a", encoding="utf-8") as result_stream:
        events.write(json.dumps({"timestamp": now(), "event": "embedded-image-ocr", "status": "started", **config, "resume_completed": len(done)}, ensure_ascii=False) + "\n")
        events.flush()
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="tesseract") as executor:
            futures: set[concurrent.futures.Future[dict[str, Any]]] = set()
            extract_and_schedule(pages, args.dataset_root, args.output_dir, done, executor, futures, result_stream, args.tesseract, args.language, args.psm, max(args.max_pending, args.workers), counters)
        all_results = list(jsonl(results_path))
        merged, merge_stats = merged_pages(pages, all_results)
        merged_path = args.output_dir / "pages_with_embedded_ocr.jsonl"
        with merged_path.open("w", encoding="utf-8", newline="\n") as stream:
            for row in merged:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        seconds = perf_counter() - started
        ok = [row for row in all_results if row.get("status") == "ok"]
        summary = {
            **config, "finished_at": now(), "wall_seconds": seconds, "source_pages": len(pages),
            "result_records": len(all_results), "successful_ocr": len(ok),
            "ocr_nonempty": sum(bool(row.get("text", "").strip()) for row in ok),
            "mean_ocr_seconds": mean(row["seconds"] for row in ok) if ok else 0.0,
            "median_ocr_seconds": median(row["seconds"] for row in ok) if ok else 0.0,
            "throughput_scheduled_per_second": counters["ocr_scheduled"] / seconds if seconds else 0.0,
            "counters": dict(counters), "merge": merge_stats,
            "merged_pages": str(merged_path),
        }
        write_json(args.output_dir / "summary.json", summary)
        events.write(json.dumps({"timestamp": now(), "event": "embedded-image-ocr", "status": "complete", **summary}, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
