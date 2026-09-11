"""Run full ViDoRe Physics OCR discovery with Tesseract + BM25.

The runner deliberately keeps the retrieval protocol identical to the visual
Physics experiments: official corpus page order, French queries, page-level
units, and the shared evaluator.  It uses ``fastparquet`` because the checked-
in ViDoRe parquet files contain nested fields that the local Arrow reader
cannot decode.

Example::

    python research/experiments/physics_tesseract_bm25.py \
        --tessdata-dir data/work/tesseract_smoke/tessdata \
        --workers 8
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import argparse
import csv
import io
import json
import statistics
import subprocess
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.benchmarks.base import GoldSpec, Question  # noqa: E402
from src.evaluation.run_retrieval import evaluate  # noqa: E402
from src.retrieval import runs  # noqa: E402
from src.retrieval.protocol import ScoredChunk  # noqa: E402
from src.retrieval.sparse import BM25Index  # noqa: E402
from src.evaluation.benchmarks.vidore_v3 import unit_id  # noqa: E402


@dataclass(frozen=True)
class PageTask:
    pdf_path: str
    doc_id: str
    file_name: str
    page_number_in_doc: int
    corpus_id: int

    @property
    def page_id(self) -> str:
        return unit_id("physics", self.doc_id, self.page_number_in_doc)


@dataclass(frozen=True)
class OCRPage:
    page_id: str
    corpus_id: int
    doc_id: str
    file_name: str
    page_number_in_doc: int
    text: str
    word_count: int
    mean_confidence: float
    render_seconds: float
    ocr_seconds: float
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "page_id": self.page_id,
            "corpus_id": self.corpus_id,
            "doc_id": self.doc_id,
            "file_name": self.file_name,
            "page_number_in_doc": self.page_number_in_doc,
            "text": self.text,
            "word_count": self.word_count,
            "mean_confidence": round(self.mean_confidence, 4),
            "render_seconds": round(self.render_seconds, 6),
            "ocr_seconds": round(self.ocr_seconds, 6),
            "error": self.error,
        }


class FastPhysicsBenchmark:
    """Minimal fastparquet-backed adapter for the shared evaluator."""

    def __init__(self, dataset_dir: Path, language: str) -> None:
        corpus = pd.read_parquet(
            dataset_dir / "corpus.parquet",
            engine="fastparquet",
            columns=["corpus_id", "doc_id", "page_number_in_doc"],
        )
        query = pd.read_parquet(
            dataset_dir / "queries.parquet",
            engine="fastparquet",
            columns=["query_id", "query", "language", "content_type", "answer"],
        )
        query = query[query.language == language].sort_values("query_id")
        qrels = pd.read_parquet(
            dataset_dir / "qrels.parquet",
            engine="fastparquet",
            columns=["query_id", "corpus_id", "score"],
        )

        by_corpus = {
            int(row.corpus_id): (str(row.doc_id), int(row.page_number_in_doc))
            for row in corpus.itertuples(index=False)
        }
        self._questions = [
            Question(
                qid=f"physics::{int(row.query_id)}",
                query=str(row.query),
                answer=str(row.answer or ""),
                answer_type="judge",
                modalities=tuple(str(x) for x in (row.content_type or []))
                or ("unknown",),
                taxonomy="vidore_v3",
            )
            for row in query.itertuples(index=False)
        ]
        self._gold: dict[str, dict[str, int]] = {}
        for row in qrels.itertuples(index=False):
            page = by_corpus.get(int(row.corpus_id))
            if page is None:
                continue
            qid = f"physics::{int(row.query_id)}"
            page_id = unit_id("physics", page[0], page[1])
            self._gold.setdefault(qid, {})[page_id] = int(row.score)

    def questions(self) -> Iterable[Question]:
        return iter(self._questions)

    def qrels(self) -> dict[str, dict[str, int]]:
        return self._gold

    def gold_docs(self, qid: str) -> GoldSpec:
        return GoldSpec(docs=tuple(self._gold.get(str(qid), {})))

    def gold_pages(self, qid: str) -> list[str] | None:
        units = self._gold.get(str(qid))
        if not units:
            return None
        return [unit.rsplit("#page=", 1)[-1] for unit in units]

    def gold_regions(self, qid: str) -> None:
        return None


def _ocr_page(
    task: PageTask,
    *,
    tesseract: str,
    tessdata_dir: str,
        language: str,
    psm: int,
) -> OCRPage:
    import fitz

    render_started = time.perf_counter()
    try:
        with fitz.open(task.pdf_path) as document:
            page = document.load_page(task.page_number_in_doc)
            pixmap = page.get_pixmap(
                matrix=fitz.Matrix(144 / 72.0, 144 / 72.0), alpha=False
            )
            image_bytes = pixmap.tobytes("png")
        render_seconds = time.perf_counter() - render_started

        ocr_started = time.perf_counter()
        command = [
            tesseract,
            "stdin",
            "stdout",
            "-l",
            language,
            "--tessdata-dir",
            tessdata_dir,
            "--psm",
            str(psm),
            "tsv",
        ]
        completed = subprocess.run(
            command,
            input=image_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        ocr_seconds = time.perf_counter() - ocr_started
        if completed.returncode != 0:
            error = completed.stderr.decode("utf-8", errors="replace").strip()
            return OCRPage(
                task.page_id, task.corpus_id, task.doc_id, task.file_name,
                task.page_number_in_doc, "", 0, 0.0, render_seconds,
                ocr_seconds, error or f"tesseract exit={completed.returncode}",
            )

        words: list[str] = []
        confidences: list[float] = []
        tsv_text = completed.stdout.decode("utf-8", errors="replace")
        for row in csv.DictReader(io.StringIO(tsv_text), delimiter="\t"):
            text = (row.get("text") or "").strip()
            if not text:
                continue
            words.append(text)
            try:
                confidence = float(row.get("conf", "-1"))
            except ValueError:
                confidence = -1.0
            if confidence >= 0:
                confidences.append(confidence)

        return OCRPage(
            task.page_id,
            task.corpus_id,
            task.doc_id,
            task.file_name,
            task.page_number_in_doc,
            " ".join(words),
            len(words),
            statistics.mean(confidences) if confidences else 0.0,
            render_seconds,
            ocr_seconds,
        )
    except Exception as exc:  # noqa: BLE001 - persist page-level failures
        elapsed = time.perf_counter() - render_started
        return OCRPage(
            task.page_id, task.corpus_id, task.doc_id, task.file_name,
            task.page_number_in_doc, "", 0, 0.0, elapsed, 0.0,
            f"{type(exc).__name__}: {exc}",
        )


def _load_tasks(dataset_dir: Path, pdf_dir: Path) -> list[PageTask]:
    corpus = pd.read_parquet(
        dataset_dir / "corpus.parquet",
        engine="fastparquet",
        columns=["corpus_id", "doc_id", "page_number_in_doc"],
    ).sort_values("corpus_id")
    metadata = pd.read_parquet(
        dataset_dir / "documents_metadata.parquet",
        engine="fastparquet",
        columns=["file_name", "doc_id"],
    )
    doc_to_file = dict(zip(metadata.doc_id.astype(str), metadata.file_name.astype(str)))
    tasks: list[PageTask] = []
    for row in corpus.itertuples(index=False):
        doc_id = str(row.doc_id)
        file_name = doc_to_file.get(doc_id)
        if not file_name:
            raise FileNotFoundError(f"No filename for doc_id={doc_id}")
        pdf_path = pdf_dir / file_name
        if not pdf_path.is_file():
            raise FileNotFoundError(pdf_path)
        tasks.append(
            PageTask(
                str(pdf_path), doc_id, file_name, int(row.page_number_in_doc),
                int(row.corpus_id),
            )
        )
    return tasks


def _run_ocr(tasks: list[PageTask], args: argparse.Namespace) -> tuple[list[OCRPage], dict[str, float]]:
    started = time.perf_counter()
    pages: list[OCRPage] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _ocr_page,
                task,
                tesseract=str(args.tesseract),
                tessdata_dir=str(args.tessdata_dir),
                language=args.ocr_language,
                psm=args.psm,
            ): task
            for task in tasks
        }
        for future in as_completed(futures):
            pages.append(future.result())
            completed += 1
            if completed % 100 == 0 or completed == len(tasks):
                print(f"ocr {completed}/{len(tasks)}", flush=True)
    pages.sort(key=lambda page: page.corpus_id)
    render_sum = sum(page.render_seconds for page in pages)
    ocr_sum = sum(page.ocr_seconds for page in pages)
    return pages, {
        "render_seconds_sum": render_sum,
        "ocr_seconds_sum": ocr_sum,
        "render_ocr_wall_seconds": time.perf_counter() - started,
    }


def _retrieve(
    pages: list[OCRPage],
    benchmark: FastPhysicsBenchmark,
    top_k: int,
    *,
    ocr_language: str,
    psm: int,
) -> tuple[list[runs.RunRecord], dict[str, float]]:
    indexable = [page for page in pages if page.text.strip()]
    payload = [
        {"chunk_id": page.page_id, "doc_id": page.page_id, "text": page.text}
        for page in indexable
    ]
    index_started = time.perf_counter()
    bm25 = BM25Index(analyzer_name="auto").build(payload)
    index_seconds = time.perf_counter() - index_started

    retrieval_started = time.perf_counter()
    records: list[runs.RunRecord] = []
    by_position = {position: page for position, page in enumerate(indexable)}
    for question in benchmark.questions():
        query_started = time.perf_counter()
        hits = bm25.search(question.query, top_k)
        scored = [
            ScoredChunk(
                chunk_id=by_position[position].page_id,
                doc_id=by_position[position].page_id,
                score=score,
                rank=rank,
                text=by_position[position].text,
            )
            for rank, (position, score) in enumerate(hits, start=1)
        ]
        records.append(
            runs.RunRecord.build(
                question.qid,
                question.query,
                "tesseract_bm25",
                "vidore.physics.page.tesseract",
                runs.params_hash(
                    {"top_k": top_k, "ocr_language": ocr_language, "psm": psm}
                ),
                scored,
                latency_ms=1000 * (time.perf_counter() - query_started),
            )
        )
    return records, {
        "bm25_index_seconds": index_seconds,
        "retrieval_seconds": time.perf_counter() - retrieval_started,
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = _arguments().parse_args()
    total_started = time.perf_counter()
    dataset_dir = args.dataset_dir
    pdf_dir = args.pdf_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.tesseract.is_file():
        raise FileNotFoundError(f"Tesseract executable not found: {args.tesseract}")
    traineddata = args.tessdata_dir / f"{args.ocr_language}.traineddata"
    if not traineddata.is_file():
        raise FileNotFoundError(f"Language data not found: {traineddata}")

    load_started = time.perf_counter()
    tasks = _load_tasks(dataset_dir, pdf_dir)
    benchmark = FastPhysicsBenchmark(dataset_dir, args.query_language)
    load_seconds = time.perf_counter() - load_started
    if args.limit is not None:
        tasks = tasks[: args.limit]

    print(
        f"pages={len(tasks)} queries={len(list(benchmark.questions()))} "
        f"workers={args.workers} ocr_language={args.ocr_language} "
        f"query_language={args.query_language}",
        flush=True,
    )
    pages, ocr_timing = _run_ocr(tasks, args)

    pages_path = args.output_dir / "ocr_pages.jsonl"
    with pages_path.open("w", encoding="utf-8") as handle:
        for page in pages:
            handle.write(json.dumps(page.as_dict(), ensure_ascii=False) + "\n")

    records, retrieval_timing = _retrieve(
        pages,
        benchmark,
        args.top_k,
        ocr_language=args.ocr_language,
        psm=args.psm,
    )
    runs.write(args.output_dir / "retrieval_french.jsonl", records)

    evaluation_started = time.perf_counter()
    metrics = {f"k={k}": evaluate(benchmark, records, k) for k in (1, 5, 10, 20, 50, 100)}
    evaluation_seconds = time.perf_counter() - evaluation_started

    nonempty = [page for page in pages if page.text.strip()]
    errors = [page for page in pages if page.error]
    result = {
        "model": "tesseract-5.5.3",
        "ocr_language": args.ocr_language,
        "query_language": args.query_language,
        "psm": args.psm,
        "workers": args.workers,
        "retrieval_unit": "one rendered PDF page OCR text",
        "page_count": len(pages),
        "nonempty_ocr_pages": len(nonempty),
        "ocr_error_pages": len(errors),
        "queries": len(list(benchmark.questions())),
        "timing_seconds": {
            "load_dataset_and_tasks": load_seconds,
            **ocr_timing,
            **retrieval_timing,
            "evaluation": evaluation_seconds,
            "total": time.perf_counter() - total_started,
        },
        "parsing_stats": {
            "mean_word_count_nonempty": statistics.mean(page.word_count for page in nonempty)
            if nonempty else 0.0,
            "mean_confidence_nonempty": statistics.mean(page.mean_confidence for page in nonempty)
            if nonempty else 0.0,
            "mean_text_chars_nonempty": statistics.mean(len(page.text) for page in nonempty)
            if nonempty else 0.0,
        },
        "metrics": metrics,
        "outputs": {
            "pages": str(pages_path),
            "retrieval": str(args.output_dir / "retrieval_french.jsonl"),
        },
    }
    _write_json(args.output_dir / "metrics.json", result)
    print(json.dumps({
        "timing_seconds": result["timing_seconds"],
        "parsing_stats": result["parsing_stats"],
        "metrics@10": {
            key: metrics["k=10"].get(key)
            for key in ("recall@10", "ndcg@10", "page_recall@10")
        },
    }, ensure_ascii=False, indent=2))
    return 0


def _arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir", type=Path,
        default=ROOT / "data/benchmark/vidore_v3/physics",
    )
    parser.add_argument(
        "--pdf-dir", type=Path,
        default=ROOT / "data/raw/benchmarks/vidore_v3/vidore_v3_physics/pdfs",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data/benchmark/vidore_v3/results/physics_tesseract_bm25",
    )
    parser.add_argument(
        "--tesseract", type=Path,
        default=Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
    )
    parser.add_argument("--tessdata-dir", type=Path, required=True)
    parser.add_argument(
        "--ocr-language", default="fra",
        help="Tesseract language code, e.g. fra or eng.",
    )
    parser.add_argument(
        "--query-language", default="french",
        choices=["english", "french", "spanish", "italian", "german", "portuguese"],
    )
    parser.add_argument("--psm", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--limit", type=int)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
