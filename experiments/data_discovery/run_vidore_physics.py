"""Run page-level BM25 discovery over a complete ViDoRe V3 subset.

This deliberately stops after the light retrieval stage.  It writes one JSONL
record per query, using the same ``chunks`` shape as the retrieval run format,
so a later evaluator can consume it without re-running discovery.

Example:

    python -m experiments.data_discovery.run_vidore_physics \
        --language french \
        --top-k 100 \
        --output data/benchmark/vidore_v3/results/physics_discovery_bm25.jsonl
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import json
import time

from .pipeline import PageIndex, PdfInspectorPageParser, build_page_index
from src.evaluation.benchmarks.vidore_v3 import ViDoreV3


def main(argv: list[str] | None = None) -> int:
    args = _arguments().parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    subset = args.subset
    pdf_root = Path(args.pdf_root) if args.pdf_root else (
        root / f"data/raw/benchmarks/vidore_v3/vidore_v3_{subset}/pdfs"
    )
    index_dir = Path(args.index_dir) if args.index_dir else (
        root / f"data/work/vidore_v3/{subset}/discovery_bm25"
    )
    output = Path(args.output) if args.output else (
        root / f"data/benchmark/vidore_v3/results/{subset}_discovery_bm25_{args.language}.jsonl"
    )

    pdfs = sorted(path for path in pdf_root.glob("*.pdf") if path.is_file())
    if not pdfs:
        raise RuntimeError(f"No PDF files found in {pdf_root}")

    benchmark = ViDoreV3(
        root=Path(args.benchmark_root) if args.benchmark_root else (root / "data/benchmark/vidore_v3"),
        subset=subset,
        language=args.language,
    )
    questions = list(benchmark.questions())
    if args.limit is not None:
        questions = questions[: args.limit]

    def vidore_unit(source_uri: str, page_index: int) -> str:
        # ViDoRe's page_number_in_doc is zero-based in the benchmark corpus.
        return f"{source_uri}#page={page_index}"

    parser = PdfInspectorPageParser(page_id_factory=vidore_unit)
    started = time.perf_counter()
    if args.rebuild_index or not (index_dir / "bm25.json").exists():
        index = build_page_index(
            pdfs,
            parser=parser,
            source_uri=lambda path: f"{subset}::{path.stem}",
        )
        index.save(index_dir)
    else:
        index = PageIndex.load(index_dir)

    if not index.pages:
        raise RuntimeError(f"No pages indexed from {pdf_root}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for question in questions:
            hits = index.search(question.query, top_k=args.top_k)
            record: dict[str, Any] = {
                "qid": question.qid,
                "query": question.query,
                "retriever_id": "bm25_discovery",
                "index_id": f"vidore_v3.{subset}.page.pdf_inspector",
                "scope": subset,
                "language": args.language,
                "chunks": [
                    {
                        "chunk_id": hit.page_id,
                        "doc_id": hit.page_id,
                        "text": hit.evidence.text,
                        "score": round(hit.score, 6),
                        "rank": hit.rank,
                    }
                    for hit in hits
                ],
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(
        f"subset={subset} language={args.language} pages={len(index.pages)} "
        f"questions={len(questions)} top_k={args.top_k} "
        f"seconds={time.perf_counter() - started:.2f} output={output}"
    )
    return 0


def _arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="physics", choices=[
        "hr", "energy", "computer_science", "physics", "finance_en",
        "finance_fr", "industrial", "pharmaceuticals",
    ])
    parser.add_argument("--language", default="french", choices=[
        "english", "french", "spanish", "italian", "german", "portuguese",
    ])
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--pdf-root")
    parser.add_argument("--benchmark-root")
    parser.add_argument("--index-dir")
    parser.add_argument("--output")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
