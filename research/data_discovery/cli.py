"""Command line entry point for the page-discovery baseline.

Example:

    python -m research.data_discovery.cli \
        --input data/raw/my-lake \
        --index-dir data/work/page-discovery \
        --query "revenue recognition" \
        --top-k-pages 12

Add ``--ingest`` and ``--pipeline-config`` to run the accurate parser and the
normal cleaning/enrichment/chunking stages on the selected pages.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import json

from .pipeline import PdfInspectorPageParser, PageIndex, build_page_index, run_on_demand


def main(argv: list[str] | None = None) -> int:
    args = _arguments().parse_args(argv)
    input_path = Path(args.input)
    files = _pdf_files(input_path)
    if not files:
        raise SystemExit(f"No PDF files found under {input_path}")

    index_dir = Path(args.index_dir)
    if args.rebuild_index or not (index_dir / "bm25.json").exists():
        index = build_page_index(files, parser=PdfInspectorPageParser())
        index.save(index_dir)
    else:
        index = PageIndex.load(index_dir)

    hits = index.search(args.query, top_k=args.top_k_pages)
    payload: dict[str, Any] = {
        "contract_version": "data-discovery-run-v1",
        "query": args.query,
        "index_dir": str(index_dir),
        "indexed_page_count": len(index.pages),
        "hits": [hit.as_dict() for hit in hits],
    }

    if args.ingest:
        from src.utils.config import load_config, resolve_parser_config

        config = load_config(args.pipeline_config)
        parser_config = config.get("parsing") or {}
        work_dir = index_dir / "on-demand-work"
        parser_config = resolve_parser_config(
            Path(__file__).resolve().parents[2],
            parser_config,
            work_dir / "parser-assets",
        )
        result = run_on_demand(
            index,
            args.query,
            parser_config=parser_config,
            top_k_pages=args.top_k_pages,
            chunking_config=(config.get("chunking_embedding") if args.chunk else None),
            project_root=Path(__file__).resolve().parents[2],
            work_dir=work_dir,
        )
        payload["selected_pages"] = result.selected_pages
        payload["ingested_documents"] = len(result.ingestion.parsed_data)
        payload["quarantined_documents"] = len(result.ingestion.quarantined_documents)
        if result.chunking_embedding is not None:
            payload["retrieval_records"] = len(result.chunking_embedding.retrieval_records)

    output = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output + "\n", encoding="utf-8")
    else:
        print(output)
    return 0


def _arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="PDF file or recursive PDF directory")
    parser.add_argument("--index-dir", required=True, help="Discovery index directory")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k-pages", type=int, default=10)
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Run accurate ingestion on selected pages",
    )
    parser.add_argument("--chunk", action="store_true", help="Also run chunking/embedding")
    parser.add_argument("--pipeline-config", default="configs/pipeline.yaml")
    parser.add_argument("--output")
    return parser


def _pdf_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() == ".pdf" else []
    return sorted(item for item in path.rglob("*.pdf") if item.is_file())


if __name__ == "__main__":
    raise SystemExit(main())
