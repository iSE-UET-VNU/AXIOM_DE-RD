"""Extract the retrievable corpus from the data lake.

    python -m src.evaluation.build_corpus --corpus PATH [--lake DIR --limit N]
    python -m src.evaluation.build_corpus --corpus PATH --from-pipeline RUN_DIR

Writes one record per successfully extracted document, plus an extraction
report. Files we cannot turn into text are reported, not hidden: extraction
coverage bounds every retrieval number that follows.

``--corpus`` is required and has no default, because a fixed filename let
``--from-pipeline`` overwrite the ``extract.py`` baseline the parser comparison
needs as its control.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from pathlib import Path

from src.utils.paths import repo_root
import argparse
import json
import time

from . import pipeline_corpus
from .extract import extract, supported_extensions
from .lake import Lake, modality_of

PROJECT_ROOT = repo_root(__file__)
DEFAULT_LAKE = PROJECT_ROOT / "[iSE Summer Challenge 2026] Data Lake"
CORPUS_MODALITIES = frozenset({"text", "table"})


def report_path_for(corpus_path: Path) -> Path:
    """The report travels with the corpus it describes, not with a shared name."""
    return corpus_path.with_name(f"{corpus_path.stem}_report.json")


def claim(corpus_path: Path, overwrite: bool) -> Path:
    """Refuse to clobber an existing corpus; a lost baseline is unrecoverable."""
    if corpus_path.exists() and not overwrite:
        raise SystemExit(
            f"{corpus_path} already exists. Pass --overwrite to replace it, or name a "
            "different --corpus. Arms are distinguished by their corpus path; reusing "
            "one destroys the control the comparison is measured against."
        )
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    return corpus_path


def build_from_pipeline(run_dir: Path, corpus_path: Path) -> None:
    """Same record shape, sourced from a pipeline run rather than extract.py."""
    started = time.perf_counter()
    parser_used = pipeline_corpus.parser_name(run_dir)

    kept = 0
    empty: list[dict[str, str]] = []
    blocks_total = chars_total = 0
    by_modality: Counter[str] = Counter()
    candidates = 0
    with corpus_path.open("w", encoding="utf-8") as handle:
        for document in pipeline_corpus.documents(run_dir):
            candidates += 1
            record = pipeline_corpus.record_of(document, parser_used)
            if record is None:
                name = (document.get("document") or {}).get("file_name", "?")
                empty.append({"doc_id": str(name), "error": "empty"})
                continue
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1
            blocks_total += len(record["blocks"])
            chars_total += sum(len(b["text"]) for b in record["blocks"])
            by_modality[record["modality"]] += 1

    report = {
        "source": "pipeline",
        "run_dir": str(run_dir),
        "parser": parser_used,
        "candidates": candidates,
        "extracted": kept,
        "failed": len(empty),
        "coverage": round(kept / candidates, 4) if candidates else 0.0,
        "blocks_total": blocks_total,
        "chars_total": chars_total,
        "chars_mean": round(chars_total / kept, 1) if kept else 0.0,
        "by_modality": dict(by_modality),
        "failure_reasons": {"empty": len(empty)} if empty else {},
        "failures": empty[:40],
        "seconds": round(time.perf_counter() - started, 2),
    }
    report_path_for(corpus_path).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"parser     : {parser_used}")
    print(f"candidates : {candidates}")
    print(f"extracted  : {kept}  ({report['coverage']:.1%} coverage)")
    print(f"failed     : {len(empty)}  (parsed but produced no text)")
    print(f"blocks     : {blocks_total}   chars={chars_total:,}  mean={report['chars_mean']:,.0f}")
    print(f"by modality: {dict(by_modality)}")
    print(f"elapsed    : {report['seconds']}s -> {corpus_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lake", default=str(DEFAULT_LAKE))
    parser.add_argument(
        "--corpus", required=True, metavar="PATH",
        help="Where to write the corpus. Required, no default: this names the arm.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace an existing corpus at --corpus. Off by default.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--modalities",
        default=",".join(sorted(CORPUS_MODALITIES)),
        help="Comma-separated lake modalities to extract.",
    )
    parser.add_argument(
        "--from-pipeline",
        default="",
        metavar="RUN_DIR",
        help="Build from data/output/<run_id> instead of parsing the lake. "
        "This is how a lift_api or chandra2 parse reaches the benchmark.",
    )
    args = parser.parse_args()
    corpus_path = claim(Path(args.corpus), args.overwrite)

    if args.from_pipeline:
        build_from_pipeline(Path(args.from_pipeline), corpus_path)
        return

    modalities = frozenset(part.strip() for part in args.modalities.split(",") if part.strip())
    lake = Lake.index(args.lake)
    candidates = lake.documents(modalities)
    if args.limit:
        candidates = candidates[: args.limit]

    started = time.perf_counter()
    failures: list[dict[str, str]] = []
    kept = 0
    blocks_total = 0
    chars_total = 0
    by_modality: Counter[str] = Counter()
    with corpus_path.open("w", encoding="utf-8") as handle:
        for path in candidates:
            doc_id = lake.doc_id(path)
            document = extract(path, doc_id)
            if not document.ok:
                failures.append({"doc_id": doc_id, "error": document.error or "empty"})
                continue
            record = {
                "doc_id": doc_id,
                "title": document.title,
                "modality": modality_of(path),
                "suffix": path.suffix.lower(),
                "blocks": [asdict(block) for block in document.blocks],
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1
            blocks_total += len(document.blocks)
            chars_total += sum(len(block.text) for block in document.blocks)
            by_modality[record["modality"]] += 1

    elapsed = time.perf_counter() - started
    report = {
        "candidates": len(candidates),
        "extracted": kept,
        "failed": len(failures),
        "coverage": round(kept / len(candidates), 4) if candidates else 0.0,
        "supported_extensions": sorted(supported_extensions()),
        "blocks_total": blocks_total,
        "chars_total": chars_total,
        "chars_mean": round(chars_total / kept, 1) if kept else 0.0,
        "by_modality": dict(by_modality),
        "failure_reasons": dict(Counter(f["error"].split(":")[0] for f in failures).most_common()),
        "failures": failures[:40],
        "seconds": round(elapsed, 2),
    }
    report_path_for(corpus_path).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"candidates : {report['candidates']}")
    print(f"extracted  : {report['extracted']}  ({report['coverage']:.1%} coverage)")
    print(f"failed     : {report['failed']}  reasons={report['failure_reasons']}")
    print(f"blocks     : {report['blocks_total']}   chars={report['chars_total']:,}  mean={report['chars_mean']:,.0f}")
    print(f"by modality: {report['by_modality']}")
    print(f"elapsed    : {report['seconds']}s -> {corpus_path}")


if __name__ == "__main__":
    main()
