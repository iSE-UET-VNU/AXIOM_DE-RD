"""Extract the retrievable corpus from the data lake.

    python -m research.harness.build_corpus [--lake DIR --out DIR --limit N]

Writes ``corpus.jsonl`` (one record per successfully extracted document) and an
extraction report. Files we cannot turn into text are reported, not hidden:
extraction coverage bounds every retrieval number that follows.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from pathlib import Path
import argparse
import json
import time

from .extract import extract, supported_extensions
from .lake import Lake, modality_of

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LAKE = PROJECT_ROOT / "[iSE Summer Challenge 2026] Data Lake"
DEFAULT_OUT = PROJECT_ROOT / "data" / "benchmark"
CORPUS_MODALITIES = frozenset({"text", "table"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lake", default=str(DEFAULT_LAKE))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--modalities",
        default=",".join(sorted(CORPUS_MODALITIES)),
        help="Comma-separated lake modalities to extract.",
    )
    args = parser.parse_args()

    modalities = frozenset(part.strip() for part in args.modalities.split(",") if part.strip())
    lake = Lake.index(args.lake)
    candidates = lake.documents(modalities)
    if args.limit:
        candidates = candidates[: args.limit]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = out_dir / "corpus.jsonl"

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
    (out_dir / "corpus_report.json").write_text(
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
