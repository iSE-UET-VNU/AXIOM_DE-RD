"""Materialise KDL-parsed page texts for the ViDoRe V3 retrieval arms.

The KDL parse lives outside data/ at data_vidore_parsed_physics/ (gitignored),
one pipeline run per subset. This flattens it to {unit_id: page text} keyed the
same way the benchmark keys gold -- subset::<file>#page=N -- so a KDL arm can be
scored against exactly the qrels the other arms use.

    python research/experiments/export_kdl_page_texts.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluation.benchmarks.vidore_v3 import unit_id
from src.evaluation.pipeline_pages import canonical_doc, documents, page_blocks

BASE = ROOT / "data_vidore_parsed_physics/output/benchmarks"
OUT = ROOT / "data/benchmark/vidore_v3/results"
RUNS = {"physics": "vidore-v3-physics-kdl",
        "pharmaceuticals": "vidore-v3-pharmaceuticals-kdl"}


def main() -> None:
    for subset, name in RUNS.items():
        run = next((BASE / name).iterdir())
        pages = {}
        for document in documents(run):
            doc = canonical_doc(document.get("document", {}).get("file_name"))
            for page, blocks in page_blocks(document).items():
                pages[unit_id(subset, doc, page)] = "\n".join(
                    b["text"] for b in blocks if b["text"].strip())
        target = OUT / f"{subset}_kdl_page_texts.json"
        target.write_text(json.dumps(pages, ensure_ascii=False), encoding="utf-8")
        lens = [len(v) for v in pages.values() if v.strip()]
        print(f"{subset:16s} run={run.name} pages={len(pages)} "
              f"non-empty={len(lens)} mean_chars={sum(lens)//len(lens)} -> {target.name}")


if __name__ == "__main__":
    main()
