"""Measure cached Physics Kf=3 hierarchical BM25 retrieval without V-SPLADE."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.data_discovery.hierarchical import CascadeConfig
from research.experiments.physics_hierarchical_retrieval import (
    HierarchicalRetriever,
    _build_corpus_and_indexes,
    _page_vector_units,
)
from src.evaluation.benchmarks.vidore_v3 import ViDoreV3


def main() -> None:
    page_ids = _page_vector_units(ROOT / "data/output/vsplade/vidore_v3_physics_48q")
    benchmark = ViDoreV3(
        root=ROOT / "data/benchmark/vidore_v3",
        subset="physics",
        language="french",
    )
    questions = sorted(
        benchmark.questions(), key=lambda item: int(item.qid.rsplit("::", 1)[1])
    )
    started = perf_counter()
    corpus, page_index, file_indexes, fine_indexes, _ = _build_corpus_and_indexes(
        ROOT / "data/output/vidore-v3-physics-kdl-pdf-inspector/32c32a45a92c45bb",
        page_ids,
        fine_units=[],
    )
    setup_seconds = perf_counter() - started
    retriever = HierarchicalRetriever(
        corpus,
        page_index=page_index,
        file_indexes=file_indexes,
        fine_indexes=fine_indexes,
        visual_scores={},
    )
    config = CascadeConfig(
        name="physics-no-v-kf3-timing",
        file_representation="all_text",
        file_pool="max",
        k_files=3,
        page_depth=1674,
        final_depth=100,
        bm25_weight=1.0,
        parent_weight=0.15,
        fine_unit="none",
        fine_weight=0.0,
        file_direct_weight=0.5,
        file_pool_source="page_base",
    )
    started = perf_counter()
    for question in questions:
        retriever.retrieve(question.qid, question.query, config)
    retrieval_seconds = perf_counter() - started
    print(
        json.dumps(
            {
                "dataset": "vidore_v3/physics",
                "variant": "hierarchical_bm25_only_kf3",
                "queries": len(questions),
                "pages": len(page_ids),
                "setup_hierarchy_plus_page_file_bm25_seconds": round(setup_seconds, 6),
                "embedding_seconds": 0.0,
                "retrieval_302q_seconds": round(retrieval_seconds, 6),
                "retrieval_ms_per_query": round(retrieval_seconds / len(questions) * 1000, 6),
            }
        )
    )


if __name__ == "__main__":
    main()
