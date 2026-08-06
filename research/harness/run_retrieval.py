"""Run retrieval arms over a benchmark and emit run records.

    python -m research.harness.run_retrieval --benchmark mmdocir --arms bm25,dense,rrf
    python -m research.harness.run_retrieval --benchmark ise --arms bm25 --embedder openrouter_te3s

Output is the run-record JSONL that ``run_answer.py`` consumes, so the same file
feeds retrieval metrics and answer metrics. Runs are cached on
``(index_id, retriever_id, params_hash, query_set_hash)``; the index identity
includes the analyzer and the embedder, so a run produced under a different
tokenizer cannot be silently reused.

Retrieval metrics are computed here at whichever granularities the benchmark
provides -- file/page/region -- and always broken down by evidence modality.
Answer metrics come later, from the same file.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence
import argparse
import json
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval import runs  # noqa: E402
from src.retrieval.index import LocalIndex  # noqa: E402
from src.retrieval.protocol import ChunkRecord  # noqa: E402
from src.retrieval.sparse import BM25Index  # noqa: E402
from src.retrieval import retrievers  # noqa: E402

from .benchmarks import load as load_benchmark  # noqa: E402

DATA = PROJECT_ROOT / "data" / "benchmark"
DEFAULT_ARMS = ("bm25", "dense", "rrf")


def build_index(
    benchmark: Any, embedder: Any, index_id: str, chunker: str = "", params: dict | None = None
) -> LocalIndex:
    """Index a benchmark's units, chunking them first if asked.

    MMDocIR ships pages and layouts -- already the retrieval granularity, so no
    chunking. The iSE lake ships whole documents, where the chunker is an
    experimental variable rather than a property of the source, so it is applied
    here and named in the index_id.
    """
    if chunker:
        from .chunking import chunk_corpus

        documents = [
            {
                "doc_id": doc.doc_id,
                "title": doc.meta.get("title", ""),
                "text": doc.text or "",
                "blocks": doc.meta.get("blocks") or [],
            }
            for doc in benchmark.corpus()
        ]
        records = [
            ChunkRecord(chunk_id=c.chunk_id, doc_id=c.doc_id, text=c.index_text)
            for c in chunk_corpus(documents, chunker, params or {})
        ]
    else:
        records = [
            ChunkRecord(
                chunk_id=doc.doc_id,
                doc_id=doc.doc_id,
                text=doc.text or "",
                page=doc.page,
                meta={"modality": doc.modality, **dict(doc.meta)},
            )
            for doc in benchmark.corpus()
        ]
    records = [r for r in records if r.text.strip()]
    payload = [{"chunk_id": r.chunk_id, "doc_id": r.doc_id, "text": r.text} for r in records]

    vectors = None
    if embedder is not None:
        import numpy as np

        texts = [r.text for r in records]
        rows: list[Any] = []
        for start in range(0, len(texts), 128):
            rows.extend(embedder.embed(texts[start : start + 128]))
        vectors = np.asarray(rows, dtype=np.float32)
        if vectors.shape[0] != len(records):
            raise ValueError("Embedder returned a different number of vectors than chunks.")

    bm25 = BM25Index(analyzer_name="auto").build(payload)
    return LocalIndex(
        index_id=f"{index_id}.{bm25.analyzer_name}",
        records=records,
        bm25=bm25,
        vectors=vectors,
        embedder=embedder,
    )


def reachability(benchmark: Any, corpus_doc_ids: set[str], qids: Sequence[str]) -> dict[str, bool]:
    """Whether each question's evidence exists in the indexed corpus at all.

    This is the coverage half of the parsing comparison and it must never be a
    filter. A question whose gold document could not be parsed is a **coverage**
    miss, not a retrieval miss; scoring it as 0.0 accuracy blames the retriever
    for the parser, and dropping it silently deletes the number that measures the
    parser. Both are wrong in opposite directions, which is why the partition is
    reported rather than applied.
    """
    out: dict[str, bool] = {}
    for qid in qids:
        gold = benchmark.gold_docs(qid)
        named_ok = all(doc in corpus_doc_ids for doc in gold.docs)
        groups_ok = all(any(doc in corpus_doc_ids for doc in g) for g in gold.any_of)
        out[str(qid)] = bool(gold.required) and named_ok and groups_ok
    return out


def evaluate(
    benchmark: Any,
    records: list[runs.RunRecord],
    k: int,
    reachable: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """Retrieval metrics at every granularity the benchmark supplies.

    The modality breakdown is nested under its taxonomy rather than flattened.
    A source can label evidence with more than one vocabulary, and one flat
    table averaging across them renders fine while comparing different things.
    """
    # One pass; the generator is re-read otherwise, which is O(n^2) on 1,658.
    questions = {q.qid: q for q in benchmark.questions()}
    by_taxonomy: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    doc_recall: list[float] = []
    page_recall: list[float] = []
    region_recall: list[float] = []
    region_graded: list[float] = []
    # Single- vs multi-evidence is where max-pooling and evidence-pooling
    # diverge. MMDocIR labels 637 multi-layout and 313 multi-page questions --
    # the same phenomenon as the iSE multi-gold set at ~20x the n.
    single_evidence: list[float] = []
    multi_evidence: list[float] = []
    # Coverage vs accuracy. Kept apart because a parser that answers 64/111 at
    # 70% beats one answering 49/111 at 78%, and one averaged number hides that.
    reach_scores: list[float] = []
    unreach_qids: list[str] = []
    # Per-question scores, so a cross-arm comparison can be recomputed on any
    # subset without re-running retrieval or re-reading gold.
    per_question: dict[str, float] = {}
    by_reach_modality: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for record in records:
        retrieved = [chunk["doc_id"] for chunk in record.chunks[:k]]
        gold = benchmark.gold_docs(record.qid)
        score = gold.recall(retrieved)
        doc_recall.append(score)
        (multi_evidence if gold.required > 1 else single_evidence).append(score)
        per_question[record.qid] = round(score, 6)

        is_reachable = True if reachable is None else reachable.get(record.qid, True)
        if is_reachable:
            reach_scores.append(score)
        else:
            unreach_qids.append(record.qid)

        question = questions.get(record.qid)
        taxonomy = getattr(question, "taxonomy", "") or "default"
        for modality in (question.modalities if question else ()) or ("unknown",):
            by_taxonomy[taxonomy][modality].append(score)
            if is_reachable:
                by_reach_modality[taxonomy][modality].append(score)

        pages = benchmark.gold_pages(record.qid)
        if pages:
            seen = {c.split("#page=")[-1].split("#")[0] for c in retrieved if "#page=" in c}
            page_recall.append(len(set(pages) & seen) / len(pages))

        regions = benchmark.gold_regions(record.qid)
        if regions:
            wanted = {r.region_id for r in regions}
            page_backed = {r.doc_id for r in regions}
            found = (wanted & set(retrieved)) | (page_backed & set(retrieved))
            region_recall.append(len(found) / len(wanted))
            # Graded box overlap, which is how the paper scores layouts. Binary
            # IoU>=0.5 is ours and is NOT comparable to their published numbers.
            graded = getattr(benchmark, "graded_region_recall", None)
            if graded is not None:
                value = graded(record.qid, retrieved)
                if value is not None:
                    region_graded.append(value)

    def mean(values: Sequence[float]) -> float | None:
        return round(sum(values) / len(values), 4) if values else None

    return {
        f"recall@{k}": mean(doc_recall),
        f"page_recall@{k}": mean(page_recall),
        # At page level this is "was the containing page retrieved", NOT the
        # paper's layout recall. The name says so, because a reader comparing it
        # to a published layout number will conflate them otherwise.
        f"{'page_backed_region' if not region_graded else 'region'}_recall@{k}": mean(region_recall),
        f"region_recall_graded@{k}": mean(region_graded),
        # Companion count: the graded mean averages best-IoU over gold regions,
        # so retrieving MORE units can add near-misses and pull it down while
        # every other metric rises. The denominator makes that legible.
        "region_recall_graded_n": len(region_graded),
        f"single_evidence_recall@{k}": mean(single_evidence),
        f"multi_evidence_recall@{k}": mean(multi_evidence),
        "single_evidence_n": len(single_evidence),
        "multi_evidence_n": len(multi_evidence),
        # Coverage is a first-class result, never a filter.
        "reachable_n": len(reach_scores),
        "unreachable_n": len(unreach_qids),
        "coverage": round(len(reach_scores) / max(len(records), 1), 4),
        f"recall@{k}|reachable": mean(reach_scores),
        # Emitted so a cross-arm comparison can intersect them. Accuracy on an
        # arm's OWN reachable set is not comparable across arms -- a parser that
        # reaches more documents has a different, not necessarily easier, subset.
        "per_question": per_question,
        "reachable_qids": sorted(
            r.qid for r in records if (reachable or {}).get(r.qid, True)
        ),
        "by_modality": {
            taxonomy: {m: mean(v) for m, v in sorted(labels.items())}
            for taxonomy, labels in sorted(by_taxonomy.items())
        },
        "by_modality_reachable": {
            taxonomy: {m: mean(v) for m, v in sorted(labels.items())}
            for taxonomy, labels in sorted(by_reach_modality.items())
        },
        "questions": len(records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="ise")
    parser.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--depth", type=int, default=100, help="Chunks written per question.")
    parser.add_argument("--embedder", default="", help="Enables the dense and fusion arms.")
    parser.add_argument("--embedder-param", action="append", default=[], metavar="K=V")
    parser.add_argument("--root", default="", help="Benchmark data root, if it needs one.")
    parser.add_argument("--level", default="page", choices=["page", "layout"])
    parser.add_argument("--text-source", default="vlm_text", choices=["vlm_text", "ocr_text"])
    parser.add_argument("--out", default=str(DATA / "runs"))
    parser.add_argument("--chunker", default="", help="Chunk document-level units, e.g. fixed_overlap.")
    parser.add_argument("--chunk-param", action="append", default=[], metavar="K=V")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    kwargs: dict[str, Any] = {}
    if args.benchmark == "mmdocir":
        kwargs = {"level": args.level, "text_source": args.text_source}
        if args.root:
            kwargs["root"] = args.root
    benchmark = load_benchmark(args.benchmark, **kwargs)

    embedder = None
    if args.embedder:
        from src.chunking_embedding.embedders import create_embedder

        params = dict(p.split("=", 1) for p in args.embedder_param if "=" in p)
        embedder = create_embedder(args.embedder, params)

    chunk_params: dict[str, Any] = {}
    for item in args.chunk_param:
        if "=" in item:
            key, value = item.split("=", 1)
            chunk_params[key] = int(value) if value.lstrip("-").isdigit() else value

    identity = ".".join(
        [args.benchmark, args.level, args.text_source,
         args.chunker or "nochunk", args.embedder or "noembed"]
    )
    index = build_index(benchmark, embedder, identity, args.chunker, chunk_params)

    questions = list(benchmark.questions())
    if args.limit:
        questions = questions[: args.limit]
    qids = [q.qid for q in questions]
    print(f"benchmark : {benchmark.name}  index_id={index.index_id}")
    print(f"corpus    : {len(index.records)} units   questions: {len(questions)}")

    corpus_docs = {r.doc_id for r in index.records}
    reachable = reachability(benchmark, corpus_docs, qids)
    covered = sum(1 for v in reachable.values() if v)
    print(f"coverage  : {covered}/{len(qids)} questions have evidence in the corpus "
          f"({covered/max(len(qids),1):.1%})")

    report: dict[str, Any] = {
        "index_id": index.index_id,
        "coverage": {
            "reachable_n": covered,
            "unreachable_n": len(qids) - covered,
            # Corpus-level coverage. Two arms are not "the same pipeline with a
            # different text source" if one indexed fewer units -- a reader will
            # assume both saw the same corpus unless this number says otherwise.
            "units_indexed": len(index.records),
        },
        "arms": {},
    }
    out_root = Path(args.out)
    for name in [a.strip() for a in args.arms.split(",") if a.strip()]:
        if name in {"dense", "rrf"} or name.startswith("alpha"):
            if embedder is None:
                print(f"  {name:9} skipped (needs --embedder)")
                continue
        arm = retrievers.build(name, index)
        path = runs.cache_path(out_root, index.index_id, arm.retriever_id, arm.params(), qids)
        if path.exists():
            records = runs.read(path)
            print(f"  {arm.retriever_id:9} cached ({len(records)} records)")
        else:
            records = []
            for question in questions:
                # MMDocIR retrieves inside one document; scope carries that so the
                # numbers stay comparable to the published leaderboard.
                scope = (
                    benchmark.scope_for(question.qid)
                    if hasattr(benchmark, "scope_for")
                    else None
                )
                started = time.perf_counter()
                hits = arm.retrieve(question.query, args.depth, scope=scope)
                records.append(
                    runs.RunRecord.build(
                        question.qid, question.query, arm.retriever_id, index.index_id,
                        runs.params_hash(arm.params()), hits,
                        scope_doc_ids=scope,
                        latency_ms=(time.perf_counter() - started) * 1000,
                    )
                )
            runs.write(path, records)
            print(f"  {arm.retriever_id:9} {len(records)} records -> {path.name}")
        report["arms"][arm.retriever_id] = evaluate(benchmark, records, args.k, reachable)

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / f"{index.index_id}.report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print()
    for name, metrics in report["arms"].items():
        # Bulk fields belong in the JSON, not the terminal.
        skip = {"by_modality", "by_modality_reachable", "per_question", "reachable_qids"}
        line = "  ".join(
            f"{k}={v}" for k, v in metrics.items() if k not in skip and v is not None
        )
        print(f"  {name:9} {line}")
        for taxonomy, labels in metrics["by_modality"].items():
            print(f"  {'':9} [{taxonomy}] {labels}")


if __name__ == "__main__":
    main()
