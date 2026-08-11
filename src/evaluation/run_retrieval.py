"""Run retrieval arms over a benchmark and emit run records.

    python -m src.evaluation.run_retrieval --benchmark mmdocir --arms bm25,dense,rrf
    python -m src.evaluation.run_retrieval --benchmark ise --arms bm25 --embedder openrouter_te3s

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

from src.utils.paths import repo_root
from dataclasses import replace
from typing import Any, Sequence
import argparse
import hashlib
import json
import sys
import time

PROJECT_ROOT = repo_root(__file__)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.env import load_dotenv_file  # noqa: E402

# Every entry point that can reach a keyed API must load .env the same way.
# Loading it in some and not others makes a run succeed or fail on which shell
# it was launched from, and the failure surfaces deep in the embedder as
# "OPENROUTER_API_KEY is not set" rather than as a missing config file.
load_dotenv_file(PROJECT_ROOT)

from src.retrieval import runs  # noqa: E402
from src.retrieval.index import LocalIndex  # noqa: E402
from src.retrieval.protocol import ChunkRecord, ScoredChunk  # noqa: E402
from src.retrieval.sparse import BM25Index  # noqa: E402
from src.retrieval import retrievers  # noqa: E402

from .benchmarks import load as load_benchmark  # noqa: E402
from .corpus_source import BenchmarkCorpus, PipelineRunCorpus, content_identity  # noqa: E402

DATA = PROJECT_ROOT / "data" / "benchmark"
DEFAULT_ARMS = ("bm25", "dense", "rrf")


def index_identity(args: Any, benchmark: Any) -> str:
    """Everything that changes the index; asserted distinct in test_identity_keys."""
    parts = [
        args.benchmark,
        args.level,
        args.text_source,
        args.chunker or "nochunk",
        args.embedder or "noembed",
        corpus_token(benchmark),
    ]
    chunk_params = _kv(getattr(args, "chunk_param", []))
    if chunk_params:
        parts.append(f"cp-{runs.stable_hash(chunk_params, 8)}")
    embed_params = _kv(getattr(args, "embedder_param", []))
    if embed_params:
        parts.append(f"ep-{runs.stable_hash(embed_params, 8)}")
    if args.prefix:
        parts.append("prefix")
    if args.rerank:
        parts.append(f"rr-{args.rerank}-{args.rerank_model}-d{args.rerank_depth}")
    return ".".join(parts)


def _kv(items: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items or []:
        if "=" in item:
            key, value = item.split("=", 1)
            out[key] = int(value) if value.lstrip("-").isdigit() else value
    return out


def corpus_token(benchmark: Any) -> str:
    return content_identity(benchmark)


def build_index(
    benchmark: Any, embedder: Any, index_id: str, chunker: str = "",
    params: dict | None = None, prefix: bool = False, analyzer: str = "auto",
) -> LocalIndex:
    """Index a corpus source's units.

    MMDocIR ships pages and layouts -- already the retrieval granularity, so no
    chunking. The iSE lake ships whole documents, where the chunker is an
    experimental variable rather than a property of the source.
    """
    source = benchmark if hasattr(benchmark, "units") \
        else BenchmarkCorpus(benchmark, chunker, params, prefix)
    records = list(source.units())
    if not records:
        raise SystemExit(
            f"{index_id}: no indexable text from {source.corpus_identity()}"
            + (f" via chunker {chunker!r}" if chunker else "")
            + ". An empty index scores 0.0 on every question without erroring, so "
            "it is refused here rather than reported as a result."
        )
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

    bm25 = BM25Index(analyzer_name=analyzer).build(payload)
    return LocalIndex(
        index_id=f"{index_id}.{bm25.analyzer_name}",
        records=records,
        bm25=bm25,
        vectors=vectors,
        embedder=embedder,
    )


def oracle_records(benchmark: Any, questions: Sequence[Any], index: Any, depth: int) -> list:
    """Gold units as a perfect ranking: the ceiling generation is read against.

    Emitted as an ordinary run so the same ``run_answer`` scores it. Graded qrels
    rank by descending relevance, and a gold unit absent from the index is
    dropped rather than faked -- an empty passage scores as a generator failure
    when it is really a corpus gap.
    """
    by_id = {record.chunk_id: record for record in index.records}
    qrels = benchmark.qrels()
    out = []
    for question in questions:
        gold = sorted(qrels.get(question.qid, {}).items(), key=lambda kv: -kv[1])
        hits = [
            ScoredChunk(chunk_id=unit, doc_id=by_id[unit].doc_id, score=float(grade),
                        rank=rank, text=by_id[unit].text)
            for rank, (unit, grade) in enumerate(gold[:depth], start=1)
            if unit in by_id
        ]
        out.append(runs.RunRecord.build(
            question.qid, question.query, "oracle", index.index_id,
            runs.params_hash({"depth": depth}), hits))
    return out


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


def graded_ndcg(benchmark: Any, records: list[runs.RunRecord], k: int) -> dict[str, Any]:
    """NDCG@k over graded relevance, for benchmarks that publish it.

    Computed here rather than in a side script: a metric needing a manual second
    step is one that eventually gets reported from a stale file.
    """
    qrels = getattr(benchmark, "qrels", None)
    if not callable(qrels):
        return {}
    try:
        import pytrec_eval
    except ImportError:
        return {f"ndcg@{k}": None, "ndcg_note": "pip install pytrec_eval-terrier"}

    gold = {q: dict(v) for q, v in qrels().items() if v}
    run = {
        r.qid: {c["doc_id"]: float(c["score"]) for c in r.chunks}
        for r in records if r.qid in gold
    }
    if not run:
        return {}
    scores = pytrec_eval.RelevanceEvaluator(gold, {f"ndcg_cut_{k}"}).evaluate(run)
    values = [v[f"ndcg_cut_{k}"] for v in scores.values()]
    return {
        f"ndcg@{k}": round(100 * sum(values) / len(values), 2),
        f"ndcg@{k}_n": len(values),
    }


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
        **graded_ndcg(benchmark, records, k),
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
    parser.add_argument(
        "--corpus", default="", metavar="PATH",
        help="Corpus file for adapters that read one (ise). Names the arm and "
        "enters the index_id, so two parsers never share a run cache.",
    )
    parser.add_argument(
        "--analyzer", default="auto", choices=["auto", "plain", "cjk_bigram"],
        help="'auto' picks per corpus by CJK presence, so two parsers can resolve "
        "differently and a comparison varies tokenizer as well as parser. Pin it "
        "when comparing corpora.",
    )
    parser.add_argument(
        "--parsed-run", default="", metavar="DIR",
        help="Index a pipeline run directory instead of the benchmark's own text. "
        "The benchmark still supplies questions and gold.",
    )
    parser.add_argument(
        "--granularity", default="page", choices=["page", "content"],
        help="Unit for --parsed-run: page text, or page text keeping table/list HTML.",
    )
    parser.add_argument("--subset", default="", help="ViDoRe V3 subset, e.g. physics.")
    parser.add_argument(
        "--language", default="", help="ViDoRe V3 query language. Required for that benchmark."
    )
    parser.add_argument("--level", default="page", choices=["page", "layout"])
    parser.add_argument("--text-source", default="vlm_text", choices=["vlm_text", "ocr_text"])
    parser.add_argument("--out", default=str(DATA / "runs"))
    parser.add_argument("--chunker", default="", help="Chunk document-level units, e.g. fixed_overlap.")
    parser.add_argument(
        "--prefix", action="store_true",
        help="Prepend the document title/filename to each chunk before indexing. "
        "Gives every chunk its provenance, which lexical matching otherwise loses "
        "when a chunk from the middle of a document carries no identifying text.",
    )
    parser.add_argument(
        "--rerank", default="", choices=["", "llm"],
        help="Rerank retrieved hits. 'llm' scores the best chunk of each distinct "
        "document via model-service.",
    )
    parser.add_argument("--rerank-model", default="llm-rerank")
    parser.add_argument("--rerank-depth", type=int, default=20)
    parser.add_argument("--chunk-param", action="append", default=[], metavar="K=V")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    kwargs: dict[str, Any] = {}
    if args.benchmark == "mmdocir":
        kwargs = {"level": args.level, "text_source": args.text_source}
        if args.root:
            kwargs["root"] = args.root
    elif args.benchmark == "vidore_v3":
        kwargs = {"subset": args.subset, "language": args.language}
        if args.root:
            kwargs["root"] = args.root
    elif args.benchmark == "ise" and args.corpus:
        kwargs = {"corpus_path": args.corpus}
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

    if args.parsed_run:
        source: Any = PipelineRunCorpus(
            args.parsed_run, subset=args.subset, granularity=args.granularity,
            chunker=args.chunker, params=chunk_params, prefix=args.prefix,
        )
    else:
        source = BenchmarkCorpus(benchmark, args.chunker, chunk_params, args.prefix)

    # Any two arms we compare must differ here, or the second reports the first's cache.
    identity = index_identity(args, source)
    index = build_index(source, embedder, identity, analyzer=args.analyzer)

    reranker = None
    if args.rerank == "llm":
        from .llm_rerank import LLMReranker

        reranker = LLMReranker(model=args.rerank_model, depth=args.rerank_depth)

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
        if name == "oracle":
            path = runs.cache_path(out_root, index.index_id, "oracle",
                                   {"depth": args.depth}, qids)
            if path.exists():
                records = runs.read(path)
                print(f"  {'oracle':9} cached ({len(records)} records)")
            else:
                records = oracle_records(benchmark, questions, index, args.depth)
                runs.write(path, records)
                print(f"  {'oracle':9} {len(records)} records -> {path.name}")
            report["arms"]["oracle"] = evaluate(benchmark, records, args.k, reachable)
            continue
        if name in {"dense", "rrf"} or name.startswith("alpha"):
            if embedder is None:
                print(f"  {name:9} skipped (needs --embedder)")
                continue
        arm = retrievers.build(name, index)
        # ``depth`` changes what each record carries, so it belongs in the run key.
        arm_params = {**dict(arm.params()), "depth": args.depth}
        path = runs.cache_path(out_root, index.index_id, arm.retriever_id, arm_params, qids)
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
                if reranker is not None and hits:
                    # Document-level budget: rerank the best chunk of each of
                    # `depth` distinct documents rather than `depth` chunks that
                    # may all come from two. Measured +5 R@1 at identical cost.
                    order = reranker.rerank(
                        question.query,
                        [(i, h.score) for i, h in enumerate(hits)],
                        [h.text for h in hits],
                        [h.doc_id for h in hits],
                    )
                    if order:
                        ranked = [hits[i] for i, _ in order if 0 <= i < len(hits)]
                        seen = {id(h) for h in ranked}
                        # A reranker may only reorder: anything it dropped keeps
                        # its retrieved order behind what it ranked, so recall
                        # can never fall below the first stage.
                        ranked.extend(h for h in hits if id(h) not in seen)
                        hits = [
                            replace(h, rank=position, score=float(len(ranked) - position))
                            for position, h in enumerate(ranked, start=1)
                        ]
                records.append(
                    runs.RunRecord.build(
                        question.qid, question.query, arm.retriever_id, index.index_id,
                        runs.params_hash(arm_params), hits,
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
