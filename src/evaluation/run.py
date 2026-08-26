"""Run the in-corpus retrieval benchmark.

    python -m src.evaluation.run                      # lexical arms only, no API spend
    python -m src.evaluation.run --dense              # adds dense + hybrid (paid embedder)
    python -m src.evaluation.run --arms fixed_512_ol,blocks

Each arm reports quality and cost together, because the hypothesis under test is
whether cheaper chunking is competitive once cost is counted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.utils.paths import repo_root
from typing import Any, Sequence
import argparse
import json
import time

import numpy as np

from .chunking import chunk_corpus
from .metrics import QueryResult, bootstrap_ci, evaluate, paired_bootstrap
from .retrieval import ANALYZERS, BM25Index, DenseIndex, aggregate, alpha_fuse, rrf

PROJECT_ROOT = repo_root(__file__)
DATA = PROJECT_ROOT / "data" / "benchmark"

ARMS: dict[str, dict[str, Any]] = {
    "fixed_512_ol": {"strategy": "fixed_overlap", "params": {"n_words": 512, "overlap": 128}},
    "fixed_256_ol": {"strategy": "fixed_overlap", "params": {"n_words": 256, "overlap": 64}},
    "fixed_1024_ol": {"strategy": "fixed_overlap", "params": {"n_words": 1024, "overlap": 128}},
    "recursive_400": {"strategy": "recursive", "params": {"target": 400}},
    "paragraph": {"strategy": "paragraph", "params": {}},
    "sentence_5": {"strategy": "sentence_group", "params": {"n": 5}},
    "blocks": {"strategy": "blocks", "params": {"target": 1200}},
    "blocks_hdr": {"strategy": "blocks", "params": {"target": 1200, "carry_header": True}},
    # Size-matched to fixed_512_ol (~3.4k chars/chunk) so the structure-aware
    # packing is compared against fixed windows at equal granularity, not at a
    # 3x smaller chunk size that confounds algorithm with size.
    "blocks_3400": {"strategy": "blocks", "params": {"target": 3400}},
    # Best of both: fixed-size windows WITH overlap, but boundaries land on
    # whole blocks, so a table row is never cut in half.
    "blocks_3400_ol": {"strategy": "blocks", "params": {"target": 3400, "overlap": 850}},
    "blocks_3400_ol_hdr": {"strategy": "blocks", "params": {"target": 3400, "overlap": 850, "carry_header": True}},
    "blocks_3400_hdr": {"strategy": "blocks", "params": {"target": 3400, "carry_header": True}},
}


@dataclass
class ArmResult:
    arm: str
    retriever: str
    chunks: int
    build_seconds: float
    query_seconds: float
    metrics: dict[str, float] = field(default_factory=dict)
    ci_recall10: tuple[float, float] = (0.0, 0.0)
    extra: dict[str, Any] = field(default_factory=dict)
    # Per-query results, kept for paired significance testing; not serialized.
    entries: list[QueryResult] = field(default_factory=list)


def load_corpus(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def load_eval(path: Path, modalities: set[str]) -> list[dict[str, Any]]:
    questions = [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]
    return [q for q in questions if q["resolvable"] and set(q["modalities"]) <= modalities]


def run_arm(
    name: str,
    corpus: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    analyzer: str,
    prefix: bool,
    aggregation: str,
    embedder: Any | None,
    top_k: int,
    reranker: Any | None = None,
    alphas: Sequence[float] = (0.5,),
) -> list[ArmResult]:
    spec = ARMS[name]
    started = time.perf_counter()
    chunks = chunk_corpus(corpus, spec["strategy"], spec["params"], prefix=prefix)
    chunk_seconds = time.perf_counter() - started

    started = time.perf_counter()
    bm25 = BM25Index(analyzer=ANALYZERS[analyzer]).build(chunks)
    lexical_build = time.perf_counter() - started

    dense: DenseIndex | None = None
    dense_build = 0.0
    if embedder is not None:
        started = time.perf_counter()
        vectors = np.asarray(embedder.embed([chunk.index_text for chunk in chunks]), dtype=np.float32)
        dense = DenseIndex().build(chunks, vectors)
        dense_build = time.perf_counter() - started

    query_vectors: dict[str, np.ndarray] = {}
    if embedder is not None:
        texts = [q["question"] for q in questions]
        # Asymmetric models want queries in a different space than passages.
        encode = getattr(embedder, "embed_query", embedder.embed)
        for question, vector in zip(questions, encode(texts)):
            query_vectors[question["qid"]] = np.asarray(vector, dtype=np.float32)

    chunk_texts = [chunk.index_text for chunk in chunks]
    runs: dict[str, list[QueryResult]] = {"bm25": [], "dense": [], "rrf": [], "rerank": []}
    for weight in alphas:
        runs[f"alpha{weight:g}"] = []
    started = time.perf_counter()
    for question in questions:
        gold = question["gold_doc_ids"]
        lexical_hits = bm25.search(question["question"], top_k=top_k)
        runs["bm25"].append(_result(question, lexical_hits, bm25.doc_ids, aggregation, gold))
        if dense is not None:
            dense_hits = dense.search(query_vectors[question["qid"]], top_k=top_k)
            runs["dense"].append(_result(question, dense_hits, dense.doc_ids, aggregation, gold))
            runs["rrf"].append(
                _result(question, rrf([lexical_hits, dense_hits], top_k=top_k), bm25.doc_ids, aggregation, gold)
            )
            for weight in alphas:
                runs[f"alpha{weight:g}"].append(
                    _result(question, alpha_fuse(lexical_hits, dense_hits, weight, top_k),
                            bm25.doc_ids, aggregation, gold)
                )
        if reranker is not None:
            # Rerank the strongest first stage available; it can only reorder what that
            # stage found. Fusion at alphas[0] beats dense alone, so prefer it.
            base = (
                alpha_fuse(lexical_hits, dense_hits, alphas[0], top_k)
                if dense is not None
                else lexical_hits
            )
            reordered = reranker.rerank(question["question"], base, chunk_texts, bm25.doc_ids)
            runs["rerank"].append(_result(question, reordered, bm25.doc_ids, aggregation, gold))
    query_seconds = time.perf_counter() - started

    results: list[ArmResult] = []
    for retriever, entries in runs.items():
        if not entries:
            continue
        results.append(
            ArmResult(
                arm=name,
                retriever=retriever,
                chunks=len(chunks),
                build_seconds=round(chunk_seconds + lexical_build + dense_build, 2),
                query_seconds=round(query_seconds, 2),
                metrics=evaluate(entries),
                ci_recall10=bootstrap_ci(entries, "recall@10"),
                extra={
                    "chunk_seconds": round(chunk_seconds, 2),
                    "analyzer": analyzer,
                    "prefix": prefix,
                    "aggregation": aggregation,
                },
                entries=entries,
            )
        )
    return results


def _result(
    question: dict[str, Any],
    hits: list[tuple[int, float]],
    doc_ids: list[str],
    aggregation: str,
    gold: list[str],
) -> QueryResult:
    ranked = [doc for doc, _ in aggregate(hits, doc_ids, aggregation)]
    return QueryResult(qid=question["qid"], ranked_docs=ranked, gold_docs=gold)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=str(DATA / "corpus.jsonl"))
    parser.add_argument("--questions", default=str(DATA / "questions.jsonl"))
    parser.add_argument("--out", default=str(DATA / "results.json"))
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--modalities", default="text,table")
    parser.add_argument("--analyzer", default="plain", choices=sorted(ANALYZERS))
    parser.add_argument("--aggregation", default="maxp", choices=["maxp", "sum_topk", "sum"])
    parser.add_argument("--prefix", action="store_true", help="Prepend title/section to each chunk.")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--keep-unreachable", action="store_true",
                        help="Score questions whose gold evidence is missing from the corpus.")
    parser.add_argument("--dense", action="store_true", help="Enable the paid embedder arms.")
    parser.add_argument("--embedder", default="openrouter_te3s")
    parser.add_argument("--embedder-param", action="append", default=[], metavar="KEY=VALUE",
                        help="Embedder parameter, e.g. model=embedding-default. Repeat for more.")
    parser.add_argument("--rerank", default=None, metavar="ALIAS",
                        help="Rerank the top --rerank-depth hits via the Model Service alias.")
    parser.add_argument("--llm-rerank", default=None, metavar="ALIAS",
                        help="Listwise LLM reranking via the Model Service LLM alias.")
    parser.add_argument("--voyage-rerank", default=None, metavar="MODEL",
                        help="Voyage cross-encoder reranking, e.g. rerank-2.5.")
    parser.add_argument("--rerank-depth", type=int, default=50)
    parser.add_argument("--dump-runs", default=None, metavar="PATH",
                        help="Write per-query rankings for failure analysis.")
    parser.add_argument("--baseline", default="alpha0.7",
                        help="Retriever to run the paired significance test against.")
    parser.add_argument("--alpha", default="0.5",
                        help="Comma-separated dense weights for alpha fusion, e.g. 0,0.25,0.5,0.75,1.")
    args = parser.parse_args()

    corpus = load_corpus(Path(args.corpus))
    questions = load_eval(Path(args.questions), {m.strip() for m in args.modalities.split(",")})
    corpus_ids = {record["doc_id"] for record in corpus}
    total = len(questions)
    if not args.keep_unreachable:
        questions = [q for q in questions if set(q["gold_doc_ids"]) <= corpus_ids]
    reachable = len(questions)
    print(f"corpus={len(corpus)} docs   questions={reachable}/{total} scored "
          f"({'all' if args.keep_unreachable else 'evidence fully in corpus'})")

    embedder = None
    if args.dense:
        from src.chunking_embedding.embedders import create_embedder
        from src.utils.env import load_dotenv_file

        load_dotenv_file(PROJECT_ROOT)
        embedder = create_embedder(args.embedder, _parse_params(args.embedder_param))

    alphas = tuple(float(part) for part in args.alpha.split(",") if part.strip())

    reranker = None
    if args.rerank:
        from .rerank import GatewayReranker

        reranker = GatewayReranker(model=args.rerank, depth=args.rerank_depth)
    elif args.llm_rerank:
        from .llm_rerank import LLMReranker

        reranker = LLMReranker(model=args.llm_rerank, depth=args.rerank_depth)
    elif args.voyage_rerank:
        from .voyage_rerank import VoyageReranker
        from src.utils.env import load_dotenv_file

        load_dotenv_file(PROJECT_ROOT)
        reranker = VoyageReranker(model=args.voyage_rerank, depth=args.rerank_depth)

    results: list[ArmResult] = []
    for name in [part.strip() for part in args.arms.split(",") if part.strip()]:
        if name not in ARMS:
            raise SystemExit(f"Unknown arm {name!r}; known: {sorted(ARMS)}")
        print(f"-- running {name} ...", flush=True)
        results.extend(
            run_arm(name, corpus, questions, args.analyzer, args.prefix, args.aggregation,
                    embedder, args.top_k, reranker, alphas)
        )

    payload = {
        "corpus_docs": len(corpus),
        "questions": len(questions),
        "gold_reachable": reachable,
        "settings": {
            "analyzer": args.analyzer,
            "aggregation": args.aggregation,
            "prefix": args.prefix,
            "top_k": args.top_k,
            "dense": args.dense,
        },
        "results": [
            {
                "arm": r.arm,
                "retriever": r.retriever,
                "chunks": r.chunks,
                "build_seconds": r.build_seconds,
                "query_seconds": r.query_seconds,
                "ci_recall@10": list(r.ci_recall10),
                **r.metrics,
                **r.extra,
            }
            for r in results
        ],
    }
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _print_table(results)
    if reranker is not None:
        stats = getattr(reranker, "stats", {})
        print(f"\nreranker {type(reranker).__name__}: {stats}")
        failed = stats.get("failed", 0) + stats.get("unparsed", 0)
        if failed:
            print(f"  WARNING: {failed} queries fell back to first-stage order. "
                  f"The 'rerank' row is NOT a clean measurement of this reranker.")
    _print_significance(results, args.baseline)
    print(f"\nwrote {args.out}")
    if args.dump_runs:
        dump = [
            {
                "arm": r.arm,
                "retriever": r.retriever,
                "qid": e.qid,
                "gold": e.gold_docs,
                "ranked": e.ranked_docs[:10],
                "hit@1": int(bool(set(e.gold_docs) & set(e.ranked_docs[:1]))),
                "gold_rank": next((i + 1 for i, d in enumerate(e.ranked_docs) if d in set(e.gold_docs)), None),
            }
            for r in results
            for e in r.entries
        ]
        Path(args.dump_runs).write_text(json.dumps(dump, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.dump_runs} ({len(dump)} per-query rows)")


def _print_significance(results: list[ArmResult], baseline: str) -> None:
    """Paired bootstrap against a baseline retriever. Overlapping CIs cannot
    separate two systems scored on the same questions; a paired test can."""
    base = next((r for r in results if r.retriever == baseline and r.entries), None)
    if base is None:
        return
    print(f"\npaired bootstrap vs {baseline} (2000 samples, same {len(base.entries)} questions)")
    print(f"{'retriever':12s} {'dR@1':>8s} {'p(R@1)':>8s} {'dR@10':>8s} {'p(R@10)':>8s}")
    for r in results:
        if r.retriever == baseline or not r.entries:
            continue
        d1 = r.metrics.get("recall@1", 0.0) - base.metrics.get("recall@1", 0.0)
        d10 = r.metrics.get("recall@10", 0.0) - base.metrics.get("recall@10", 0.0)
        p1 = paired_bootstrap(r.entries, base.entries, "recall@1")
        p10 = paired_bootstrap(r.entries, base.entries, "recall@10")
        star = "  *" if p1 < 0.05 else ""
        print(f"{r.retriever:12s} {d1:+8.3f} {p1:8.3f} {d10:+8.3f} {p10:8.3f}{star}")


def _parse_params(values: list[str]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for raw in values:
        if "=" not in raw:
            raise SystemExit(f"Expected KEY=VALUE, got {raw!r}")
        key, value = raw.split("=", 1)
        try:
            params[key.strip()] = json.loads(value)
        except json.JSONDecodeError:
            params[key.strip()] = value
    return params


def _print_table(results: list[ArmResult]) -> None:
    header = f"{'arm':16s} {'retr':9s} {'chunks':>7s} {'R@1':>6s} {'R@5':>6s} {'R@10':>6s} {'MRR':>6s} {'nDCG':>6s} {'CI(R@10)':>16s} {'build_s':>8s}"
    print("\n" + header)
    print("-" * len(header))
    for r in sorted(results, key=lambda x: -x.metrics.get("recall@10", 0)):
        ci = f"[{r.ci_recall10[0]:.2f},{r.ci_recall10[1]:.2f}]"
        print(
            f"{r.arm:16s} {r.retriever:9s} {r.chunks:7d} "
            f"{r.metrics['recall@1']:6.3f} {r.metrics['recall@5']:6.3f} {r.metrics['recall@10']:6.3f} "
            f"{r.metrics['mrr@10']:6.3f} {r.metrics['ndcg@10']:6.3f} {ci:>16s} {r.build_seconds:8.1f}"
        )


if __name__ == "__main__":
    main()
