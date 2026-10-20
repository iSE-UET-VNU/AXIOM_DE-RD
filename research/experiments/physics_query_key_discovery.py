"""Probe query decomposition as a discovery signal for Physics retrieval.

The experiment keeps the page index fixed and retrieves independently with:

* the full query;
* content-only tokens;
* individual content tokens;
* content-token bigrams/trigrams;
* punctuation-delimited clauses.

It reports per-key-type retrieval, an oracle best-key ceiling, key-list union
coverage, and simple qrel-free primary-key heuristics.  The oracle rows are
diagnostic only: qrels never choose a key in the heuristic or fusion rows.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import json
import math
import re
import sys
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.data_discovery.hierarchical import HierarchyCorpus, build_bm25  # noqa: E402
from src.chunking_embedding.lexical import analyze  # noqa: E402
from src.evaluation.benchmarks.vidore_v3 import ViDoreV3  # noqa: E402


PARSED_RUN = ROOT / "data" / "output" / "vidore-v3-physics-kdl-pdf-inspector" / "32c32a45a92c45bb"
BENCHMARK_ROOT = ROOT / "data" / "benchmark" / "vidore_v3"
OUTPUT_DIR = ROOT / "data" / "benchmark" / "vidore_v3" / "results" / "physics_query_key_discovery"

DEPTH = 100
EVAL_K = (10, 20, 50, 100)
RRF_K = 60

# Function words are removed only for the content-only/key candidates.  This
# is intentionally a small fixed list, not a benchmark-fitted stopword list.
FRENCH_FUNCTION_WORDS = frozenset(
    "a au aux avec ce ceci cela ces cette d de des du elle elles en es est et eux "
    "il ils j je l la le les leur leurs lui ma mais me mes moi mon même ne nos notre "
    "nous on ou par pas pour que quel quelle quelles quels qui quoi se seront son "
    "sont sous sur ta te tes toi ton tu un une vos votre vous y à après avant dans "
    "entre dont où peut peuvent comment pourquoi plusieurs plus moins comme cette "
    "cet ces est-il est-elle".split()
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _idf(index: Any) -> dict[str, float]:
    total = len(index.chunk_ids)
    return {
        term: math.log(1.0 + (total - len(posting) + 0.5) / (len(posting) + 0.5))
        for term, posting in index.postings.items()
    }


def _content_tokens(query: str) -> list[str]:
    return [
        token
        for token in analyze(query)
        if token not in FRENCH_FUNCTION_WORDS
        and (len(token) >= 3 or any(char.isdigit() for char in token))
    ]


def _query_keys(query: str, term_idf: Mapping[str, float]) -> list[dict[str, str]]:
    content = _content_tokens(query)
    output: list[dict[str, str]] = [{"type": "full", "key": query}]
    if content:
        output.append({"type": "content_only", "key": " ".join(content)})
    output.extend({"type": "token", "key": token} for token in content)
    for size in (2, 3):
        output.extend(
            {
                "type": f"ngram{size}",
                "key": " ".join(content[start : start + size]),
            }
            for start in range(len(content) - size + 1)
        )
    clauses = [clause.strip() for clause in re.split(r"[?!.;:]+", query) if clause.strip()]
    if len(clauses) > 1:
        output.extend(
            {"type": "clause", "key": clause}
            for clause in clauses
            if len(analyze(clause)) >= 2
        )

    # Stable de-duplication avoids counting a repeated token/phrase twice.
    seen: set[str] = set()
    unique: list[dict[str, str]] = []
    for item in output:
        key = item["key"]
        if key and key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def _retrieve(index: Any, key: str) -> dict[str, Any]:
    hits = index.search(key, DEPTH)
    return {
        "ids": [index.chunk_ids[position] for position, _score in hits],
        "scores": [float(score) for _position, score in hits],
    }


def _page_metrics(ids: Sequence[str], gold: set[str]) -> dict[str, float | bool]:
    return {
        f"recall_{k}": len(set(ids[:k]) & gold) / len(gold)
        for k in EVAL_K
    } | {
        "hit_10": bool(set(ids[:10]) & gold),
    }


def _file_recall_at3(ids: Sequence[str], gold_pages: set[str]) -> float:
    gold_files = {page.split("#page=", 1)[0] for page in gold_pages}
    candidate_files: list[str] = []
    for page in ids[:DEPTH]:
        file_id = page.split("#page=", 1)[0]
        if file_id not in candidate_files:
            candidate_files.append(file_id)
        if len(candidate_files) >= 3:
            break
    return len(set(candidate_files) & gold_files) / len(gold_files)


def _rrf_rank(results: Sequence[Mapping[str, Any]]) -> list[str]:
    fused: dict[str, float] = defaultdict(float)
    best_rank: dict[str, int] = {}
    for result in results:
        for rank, page_id in enumerate(result["ids"], start=1):
            fused[str(page_id)] += 1.0 / (RRF_K + rank)
            best_rank[str(page_id)] = min(best_rank.get(str(page_id), 10**9), rank)
    return [
        page_id
        for page_id, _score in sorted(
            fused.items(), key=lambda item: (-item[1], best_rank[item[0]], item[0])
        )
    ]


def _metrics_for_rank(
    rank: Sequence[str], qrels: Mapping[str, int]
) -> dict[str, float]:
    gold = set(qrels)
    values = _page_metrics(rank, gold)
    return {
        **{key: float(value) for key, value in values.items() if key != "hit_10"},
        "hit_10": float(values["hit_10"]),
        "file_recall_at3": _file_recall_at3(rank, gold),
    }


def _aggregate(metrics: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not metrics:
        return {}
    keys = sorted(
        key
        for key, value in metrics[0].items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    )
    return {key: float(np.mean([float(row[key]) for row in metrics])) for key in keys}


def _query_rows(
    qids: Sequence[str],
    questions: Mapping[str, Any],
    qrels: Mapping[str, Mapping[str, int]],
    index: Any,
    term_idf: Mapping[str, float],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    per_query: list[dict[str, Any]] = []
    type_metrics: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for qid in qids:
        keys = _query_keys(questions[qid].query, term_idf)
        gold = set(qrels[qid])
        results: list[dict[str, Any]] = []
        for item in keys:
            retrieval = _retrieve(index, item["key"])
            metrics = _metrics_for_rank(retrieval["ids"], qrels[qid])
            row = {**item, **retrieval, **metrics}
            results.append(row)
            type_metrics[item["type"]].append(
                {"qid": qid, "key": item["key"], **metrics}
            )

        full = next(row for row in results if row["type"] == "full")
        content = next(
            (row for row in results if row["type"] == "content_only"),
            full,
        )
        oracle = max(results, key=lambda row: (row["recall_10"], row["recall_100"]))
        token_rows = [row for row in results if row["type"] == "token"]
        phrase_rows = [
            row for row in results if row["type"] in {"ngram2", "ngram3"} and row["ids"]
        ]
        rare_token = max(token_rows, key=lambda row: term_idf.get(row["key"], 0.0), default=full)
        rare_phrase = max(
            phrase_rows,
            key=lambda row: sum(term_idf.get(token, 0.0) for token in analyze(row["key"])),
            default=full,
        )
        gap_primary = max(results, key=lambda row: row["scores"][0] - row["scores"][min(9, len(row["scores"]) - 1)] if row["scores"] else 0.0)
        rrf_rank = _rrf_rank(results)
        all_key_ids = list(dict.fromkeys(page_id for row in results for page_id in row["ids"]))
        all_key_top10_hit = any(row["hit_10"] for row in results)
        per_query.append(
            {
                "qid": qid,
                "query": questions[qid].query,
                "gold_page_count": len(gold),
                "key_count": len(results),
                "full": {**full, "ids": full["ids"][:10]},
                "content_only": {**content, "ids": content["ids"][:10]},
                "oracle_best_key": {
                    "type": oracle["type"],
                    "key": oracle["key"],
                    "recall_10": oracle["recall_10"],
                    "recall_100": oracle["recall_100"],
                    "hit_10": oracle["hit_10"],
                    "file_recall_at3": oracle["file_recall_at3"],
                },
                "heuristics": {
                    "rarest_token": {
                        "key": rare_token["key"],
                        "recall_10": rare_token["recall_10"],
                        "recall_100": rare_token["recall_100"],
                        "hit_10": rare_token["hit_10"],
                        "file_recall_at3": rare_token["file_recall_at3"],
                    },
                    "rarest_phrase": {
                        "key": rare_phrase["key"],
                        "recall_10": rare_phrase["recall_10"],
                        "recall_100": rare_phrase["recall_100"],
                        "hit_10": rare_phrase["hit_10"],
                        "file_recall_at3": rare_phrase["file_recall_at3"],
                    },
                    "largest_top1_top10_gap": {
                        "type": gap_primary["type"],
                        "key": gap_primary["key"],
                        "recall_10": gap_primary["recall_10"],
                        "recall_100": gap_primary["recall_100"],
                        "hit_10": gap_primary["hit_10"],
                        "file_recall_at3": gap_primary["file_recall_at3"],
                    },
                },
                "all_key_union": {
                    "candidate_recall": len(set(all_key_ids) & gold) / len(gold),
                    "candidate_hit": bool(set(all_key_ids) & gold),
                    "hit_10_any_key": all_key_top10_hit,
                    "candidate_count": len(all_key_ids),
                },
                "rrf_all_keys": _metrics_for_rank(rrf_rank, qrels[qid]),
            }
        )
    return per_query, type_metrics


def _summary(
    per_query: Sequence[Mapping[str, Any]],
    type_metrics: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    by_type: dict[str, Any] = {}
    for key_type, rows in sorted(type_metrics.items()):
        by_type[key_type] = {
            "key_instances": len(rows),
            "query_instances": len({row["qid"] for row in rows}),
            "metrics": _aggregate(rows),
        }

    def rows_for(path: str) -> list[dict[str, Any]]:
        output = []
        for row in per_query:
            value: Any = row
            for part in path.split("."):
                value = value[part]
            output.append(value)
        return output

    oracle = rows_for("oracle_best_key")
    full = rows_for("full")
    content = rows_for("content_only")
    union = rows_for("all_key_union")
    rrf = rows_for("rrf_all_keys")
    heuristic_rows = {
        name: rows_for(f"heuristics.{name}")
        for name in ("rarest_token", "rarest_phrase", "largest_top1_top10_gap")
    }
    return {
        "by_key_type": by_type,
        "full_query": _aggregate(full),
        "content_only": _aggregate(content),
        "oracle_best_key": {
            **_aggregate(oracle),
            "hit_10_queries": sum(bool(row["hit_10"]) for row in oracle) / len(oracle),
            "best_key_type_counts": dict(Counter(row["type"] for row in oracle)),
        },
        "heuristics": {
            name: {
                **_aggregate(rows),
                "hit_10_queries": sum(bool(row["hit_10"]) for row in rows) / len(rows),
            }
            for name, rows in heuristic_rows.items()
        },
        "all_key_union": {
            "candidate_recall": float(np.mean([row["candidate_recall"] for row in union])),
            "candidate_hit": float(np.mean([row["candidate_hit"] for row in union])),
            "hit_10_any_key_queries": sum(bool(row["hit_10_any_key"]) for row in union) / len(union),
            "mean_candidate_count": float(np.mean([row["candidate_count"] for row in union])),
        },
        "rrf_all_keys": _aggregate(rrf),
        "rescue": {
            "full_hit_10_queries": sum(bool(row["hit_10"]) for row in full),
            "any_key_hit_10_queries": sum(bool(row["hit_10_any_key"]) for row in union),
            "queries_rescued_at_hit_10": sum(
                not bool(full_row["hit_10"]) and bool(union_row["hit_10_any_key"])
                for full_row, union_row in zip(full, union)
            ),
            "full_candidate_recall_100": float(np.mean([row["recall_100"] for row in full])),
            "union_candidate_recall": float(np.mean([row["candidate_recall"] for row in union])),
        },
    }


def _distribution(benchmark: ViDoreV3, qrels: Mapping[str, Mapping[str, int]]) -> dict[str, Any]:
    queries = list(benchmark._queries)
    gold_pages = [len(qrels[query.qid]) for query in queries]
    query_keys = [_query_keys(query.query, {}) for query in queries]
    key_counts = [len(keys) for keys in query_keys]
    return {
        "queries": len(queries),
        "gold_pages": {
            "min": int(min(gold_pages)),
            "median": float(np.median(gold_pages)),
            "p90": float(np.percentile(gold_pages, 90)),
            "max": int(max(gold_pages)),
            "mean": float(np.mean(gold_pages)),
        },
        "key_candidates_per_query": {
            "min": int(min(key_counts)),
            "median": float(np.median(key_counts)),
            "p90": float(np.percentile(key_counts, 90)),
            "max": int(max(key_counts)),
            "mean": float(np.mean(key_counts)),
        },
    }


def _write_markdown(report: Mapping[str, Any], path: Path) -> None:
    summary = report["summary"]
    distribution = report["query_distribution"]
    lines = [
        "# Physics query-key discovery probe",
        "",
        "The page index and qrels are fixed. Keys are generated deterministically",
        "from the French query; qrels are used only for evaluation and oracle rows.",
        "",
        "## Setup",
        "",
        f"- Queries: **{distribution['queries']}**; gold pages/query median **{distribution['gold_pages']['median']:.0f}**, p90 **{distribution['gold_pages']['p90']:.0f}**, range **{distribution['gold_pages']['min']}–{distribution['gold_pages']['max']}**.",
        f"- Key candidates/query: mean **{distribution['key_candidates_per_query']['mean']:.1f}**, median **{distribution['key_candidates_per_query']['median']:.0f}**.",
        "- Key families: full query, content-only query, tokens, content bigrams/trigrams, and punctuation-delimited clauses.",
        "",
        "## Retrieval by key family",
        "",
        "| Key family | Instances | Page recall@10 | Page recall@100 | Page hit@10 | File recall@3 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key_type, row in summary["by_key_type"].items():
        m = row["metrics"]
        lines.append(
            f"| {key_type} | {row['key_instances']} | {m['recall_10']:.2%} | {m['recall_100']:.2%} | {m['hit_10']:.2%} | {m['file_recall_at3']:.2%} |"
        )
    lines += [
        "",
        "The token/ngram rows average over every key instance, not over one selected key per query.",
        "",
        "## Query-level comparison",
        "",
        "| Strategy | Page recall@10 | Page recall@100 | Page hit@10 | File recall@3 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in [
        ("full query", summary["full_query"]),
        ("content-only query", summary["content_only"]),
        ("rarest token heuristic", summary["heuristics"]["rarest_token"]),
        ("rarest phrase heuristic", summary["heuristics"]["rarest_phrase"]),
        ("largest top1-top10 gap heuristic", summary["heuristics"]["largest_top1_top10_gap"]),
        ("RRF over all keys", summary["rrf_all_keys"]),
        ("oracle best key (diagnostic)", summary["oracle_best_key"]),
    ]:
        lines.append(
            f"| {name} | {row['recall_10']:.2%} | {row['recall_100']:.2%} | {row.get('hit_10', row.get('hit_10_queries', 0)):.2%} | {row['file_recall_at3']:.2%} |"
        )
    lines.append(
        f"| all-key union (coverage only) | -- | {summary['all_key_union']['candidate_recall']:.2%} | "
        f"{summary['all_key_union']['hit_10_any_key_queries']:.2%} any-key | -- |"
    )
    lines += [
        "",
        "## Oracle and rescue analysis",
        "",
        f"- Oracle best-key type counts: `{summary['oracle_best_key']['best_key_type_counts']}`.",
        f"- Full-query top-10 hit queries: **{summary['rescue']['full_hit_10_queries']} / {distribution['queries']}**.",
        f"- Queries where at least one key hits top-10: **{summary['rescue']['any_key_hit_10_queries']} / {distribution['queries']}**.",
        f"- Top-10 queries rescued by key union: **{summary['rescue']['queries_rescued_at_hit_10']}**.",
        f"- Full-query candidate recall@100 is **{summary['rescue']['full_candidate_recall_100']:.2%}**; the union contains **{summary['rescue']['union_candidate_recall']:.2%}** of gold pages before a final union ranking.",
        "",
        "The oracle row is an upper bound because it chooses the key after seeing qrels. It proves that useful primary keys exist for many queries, but it is not an executable policy.",
        "",
        "## Conclusion",
        "",
        "- The full-query score is not the only useful representation. A union of independently retrieved keys broadens candidate coverage and rescues queries that full BM25 misses.",
        "- Removing function words is a small executable improvement; selecting the rarest single token is not.",
        "- The best key often remains the full query, while token and phrase keys provide complementary rescue candidates. This supports discovery/consensus rather than blind replacement of the full query.",
        "- The next step is a qrel-free key-quality estimator based on key result stability, lexical specificity and cross-key agreement, followed by a held-out cost-aware fusion. Do not use the oracle best key as a production result.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    benchmark = ViDoreV3(root=BENCHMARK_ROOT, subset="physics", language="french")
    questions_list = sorted(benchmark.questions(), key=lambda question: int(question.qid.rsplit("::", 1)[1]))
    questions = {question.qid: question for question in questions_list}
    qids = [question.qid for question in questions_list]
    qrels = benchmark.qrels()
    corpus = HierarchyCorpus.from_parsed_run(PARSED_RUN, subset="physics")
    index = build_bm25((page_id, corpus.pages[page_id].text) for page_id in corpus.page_order)
    term_idf = _idf(index)
    per_query, type_metrics = _query_rows(qids, questions, qrels, index, term_idf)
    report = {
        "contract_version": "physics-query-key-discovery-v1",
        "dataset": {
            "subset": "physics",
            "language": "french",
            "queries": len(qids),
            "pages": len(corpus.pages),
            "files": len(corpus.files),
            "qrel_pages": sum(len(qrels[qid]) for qid in qids),
        },
        "protocol": {
            "page_index": str(PARSED_RUN),
            "retriever": "fixed page BM25 over KDL + pdf-inspector page text",
            "depth": DEPTH,
            "qrel_leakage": "qrels are evaluation-only except the explicitly named oracle best-key diagnostic",
            "key_generation": "fixed French function-word list, content tokens, bigrams/trigrams, punctuation clauses",
        },
        "query_distribution": _distribution(benchmark, qrels),
        "summary": _summary(per_query, type_metrics),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT_DIR / "per_query.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in per_query),
        encoding="utf-8",
    )
    _write_markdown(report, OUTPUT_DIR / "report.md")
    print(f"Wrote {OUTPUT_DIR / 'report.md'}")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
