"""Diagnose whether QCAR's query-clustering assumption fits the 220 queries.

The clustering itself uses only cached query embeddings.  Page qrels are used
after clustering to test the operational assumption: queries in one cluster
should have similar retrieval-depth needs (the paper's k*).  This is a
diagnostic, not a deployment claim, because the 220 queries are a benchmark
sample rather than a historical production query log.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "experiments/adaptive-k-retrieval"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_qcar_faithful import (  # type: ignore  # noqa: E402
    _kstar,
    _load_query_vectors,
    load_rows,
)


DEFAULT_OUTPUT = ROOT / "data/work/qcar_query_clustering_diagnostic_20261007"
DEFAULT_CLUSTERS = (2, 3, 5, 10, 20, 40, 80)


def _projection_configs(vectors: np.ndarray) -> dict[str, np.ndarray]:
    from sklearn.decomposition import PCA

    configurations: dict[str, np.ndarray] = {"raw": vectors}
    for components in (10, 20, 50):
        actual = min(components, vectors.shape[0] - 1, vectors.shape[1])
        configurations[f"pca_{actual}"] = PCA(
            n_components=actual, random_state=2026
        ).fit_transform(vectors)

    try:
        import umap

        # Include a small visualization space and the paper-like high-
        # dimensional space.  n_neighbors is capped by the 220-query sample.
        for components, neighbors in ((2, 50), (10, 50), (190, 200)):
            configurations[f"umap_{components}_nn{neighbors}"] = umap.UMAP(
                n_components=min(components, vectors.shape[0] - 2),
                n_neighbors=min(neighbors, vectors.shape[0] - 1),
                min_dist=0.0,
                metric="cosine",
                random_state=2026,
                transform_seed=2026,
                n_jobs=1,
            ).fit_transform(vectors)
    except ImportError:
        configurations["umap_unavailable"] = np.empty((len(vectors), 0))
    return configurations


def _purity(labels: np.ndarray, sources: Sequence[str]) -> float:
    grouped: dict[int, list[str]] = defaultdict(list)
    for label, source in zip(labels, sources):
        grouped[int(label)].append(source)
    return float(
        sum(max(Counter(values).values()) for values in grouped.values()) / len(labels)
    )


def _cluster_stats(
    representation: np.ndarray,
    labels: np.ndarray,
    sources: Sequence[str],
    kstars: np.ndarray,
    requested_clusters: int,
    max_k: int,
) -> dict[str, Any]:
    from sklearn.metrics import (
        adjusted_rand_score,
        calinski_harabasz_score,
        davies_bouldin_score,
        normalized_mutual_info_score,
        silhouette_score,
    )

    actual_clusters = len(set(int(value) for value in labels))
    metrics: dict[str, Any] = {
        "requested_clusters": int(requested_clusters),
        "actual_clusters": int(actual_clusters),
        "cluster_sizes": dict(
            sorted(Counter(int(value) for value in labels).items())
        ),
        "min_cluster_size": int(np.min(np.bincount(labels))),
        "max_cluster_size": int(np.max(np.bincount(labels))),
        "singleton_clusters": int(np.sum(np.bincount(labels) == 1)),
        "source_ari": float(adjusted_rand_score(sources, labels)),
        "source_nmi": float(normalized_mutual_info_score(sources, labels)),
        "source_purity": _purity(labels, sources),
    }
    if actual_clusters > 1 and actual_clusters < len(labels):
        metrics["silhouette"] = float(silhouette_score(representation, labels))
        metrics["davies_bouldin"] = float(davies_bouldin_score(representation, labels))
        metrics["calinski_harabasz"] = float(
            calinski_harabasz_score(representation, labels)
        )
    else:
        metrics["silhouette"] = None
        metrics["davies_bouldin"] = None
        metrics["calinski_harabasz"] = None

    global_mean = float(np.mean(kstars))
    total_ss = float(np.sum((kstars - global_mean) ** 2))
    within_ss = 0.0
    absolute_errors: list[float] = []
    log_kstars = np.log1p(kstars)
    log_global_mean = float(np.mean(log_kstars))
    log_total_ss = float(np.sum((log_kstars - log_global_mean) ** 2))
    log_within_ss = 0.0
    cluster_budgets: dict[str, int] = {}
    covered = []
    for cluster in sorted(set(int(value) for value in labels)):
        values = kstars[labels == cluster]
        cluster_mean = float(np.mean(values))
        cluster_std = float(np.std(values))
        within_ss += float(np.sum((values - cluster_mean) ** 2))
        absolute_errors.extend(np.abs(values - cluster_mean).tolist())
        log_values = np.log1p(values)
        log_cluster_mean = float(np.mean(log_values))
        log_within_ss += float(np.sum((log_values - log_cluster_mean) ** 2))
        budget = int(np.clip(math.ceil(cluster_mean + 2.0 * cluster_std), 1, max_k))
        cluster_budgets[str(cluster)] = budget
        covered.extend((values <= budget).tolist())
    metrics.update(
        {
            "kstar_mean": global_mean,
            "kstar_median": float(np.median(kstars)),
            "kstar_p90": float(np.percentile(kstars, 90)),
            "kstar_within_cluster_mae": float(np.mean(absolute_errors)),
            "kstar_cluster_r2": (
                float(1.0 - within_ss / total_ss) if total_ss > 0 else 0.0
            ),
            "log_kstar_cluster_r2": (
                float(1.0 - log_within_ss / log_total_ss)
                if log_total_ss > 0
                else 0.0
            ),
            "mean_cluster_budget_mean2std": float(np.mean(list(cluster_budgets.values()))),
            "cluster_budget_coverage": float(np.mean(covered)),
            "cluster_budgets": cluster_budgets,
        }
    )
    return metrics


def _domain_diagnostics(
    vectors: np.ndarray,
    sources: Sequence[str],
    kstars: np.ndarray,
    clusters: Sequence[int],
    max_k: int,
) -> list[dict[str, Any]]:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    rows: list[dict[str, Any]] = []
    for source in sorted(set(sources)):
        indices = np.asarray([i for i, value in enumerate(sources) if value == source])
        source_vectors = vectors[indices]
        components = min(20, len(indices) - 1, source_vectors.shape[1])
        representation = PCA(n_components=max(1, components), random_state=2026).fit_transform(
            source_vectors
        )
        for requested in clusters:
            if requested >= len(indices):
                continue
            labels = KMeans(
                n_clusters=requested, n_init=20, random_state=2026
            ).fit_predict(representation)
            metrics = _cluster_stats(
                representation,
                labels,
                [source] * len(indices),
                kstars[indices],
                requested,
                max_k,
            )
            metrics.update(
                {
                    "source": source,
                    "embedding_dimension": int(vectors.shape[1]),
                    "queries": int(len(indices)),
                    "projection": f"PCA_{components}",
                }
            )
            rows.append(metrics)
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--candidate-depth", type=int, default=200)
    parser.add_argument(
        "--kstar-depth",
        type=int,
        default=100,
        help="Depth used for the paper-style k* curve; QCAR uses 100 by default.",
    )
    parser.add_argument("--epsilon", type=float, default=0.001)
    parser.add_argument("--clusters", type=int, nargs="+", default=list(DEFAULT_CLUSTERS))
    parser.add_argument("--embeddings", nargs="+", choices=("qwen", "te3s"), default=["qwen", "te3s"])
    args = parser.parse_args(argv)
    if args.kstar_depth > args.candidate_depth:
        raise ValueError("--kstar-depth cannot exceed --candidate-depth")

    rows, _ = load_rows("hybrid", args.candidate_depth)
    qids = [str(row["qid"]) for row in rows]
    sources = [str(row["source"]) for row in rows]
    kstars = np.asarray(
        [
            _kstar(
                row["ranked"],
                set(row["gold"]),
                args.kstar_depth,
                args.epsilon,
                "max",
            )
            for row in rows
        ],
        dtype=np.float64,
    )
    global_budget = int(
        np.clip(
            math.ceil(np.mean(kstars) + 2.0 * np.std(kstars)),
            1,
            args.kstar_depth,
        )
    )

    all_results: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []
    domain_results: list[dict[str, Any]] = []
    for embedding in args.embeddings:
        vectors = _load_query_vectors(embedding, qids)
        vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9)
        representations = _projection_configs(vectors)
        for projection, representation in representations.items():
            if representation.shape[1] == 0:
                continue
            for requested in args.clusters:
                if requested >= len(rows):
                    continue
                from sklearn.cluster import KMeans

                labels = KMeans(
                    n_clusters=requested, n_init=20, random_state=2026
                ).fit_predict(representation)
                metrics = _cluster_stats(
                    representation,
                    labels,
                    sources,
                    kstars,
                    requested,
                    args.kstar_depth,
                )
                metrics.update(
                    {
                        "embedding": embedding,
                        "projection": projection,
                        "queries": len(rows),
                    }
                )
                all_results.append(metrics)
                if embedding == "qwen" and projection == "umap_190_nn200" and requested == 80:
                    for row, label, kstar in zip(rows, labels, kstars):
                        assignments.append(
                            {
                                "qid": row["qid"],
                                "source": row["source"],
                                "query": row["query"],
                                "cluster": int(label),
                                "kstar": int(kstar),
                            }
                        )
        domain_results.extend(
            _domain_diagnostics(
                vectors, sources, kstars, (2, 3, 5, 10), args.kstar_depth
            )
        )

    report = {
        "protocol": {
            "benchmark": "Physics 80 + Industrial 75 + OHR-Bench 65",
            "queries": len(rows),
            "pages": 5091,
            "ranking": "PP-OCRv5 + Qwen3-Embedding-8B; 0.2 normalized BM25 + 0.8 normalized dense",
            "candidate_depth": args.candidate_depth,
            "kstar_depth": args.kstar_depth,
            "qrels_used_for": "post-clustering k* homogeneity diagnostics only",
            "qrels_not_used_for": "embedding, projection, clustering, or assignment",
            "kstar": "last NDCG gain >= epsilon, epsilon=%.6f, empty prefix treated as hard (k=max)" % args.epsilon,
            "global_mean2std_budget": global_budget,
            "global_kstar_coverage": float(np.mean(kstars <= global_budget)),
        },
        "kstar_summary": {
            "mean": float(np.mean(kstars)),
            "median": float(np.median(kstars)),
            "p90": float(np.percentile(kstars, 90)),
            "max": int(np.max(kstars)),
            "histogram": {
                str(int(value)): int(np.sum(kstars == value))
                for value in sorted(set(kstars))
            },
        },
        "overall": all_results,
        "within_domain": domain_results,
        "paper_like_assignments": assignments,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "cluster_assignments_qwen_umap190_k80.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for row in assignments:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    rows_sorted = sorted(
        all_results,
        key=lambda row: (
            -(row["kstar_cluster_r2"] if row["kstar_cluster_r2"] is not None else -1),
            row["kstar_within_cluster_mae"],
        ),
    )
    silhouette_sorted = sorted(
        [row for row in all_results if row["silhouette"] is not None],
        key=lambda row: row["silhouette"],
        reverse=True,
    )
    lines = [
        "# QCAR query-clustering diagnostic",
        "",
        "Clustering uses only query embeddings. Qrels are used after clustering to test whether clusters have homogeneous retrieval-depth needs.",
        "",
        f"Global k* mean={np.mean(kstars):.2f}, median={np.median(kstars):.0f}, p90={np.percentile(kstars, 90):.0f}; global mean+2std budget={global_budget}, coverage={np.mean(kstars <= global_budget):.2%}.",
        "",
        "## Best retrieval-depth homogeneity",
        "",
        "| Embedding | Projection | K | Silhouette | Source purity | k* R2 | log(k*) R2 | Within-cluster MAE | Budget coverage |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows_sorted[:15]:
        lines.append(
            f"| {row['embedding']} | {row['projection']} | {row['requested_clusters']} | "
            f"{row['silhouette'] if row['silhouette'] is not None else float('nan'):.3f} | "
            f"{row['source_purity']:.2%} | {row['kstar_cluster_r2']:.3f} | "
            f"{row['log_kstar_cluster_r2']:.3f} | {row['kstar_within_cluster_mae']:.2f} | "
            f"{row['cluster_budget_coverage']:.2%} |"
        )
    lines += [
        "",
        "## Best geometric separation",
        "",
        "| Embedding | Projection | K | Silhouette | Davies-Bouldin | Source ARI | Source NMI | Source purity | Min cluster | Singleton clusters |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in silhouette_sorted[:15]:
        lines.append(
            f"| {row['embedding']} | {row['projection']} | {row['requested_clusters']} | "
            f"{row['silhouette']:.3f} | {row['davies_bouldin']:.3f} | {row['source_ari']:.3f} | "
            f"{row['source_nmi']:.3f} | {row['source_purity']:.2%} | {row['min_cluster_size']} | "
            f"{row['singleton_clusters']} |"
        )
    lines += [
        "",
        "## Within-domain diagnostic",
        "",
        "| Source | K | Silhouette | k* R2 | log(k*) R2 | Within-cluster MAE | Budget coverage |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in domain_results:
        if row["projection"] == "PCA_20" and row["requested_clusters"] in (3, 5, 10):
            lines.append(
                f"| {row['source']} | {row['requested_clusters']} | {row['silhouette']:.3f} | "
                f"{row['kstar_cluster_r2']:.3f} | {row['log_kstar_cluster_r2']:.3f} | "
                f"{row['kstar_within_cluster_mae']:.2f} | {row['cluster_budget_coverage']:.2%} |"
            )
    lines += [
        "",
        "## Interpretation",
        "",
        "- Source/domain purity is only a coarse proxy: QCAR assumes recurring retrieval intents, not merely Physics/Industrial/OHR labels.",
        "- k* homogeneity is the operational test. A useful query cluster should reduce within-cluster k* variance and make a cluster budget cover its members without simply selecting the global maximum.",
        "- This full-set diagnostic is exploratory; the earlier QCAR evaluation uses five-fold OOF budgets to avoid using a test query's qrels for its own assignment/budget.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", args.output_dir / "report.md")
    print("global kstar", float(np.mean(kstars)), float(np.median(kstars)), global_budget, float(np.mean(kstars <= global_budget)))
    print("top homogeneity")
    for row in rows_sorted[:10]:
        print(row["embedding"], row["projection"], row["requested_clusters"], row["kstar_cluster_r2"], row["kstar_within_cluster_mae"], row["silhouette"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
