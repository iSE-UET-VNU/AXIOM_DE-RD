"""Document-level retrieval metrics with bootstrap confidence intervals.

Gold evidence is a set of documents per question, so recall is set-based and a
question with several evidence files is only fully satisfied when all of them
are retrieved. Confidence intervals matter here: 54 questions can separate large
effects and cannot separate small ones, and the interval says which is which.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import math
import random


@dataclass(frozen=True)
class QueryResult:
    qid: str
    ranked_docs: list[str]
    gold_docs: list[str]


def recall_at_k(result: QueryResult, k: int) -> float:
    gold = set(result.gold_docs)
    if not gold:
        return 0.0
    return len(gold & set(result.ranked_docs[:k])) / len(gold)


def hit_at_k(result: QueryResult, k: int) -> float:
    return 1.0 if set(result.gold_docs) & set(result.ranked_docs[:k]) else 0.0


def mrr_at_k(result: QueryResult, k: int) -> float:
    gold = set(result.gold_docs)
    for rank, doc in enumerate(result.ranked_docs[:k], start=1):
        if doc in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(result: QueryResult, k: int) -> float:
    gold = set(result.gold_docs)
    if not gold:
        return 0.0
    gain = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc in enumerate(result.ranked_docs[:k], start=1)
        if doc in gold
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(gold), k) + 1))
    return gain / ideal if ideal else 0.0


METRICS = {
    "recall@1": lambda r: recall_at_k(r, 1),
    "recall@5": lambda r: recall_at_k(r, 5),
    "recall@10": lambda r: recall_at_k(r, 10),
    "recall@20": lambda r: recall_at_k(r, 20),
    "hit@1": lambda r: hit_at_k(r, 1),
    "hit@5": lambda r: hit_at_k(r, 5),
    "hit@10": lambda r: hit_at_k(r, 10),
    "mrr@10": lambda r: mrr_at_k(r, 10),
    "ndcg@10": lambda r: ndcg_at_k(r, 10),
}


def evaluate(results: Sequence[QueryResult]) -> dict[str, float]:
    if not results:
        return {name: 0.0 for name in METRICS}
    return {
        name: round(sum(fn(result) for result in results) / len(results), 4)
        for name, fn in METRICS.items()
    }


def bootstrap_ci(
    results: Sequence[QueryResult],
    metric: str = "recall@10",
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 20260729,
) -> tuple[float, float]:
    if not results:
        return (0.0, 0.0)
    fn = METRICS[metric]
    values = [fn(result) for result in results]
    rng = random.Random(seed)
    size = len(values)
    means = []
    for _ in range(samples):
        means.append(sum(values[rng.randrange(size)] for _ in range(size)) / size)
    means.sort()
    low = means[int((1 - confidence) / 2 * samples)]
    high = means[min(samples - 1, int((1 + confidence) / 2 * samples))]
    return (round(low, 4), round(high, 4))


def paired_bootstrap(
    left: Sequence[QueryResult],
    right: Sequence[QueryResult],
    metric: str = "recall@10",
    samples: int = 2000,
    seed: int = 20260729,
) -> float:
    """Two-sided p-value that two systems differ on the same questions."""
    fn = METRICS[metric]
    by_qid = {result.qid: fn(result) for result in right}
    paired = [(fn(result), by_qid[result.qid]) for result in left if result.qid in by_qid]
    if not paired:
        return 1.0
    observed = sum(a - b for a, b in paired) / len(paired)
    rng = random.Random(seed)
    size = len(paired)
    extreme = 0
    for _ in range(samples):
        total = 0.0
        for _ in range(size):
            a, b = paired[rng.randrange(size)]
            difference = a - b
            total += difference if rng.random() < 0.5 else -difference
        if abs(total / size) >= abs(observed):
            extreme += 1
    return round(extreme / samples, 4)


def mcnemar(
    left: dict[str, bool], right: dict[str, bool]
) -> dict[str, float | int]:
    """Exact McNemar test on paired binary outcomes (answer correctness).

    Every arm answers the same questions, so the comparison is paired and only
    the discordant pairs carry information: questions both arms get right, or
    both get wrong, say nothing about which is better. At n<100 that leaves few
    informative pairs, which is exactly the power problem this reports rather
    than hides -- ``discordant`` is the effective sample size, not ``n``.
    """
    shared = sorted(set(left) & set(right))
    only_left = sum(1 for qid in shared if left[qid] and not right[qid])
    only_right = sum(1 for qid in shared if right[qid] and not left[qid])
    discordant = only_left + only_right
    if discordant == 0:
        return {"n": len(shared), "discordant": 0, "left_only": 0, "right_only": 0, "p": 1.0}
    # Exact two-sided binomial test at p=0.5 over the discordant pairs.
    tail = sum(
        math.comb(discordant, i) for i in range(min(only_left, only_right) + 1)
    )
    p = min(1.0, 2.0 * tail / (2.0**discordant))
    return {
        "n": len(shared),
        "discordant": discordant,
        "left_only": only_left,
        "right_only": only_right,
        "p": round(p, 4),
    }


def minimum_detectable_effect(n: int, discordant_rate: float = 0.3) -> float:
    """Roughly the smallest accuracy gap McNemar can resolve at 80% power.

    Reported alongside every comparison so a null result is read as "too small
    to see at this n" rather than as "no difference".
    """
    discordant = max(1.0, n * discordant_rate)
    return round(1.96 * math.sqrt(discordant) / max(1, n), 4)
