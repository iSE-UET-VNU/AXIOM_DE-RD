"""SEP -- Structural Evidence Propagation over a served pool.

The physics diagnostic showed the dominant failure is not "the gold file is
missing" (7%) but "the gold file is already in the top-10 and the wrong pages
within it rank" (69%), and that a rank-11..100 page adjacent to a returned page
is ~4.4x more likely to be gold than a non-adjacent one. SEP turns that free
structural prior into a score:

    s'(c) = lam * s(c) + (1 - lam) * [ beta * A_file(c) + (1 - beta) * N(c) ]

    N(c)       distance-decayed evidence from neighbouring pages of the same file
    A_file(c)  top-m mean of the file's pool scores (mean, not sum: sum rewards
               long documents)

The tree is the corpus's own file -> page structure, already encoded in the unit
id, so this needs no parser, no LLM and no API call. It reorders only the served
pool, so recall at the pool depth is unchanged by construction.
"""

from __future__ import annotations

from collections import defaultdict
import re

# Defaults verified on ViDoRe V3 physics (docs/vidore_v3_results.md).
LAMBDA, W, GAMMA, BETA, TOPM = 0.5, 2, 0.5, 0.75, 3

# Accepts both id conventions in the repo: the benchmark unit id
# ``physics::File#page=0`` and the discovery page id ``a/b/File.pdf#page=1``.
# SEP only uses same-file grouping and *relative* page distance, so a uniform
# 0/1-based offset between the two is harmless.
UNIT = re.compile(r"^(?:[^:/]+::)?(?P<file>.+)#page=(?P<page>\d+)$")


def split(unit_id: str) -> tuple[str, int] | None:
    m = UNIT.match(unit_id)
    return (m["file"], int(m["page"])) if m else None


def minmax(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    low, high = min(values.values()), max(values.values())
    span = high - low
    if span <= 1e-12:
        return {k: 1.0 for k in values}
    return {k: (v - low) / span for k, v in values.items()}


def propagate(scores: dict[str, float], lam: float = LAMBDA, w: int = W,
              gamma: float = GAMMA, beta: float = BETA, topm: int = TOPM,
              radius: int = 0) -> dict[str, float]:
    """One query's pool scores -> SEP-rescored pool scores.

    `radius` bounds the aggregation node: pages within +/-radius of c in the same
    file. radius=0 means the whole file, i.e. the tree's root.
    """
    base = minmax(scores)
    by_file: dict[str, dict[int, float]] = defaultdict(dict)
    for unit, value in base.items():
        if (p := split(unit)):
            by_file[p[0]][p[1]] = value

    neighbour: dict[str, float] = {}
    aggregate: dict[str, float] = {}
    for unit in base:
        p = split(unit)
        if p is None:
            neighbour[unit] = aggregate[unit] = 0.0
            continue
        pages = by_file[p[0]]
        neighbour[unit] = sum(
            (gamma ** abs(d)) * pages[p[1] + d]
            for d in range(-w, w + 1)
            if d and (p[1] + d) in pages)
        scope = (pages.values() if radius <= 0 else
                 [v for pg, v in pages.items() if abs(pg - p[1]) <= radius])
        top = sorted(scope, reverse=True)[:topm]
        aggregate[unit] = sum(top) / len(top) if top else 0.0

    neighbour, aggregate = minmax(neighbour), minmax(aggregate)
    return {u: lam * base[u] + (1 - lam) * (beta * aggregate[u] + (1 - beta) * neighbour[u])
            for u in base}
