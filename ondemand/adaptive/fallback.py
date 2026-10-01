import sys
from collections import Counter

COUNTS = Counter()
SEEN = set()
WIDTH = 78


def warn(kind, detail):
    COUNTS[kind] += 1
    if kind in SEEN:
        return
    SEEN.add(kind)
    bar = "!" * WIDTH
    print(f"\n{bar}\n!!! FALLBACK: {kind}\n!!! {detail}\n"
          f"!!! The affected queries are NOT using the tool you configured.\n"
          f"!!! Only the first occurrence is printed; the run summary gives the total.\n{bar}\n",
          file=sys.stderr, flush=True)


def reset():
    COUNTS.clear()
    SEEN.clear()


def snapshot():
    return dict(COUNTS)


def since(before):
    return {k: v - before.get(k, 0) for k, v in COUNTS.items() if v - before.get(k, 0) > 0}


def banner(per_query, total):
    if not per_query:
        return
    bar = "!" * WIDTH
    lines = [f"!!!   {kind}: {len(queries)} of {total} queries ({sum(queries.values())} calls)"
             for kind, queries in sorted(per_query.items())]
    print(f"\n{bar}\n!!! THIS RUN USED FALLBACKS. Do not report it as the configured pipeline.\n"
          + "\n".join(lines) + f"\n{bar}\n", file=sys.stderr, flush=True)
