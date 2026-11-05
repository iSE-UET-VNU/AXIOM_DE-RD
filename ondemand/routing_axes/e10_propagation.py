import statistics as st
from collections import defaultdict

import numpy as np

from ondemand.bench import BENCH, load_jsonl, work
from ondemand.light_retrieval import chunk_vectors
from ondemand.routing_axes.tight_coupling_common import ENC, GROUPS, W, boot, doc_of, doc_pages, fmt, group, mean_r, metrics, stored

ALPHAS = (0.05, 0.1, 0.3, 0.5, 0.7, 0.9)
LENGTHS = (1, 2, 4, 8, 16, 32)
FOLDS, FOLD_SEED, FIXED_ALPHA = 5, 0, 0.5


def page_no(unit):
    return int(unit.split("#page=")[1])


def solve(w):
    d = w.sum(1)
    isolated = d == 0
    root = np.where(isolated, 0.0, 1.0 / np.sqrt(np.where(isolated, 1.0, d)))
    s = w * np.outer(root, root)
    s[isolated, isolated] = 1.0
    return np.linalg.eigh(s)


def diffuse(system, y, a):
    vals, vecs = system
    return (1 - a) * vecs @ ((vecs.T @ y) / (1 - a * vals))


def toy_check():
    w = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=float)
    got = diffuse(solve(w), np.array([1.0, 0.0, 0.5]), 0.5)
    if not np.allclose(got, [2 / 3, 1 / 3, 0.5]):
        raise SystemExit(f"toy check failed: {got}")
    print("toy check: connected pair and isolated page match hand-computed values")


def page_vectors():
    rows = load_jsonl(W / "light_prep/pages_ppocrv5.jsonl")
    owners, vectors = chunk_vectors(rows, work("embedding_cache", "te3s_ppocr", bench=BENCH), False)
    total = np.zeros((len(rows), vectors.shape[1]), dtype=np.float64)
    np.add.at(total, owners, vectors)
    has = np.zeros(len(rows), dtype=bool)
    has[owners] = True
    total[has] /= np.linalg.norm(total[has], axis=1, keepdims=True)
    index = {r["page_id"]: i for i, r in enumerate(rows)}
    return total, has, index


def main():
    toy_check()
    members = GROUPS["vidore"]
    vectors, has, index = page_vectors()
    docs = sorted({doc_of(p) for q in members for p in ENC[q]})
    order = {d: sorted(doc_pages[d], key=page_no) for d in docs}
    if any(p not in index for d in docs for p in order[d]):
        raise SystemExit("candidate page missing from pages_ppocrv5.jsonl")

    pairs = {}
    for d in docs:
        rows = np.array([index[p] for p in order[d]])
        text = rows[has[rows]]
        c = vectors[text] @ vectors[text].T
        pairs[d] = c[np.triu_indices(len(text), 1)]
    overall = float(np.median(np.concatenate(list(pairs.values()))))
    fallback_docs = sum(len(pairs[d]) == 0 for d in docs)
    print(f"documents {len(docs)}; without two text pages, using overall median cosine {overall:.4f}: {fallback_docs}")

    sim = {}
    for d in docs:
        rows = np.array([index[p] for p in order[d]])
        neutral = float(np.median(pairs[d])) if len(pairs[d]) else overall
        c = vectors[rows] @ vectors[rows].T
        missing = ~has[rows]
        c[missing, :] = neutral
        c[:, missing] = neutral
        c = np.clip(c, 0, None)
        np.fill_diagonal(c, 0)
        sim[d] = c

    systems = {}

    def system(d, length):
        if (d, length) not in systems:
            numbers = np.array([page_no(p) for p in order[d]], dtype=float)
            systems[d, length] = solve(np.exp(-np.abs(numbers[:, None] - numbers[None, :]) / length) * sim[d])
        return systems[d, length]

    def rank(q, a, length):
        scores = ENC[q]
        low, high = min(scores.values()), max(scores.values())
        span = high - low
        y = {p: (s - low) / span if span > 0 else 1.0 for p, s in scores.items()}
        final = {}
        for d in {doc_of(p) for p in scores}:
            start = np.array([y.get(p, 0.0) for p in order[d]])
            for p, f in zip(order[d], diffuse(system(d, length), start, a)):
                final[p] = round(float(f), 10)
        return sorted(final, key=lambda p: (-final[p], p not in scores, p))[:10]

    base = metrics({q: stored[q] for q in members})
    print(f"baseline R@10 {mean_r(base, members):.2f} n {len(members)}")
    zero = {q: rank(q, 0.0, LENGTHS[0]) for q in members}
    wrong = [q for q in members if zero[q][:len(stored[q])] != stored[q]]
    print(f"a=0 check: differs from stored baseline on {len(wrong)} queries; encoded pages min {min(len(ENC[q]) for q in members)}")
    if wrong:
        raise SystemExit(f"a=0 does not reproduce baseline: {wrong[:5]}")

    rng = np.random.default_rng(FOLD_SEED)
    by_group = defaultdict(list)
    for q in sorted(members):
        by_group[group[q]].append(q)
    fold = {}
    for g in sorted(by_group):
        for i, position in enumerate(rng.permutation(len(by_group[g]))):
            fold[by_group[g][position]] = i % FOLDS

    grid = {(a, length): metrics({q: rank(q, a, length) for q in members}) for a in ALPHAS for length in LENGTHS}
    out = {}
    for f in range(FOLDS):
        train = [q for q in members if fold[q] != f]
        test = [q for q in members if fold[q] == f]
        a, length = min(grid, key=lambda c: (-round(mean_r(grid[c], train), 10), c[0], c[1]))
        print(f"fold {f}: train {len(train)} test {len(test)} chose a={a} l={length} train R@10 {mean_r(grid[a, length], train):.2f}")
        for q in test:
            out[q] = grid[a, length][q]

    delta = [out[q][0] - base[q][0] for q in members]
    ci = boot(delta)
    print(f"OUT-OF-FOLD R@10 {mean_r(out, members):.2f} vs baseline {mean_r(base, members):.2f}: {fmt(ci)} {'PASS' if ci[1] > 0 else 'FAIL'}")

    own = {}
    for q in members:
        page_by_doc = defaultdict(list)
        for p in stored[q]:
            page_by_doc[doc_of(p)].append(page_no(p))
        gaps = [b - a for numbers in page_by_doc.values() for a, b in zip(sorted(numbers), sorted(numbers)[1:])]
        own[q] = st.median(gaps) if gaps else None
    known = [v for v in own.values() if v is not None]
    middle = st.median(known)
    fixed = metrics({q: rank(q, FIXED_ALPHA, own[q] if own[q] is not None else middle) for q in members})
    print(f"reference a={FIXED_ALPHA} label-free l (median {middle}, fallback on {len(members) - len(known)} queries): "
          f"R@10 {mean_r(fixed, members):.2f} {fmt(boot([fixed[q][0] - base[q][0] for q in members]))}")


if __name__ == "__main__":
    main()
