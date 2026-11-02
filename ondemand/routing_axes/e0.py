import json
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import spearmanr

from ondemand.adaptive.chunk_rank import shingles
from ondemand.bench import FLAWED_QIDS, doc_of, gold as load_gold, group_of, load_jsonl, queries as load_queries, work
from ondemand.evaluate import as_run, evaluate
from ondemand.light_retrieval import language
from ondemand.text import enfr, real_text

BOOT, SEED = 10000, 0
W = work()
queries, gold_all = load_queries(), load_gold()
gx = json.loads((W / "light_prep/gate_ppocrv5_ungated_xgbK214.json").read_text())
gf = json.loads((W / "light_prep/gate_ppocrv5_ungated.json").read_text())
qids = sorted(q for q in gx["gate"] if q not in FLAWED_QIDS)
gold = {q: {u: g for u, g in gold_all[q].items() if g > 0} for q in qids}
group = {q: group_of(q, queries[q]) for q in qids}
GROUPS = {"all": qids, "vidore": [q for q in qids if group[q].startswith("vidore")]}
for name in sorted(set(group.values())):
    GROUPS[name] = [q for q in qids if group[q] == name]
SHOW = ["vidore", "vidore_industrial", "vidore_physics", "ohrbench:scan", "ohrbench:handwritten", "all"]
keys = json.loads((W / "colvec/colvec_keys.json").read_text())
rows = json.loads((W / "colvec/colvec_qids.json").read_text())
col = {k: i for i, k in enumerate(keys)}
row = {q: i for i, q in enumerate(rows)}
full = np.load(W / "routing_axes/k3/k3_out_shard0/evie_scores_2048.npy")
v2 = np.load(W / "routing_axes/k13/scores_v2_all.npy")
print("matrices", full.shape, v2.shape, "nan", int(np.isnan(full).sum()), int(np.isnan(v2).sum()))
pages = {r["page_id"]: r for r in load_jsonl(W / "light_prep/pages_ppocrv5.jsonl")}
doc_pages = defaultdict(list)
for k in keys:
    doc_pages[doc_of(k)].append(k)


def run_of(tag):
    return {r["query_id"]: r for r in load_jsonl(W / "results" / tag / "per_query.jsonl")}


def metrics(ranked):
    sub = {q: as_run(ranked[q]) for q in ranked}
    _, _, per = evaluate(sub, {q: gold[q] for q in ranked}, 10)
    return {q: (100 * per[q]["recall_10"], 100 * per[q]["ndcg_cut_10"]) for q in per}


def boot(delta):
    rng = np.random.default_rng(SEED)
    d = np.asarray(delta, float)
    m = d[rng.integers(0, len(d), size=(BOOT, len(d)))].mean(axis=1)
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def table(name, m, ref=None):
    for g in SHOW:
        q = GROUPS[g]
        r, n = np.mean([m[x][0] for x in q]), np.mean([m[x][1] for x in q])
        line = f"  {name:34s} {g:22s} n={len(q):3d} R@10 {r:6.2f} NDCG@10 {n:6.2f}"
        if ref:
            a, l, h = boot([m[x][0] - ref[x][0] for x in q])
            b, bl, bh = boot([m[x][1] - ref[x][1] for x in q])
            line += f" | dR {a:+.2f} [{l:+.2f},{h:+.2f}] dN {b:+.2f} [{bl:+.2f},{bh:+.2f}]"
        print(line)


def rank_by(mat, q, pool):
    r = row[q]
    return sorted(pool, key=lambda p: (-float(mat[r, col[p]]), p))


def stop_rule(mat, q, pool, light_top):
    docs = {doc_of(p) for p in light_top[:10]}
    order = [p for p in pool if doc_of(p) in docs] + [p for p in pool if doc_of(p) not in docs]
    r, scores, streak = row[q], {}, 0
    for p in order:
        best = sorted(scores.values(), reverse=True)
        s = float(mat[r, col[p]])
        streak = 0 if len(best) < 10 or s > best[9] else streak + 1
        scores[p] = s
        if streak >= 10:
            break
    return scores


pool_x = {q: gx["light_run_top100"][q][:min(gx["adaptive_k"][q], len(gx["light_run_top100"][q]))] for q in qids}
pool_f = {q: gf["light_run_top100"][q][:59] for q in qids}

print("\n===== S1 baseline reproduced from the score matrix =====")
base_rec = run_of("xk_evie_v2_stop10_214")
fx_rec = run_of("fx59_evie_v2_stop10")
encoded_x, encoded_f, base, fx = {}, {}, {}, {}
for q in qids:
    encoded_x[q] = stop_rule(v2, q, pool_x[q], pool_x[q])
    base[q] = sorted(encoded_x[q], key=lambda p: (-encoded_x[q][p], p))[:10]
    encoded_f[q] = stop_rule(v2, q, pool_f[q], pool_f[q])
    fx[q] = sorted(encoded_f[q], key=lambda p: (-encoded_f[q][p], p))[:10]
same = sum(base[q] == [p["unit"] for p in base_rec[q]["branch_pages"]] for q in qids)
samef = sum(fx[q] == [p["unit"] for p in fx_rec[q]["branch_pages"]] for q in qids)
print(f"top-10 lists identical to stored run: xgbK {same}/{len(qids)}, fx59 {samef}/{len(qids)}")
stored = {q: [p["unit"] for p in base_rec[q]["branch_pages"]] for q in qids}
m_base = metrics(stored)
table("baseline xk_evie_v2_stop10_214", m_base)
for g in SHOW:
    print(f"  encoded pages/q {g:22s} {np.mean([len(encoded_x[q]) for q in GROUPS[g]]):6.2f} of pool {np.mean([len(pool_x[q]) for q in GROUPS[g]]):6.2f}")

print("\n===== S2 128-dim vs 2048-dim, full EVIE, xgbK pool, no stop =====")
d = W / "routing_axes/k3/k3_out_shard0"
pidx = json.loads((d / "evie_pages_128_index.json").read_text())
qidx = json.loads((d / "evie_queries_128_index.json").read_text())
pemb = np.load(d / "evie_pages_128.npy", mmap_mode="r")
qemb = np.load(d / "evie_queries_128.npy").astype(np.float32)
pstart = np.concatenate([[0], np.cumsum(pidx["lengths"])])
qstart = np.concatenate([[0], np.cumsum(qidx["lengths"])])
ppos = {p: i for i, p in enumerate(pidx["page_ids"])}
qpos = {q: i for i, q in enumerate(qidx["query_ids"])}
o2048, o128, rho, overlap = {}, {}, {}, {}
for q in qids:
    i = qpos[q]
    Q = qemb[qstart[i]:qstart[i + 1]]
    s = {p: float((Q @ np.asarray(pemb[pstart[ppos[p]]:pstart[ppos[p] + 1]], dtype=np.float32).T).max(axis=1).sum())
         for p in pool_x[q]}
    o128[q] = sorted(pool_x[q], key=lambda p: (-s[p], p))
    o2048[q] = rank_by(full, q, pool_x[q])
    overlap[q] = len(set(o128[q][:10]) & set(o2048[q][:10])) / max(min(10, len(pool_x[q])), 1)
    if len(pool_x[q]) > 2:
        rho[q] = spearmanr([s[p] for p in pool_x[q]], [full[row[q], col[p]] for p in pool_x[q]]).statistic
m2048 = metrics({q: o2048[q][:10] for q in qids})
m128 = metrics({q: o128[q][:10] for q in qids})
table("2048 order (xk_evie)", m2048)
table("128 order", m128, m2048)
for g in SHOW:
    q = GROUPS[g]
    print(f"  fidelity {g:22s} top-10 overlap {100 * np.mean([overlap[x] for x in q]):5.1f}%  median Spearman {np.median([rho[x] for x in q if x in rho]):.3f}")

print("\n===== S3 xgbK vs fx59 =====")
m_fx = metrics({q: [p["unit"] for p in fx_rec[q]["branch_pages"]] for q in qids})
table("fx59 + V2 + stop:10 (vs baseline)", m_fx, m_base)
mf_full = metrics({q: rank_by(full, q, pool_f[q])[:10] for q in qids})
table("fx59 full EVIE (vs xgbK full EVIE)", mf_full, m2048)
for g in SHOW:
    print(f"  K {g:22s} xgbK mean {np.mean([len(pool_x[q]) for q in GROUPS[g]]):6.2f}, fx59 {np.mean([len(pool_f[q]) for q in GROUPS[g]]):6.2f}; encoded xgbK {np.mean([len(encoded_x[q]) for q in GROUPS[g]]):6.2f}, fx59 {np.mean([len(encoded_f[q]) for q in GROUPS[g]]):6.2f}")

print("\n===== S4 where baseline recall goes, in R@10 points =====")
CLASSES = ["hit", "crowded by gold", "encoded, doc in top10", "encoded, doc absent", "stop-halted, doc in top10",
           "stop-halted, doc absent", "outside K, in light 100", "outside light 100"]
pts = {g: Counter() for g in SHOW}
cnt = {g: Counter() for g in SHOW}
for q in qids:
    top = stored[q]
    docs = {doc_of(p) for p in top}
    all_gold_top = len(top) >= 10 and all(p in gold[q] for p in top[:10])
    light100 = set(gx["light_run_top100"][q])
    for u in gold[q]:
        if u in top:
            c = "hit"
        elif all_gold_top:
            c = "crowded by gold"
        elif u in encoded_x[q]:
            c = "encoded, doc in top10" if doc_of(u) in docs else "encoded, doc absent"
        elif u in pool_x[q]:
            c = "stop-halted, doc in top10" if doc_of(u) in docs else "stop-halted, doc absent"
        elif u in light100:
            c = "outside K, in light 100"
        else:
            c = "outside light 100"
        for g in SHOW:
            if q in GROUPS[g]:
                pts[g][c] += 100 / len(gold[q]) / len(GROUPS[g])
                cnt[g][c] += 1
for g in SHOW:
    print(f"  {g}: total {sum(cnt[g].values())} gold pages")
    for c in CLASSES:
        print(f"    {c:28s} pages {cnt[g][c]:4d}  points {pts[g][c]:6.2f}")
    cap = np.mean([100 * min(len(gold[q]), 10) / len(gold[q]) for q in GROUPS[g]])
    print(f"    {'cap with 10 slots':28s} {'':10s}  points {cap:6.2f}")

print("\n===== S5 ceilings: full-corpus EVIE and perfect document choice =====")
variants = {}
for label, mat in (("full EVIE", full), ("skip_blank", v2)):
    variants[f"{label} pool, no stop"] = {q: rank_by(mat, q, pool_x[q])[:10] for q in qids}
    variants[f"{label} full corpus"] = {q: rank_by(mat, q, keys)[:10] for q in qids}
    variants[f"{label} oracle doc, pool"] = {q: rank_by(mat, q, [p for p in pool_x[q] if doc_of(p) in {doc_of(u) for u in gold[q]}])[:10] for q in qids}
    variants[f"{label} oracle doc, all pages"] = {q: rank_by(mat, q, [p for d_ in {doc_of(u) for u in gold[q]} for p in doc_pages[d_]])[:10] for q in qids}
for name, ranked in variants.items():
    table(name, metrics(ranked), m_base)
print("  skip_blank vs full EVIE, same pages")
for scope in ("pool, no stop", "full corpus", "oracle doc, pool", "oracle doc, all pages"):
    table(f"skip_blank {scope}", metrics(variants[f"skip_blank {scope}"]), metrics(variants[f"full EVIE {scope}"]))
cap = {q: (100 * min(len(gold[q]), 10) / len(gold[q]), 0.0) for q in qids}
for g in SHOW:
    q = GROUPS[g]
    corpus = {x: rank_by(full, x, keys)[:10] for x in q}
    gdocs = {x: {doc_of(u) for u in gold[x]} for x in q}
    print(f"  {g:22s} cap R@10 {np.mean([cap[x][0] for x in q]):6.2f} | multi-doc gold q {sum(len(gdocs[x]) > 1 for x in q):3d} | "
          f"EVIE #1 page in gold doc: corpus {100 * np.mean([doc_of(corpus[x][0]) in gdocs[x] for x in q]):5.1f}% baseline {100 * np.mean([doc_of(stored[x][0]) in gdocs[x] for x in q]):5.1f}% | "
          f"top-10 slots in gold doc: corpus {100 * np.mean([np.mean([doc_of(p) in gdocs[x] for p in corpus[x]]) for x in q]):5.1f}% baseline {100 * np.mean([np.mean([doc_of(p) in gdocs[x] for p in stored[x]]) for x in q]):5.1f}%")

print("\n===== S6 confound table =====")
light_rec = run_of("xk_light")
fxl_rec = run_of("fx59_light")
light10 = {q: [p["unit"] for p in light_rec[q]["branch_pages"][:10]] for q in qids}
SETS = ["V+L-", "V+L+", "V-L+", "V-L-"]
shing = {}


def shing_of(p):
    if p not in shing:
        shing[p] = shingles(pages[p]["text"])
    return shing[p]


def near_copy(p):
    s = shing_of(p)
    if not s:
        return 0.0
    for other in doc_pages[doc_of(p)]:
        if other != p:
            t = shing_of(other)
            if t and len(s & t) >= 0.6 * min(len(s), len(t)):
                return 1.0
    return 0.0


FEATS = ["cross_lang", "lang_known", "overlap", "zero_overlap", "light_rank", "words", "near_copy", "native", "doc_in_light10", "doc_in_base10", "light1_gold_doc", "base1_gold_doc"]
data = []
for q in qids:
    if not group[q].startswith("vidore"):
        continue
    qtext = queries[q]["query"]
    ql = language(qtext)
    stems = set(enfr(qtext))
    lr = {p: i + 1 for i, p in enumerate(gx["light_run_top100"][q])}
    gdocs = {doc_of(u) for u in gold[q]}
    for u in gold[q]:
        text = real_text(pages[u]["text"])
        pl = language(text)
        pstems = set(enfr(text))
        known = ql is not None and pl is not None
        ov = len(stems & pstems) / max(len(stems), 1)
        data.append({"q": q, "group": group[q],
                     "set": ("V+" if u in stored[q] else "V-") + ("L+" if u in light10[q] else "L-"),
                     "cross_lang": float(ql != pl) if known else np.nan, "lang_known": float(known),
                     "overlap": ov, "zero_overlap": float(ov == 0), "light_rank": lr.get(u, 101),
                     "words": len(text.split()), "near_copy": near_copy(u),
                     "native": float(pages[u]["text_source"] == "pdf_inspector"),
                     "doc_in_light10": float(doc_of(u) in {doc_of(p) for p in light10[q]}),
                     "doc_in_base10": float(doc_of(u) in {doc_of(p) for p in stored[q]}),
                     "light1_gold_doc": float(doc_of(light10[q][0]) in gdocs),
                     "base1_gold_doc": float(doc_of(stored[q][0]) in gdocs),
                     "n_gold": len(gold[q])})


def cluster_diff(rows_, feat, a="V+L-", b="V+L+"):
    qs = sorted({r["q"] for r in rows_})
    idx = {q: i for i, q in enumerate(qs)}
    S = np.zeros((len(qs), 2))
    N = np.zeros((len(qs), 2))
    for r in rows_:
        if r["set"] in (a, b) and not np.isnan(r[feat]):
            j = 0 if r["set"] == a else 1
            S[idx[r["q"]], j] += r[feat]
            N[idx[r["q"]], j] += 1
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(qs), size=(BOOT, len(qs)))
    ss, nn = S[draws].sum(axis=1), N[draws].sum(axis=1)
    m = ss / np.maximum(nn, 1)
    delta = m[:, 0] - m[:, 1]
    return (S[:, 0].sum() / max(N[:, 0].sum(), 1) - S[:, 1].sum() / max(N[:, 1].sum(), 1),
            np.percentile(delta, 2.5), np.percentile(delta, 97.5))


for g in ["vidore", "vidore_industrial", "vidore_physics"]:
    sub = [r for r in data if g == "vidore" or r["group"] == g]
    print(f"  {g}")
    print("    " + f"{'feature':18s}" + "".join(f"{s:>10s}" for s in SETS) + "   V+L- minus V+L+ [95% CI, query bootstrap]")
    print("    " + f"{'gold pages':18s}" + "".join(f"{sum(r['set'] == s for r in sub):10d}" for s in SETS))
    print("    " + f"{'queries':18s}" + "".join(f"{len({r['q'] for r in sub if r['set'] == s}):10d}" for s in SETS))
    for f in FEATS:
        vals = "".join(f"{np.nanmean([r[f] for r in sub if r['set'] == s]) if any(r['set'] == s for r in sub) else float('nan'):10.3f}" for s in SETS)
        a, l, h = cluster_diff(sub, f)
        print(f"    {f:18s}{vals}   {a:+.3f} [{l:+.3f},{h:+.3f}]")
    print(f"    {'median light_rank':18s}" + "".join(f"{np.median([r['light_rank'] for r in sub if r['set'] == s]):10.1f}" for s in SETS))
    print(f"    {'median words':18s}" + "".join(f"{np.median([r['words'] for r in sub if r['set'] == s]):10.1f}" for s in SETS))

c = Counter()
for q in GROUPS["vidore"]:
    l = {p["unit"] for p in fxl_rec[q]["branch_pages"][:10]}
    for u in gold[q]:
        c[("V+" if u in stored[q] else "V-") + ("L+" if u in l else "L-")] += 1
print("  sets with fx59_light instead of xk_light:", dict(sorted(c.items())))
same_l = sum(set(light10[q]) == {p["unit"] for p in fxl_rec[q]["branch_pages"][:10]} for q in GROUPS["vidore"])
print(f"  light top-10 identical xk_light vs fx59_light: {same_l}/{len(GROUPS['vidore'])}")
print(f"  all near-copy share of gold pages: physics {100 * np.mean([r['near_copy'] for r in data if r['group'] == 'vidore_physics']):.1f}%, industrial {100 * np.mean([r['near_copy'] for r in data if r['group'] == 'vidore_industrial']):.1f}%")

print("\n  V+L- pages: does the win survive at each scorer?")
t2048 = {q: set(o2048[q][:10]) for q in qids}
t128 = {q: set(o128[q][:10]) for q in qids}
vl = [(q, u) for q in GROUPS["vidore"] for u in gold[q] if u in stored[q] and u not in light10[q]]
print(f"    n={len(vl)} in full-EVIE 2048 top 10: {sum(u in t2048[q] for q, u in vl)}, in 128 top 10: {sum(u in t128[q] for q, u in vl)}")
