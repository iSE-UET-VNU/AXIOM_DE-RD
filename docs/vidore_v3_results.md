# ViDoRe V3 — results ledger

Every number we have against the number it is meant to compare to. Paper figures
are from `aclanthology.org/2026.acl-long.755` (Table 3, Table 10, Table 12).

Scope: **physics, French only** for the ladder and end-to-end; the English Oracle
generation run predates that narrowing and is kept for its judge-stability
measurement.

---

## 1. Retrieval — physics, French, NDCG@10

302 queries, 1,674 pages, 42 documents, open-corpus within the subset.

### 1a. The only one-to-one comparison

| System | NDCG@10 | Source |
|---|---:|---|
| BM25S (published) | **39.8** | paper Table 10 |
| BM25S, French stemmer + stopwords, our data | 40.15 | verified locally |
| BM25S, French stopwords, our data | 39.71 | verified locally |
| BM25S, no stemmer/stopwords, our data | 39.00 | verified locally |
| **our BM25**, `analyzer=french` | **40.16** | E0a, `physics_french_analyzer.py` |
| **our BM25**, `analyzer=plain` | **37.45** | our ladder |

**Why the gap.** We installed the paper's own `bm25s` and ran it over our loader's
corpus, our queries, our qrels, scored by our `pytrec_eval` call. It brackets the
published 39.8. The corpus, the language filter, the page-id join and the NDCG
computation are therefore correct. The −2.7 is entirely tokenization.

**Closed (E0a).** `analyzer=french` in `src/retrieval/sparse.py` reaches 40.16,
matching `bm25s`'s 40.15 on identical inputs. Decomposed by
`physics_french_analyzer.py`, which reproduces every row below:

| step | NDCG@10 | Δ |
|---|---:|---:|
| `plain` | 37.45 | — |
| + split elisions (`l'énergie` → `énergie`) | 38.76 | +1.31 |
| + French stopwords | 39.79 | +1.02 |
| + Snowball stemming | **40.16** | +0.37 |

The largest single term is elision, which the earlier "stopwords and stemming"
attribution missed: `analyze` joins on the apostrophe, so `l'énergie` was one
token and a query for `énergie` could not match it. French elides constantly.

**It does not reach the fusion.** Under α=0.7 the french analyzer scores 44.02 vs
plain's 44.15 (E0b) — the dense leg already recovers what stemming does, and BM25
carries 0.3 weight. Keep it for BM25-only serving; it is not a pipeline win.

### 1b. Our arms (no published counterpart — these are fusions, not single models)

| Index | Items | Chars | BM25 | Dense | RRF | α=0.7 |
|---|---:|---:|---:|---:|---:|---:|
| ViDoRe page text | 1,674 | 1.08 M | 37.5 | 41.1 | 42.4 | **44.1** |
| ViDoRe + fixed-512 | 1,685 | 1.09 M | 37.5 | 41.1 | 42.3 | 44.2 |
| chandra2 page | 1,674 | 2.05 M | 38.3 | 39.6 | 41.5 | 42.9 |
| chandra2 + fixed-512 | 1,710 | 2.09 M | 38.2 | 39.5 | 41.5 | 42.9 |
| chandra2 + block chunking | 2,636 | 1.89 M | 35.9 | 39.0 | 40.8 | 42.5 |
| chandra2 − image descriptions | 1,674 | 1.12 M | 37.0 | 38.5 | 40.7 | 41.9 |
| chandra2 − headers/footers | 1,674 | 1.86 M | 36.0 | 39.1 | 40.3 | 41.7 |
| chandra2 prose only | 1,628 | 0.93 M | 35.9 | 38.1 | 41.1 | 41.6 |

Dense is `openai/text-embedding-3-small`, which appears nowhere in the paper.
α=0.7 fuses BM25 and dense; every paper row is a single retriever. **Do not read
44.1 against Jina-v4's 44.0.**

Reference band from paper Table 10, physics column — textual retrievers:
BGE-M3 38.3, BM25S 39.8, Qwen3-0.6B 43.8, Jina-v4 44.0, Qwen3-8B 45.8.
Visual retrievers run 43.2–48.5; we have no visual arm.

### 1b-ii. Is α=0.7 tuned on test? (E0b)

Queries split 50/50 by `sha256(qid)`, α selected on one half and scored on the
other, both directions. `physics_alpha_holdout.py`.

| analyzer | best α in-sample | in-sample | α=0.7, all 302 | **held-out** |
|---|---:|---:|---:|---:|
| plain | 0.5 | 44.71 | 44.15 | **43.39** |
| french | 0.5 | 44.57 | 44.02 | **43.99** |

Two things worth stating. **α=0.7 is not the in-sample optimum** — 0.5 is, at
44.71 — so 44.15 was never the best number available on these queries, which is
the opposite of the overfit it was suspected of. And the honest out-of-sample
figure is **43.39**, so the selection cost is ~0.8, not several points.

α is weakly identified: the curve is flat from 0.4 to 0.7 (44.13 / 44.71 / 44.29 /
44.15) and the fold-selected α swings 0.45 → 0.75 across halves. Report the band,
not the argmax.

### 1b-iii. Reranking (E1) — the largest single gain measured here

Voyage `rerank-2.5` over the served α=0.7 top-20, scored on the identical
candidate pool so the only difference between arms is the ordering of the prefix.
`physics_rerank_voyage.py`, then `physics_rerank_significance.py`.

| Arm | NDCG@10 | Δ | p |
|---|---:|---:|---:|
| served α=0.7 | 44.15 | — | — |
| **+ rerank-2.5, top-20** | **49.23** | **+5.08** | **0.0001** |

146 queries better, 83 worse, 73 tied. This is the first change in the ladder
worth more than ~1 point, and it confirms the §1 diagnosis of
[retrieval_diagnosis_and_plan.md](retrieval_diagnosis_and_plan.md): the pages were
already in the pool and ordered badly.

**Which corpus this is.** `index=vidore_page` — **ViDoRe V3's own supplied page
text**, not chandra2 and not KDL. BM25 uses `analyzer=plain`, dense is
`openai/text-embedding-3-small`, pool depth 100, rerank top-20.

That is the arm involving **none of our parsing work**, so it measures their text
against their text and is the wrong row to represent our pipeline. The chandra2
arm (α=0.7 = 42.90, reproduced exactly from `physics_served_pool_chandra.json`) is
the one that does. `physics_chandra_pool.py` builds its pool; the rerank pass over
it is the comparison that matters and is reported separately below.

**A truncation asymmetry to carry into that comparison.** chandra2 pages are
richer — mean 1,234 chars against ViDoRe's 767 — so at the same 1200-char budget
**50.5% of chandra2 passages are clipped versus 11.5% of ViDoRe's**. Identical
protocol, unequal handicap, and it runs against chandra2. The budget is set by
Voyage's 10K TPM free-tier cap at depth 20, not by choice.

**Reranked vs reranked** — paper Table 2, physics (French) column:

| Pipeline | NDCG@10 | Δ from rerank |
|---|---:|---:|
| Jina-v4 textual | 43.6 | — |
| **Jina-v4 textual + zerank-2** | **48.2** | +4.6 |
| Jina-v4 visual | 46.6 | — |
| Jina-v4 visual + jina-reranker-m0 | 46.9 | +0.3 |
| our α=0.7, `vidore_page` | 44.15 | — |
| **our α=0.7 + rerank-2.5** | **49.23** | **+5.08** |

So we do edge the paper's best physics pipeline, 49.23 vs **48.2**, and our rerank
delta (+5.08) is slightly larger than zerank-2's on this dataset (+4.6).

**The +13.2 headline does not apply here.** Physics is the dataset where zerank-2
helps *least* of the ten — +4.6 against a 10-dataset average of +13.2 (Fin. gets
+20.8). Any earlier extrapolation of +13.2 onto physics was wrong. Our +5.08
landing near +4.6 is the expected magnitude for this dataset, not a shortfall.

Still not like-for-like: ours is a **fusion plus a cross-encoder**, theirs a single
retriever plus a cross-encoder. And this row is the `vidore_page` arm — see below.

Three things this number is not:

- **Not zerank-2.** The plan predicted +8/+14 for zerank-2 over Jina-v4
  candidates. There is no ZeroEntropy key; this is a different model on a
  different pool. +5.08 neither confirms nor refutes that band.
- **Not depth-tuned.** Depth 20 is `RETRIEVAL_K3`, what production already serves.
  The free Voyage tier (3 RPM / 10K TPM) cannot fit a depth-50 request at all, so
  the depth ablation is unmeasured, not chosen.
- **Not an untruncated measurement.** 11.5% of passages are clipped at 1200 chars
  by the token budget. A clipped gold page can lose what made it gold, so +5.08 is
  a **floor** under an untruncated reranker. It does not explain the 83 losers
  though: they average 13.7% clipped against the winners' 11.4%.

**QA was not measured for this arm.** §2 of the plan bounds any end-to-end gain at
~3.6pp credited by the oracle gap, and E3 was not run. Do not infer a QA gain from
this row.

### 1c. Ablations (paired permutation, 10,000 resamples, α=0.7 unless noted)

| Contrast | Δ | p | |
|---|---:|---:|---|
| + sub-page chunking (blocks) | −0.39 | 0.72 | n.s. |
| + image descriptions | +0.99 | 0.25 | n.s. |
| + headers/footers | +1.16 | 0.23 | n.s. |
| chandra2 vs ViDoRe text | −1.26 | 0.12 | n.s. |
| + headers/footers, **BM25 arm** | **+2.27** | **0.0012** | significant |
| + sub-page chunking, **BM25 arm** | **−2.36** | **0.0044** | significant |

Confirms the paper's own justification: *"Chunking within pages or providing image
descriptions did not improve our results. Thus, we report the results of the
simplest pipeline."* Both hold. The simplest arm is also our best.

---

## 2. End-to-end — physics, French

| Pipeline | Context | Generator | NDCG@10 | correct_only | +partial | Paper (Phys.) |
|---|---|---|---:|---:|---:|---:|
| Oracle (gold pages) | Text | gpt-4o-mini | — | 55.6% | 93.4% | **71.2** (Gemini 3 Pro) |
| α=0.7 top-10, ViDoRe | Text | gpt-4o-mini | 44.2 | 54.3% | 89.7% | 69.2 (Jina-v4 + zerank-2) |
| α=0.7 top-10, chandra2 | Text | gpt-4o-mini | 42.9 | 56.6% | 89.7% | 64.9 (ColEmbed-3B-v2) |

| Contrast | correct_only | p | +partial | p |
|---|---:|---:|---:|---:|
| Oracle → retrieved ViDoRe | −1.32pp | 0.72 n.s. | **−3.64pp** | **0.030** |
| Oracle → retrieved chandra2 | +0.99pp | 0.84 n.s. | −3.64pp | 0.077 n.s. |
| ViDoRe → chandra2 | +2.32pp | 0.44 n.s. | 0.00pp | 1.00 n.s. |

**Retrieval penalty replicates.** Theirs 71.2 → 69.2 = −2.0. Ours 55.6 → 54.3 = −1.3.

**Absolute level does not, and should not.** −15.6 against 71.2 is gpt-4o-mini
against Gemini 3 Pro, not a pipeline difference.

**Gold coverage is low and it barely matters.** Retrieved context carries 2.34 of
7.21 gold pages; 43/302 queries get zero gold pages. Answer quality is still
statistically indistinguishable from Oracle on `correct_only`. Physics gold sets
are redundant — two of seven pages usually suffice.

**Retrieval loss shows up as partial answers, not wrong ones.** The only
significant end-to-end effect is −3.64pp on `correct_plus_partial`. Reporting
`correct_only` alone would have missed it entirely.

---

## 3. English Oracle generation (superseded scope, kept for judge stability)

5 English subsets, 1,489 queries, uncapped context.

| Subset | n | correct_only | +partial |
|---|---:|---:|---:|
| computer_science | 215 | 79.5% | 97.2% |
| finance_en | 309 | 59.2% | 90.0% |
| pharmaceuticals | 364 | 59.6% | 91.8% |
| hr | 318 | 55.0% | 88.7% |
| industrial | 283 | 51.2% | 91.2% |
| **global** | **1,489** | **59.8%** | **91.4%** |

Paper global Oracle/Text = 70.6, which sits **inside** our 59.8–91.4 band. We
cannot claim to match or miss it.

**Judge stability, two full passes on identical generator outputs:**
pass 1 59.8% → pass 2 60.8%, spread **1.0pp**, label agreement **93.1%**.

| pass 1 ↓ / pass 2 → | Correct | Partially Correct | Incorrect |
|---|---:|---:|---:|
| Correct | 856 | 35 | 0 |
| Partially Correct | 49 | 411 | 10 |
| Incorrect | 1 | 7 | 120 |

No Correct ↔ Incorrect flips in either direction beyond 1 case. The three-way
band is a property of the rubric, not judge noise.

---

## 4. Standing caveats for every table

1. **8 public subsets, not 10** — nuclear and telecom are private hold-outs.
   Physics is 1 of the 8, French only.
2. **Our BM25 is not BM25S** — no French stopwords or stemming, verified worth
   −2.3 NDCG on identical inputs.
3. **Our judge is gpt-4o; theirs is GPT-5.2 at medium reasoning effort.** Their
   stability figures do not transfer; ours is measured above.
4. **`correct_only` alongside `correct_plus_partial`, always.** The band is wide
   enough that quoting one picks the conclusion.
5. **Our generator is gpt-4o-mini; theirs is Gemini 3 Pro.** Dominates absolute
   answer quality; internal Oracle-minus-retrieved deltas are the comparable part.
6. **We have no visual retriever.** Compare only against the textual block.

---

## 5. Artifacts

`data/benchmark/vidore_v3/results/`

| File | Contents |
|---|---|
| `physics_e2e_oracle.json` | 302 rows: query, gold answer, model answer, judge label, retrieved unit ids + scores + is_gold |
| `physics_e2e_retrieved_vidore.json` | as above, α=0.7 top-10 over ViDoRe text |
| `physics_e2e_retrieved_chandra2.json` | as above, over the chandra2 parse |
| `physics_e2e_summary.json` | the three-arm summary |
| `physics_retrieval_ladder.json` | 8 indexes × 4 retrievers, with per-query NDCG@10 |
| `english_oracle_generation/*.json` | 1,489 generation rows + `stability.json` second-pass verdicts |

Retrieved rankings were re-derived and reconcile with the recorded `gold_hit` on
302/302 rows in all three arms.

**Not stored:** the rendered context string (reconstructible from `retrieved` plus
the corpus) and the judge's rationale (only the three-way label is kept).

---

## 6. Refactor gates — `research/harness` to `src/evaluation`

**Test baseline: 293 passing, 0 errors** at the merge of `origin/baseline`
(`0108649`) plus the conflict resolution (`9152841`). 304 after the registry
spanning test. Any drop below is a regression.

**Gate 3 reference** is frozen read-only outside the working tree at
`data/benchmark/vidore_v3/results/gate3_baseline.json`. The check is a SHA-256 over the
sorted per-question NDCG@10 map, so it is exact rather than tolerance-based --
an aggregate can match while individual questions move.

| arm | NDCG@10 | per-question SHA-256 |
|---|---:|---|
| `vidore_page::bm25` | 37.45 | `e9b83d23937e1ce7` |
| `vidore_page::dense` | 41.15 | `ff4d8f317bdd5d9c` |
| `vidore_page::rrf` | 42.35 | `fdb181bc513309ea` |
| `vidore_page::alpha0.7` | 44.15 | `5578c6c03e9be50d` |
| `chandra_page::bm25` | 38.28 | `7abd531b6859d900` |
| `chandra_page::dense` | 39.58 | `4685c1c4778830cb` |
| `chandra_page::rrf` | 41.45 | `6a0099295d29bb7b` |
| `chandra_page::alpha0.7` | 42.90 | `a248fce3531bdd37` |

Verified identical **before** the move as well as after, so a mismatch could not
be attributed to the refactor when the cause was nondeterminism.

    python research/experiments/gate_check.py <frozen reference>

**Machine-tuned config.** `configs/pipeline.yaml` keeps `chandra2.max_workers: 48`
and `max_output_tokens: 4096` against upstream's 256 / 12384, which are sized for
the notebook's H100/A100 branch. The file is JSON parsed by `json.loads` despite
its `.yaml` extension, so this cannot be recorded as an inline comment -- it is
recorded here instead, and the next merge should not silently restore upstream's
numbers.

---

## 7. SEP — Structural Evidence Propagation (2026-08-24)

### The idea

DISRetrieval (arXiv 2506.06313) builds an RST discourse tree over a document and
lets a query-relevant internal node promote the top-k leaves of its own subtree.
Two of its own ablations say which half of that is load-bearing: RQ3 finds
summary-based retrieval *underperforms* the leaf baseline, and RQ4 finds swapping
the node-summary LLM moves results <0.5%. So the expensive half — LLM-summarised
internal nodes — is the half the paper's evidence says to drop.

What remains is aggregation-and-promotion, and that needs a tree, not an RST
parser. Our corpus already ships one: `file → page`, encoded directly in the unit
id `subset::<file>#page=N`. SEP scores the existing α=0.7 pool, aggregates over
that tree, and blends the aggregate back:

    s'(c) = λ·s(c) + (1−λ)·[ β·A_file(c) + (1−β)·N(c) ]

`A_file` is the top-m mean of the file's pool scores (mean, not sum — sum rewards
long documents). `N` is distance-decayed evidence from neighbouring pages.
Reorders the pool only, so recall@100 is unchanged and every delta is ordering.
No API calls, no GPU, no index change.

### Why ordering, and at which level (`physics_structure_diagnostic.py`)

Recall was already known to be near-saturated (any-gold@100 = 98.3%). The open
question was which structural level carries the fix. On physics:

| failure shape | vidore_page | chandra_page |
|---|---|---|
| B: gold file present in top-10, wrong pages within it | **68.9%** | **70.5%** |
| C: all gold already in top-10 | 24.2% | 23.8% |
| A: gold file absent from top-10 | 7.0% | 5.6% |

So "promote a missing document" addresses ~7% of queries — a plausible
explanation for why the headers/footers, chunking and image-description
ablations all landed n.s. at α=0.7 (§1). The signal is *within* the file.

Decomposing the lift over rank-11..100 candidates settles what to aggregate:

| stratum | n | gold rate |
|---|---|---|
| same file as a top-10 page, adjacent (±2) | 3,563 | 10.64% |
| same file, not adjacent | 7,683 | 6.53% |
| different file | 15,934 | **0.46%** |

Same-file membership is a **16.9×** lift; adjacency adds only **1.63×** on top.
The headline "4.4× adjacency lift" is mostly same-file in disguise. Cross-file
candidates are gold 0.46% of the time — they are almost pure distractor, and
displacing them is where the gain comes from.

Gold is *not* contiguous (74.6% of runs are a single page, mean run 1.42), so a
span model is not the mechanism; gold is dispersed through the relevant file.

### Result — physics

Paired permutation, 10,000 resamples, vs the α=0.7 baseline on identical pools:

| index | baseline | SEP (λ=0.5) | delta | p | better/worse/tied |
|---|---|---|---|---|---|
| vidore_page  | 44.15 | 46.17 | **+2.02** | 0.0026 | 125/82/95 |
| chandra_page | 42.90 | 44.59 | **+1.69** | 0.0083 | 126/75/101 |

The effect decays smoothly to zero as λ→0.9 (+0.19, p=0.41), which is the shape
a real effect has rather than a spike.

**Selection caveat, stated plainly.** `β=0.75` was chosen from a grid sweep on
physics, not fixed a priori. The lift table justifies `β > 0.5`, not `0.75`
specifically. So the physics p-value is optimistic. `chandra_page` re-tests the
same 302 queries under a different parse — it establishes parse-robustness, not
selection-robustness. The honest physics bracket is **+1 to +2**, stable across
β ∈ [0.5, 0.75] and λ ∈ [0.5, 0.7].

### SEP does not generalise — and the precondition predicts where

SEP needs file-level evidence to constrain *which pages* are relevant, which only
holds when documents are small. Both statistics below are free (corpus + qrels,
no embeddings, no retrieval). Predictions were **registered before testing** in
`docs/sep_prescreen_prediction.md`.

| subset | pages/file | gold as % of its file | predicted | measured (best λ) |
|---|---|---|---|---|
| physics          |  39.9 | 14.6% | helps (in-sample) | **+2.02**, p=0.0026 |
| pharmaceuticals  |  44.5 | 12.0% | helps | **+2.17**, p=0.0005 |
| hr               |  79.3 |  9.5% | weak/marginal | +0.54, p=0.14 n.s. |
| industrial       | 194.2 |  5.2% | fails | −0.14, p=0.82 n.s. |
| computer_science | 680.0 |  0.8% | fails hardest | −0.09, p=0.87 n.s. |

**5/5, with graded resolution.** `pharmaceuticals` is the load-bearing test: a
different subset and domain, config completely unchanged, +2.17 at p=0.0005.
That is the one result here not exposed to the selection caveat above.

Mechanism of the failure: on `industrial`, gold occupies 5.2% of a 194-page file,
so the file aggregate averages the signal away. Gold rate by distance to the
nearest top-10 anchor makes it concrete —

| distance | physics | industrial |
|---|---|---|
| ≤1  | 24.4× | 6.6× |
| ≤5  | 19.2× | **8.5×** |
| ≤50 |  8.7× | 4.4× |
| >50 | 13.5× (n=80) | **2.1×** (n=3,460) |

Industrial does have local signal, but most of its same-file mass sits in the
diluting >50 tail. Bounding the aggregation node to a page window (radius 5–40)
was tried to rescue this and **does not work**: no radius is significant on
industrial, and on physics the whole-file node stays best. Recorded as a dead
end rather than retried.

### Standing

SEP is a **corpus-shape-conditional** technique, not a general improvement. The
screen above is the deployment test and costs nothing to run on a new corpus.
Rough threshold: gold ≳10% of its own file, i.e. chapter-sized documents.
Reproduce with:

```bash
python research/experiments/physics_structure_diagnostic.py
python research/experiments/physics_sep_test.py
python research/experiments/vidore_sep_holdout.py --subset pharmaceuticals --language english
```

---

## 8. DCW — refuted (2026-08-24)

**Hypothesis.** 69% of failures are "right file, wrong pages within it". A
bi-encoder should be structurally bad at that: every page of a document shares
its topical vocabulary, so each page embedding carries a large document-topic
component that is near-identical across the file — the very thing that makes the
file findable and its pages indistinguishable. Factorise it:

    e_hat_p = e_p − κ·μ_f
    score   = a·cos(q, μ_f) + b·cos(q, e_hat_p)

**Registered prediction.** DCW should behave opposite to SEP — μ_f is better
estimated and there is more to disambiguate when files are large, so it should
get *stronger* on `industrial` (194 pages/file) where SEP dies.

**Result: refuted, with the sign backwards.** Every one of 16 configurations is
worse on both subsets, monotonically in κ, and it fails *harder* on the large-
document subset:

| worst case (κ=1.0, a=0, b=1) | baseline | DCW | delta | p |
|---|---|---|---|---|
| physics    | 44.15 | 38.43 | **−5.73** | 0.0005 |
| industrial | 45.19 | 35.68 | **−9.50** | 0.0005 |

Even the gentlest setting (κ=0.25) never beats baseline on either subset.

**What it teaches.** The document-topic component is *signal, not confound*.
ViDoRe queries are largely topical, so most of a page's relevance genuinely is
"is this document about this topic"; the within-document residual is dominated
by page idiosyncrasy (headers, captions, layout noise) rather than by
discriminative content. This is consistent with §7 rather than contradicting it:
SEP gains by **amplifying** the document component (β=0.75 toward the file
aggregate), and DCW loses by removing it. Both point the same way.

The corollary is the useful part: within-document page discrimination is **not
recoverable from bi-encoder geometry**. The only thing measured to do it is
token-level query–page interaction — the cross-encoder, at +5.08 (§1). That is
also what the ColEmbed report concludes from the other direction, where
bi-encoder + reranker matches a late-interaction model at 1/2700th the storage.

Reproduce: `python research/experiments/vidore_dcw.py --subset industrial --language english`

---

## 9. KDL arm, and the granularity refutation (2026-08-25)

### KDL baseline — the parse we actually ship

Page-level, physics/French, from `data_vidore_parsed_physics/output/benchmarks/`:

| arm | physics | pharmaceuticals (en) |
|---|---|---|
| bm25 | 37.46 | — |
| dense | 39.78 | — |
| **α=0.7** | **43.03** | **56.36** |

Sits between `chandra_page` (42.90) and `vidore_page` (44.15). Note the CSV's
"Baseline Legacy" 44.2 is a *different* configuration — fixed-512 chunking — not
this page-level arm.

### SEP transfers to KDL

Config unchanged from physics (w=2, γ=0.5, β=0.75, top-m=3):

| KDL arm | baseline | SEP | delta | p |
|---|---|---|---|---|
| physics/french | 43.03 | 44.75 | **+1.73** | 0.0119 |
| pharmaceuticals/english | 56.36 | 57.77 | **+1.41** | 0.0021 |

As predicted: SEP never reads text, only `file#page` structure and pool scores,
so it is parse-agnostic. Gains are slightly smaller than on ViDoRe's own text
(+2.02) but hold on both subsets.

### Sub-page granularity is refuted — pooling helps, it does not hurt

Hypothesis was that one embedding per page (12.8 blocks, ~1,330 chars) drowns
the one block a query actually matches, and that MaxSim over sub-page units
would recover it — late interaction at affordable granularity.

**Wrong, monotonically.** BM25 over KDL physics, MaxSim aggregated back to pages:

| grouping | units/page | NDCG@10 | delta |
|---|---|---|---|
| **page (pooled)** | 1.0 | **37.46** | — |
| section (SectionHeader boundaries) | 1.6 | 35.51 | −1.94 |
| typed (tables/figures split out) | 3.8 | 33.64 | −3.82 |
| fixed-5 blocks | 2.8 | 34.69 | −2.77 |
| fixed-3 blocks | 4.4 | 32.46 | −5.00 |
| per block | 12.1 | 26.30 | −11.16 |

Random-grouping controls at matched unit count (`rand3` −6.22 vs `fixed3` −5.00;
`rand5` −4.89 vs `fixed5` −2.77) isolate the cause: contiguity is worth only
~1.2 points, while **granularity itself costs 5+**. It is fragmentation, not
grouping quality.

Confirmed with dense embeddings, so it is not a BM25 length-normalisation
artifact: section-unit MaxSim **38.60** vs pooled page **39.78** (−1.18).

**Where the reasoning went wrong.** ColEmbed's late interaction works because the
model is *trained* with a MaxSim objective. Bolting MaxSim onto
`text-embedding-3-small`, trained for pooled-sequence similarity, is a different
operation with no reason to work. The paper's result does not transfer to an
off-the-shelf bi-encoder.

### Block type does not discriminate gold

Share of pages containing at least one block of each type, gold vs non-gold
(962 gold / 712 non-gold pages):

| type | gold | non-gold | ratio |
|---|---|---|---|
| Table | 20.2% | 19.5% | 1.03 |
| Figure | 63.4% | 71.2% | 0.89 |
| EquationBlock | 20.0% | 24.4% | 0.82 |
| SectionHeader | 39.5% | 49.2% | 0.80 |

No type is enriched in gold. The `boost_fields {title: 2.0, figures: 0.7}`
weights declared in `chunking_embedding`'s `hybrid_default` profile have **no
empirical basis on this benchmark** — they should not be implemented on the
strength of intuition.

### The pattern across all four negative results

Localise → lose. Aggregate → win.

| intervention | direction | result |
|---|---|---|
| DCW centroid removal (§8) | localise | −5.73 / −9.50 |
| sub-page MaxSim | localise | −1.18 to −11.16 |
| type-aware boosting | localise | no basis |
| page pooling (status quo) | aggregate | best single-page scorer |
| SEP file aggregation (§7) | aggregate | +1.4 to +2.2 |

Consistent with the corpus: **962 of 1,674 physics pages (57%) are gold for some
query**, 7.21 gold pages per query, dispersed rather than contiguous. These are
*topical* queries over a small topical corpus — evidence is diffuse, not
localised. This is not needle-in-a-haystack retrieval, and methods designed for
that lose here.

The one localising method that *does* win is the cross-encoder (+5.08), and the
distinction is instructive: it is **trained** for query–document interaction
rather than assembled from a pooled embedder at inference time.

Reproduce:
```bash
python research/experiments/vidore_granularity_test.py
python research/experiments/vidore_sep_kdl.py
```

---

## 10. Results table — SEP standalone vs SEP on light preparation (2026-08-25)

Same 42 physics PDFs, same gold, same 302 French queries. Two preparation arms:

- **KDL** — the accurate parse (GPU, vLLM), what the pipeline ships
- **light prep** — `PdfInspectorPageParser` native text only, CPU, the light
  preparation stage of the on-demand branch

SEP config unchanged from §7 (w=2, γ=0.5, β=0.75, top-m=3); λ reported as a
curve, not tuned. Paired permutation, 10,000 resamples, vs each arm's own α=0.7.

### KDL — 1,674 pages, mean 1,330 chars/page

| arm | NDCG@10 | R@10 | delta | p |
|---|---|---|---|---|
| bm25 | 37.46 | 39.71 | | |
| dense | 39.78 | 42.38 | | |
| α=0.7 | 43.03 | 46.49 | | |
| **α=0.7 + SEP (λ=0.5)** | **44.75** | **47.42** | **+1.72** | 0.0119 |
| α=0.7 + SEP (λ=0.6) | 44.50 | 47.28 | +1.47 | 0.0085 |
| α=0.7 + SEP (λ=0.7) | 44.11 | 47.21 | +1.08 | 0.0167 |

### Light preparation — 1,674 pages, mean 623 chars/page, 12s CPU

| arm | NDCG@10 | R@10 | delta | p |
|---|---|---|---|---|
| bm25 | 36.62 | 37.92 | | |
| dense | 40.70 | 44.34 | | |
| α=0.7 | 43.09 | 46.29 | | |
| **α=0.7 + SEP (λ=0.5)** | **44.62** | **47.05** | **+1.53** | 0.0342 |
| α=0.7 + SEP (λ=0.6) | 44.60 | 47.00 | +1.51 | 0.0120 |
| α=0.7 + SEP (λ=0.7) | 44.00 | 46.80 | +0.91 | 0.0692 n.s. |

### The finding: for *retrieval*, KDL buys nothing over light preparation

Paired, same queries:

| comparison | light prep | KDL | delta | p | |
|---|---|---|---|---|---|
| α=0.7 | 43.09 | 43.03 | +0.07 | 0.9447 | indistinguishable |
| α=0.7 + SEP | 44.62 | 44.75 | −0.13 | 0.8951 | indistinguishable |

Light preparation extracts **half the text** (623 vs 1,330 chars/page) in
**12 seconds of CPU** (139.5 pages/s) against a GPU vLLM run, and ranks exactly
as well. The legs differ in the expected directions — light prep's BM25 is worse
(36.62 vs 37.46, less text to match) while its dense leg is *better* (40.70 vs
39.78, shorter cleaner text embeds better) — and the fusion cancels the two out.

This does **not** say KDL is worthless. It says KDL's extra content — tables,
figures, formulas, layout — does not change *which pages rank*. Its value, if
any, is downstream in generation, where a table's contents matter for answering
even though they did not matter for finding the page. That is untested here and
should not be assumed in either direction.

SEP holds on both arms (+1.72 / +1.53), which is expected: it never reads text,
only `file#page` structure and pool scores.

### Alignment note

`pdf-inspector` page numbers are 1-based; KDL and the ViDoRe gold are 0-based.
The −1 offset is *verified* by exact unit-id overlap with the KDL parse
(1,674/1,674), not assumed — checking overlap against gold alone is not
discriminating, since both offsets yield 962 matching ids by set membership
while pointing at different physical pages.

Reproduce: `python research/experiments/vidore_sep_results.py`

---

## 11. Correction — SEP against the *production* baseline (2026-08-25)

§10 measured SEP against a page-level arm (43.03). That is **not** the shipped
configuration. The CSV's "Baseline Legacy" is KDL → **fixed_512/128 chunking** →
MaxP to pages → hybrid α=0.7, and it records **44.2 / 47.47** — i.e. 1.2 NDCG
*above* the arm SEP was being credited against. Re-measured properly:

| arm | NDCG@10 | R@10 | delta | p |
|---|---|---|---|---|
| Baseline Legacy reproduction | 43.86 | 46.73 | | |
| **+ SEP (λ=0.5)** | **46.27** | **48.88** | **+2.41** | 0.0004 |
| + SEP (λ=0.6) | 45.94 | 47.91 | +2.08 | 0.0002 |
| + SEP (λ=0.7) | 45.27 | 47.33 | +1.41 | 0.0037 |
| + SEP (λ=0.8) | 44.94 | 47.21 | +1.08 | 0.0100 |

The reproduction lands at 43.86 / 46.73 against the recorded 44.2 / 47.47 —
within ~0.35 NDCG, attributable to KDL run variation or the "lake-subset"
retrieval scope. Close enough to treat as faithful, but the gap should be
closed before this row is quoted as production.

**SEP is additive to chunking, not a substitute for it.** The gain against the
production baseline (+2.41) is *larger* than against the page-level arm (+1.72),
so the earlier concern that SEP merely recovers what fixed-512 chunking provides
is refuted.

### Standing correction on "~50"

An earlier note projected "SEP (+1.7) and rerank (+5) moves NDCG@10 to ~50."
That was **arithmetic on two gains never measured together** and should not have
been stated as a number. The SEP × rerank composition is still untested. What is
measured is 44.2 → **46.27**, i.e. **+2.07 over the recorded production number**.
Whether reranking stacks additively on top of that is exactly the open
experiment, and the diagnostic gives reason to think it might (SEP changes which
20 pages reach the reranker, and same-file candidates are gold 7.83% vs 0.46%
cross-file) — but *might* is not a result.

### Bug found while reproducing

`fixed_overlap` returns `Span` as a plain `(start, end)` **tuple**, not an object
with `.start`/`.end`. A `hasattr(sp, "start")` guard silently fell through to
`str(sp)`, embedding the literal text `"(0, 3985)"` as every chunk and producing
NDCG 0.46. Worth knowing for any future code consuming the chunker directly.

Reproduce: `python research/experiments/vidore_prod_baseline_sep.py`

---

## 12. Signal headroom scan — where the unexploited signal actually is (2026-08-25)

SEP tunes a coarse signal (file structure) that the fusion partly captures
already, which is why it lands at +2.4. This scan asks a different question:
across the signals we do *not* use, which has the most headroom? Oracles are
never results — they are upper bounds that say whether a predictor is worth
building.

### Query-adaptive fusion is the largest unexploited signal found

α is fixed at 0.7 for every query. It should not be:

| | NDCG@10 |
|---|---|
| production, fixed α=0.7 | 43.03 |
| best fixed α (0.5) | 43.56 |
| **oracle per-query α** | **52.16** |
| oracle over just {0.0, 0.7, 1.0} | 50.52 |

**Headroom +8.60** over the best fixed α — roughly 4× what SEP delivers. And
**113 of 302 queries (37%) are best served by pure BM25 (α=0.0)** while 30 want
pure dense; we force a dense-dominant blend on all of them.

**The headroom is real, not winner's curse.** Picking the max over 11 noisy
options inflates, so: give each query the best α *of a randomly chosen other
query*. That scores **39.87** (sd 0.58) — worse than any fixed α — and random α
scores 41.14. If the oracle were selection noise, the shuffled version would
match it. It does not, by 12 points. The per-query choice is genuinely
query-specific.

The gain is also concentrated: 66% of it sits in 50 queries, and the 43 queries
where production scores exactly 0 have a mean oracle gain of 9.7.

### But nothing cheap predicts it

Seven features, all effectively uncorrelated with the best α:

| feature | pearson r |
|---|---|
| query length | +0.128 |
| has interrogative word | +0.094 |
| dense max score | +0.061 |
| ends with "?" | +0.059 |
| bm25 score gap (relative) | +0.054 |
| bm25/dense top-10 overlap | +0.049 |
| **stopword ratio** | **+0.013** |

The stopword hypothesis deserves a specific retraction. Eyeballing five samples
per group suggested BM25-favouring queries were terse keyword strings
(*"valeur commutateur [X, P] oscillateur harmonique quantique"*) and
dense-favouring ones were full questions. Measured over all 302, function-word
density is **0.372 vs 0.380** — indistinguishable. That was confirmation bias
from a handful of examples. A held-out router built on it returns −0.15 / +0.47
against production, i.e. nothing.

### Feeding the reranker the union instead of the fusion — refuted

If we cannot decide *a priori* which leg to trust, an appealing move is to stop
discarding the loser's candidates. At matched budget it is worse:

| K | fused α=0.7 recall | union (bm25 K/2 + dense K/2) | delta |
|---|---|---|---|
| 10 | 46.49 | 40.77 | **−5.72** |
| 20 | 58.18 | 53.30 | **−4.87** |

Fusion scores every item with *both* signals; union merely concatenates two
partial views. The complementarity is nonetheless real — each leg exclusively
finds ~11% of gold (bm25-only 10.9%, dense-only 13.6% at K=20) — but fusion is
already the better way to exploit it.

### Standing

A large, verified, query-specific signal (+8.60) with **no known cheap
predictor**. That is a more useful place to be than another +2 on a saturated
signal, but it is not yet a method. Capturing it needs something that models the
query–corpus *interaction* rather than query surface form — which is what a
cross-encoder does, and suggests using one as a **router** rather than only as a
reranker. Untested.

Reproduce: `python research/experiments/vidore_signal_headroom.py`

---

## 13. The French analyzer was rejected under a confound — but it still loses (2026-08-25)

The ledger records the `french` analyzer as +2.71 on BM25 alone but dropped for
−0.13 "on the fusion". That fusion was **α=0.7 — dense-dominant**, i.e. the exact
setting where a better BM25 leg is suppressed. §12 then showed 37% of queries
want pure BM25. So the rejection was tested under a confound.

**Confirmed.** Production config (KDL + fixed_512/128 + MaxP), french vs plain:

| α | delta | p |
|---|---|---|
| 0.7 — where it was rejected | +0.09 | 0.8808 |
| 0.6 | +0.49 | 0.4509 |
| **0.5** | **+1.65** | **0.0200** |
| 0.4 | +1.24 | 0.1140 |

At α=0.5 the analyzer is worth a significant +1.65. The original −0.13 was an
artifact of the fusion weight, not a property of the analyzer.

Also worth noting: production's α=0.7 is not optimal on this config either —
plain α=0.6 scores 44.16 vs 43.86 at 0.7.

### But it does not survive contact with SEP

Selecting `(analyzer, α, λ)` on one fold and scoring the other is **unstable**:
fold0→fold1 picks `french/0.5/0.7` and returns +0.30 (n.s.); fold1→fold0 picks
`plain/0.6/0.5` and returns +2.24 (p=0.020). A single held-out split cannot pick
the configuration.

Ranking instead by each config's **worst** fold — which is what guards against
fold-luck — every robust configuration is `plain`:

| config | fold0 | fold1 | all | vs production | p |
|---|---|---|---|---|---|
| **plain, α=0.7, SEP λ=0.5** | 46.22 | 46.33 | **46.27** | **+2.41** | 0.0004 |
| plain, α=0.5, SEP λ=0.5 | 46.63 | 46.16 | 46.41 | +2.55 | 0.0023 |
| plain, α=0.6, SEP λ=0.6 | 46.47 | 46.11 | 46.30 | +2.44 | 0.0003 |
| plain, α=0.6, SEP λ=0.5 | 46.09 | 46.39 | 46.23 | +2.38 | 0.0008 |

French does not appear. SEP and the French analyzer capture **overlapping**
signal — both recover lexically-matched pages the dense-dominant fusion buries —
and SEP captures more of it. Stacking them is worse than SEP alone.

### Standing result

**`plain`, α=0.7, SEP λ=0.5 → NDCG@10 46.27, R@10 48.88**, against the
production baseline's 43.86/46.73 reproduction and the recorded 44.2/47.47.
Fold0 46.22 / fold1 46.33 — the two halves agree to 0.11, which is the strongest
evidence available here that this is not fold-luck.

Reproduce: `python research/experiments/vidore_french_alpha_confound.py`

---

## 14. DAT — dynamic alpha tuning: ceiling test (2026-08-25)

§12 found +8.60 of real per-query α headroom and no predictor, having tried
seven **query-surface** features (all r < 0.13). The literature says why that
failed. *DAT: Dynamic Alpha Tuning for Hybrid Retrieval in RAG*
([arXiv 2503.23013](https://arxiv.org/abs/2503.23013)) uses a **post-retrieval**
signal instead: an LLM scores the effectiveness of the **top-1 result from each
leg**, and those two scores are normalised into α. The information is not in the
query, it is in how well each leg actually did.

### Ceiling, using a perfect judge (true qrels grade of each leg's top-1)

| | NDCG@10 | vs production |
|---|---|---|
| production α=0.7 | 43.86 | |
| best fixed α (0.6) | 44.16 | +0.30 |
| **DAT, perfect judge** | **47.13** | **+3.27** (p=0.0003) |
| SEP alone | 46.27 | +2.41 |
| **DAT + SEP (λ=0.5)** | **48.87** | **+5.01** (p=0.0001) |

DAT recovers ~35% of the oracle-α headroom on its own, and **adds +2.60 on top
of SEP** (p=0.0007) — the two are complementary, not redundant. SEP reorders
within a fixed fusion; DAT changes the fusion itself.

### Controls

- **Not the fallback.** α falls back to 0.5 when the judge cannot separate the
  legs, and 0.5/0.6 already beat production's 0.7. Against the *best fixed* α
  DAT is still +2.94 (p=0.0001).
- **Not winner's curse.** Giving each query *another query's* judge scores
  yields **41.94** (sd 0.74) versus 47.10 for the true scores — below every
  fixed α. The signal is genuinely query-specific.
- **The judge is idle on half the queries.** Both legs' top-1 are non-gold on
  144/302 (48%), and the two agree on 199/302 (66%), so α falls back to 0.5
  there. The whole +3.27 comes from differentiating ~34% of queries.
- Graded (0/1/2) and binary judges are indistinguishable (47.10 vs 47.13), so
  the judge only needs to answer "is this relevant", not "how relevant".

### Status: this is a CEILING, not a result

It substitutes ground-truth relevance for the LLM judge. A real judge will be
worse — how much worse is exactly what determines whether DAT ships. What the
ceiling establishes is that the approach is **worth the API spend to test**
(one call per query), and that its target is ~48.9 in combination with SEP.

Reproduce: `python research/experiments/vidore_dat_ceiling.py`

---

## 15. DAT with a real LLM judge — the ceiling does not materialise (2026-08-25)

§14 put DAT's ceiling at +3.27 alone and +5.01 with SEP. Measured with actual
judges on the production config (302 queries, one call each):

| arm | NDCG@10 | R@10 | vs production | p |
|---|---|---|---|---|
| production α=0.7 | 43.86 | 46.73 | | |
| DAT, gpt-4o-mini, graded 0–5 | 44.07 | 46.91 | +0.21 | 0.718 n.s. |
| DAT, gpt-4o-mini, binary | 43.10 | 45.40 | −0.76 | 0.298 n.s. |
| DAT, gpt-4o, binary | 44.57 | 45.91 | +0.71 | 0.365 n.s. |
| **SEP only** | **46.27** | **48.88** | **+2.41** | **0.0004** |
| DAT (gpt-4o) + SEP | 46.50 | 48.45 | +2.64 | 0.0032 |

DAT recovers ~6–22% of its ceiling and adds only +0.23 on top of SEP (against
+2.60 for a perfect judge), with *lower* recall. It does not ship.

### Why: the judge is at chance

Accuracy of the binary relevance verdict on each leg's top-1, against qrels:

| judge | accuracy | precision | recall | says YES |
|---|---|---|---|---|
| gpt-4o-mini | 50.2% | 40.6% | 88.7% | 77.2% |
| gpt-4o | 52.2% | 41.8% | 91.1% | 76.8% |

Base rate of a top-1 actually being relevant is **35.3%**, and both judges answer
YES on ~77% of cases. They affirm nearly everything, so the verdict carries
almost no information. This is a judge *capability* limit on French physics
prose, not a prompting problem — gpt-4o buys 2 points of accuracy over
gpt-4o-mini and neither is usefully above chance.

The two failure modes follow directly from that. A graded 0–5 judge hedges both
passages into the mid-range, α collapses to ~0.5 and nothing changes
(distribution `{0.4:34, 0.5:85, 0.6:80, 0.7:48, 0.8:49}`, never an extreme).
A binary judge produces correctly extreme α (`{0.0:50, 0.5:186, 1.0:66}`) but on
near-random verdicts, which is actively harmful.

**The ceiling remains valid and worth revisiting** with a judge that can actually
discriminate — a cross-encoder relevance model rather than a generative LLM is
the obvious candidate, since that is precisely what cross-encoders are trained
for and they already deliver +5.08 here as rerankers.

### Bug worth recording

The judgement checkpoint was keyed by `mode` only, so running a second judge
silently reloaded the first judge's verdicts and reported them as the new
model's result — an exactly-identical results table and "judge done in 0s" were
the only tells. Now keyed by `mode + judge`.

Reproduce:
```bash
python research/experiments/vidore_dat_llm.py --mode binary --judge openai/gpt-4o
```

---

## 16. Solution row — SEP end-to-end, ViDoRe V3 physics (2026-08-25)

Production config (KDL + pdf-inspector, fixed_512/128, MaxP, α=0.7), 302 French
questions, top-10 full-page context, DeepSeek V4 Flash generation, GPT-4o judge.
Both arms run over the same questions so every delta is paired.

| | NDCG@10 | R@10 | Correct_only | Correct+partial |
|---|---|---|---|---|
| CSV Baseline Legacy (recorded) | 44.2 | 47.47 | 51.66 | 90.4 |
| baseline reproduction | 43.86 | 46.73 | 50.17 | 91.03 |
| **+ SEP (λ=0.5)** | **46.27** | **48.88** | **51.83** | **89.70** |

Paired permutation, 10,000 resamples:

| metric | delta | p | better/worse/tied |
|---|---|---|---|
| NDCG@10 | **+2.41** | **0.0004** | 127/80/95 |
| Correct_only | +1.33 | 0.6799 | 28/24/248 |
| Correct+partial | −1.33 | 0.3855 | 4/8/288 |

The E2E path reproduces the recorded row closely (50.17/91.03 vs 51.66/90.4),
so the QA harness is sound.

### The retrieval gain does not reach QA

Both QA deltas are non-significant, and they point in opposite directions. This
is the predicted outcome, not a surprise: §1 measured the oracle-vs-retrieved QA
gap at ~3.6pp credited, because gold is redundant (7.21 gold pages available per
query, ~2 needed to answer). Reordering pages that were *already retrieved*
mostly reshuffles evidence the generator could already use — 248 of 300
questions do not change label at all on correct_only, and 288 of 300 do not
change on credited.

**Read the row accordingly: SEP is a retrieval result, not an end-to-end one.**
Quoting +2.41 NDCG@10 is supported; quoting a QA improvement is not.

### Cost

| | baseline | + SEP |
|---|---|---|
| online retrieval latency | 18.5 ms/query | **18.8 ms/query** |
| offline parsing / chunk+embed+index | unchanged | unchanged |
| API calls, GPU, index changes | — | **none** |

SEP is arithmetic over scores already computed: +0.3 ms/query and nothing else.

Reproduce: `python research/experiments/vidore_sep_e2e.py`

---

## 17. The generator, not retrieval, is where the QA number is (2026-08-25)

§16 showed SEP's +2.41 NDCG@10 producing no QA movement. The obvious next
question — asked far too late in this work — is what *does* move QA. Holding
retrieval **byte-identical** (α=0.7 baseline ranking for every arm) and swapping
only the generator, n=120 physics questions, GPT-4o judge:

| generator | Correct_only | Correct+partial |
|---|---|---|
| deepseek-v4-flash (current production) | 44.54 | 86.55 |
| **openai/gpt-5.2** | **71.67** | **95.83** |
| anthropic/claude-sonnet-4.5 | 63.33 | 86.67 |

Paired permutation vs deepseek, 10,000 resamples, same questions:

| | Correct_only | Correct+partial |
|---|---|---|
| gpt-5.2 | **+27.73** (p=0.0001) | **+9.24** (p=0.0009) |
| claude-sonnet-4.5 | +19.33 (p=0.0002) | +0.84 (n.s.) |

### Scale, against every retrieval change measured this session

| change | Correct_only |
|---|---|
| SEP (+2.41 NDCG@10, p=0.0004) | +1.33 **n.s.** |
| **generator swap** | **+27.73, p=0.0001** |

Roughly **20× the effect, from a one-line config change, with retrieval
untouched.** gpt-5.2's 71.67 also lands on the paper's 71.2% (Gemini 3 Pro),
confirming the entire remaining QA gap was the generator.

This was foreseeable from evidence already in this ledger: the oracle-retrieval
QA ceiling was 55.6% with gpt-4o-mini against the paper's 71.2%, i.e. a ~15
point discount that no retrieval work could close, and the oracle-vs-retrieved
gap was only ~3.6pp. Both numbers were recorded before any of the retrieval
experiments in §7–§15 were run.

**Methodological lesson worth keeping:** measure the ceiling of each component
before optimising any of them. Retrieval had ~3.6pp of end-to-end headroom and
consumed the session; generation had ~27pp and took one experiment.

### Status: n=120, full run blocked

The full 302-query run aborted — the OpenRouter key hit its **monthly limit**
(244/302 calls returned HTTP 403). The n=120 sweep above completed before that
and is a clean paired comparison. Numbers should be refreshed at n=302 once the
key is topped up; the checkpoints are resumable and keyed by generator.

Reproduce (needs OpenRouter budget):
```bash
python research/experiments/vidore_sep_e2e.py --generator openai/gpt-5.2
```

---

## 18. ColQwen2 — the first visual arm that actually helps (2026-08-26)

§7-8 of `docs/sep_giai_thich.md` and the CLIP arm (§ above, `vidore_visual_eval.py`)
both found visual signal too weak to matter: CLIP ViT-B/32 scored 4.45 NDCG@10 and
fusion never beat text at any weight. That used a natural-image encoder at 224px on
dense French text pages — the honest prediction going in was "not visual is useless,
CLIP-at-224px is useless here."

Ran `vidore/colqwen2-v1.0` (Qwen2-VL-2B backbone, late-interaction/MaxSim, real
document-VLM) on the same 1,674 rendered pages, via `ColQwen2_visual_arm_physics.ipynb`
on Colab T4 + `research/experiments/physics_colqwen_eval.py` locally (no GPU, no API —
scores exported as a flat 302×1,674 matrix). Text side is `physics_served_pool.json`'s
cached α=0.7 order, same convention as the local rerank arm.

| arm | NDCG@10 | R@10 | Δ vs text | p |
|---|---:|---:|---:|---:|
| text α=0.7 (served pool) | 44.15 | 47.58 | — | — |
| ColQwen2 visual-only | 45.70 | 47.99 | +1.54 | 0.2293 n.s. |
| **+ ColQwen2 fusion, best w≈0.7** | **47.37** | — | **+3.22** | **0.0014** |

Fusion sweep, w_visual from 0.1 to 1.0 — the band 0.4–0.8 all land 46.9–47.4, all
significant (p ≤ 0.005 except the tails); 0.9 is marginal (p=0.030); 1.0 (visual only)
is the n.s. row above:

| w_visual | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|---|---|---|---|---|
| NDCG@10 | 44.94 | 45.60 | 46.12 | 46.87 | 47.12 | 47.08 | 47.37 | 47.16 | 46.69 |
| p | 0.0059 | 0.0017 | 0.0005 | 0.0005 | 0.0006 | 0.0015 | 0.0014 | 0.0046 | 0.0300 |

**Read it as a band (w≈0.5–0.8, NDCG@10≈47.1–47.4), not an argmax at 0.7** — same
convention as the SEP λ sweep. This is the largest verified retrieval-only gain in
this ledger: bigger than SEP (+2.41) alone, and this is on top of α=0.7 with no
reranker in the loop at all.

### Standalone visual is not significant — the win is specifically fusion

ColQwen2 alone (45.70) is numerically above text (44.15) but p=0.229. Do not quote
"ColQwen2 beats text." What is supported: **fused, it moves the number**; alone, it
does not clear significance on 302 queries. This matters for cost — you cannot ship
visual-only and expect this gain; the text leg is still required.

### Complementarity vs the CLIP arm — night and day

| | CLIP (224px) | ColQwen2 |
|---|---:|---:|
| gold@10 found ONLY by visual | 1.17% | 7.92% |
| gold@10 found ONLY by text | 42.86% | 7.52% |

CLIP's only-text share (42.86%) says it almost never contributes anything text
doesn't already have. ColQwen2's only-visual (7.92%) is close to only-text (7.52%) —
genuinely complementary coverage, not a weak echo of the text leg.

### It does not surgically fix the figure-dependent 473/516-page group

Of the gold pages text misses entirely (rank≥100 in the served pool — 516 here,
vs the 473 counted in `vidore_why_missed.py` from a separate BM25+dense
reconstruction over KDL text; same phenomenon, different snapshot):

| | recovered |
|---|---:|
| in ColQwen2 visual top-10 | 5 / 516 (1.0%) |
| in ColQwen2 visual top-100 | 170 / 516 (32.9%) |

ColQwen2 does **not** promote these specific pages to the top of its own ranking —
only 1% reach its top-10. But a third of them are somewhere in its top-100, which is
real signal a reranker could exploit; it is just not concentrated enough to act as a
standalone fix for the figure-dependent group. The overall +3.22 gain is coming from
broader reordering across all 302 queries, not from surgically patching this group.

### Standing

The first visual experiment this session that clears significance. Composes with
SEP untested (both reorder the served pool independently); composing with the
Voyage/local reranker also untested. Both are natural next steps and neither needs
new infrastructure — same served-pool JSON, same permutation harness.

Reproduce: `python research/experiments/physics_colqwen_eval.py` (needs
`data/work/vidore_physics_colqwen/{physics_colqwen_scores.npy,_keys.json,_qids.json}`
from the Colab notebook — no GPU, no API for this step).

---

## 19. Local reranker — refuted, and it clarifies *why* rerank works (2026-08-27)

§1b-iii measured Voyage `rerank-2.5` at +5.08 (depth-20), capped there by the free
tier's 3 RPM / 10K TPM. `Xenova/bge-reranker-base` via ONNX Runtime on CPU (same
torch-2.2.2 workaround as the CLIP and ColQwen2 arms — no API, no GPU) was meant to
extend that lever past the API cap, matching the priority list in
`docs/phan_tich_first_principles.md` item 2. `research/experiments/physics_rerank_local.py`.

| depth | NDCG@10 | Δ | p | better/worse/tied |
|---|---:|---:|---:|---|
| served α=0.7 | 44.15 | — | — | — |
| 10 | 43.21 | −0.95 | 0.2979 n.s. | 100/112/90 |
| 20 (same depth Voyage was measured at) | 41.26 | **−2.89** | **0.0181** | 88/143/71 |
| 50 | 38.82 | −5.34 | 0.0006 | 85/152/65 |
| 100 | 37.56 | **−6.59** | **0.0001** | 76/161/65 |

**Not a depth problem — a model problem.** At the exact depth Voyage was measured
(20), Voyage gets +5.08 and this reranker gets −2.89: opposite sign, same setting.
Damage grows monotonically with depth because the reranker's own bad calls
compound — more candidates reordered, more chances to promote a false positive
into the top-10 that the α=0.7 fusion had correctly excluded. This is computed as
a free post-hoc slice of the depth-100 checkpoint (`physics_rerank_local_depth_sweep.py`),
not a re-run, so all four rows share one scoring pass.

### What this changes in the priority list

`phan_tich_first_principles.md` item 2 read "a trained ranker helps, at ceiling
71.68" — true, but incomplete: **it has to be a *good* trained ranker.**
bge-reranker-base (278M, general multilingual) is trained, and it makes things
markedly worse. The one working example remains Voyage rerank-2.5, a much
stronger commercial model. The lesson is not "any cross-encoder helps" — it is
"a weak cross-encoder actively hurts more than doing nothing," which raises the
bar for what a free/local substitute would need to clear before it is worth
depth-extending.

### Standing

Refuted as a way to extend the rerank lever for free. Worth trying a stronger
open reranker (e.g. `BAAI/bge-reranker-v2-m3`, larger, more recent) if the ONNX
export exists, but do not assume "trained beats untrained" transfers without
re-measuring — this result is exactly the counter-example.

Reproduce: `python research/experiments/physics_rerank_local.py --depth 100`,
then `python research/experiments/physics_rerank_local_depth_sweep.py` for the
depth breakdown from the same cached scores.

---

## 20. Stacking the three verified levers — Voyage alone is the ceiling (2026-08-27)

handoff §5 priority 2: SEP (+2.41), Voyage rerank-2.5 (+5.08) and ColQwen2
fusion (+3.22) were each measured against a slightly different baseline and
never composed. `physics_stack.py` runs all of them on the **one**
`vidore_page` α=0.7 pool (44.15), every arm scored on the same qrels, paired
permutation, 10,000 resamples. Everything cached — no API, no GPU. Compose
order is retrieve → rerank → structural/visual reorder: Voyage is applied to
the base-pool top-20 (fully cached, reproduces §1b-iii's 49.23 exactly) and
SEP / ColQwen2 reorder the result. Fusion arms reported as a `w`-band, not the
argmax (§1b-ii/§18 convention).

| arm | NDCG@10 | vs | Δ | p |
|---|---:|---|---:|---:|
| baseline α=0.7 | 44.15 | — | — | — |
| SEP (λ=0.5) | 46.17 | baseline | +2.02 | 0.0026 |
| ColQwen2 fusion (w 0.2–0.8) | 44.7–47.5 | baseline | up to +3.38 | ≤0.0001 at w≥0.4 |
| **Voyage rerank-2.5 top-20** | **49.23** | baseline | **+5.07** | 0.0001 |
| SEP + ColQwen2 (w 0.2–0.8) | 47.1–48.3 | baseline | +2.9…+4.1 | ≤0.0001 |
| **Voyage + ColQwen2 (w 0.2–0.8)** | **47.5–49.8** | **Voyage** | **+0.6 … −1.8** | **0.34 → n.s. at every w** |
| Voyage + SEP + ColQwen2 (w 0.2–0.8) | 47.1–48.3 | Voyage | −1.0…−2.2 | n.s. → 0.04 |

**Nothing stacks onto the reranker.** Voyage rerank alone scores 49.23. Fusing
ColQwen2 on top of the reranked order does **not** significantly beat it —
peak +0.61 at w=0.3 (p=0.34), negative for w≥0.4. SEP+ColQwen2 without the
reranker (48.3) is also below Voyage alone. All three levers fix the same
failure (§7: gold file in top-10, wrong pages within it ranked); once the
cross-encoder has reordered the top-20 on token-level interaction, the
structural prior and the ColQwen2-2B visual arm have nothing left to add.

The one informative signal in the ColQwen2 sweep: **the useful visual weight
collapses once Voyage runs** — peak w moves 0.7 (over the raw fusion, +3.38)
→ 0.3 (over the reranked order, +0.61 n.s.). The visual arm was mostly
recovering ordering the reranker recovers better.

**SEP after reranking is not cleanly measurable from cache** and the rows above
carry a `[not clean]` flag in the script. SEP's file aggregate is built for a
real score decay; fed a rank proxy of the reranked order (a linear ramp) it
mis-fires. Direction (SEP does not help post-rerank) is expected — §7's
mechanism needs a badly-ordered pool — but do not quote the −7 delta. A clean
test needs the reranker's real scores carried through, or a fresh Voyage run
over the SEP-reordered pool (the aborted `_scores_sep.json` covers 31/302).

**Ceiling from the current toolkit: 49.23**, i.e. Voyage rerank and nothing
else. For reference, the ViDoRe V3 leaderboard SOTA on physics is
nemotron-colembed-vl-8b-v2 at **50.84** ([arXiv 2602.03992](https://arxiv.org/abs/2602.03992)
Table 2) — but that is an average over 6 query languages and our ladder is
French-only, so it is an approximate anchor, not a like-for-like target (cf.
§4). Any material gain above ~49 needs a stronger *component* — see
`docs/retrieval_research_plan.md`.

Reproduce: `python research/experiments/physics_stack.py`

---

## 21. SEP and ColQwen2 on the KDL / light-prep pools — the fillable table rows (2026-08-27)

§18/§20 measured Voyage and ColQwen2 on `vidore_page` (ViDoRe V3's own supplied
text, baseline 44.15) — an arm that uses **none of our parsing**. The results
table needs rows on the pool the pipeline actually ships: **KDL → fixed_512/128
chunks → MaxP to pages → hybrid α=0.7** (the CSV "Baseline Legacy" recipe,
43.86/46.73 reproduction) and the light-preparation equivalent. `physics_kdl_arms.py`,
all cached — no API, no GPU. Paired permutation, 10,000 resamples.

### KDL (Baseline Legacy recipe) — baseline 43.86 / 46.73

| arm | NDCG@10 | R@10 | Δ | p | vs |
|---|---:|---:|---:|---:|---|
| baseline α=0.7 | 43.86 | 46.73 | — | — | — |
| + SEP (λ=0.5) | 46.27 | 48.88 | +2.41 | 0.0004 | baseline |
| + ColQwen2 fusion (w 0.6–0.8) | 47.4–47.5 | ~50.0 | +3.5…+3.7 | ≤0.0005 | baseline |
| **+ SEP + ColQwen2 (w 0.6–0.7)** | **48.2–48.4** | ~50.6 | **+1.95…+2.08** | 0.02–0.03 | **SEP** |

Total for the SEP+ColQwen2 stack vs baseline: **43.86 → ~48.3 (+4.5)**.

### light-prep (pdf-inspector native text, fixed_512/128) — baseline 43.02 / 45.72

| arm | NDCG@10 | R@10 | Δ | p | vs |
|---|---:|---:|---:|---:|---|
| baseline α=0.7 | 43.02 | 45.72 | — | — | — |
| + SEP (λ=0.5) | 45.56 | 48.08 | +2.54 | 0.0001 | baseline |
| + ColQwen2 fusion (w 0.7–0.8) | 47.2–47.6 | ~50.2 | +4.2…+4.6 | 0.0001 | baseline |
| **+ SEP + ColQwen2 (w=0.6)** | **48.45** | 51.16 | **+2.89** | 0.0018 | **SEP** |

Total for the stack vs baseline: **43.02 → 48.45 (+5.4)**.

### What this establishes

1. **SEP + ColQwen2 *does* stack when there is no reranker** — +2 on top of SEP,
   significant, on both parses. Opposite of §20, where SEP + ColQwen2 *after
   Voyage* added nothing. Reconciles cleanly: on the raw fusion order the
   within-file ordering is still bad, so SEP and the visual arm each have work to
   do; once the cross-encoder has done that work they do not.
2. **The free stack (SEP + ColQwen2, ~48.3 on KDL) gets within ~1 point of
   Voyage-alone (49.23)** — no API, one-time GPU pass for the visual index.
3. **KDL retrieval buys ~0.8 NDCG over light-prep at the fusion** (43.86 vs
   43.02); the levers behave the same on both, consistent with §10 (KDL's extra
   content changes generation, not which pages rank).

### Still missing: Voyage on the KDL pool

Voyage rerank was only ever run on `vidore_page`. On the KDL pool it needs a
fresh ~100 min API run (free-tier 3 RPM). `physics_kdl_arms.py` dumps
`physics_KDL_pool.json` / `physics_light_prep_pool.json` as candidate lists;
feed `physics_rerank_voyage.py --pool ... --texts ...` when budget allows.

### QA (end-to-end) still only measured for SEP

§16: SEP E2E on this exact KDL pool = 51.83 correct_only / 89.70 credited (both
n.s. vs baseline). ColQwen2 fusion and the SEP+ColQwen2 stack have **no QA
measurement**.

Reproduce: `python research/experiments/physics_kdl_arms.py`
