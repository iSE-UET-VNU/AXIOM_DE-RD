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
