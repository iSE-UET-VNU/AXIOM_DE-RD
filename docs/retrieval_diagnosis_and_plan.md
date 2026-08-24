# Why our retrieval underperforms, and what to do about it

Companion to [vidore_v3_results.md](vidore_v3_results.md). Every number below is
either already in that ledger, computed here from data on disk at zero API cost,
or cited from the ViDoRe V3 paper / model cards.

Reproduce with:

    python research/experiments/physics_recall_curve.py     # recall@k curve
    python research/experiments/physics_rerank_ceiling.py   # oracle-rerank ceiling
    python research/experiments/physics_gold_buckets.py     # QA by delivered gold

---

## 1. The diagnosis: we retrieve the right pages and rank them badly

The intuitive explanation for a 44.15 NDCG@10 was "our dense leg is
`openai/text-embedding-3-small`, which is weak and English-centric, against the
paper's Jina-v4 and Qwen3-8B." **The data does not support that.** A recall curve
over the existing cached embeddings settles it:

| index | arm | R@5 | R@10 | R@20 | R@50 | R@100 | R@200 | any-gold@10 | any-gold@100 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vidore_page | bm25 | 30.1 | 39.6 | 50.8 | 62.8 | 69.4 | 77.5 | 78.5 | 95.0 |
| vidore_page | dense | 32.3 | 44.6 | 55.7 | 72.5 | 83.1 | 90.2 | 83.4 | 98.3 |
| vidore_page | rrf | 34.7 | 45.5 | 58.8 | 72.3 | 81.4 | 89.2 | 84.8 | 97.7 |
| **vidore_page** | **α=0.7** | 34.4 | **47.5** | 59.9 | 74.4 | **84.2** | 90.2 | 85.8 | **98.3** |
| chandra_page | α=0.7 | 34.8 | 47.0 | 58.4 | 74.6 | 84.5 | 90.6 | 85.1 | 98.7 |

`any-gold@100 = 98.3%`. For all but ~5 of 302 queries, at least one gold page is
already inside our top-100. 84% of *all* gold pages are there. The first-stage
retriever is doing its job — **the loss is ordering, not candidate generation.**

### How much is recoverable by reordering alone

Fixing the candidate pool at the production DEPTH=100 α=0.7 fusion and
oracle-reordering a prefix of the served ranking (graded, since qrels carry
relevance 1 and 2):

| | NDCG@10 | Δ |
|---|---:|---:|
| served α=0.7 (our number today) | 44.15 | — |
| oracle reorder of served top-10 | 60.49 | +16.34 |
| oracle reorder of served top-20 | 71.68 | +27.52 |
| oracle reorder of served top-50 | **83.88** | **+39.72** |
| oracle reorder of served top-100 | 90.27 | +46.12 |

Read the first row carefully: **reordering only the ten pages we already return
— retrieving nothing new — is worth +16.3 NDCG@10.** The pages are in hand and
in the wrong order.

### The paper found the same thing independently

From the ViDoRe V3 paper's own ablation: adding the textual reranker **zerank-2
to the Jina-v4 text pipeline yields +13.2 average NDCG@10** (to 63.6 avg,
overtaking the visual retriever's 57.6). The visual reranker on a visual pipeline
adds only +0.2, and degrades on 4 of 10 datasets.

So the single largest textual lever in the benchmark's own experiments is exactly
the component we do not have. `src/retrieval/rerank.py` raises
`NotImplementedError`.

**Two independent lines of evidence — our oracle ceiling and the paper's measured
ablation — point at reranking.** That convergence is the finding.

### Where we actually sit against the physics column

**Not like-for-like:** every paper row below is a *single retriever*; ours is a
BM25+dense *fusion*. The table places us in the band, it does not say we beat
Jina-v4.

| System | NDCG@10 | |
|---|---:|---|
| BGE-M3 | 38.3 | paper |
| our BM25 (`analyzer=plain`) | 37.45 | ours |
| BM25S | 39.8 | paper |
| Qwen3-0.6B | 43.8 | paper |
| Jina-v4 | 44.0 | paper |
| **our α=0.7 fusion** | **44.15** | ours |
| Qwen3-8B | 45.8 | paper |

Given that caveat, the honest statement is that our first stage is competitive —
it lands in the same band as the paper's textual retrievers — and that our
pipeline stops one stage too early. The gap is not the retriever.

### Secondary, already-quantified gaps

- **BM25 tokenization.** 37.45 vs BM25S 39.8 on identical inputs; the ledger
  already attributes the −2.3 entirely to missing French stopwords and stemming.
  This is a known-value fix, not an experiment.
- **No visual arm.** Visual retrievers run 43.2–48.5 on physics. Out of scope
  for a text pipeline, but it caps where this line of work can end.
- **α=0.7 was selected on the same 302 queries we report on.** Any new fusion
  weight needs a held-out split or an explicit "tuned on test" label.

---

## 2. Why better retrieval will *not* obviously show up in QA

This is the part worth being careful about, and your instinct was right.

Our measured oracle-minus-retrieved gap is **−1.3pp `correct_only` (n.s.)** and
**−3.64pp `correct_plus_partial` (p=0.030)**. Since no retrieval system can beat
oracle, **that gap is the entire budget available to retrieval work end-to-end.**
Reranking cannot buy more than ~3.6pp credited, no matter how good it gets.

The mechanism is gold-set redundancy: physics queries carry 7.21 gold pages on
average (median 5), retrieved context delivers 2.34 of them, and two of seven
usually suffice. Extra gold pages have sharply diminishing returns.

### But the aggregate hides where the loss lives

Bucketing the 302 answers by how many gold pages retrieval actually delivered:

**ViDoRe text arm** (overall 54.3% correct_only)

| bucket | n | correct_only | +partial |
|---|---:|---:|---:|
| 0 gold | 43 | **34.9%** | **76.7%** |
| 1–2 gold | 157 | 54.1% | 89.2% |
| 3+ gold | 102 | 62.7% | 96.1% |

**chandra2 arm** (overall 56.6%)

| bucket | n | correct_only | +partial |
|---|---:|---:|---:|
| 0 gold | 44 | **29.5%** | **59.1%** |
| 1–2 gold | 153 | 62.7% | 93.5% |
| 3+ gold | 105 | 59.0% | 97.1% |

The ~43 queries (14%) that receive **zero** gold pages answer at 29–35% against a
~55% baseline. That is where all the recoverable QA loss is concentrated.

**Caveat, and it matters:** these buckets are self-selected — queries where
retrieval succeeds are also easier queries — so the gradient is contaminated by
difficulty, not a clean dose-response. The tell is in the chandra2 arm, where
`1–2 gold` (62.7%) scores *higher* than `3+ gold` (59.0%). Non-monotonic, so it
cannot be purely causal.

Treating it as a ceiling rather than a forecast: fixing every zero-gold query
would be worth **+3.2pp (ViDoRe) / +4.6pp (chandra2) `correct_only`**. That
agrees in magnitude with the measured oracle gap (−1.3 / −3.64pp), which is a
useful consistency check — two different routes to "a few points, not twenty."

And the good news from §1: `any-gold@100 = 98.3%` means nearly every one of those
43 queries *does* have a gold page sitting in the top-100. A reranker is exactly
the mechanism that pulls it into the top-10.

### The real QA ceiling is the generator, not retrieval

Our oracle scores 55.6% `correct_only` with gpt-4o-mini against the paper's 71.2%
with Gemini 3 Pro. With full gold context in hand, our generator converts it at a
15-point discount. **No retrieval work closes that.** If the headline QA number is
the goal, the generator is the lever; if a defensible pipeline is the goal,
retrieval is.

### Three measurement caveats to carry forward

1. The paper's 10pp oracle gap is on **hard queries, image context, Gemini 3 Pro**.
   Ours is −1.3pp on the full query mix, text, gpt-4o-mini. Different
   measurements — the difference is not a finding about our pipeline.
2. The paper's headline BM25 figure of 20.3 is a **10-dataset average**, not
   physics. Do not put our 37.45 next to it.
3. **The KDL and chandra2 rows in the LaTeX table are still confounded.** KDL ran
   deepseek-v4-flash, chandra2 ran gpt-4o-mini. They sit adjacent as if
   comparable; they are not. Either re-run one to match or annotate the table.

---

## 3. Experiments, ranked by information per dollar

### Tier 0 — free, no API key

**E0a. French BM25 analyzer. — DONE.** `analyzer=french` in `src/retrieval/sparse.py`:
elision splitting, French stopwords, Snowball stemming. **BM25 37.45 → 40.16
(+2.71)**, matching `bm25s`'s 40.15. The dominant term was not stopwords or
stemming but **elision** (+1.31): `analyze` joined on the apostrophe, so
`l'énergie` was a single token that `énergie` could not match.

The "no prediction for the fused arms" caveat held: **α=0.7 fusion is 44.02 vs
plain's 44.15**, i.e. the +2.71 does not propagate at all. Keep the analyzer for
BM25-only serving; it is not a pipeline win.

**E0b. α on a held-out split. — DONE.** `physics_alpha_holdout.py`, 50/50 by
`sha256(qid)`, both directions. **α=0.7 is not the in-sample optimum** (0.5 is, at
44.71), so the tuned-on-test worry was misplaced. Honest out-of-sample = **43.39**,
a ~0.8 selection cost. The α curve is flat 0.4–0.7 and the fold-selected α swings
0.45 → 0.75 — α is weakly identified; report the band, not the argmax.

### Tier 1 — the main event, ~$0.25

**E1. Cross-encoder reranking of the α=0.7 pool. — DONE. 44.15 → 49.23 (+5.08,
p=0.0001), 146 better / 83 worse / 73 tied.** Voyage `rerank-2.5` over the served
top-20. The diagnosis in §1 is confirmed: reordering pages we already retrieved is
the largest lever in this ladder by a factor of five. Full row and caveats in
[vidore_v3_results.md](vidore_v3_results.md) §1b-iii.

**Two premises in this section were wrong, and both change the work:**

1. `src/retrieval/rerank.py` **does not raise `NotImplementedError`.** It already
   ships `GatewayReranker` (Cohere-shaped `/inference/reranks`) and `LlmReranker`
   (listwise). `src/evaluation/voyage_rerank.py` exists too, wired into
   `run.py --voyage-rerank`. The mechanism was already built.
2. **There is no ZeroEntropy key.** `.env` holds `OPENROUTER_API_KEY`,
   `VOYAGE_API_KEY`, `VLLM_*`. So the arm actually run is **Voyage `rerank-2.5`**,
   a different model — the +8/+14 band below was anchored on zerank-2 over
   Jina-v4 candidates and is **not** a prediction for this arm. Label results by
   the model that ran.

**Depth is 20, not 50.** The Voyage account is free-tier (3 RPM / 10K TPM,
confirmed by probe) and a depth-50 request at full passage length is ~21K tokens —
it cannot fit in one request at any pacing. Depth 20 × 1200 chars ≈ 8K tokens
fits untruncated. This is also `RETRIEVAL_K3`, the depth production already
serves, so it is a pre-registered value rather than one tuned on the 302 — which
satisfies §7's own warning. The depth-20 oracle ceiling is 71.68 vs served 44.15,
so there is no shortage of headroom to detect. **The depth ablation needs a paid
Voyage tier or a local reranker** (blocked here: torch 2.2.2 vs transformers
5.14.1, no GPU).

- Rerank the served top-50 → top-10. `physics_rerank_ceiling.py` writes the
  served top-100 per query to
  `data/benchmark/vidore_v3/results/physics_served_pool.json`, so the candidate
  lists are on disk and the rerank pass needs no retrieval recompute.
- **Cost:** 302 queries × 50 pages × ~310 tokens (chandra2 avg 1,225 chars/page)
  ≈ 4.7M tokens at $0.025/1M ≈ **$0.12**. Top-100 ≈ $0.23.
- **Predicted:** +8 to +14 NDCG@10, with the anchor stated honestly — the paper's
  +13.2 is a **10-dataset average** for zerank-2 on top of *Jina-v4's* candidates,
  not a physics figure, and physics behaves atypically in this benchmark (French
  outscores English on it). Our pool is a BM25+3-small fusion at R@100 = 84.2, so
  the transfer is not clean in either direction. Our own depth-50 oracle ceiling
  (+39.7) says only that the headroom exists. Treat the band as falsifiable, not
  as a target.
- **Ablate the depth** (top-20 / 50 / 100) in the same run — candidates are
  cached, so the marginal cost is only the extra rerank calls.

zerank-2 is multilingual (100+ languages, trained for cross-lingual retrieval),
which matters for French physics, and is instruction-following, so the query can
carry a task hint. Weights are on HuggingFace if the API key becomes the
constraint; variants `zerank-2-nano` (0.6B) and `zerank-2-small` (1.7B) exist.

**Report on the same per-question SHA-256 gate** as the other arms so this lands
in the gate3 ledger rather than floating loose.

### Tier 2 — cheap dense swap, ~$0 to a few dollars

**E2. Multilingual embedder for the dense leg.** Deprioritized by the recall
curve, but worth one pass because it lifts the *pool* the reranker draws from
(R@100 84.2 → higher means a better ceiling). Jina-v4 or Qwen3-Embedding-0.6B,
run locally via the existing `LocalEmbedder` path (`BAAI/bge-m3` is already the
default there) — so this can be **$0 on local GPU**.
- 1,674 page encodes + 302 query encodes. Trivially small.
- Mind the query/document prefix asymmetry: `index.encode_query` already exists
  to preserve it, and `axiom_gateway.py` already distinguishes `input_type`.
  Getting the prefix wrong silently degrades one arm and not another.

### Tier 3 — the expensive one, budget for exactly one

**E3. Context-size sweep, k ∈ {3, 5, 10}, oracle vs reranked.**

Here is the problem with measuring QA: at top-10 with 2.34 gold pages delivered,
redundancy saturates and the oracle gap is 1.3pp — **too small to detect on
n=302 at any reasonable power.** Reranking will look excellent on NDCG@10 and be
statistically invisible on QA. An experiment run at k=10 on `correct_only` is
designed to find nothing.

At **k=3**, precision at the very top governs what the generator sees, redundancy
cannot paper over a bad ranking, and the oracle-vs-retrieved gap should widen into
detectable range. That is the regime where reranking converts into answer quality.

- Arms: `α=0.7 @ k=3` vs `α=0.7+zerank-2 @ k=3`. Two generation passes.
- **Power it on `correct_plus_partial`**, which is where the only significant
  end-to-end effect in all our data lives (−3.64pp, p=0.030). `correct_only` has
  never moved significantly in any contrast we have run.
- Paired permutation test on per-question labels, same as the existing ablations.
- **Cost:** 2 × 302 generations + 2 × 302 judgments. Smaller contexts mean fewer
  input tokens than the runs already done.

If the generation budget will not stretch, the honest alternative is to run E1
only, report the NDCG gain, and **state plainly that QA gain is bounded at
+3.6pp credited by the oracle gap and was not measured this round.** Reporting an
NDCG win and implying QA follows is the one thing to avoid.

---

## 4. Recommended sequence — status

1. **E0a + E0b — done.** BM25 +2.71 (elision, not stemming, was the missing
   piece); α=0.7 vindicated as *not* the in-sample optimum, ~0.8 selection cost.
2. **E1 — done. 44.15 → 49.23 (+5.08, p=0.0001).** Voyage `rerank-2.5` at depth
   20, not zerank-2 at depth 50; see §3 for why both substitutions were forced.
3. **E2 — blocked on the environment, not deprioritized.** `LocalEmbedder` needs
   torch >= 2.4 for transformers 5.14.1; this machine has torch 2.2.2 with no CUDA
   or MPS. Same break blocks running zerank-2 weights locally.
4. **E3 — not run.** Deliberately: §3's own fallback. We report the NDCG gain and
   state that the QA gain is bounded at ~+3.6pp credited by the oracle gap and was
   **not measured this round**.

End state reached: physics NDCG@10 **49.23** on the `vidore_page` arm, above every
**un-reranked** retriever in the paper's physics column (visual included) — as a
fusion+rerank pipeline, not a single model. Short of the "mid-50s" hoped for,
because the arm that ran was a different reranker at a shallower depth than the
plan assumed.

**Reranked vs reranked, we edge it.** Paper Table 2, physics column: Jina-v4
textual 43.6 → **48.2** with zerank-2 (+4.6); visual 46.6 → 46.9 with
jina-reranker-m0 (+0.3). Our 49.23 is above 48.2, and our +5.08 is slightly larger
than zerank-2's +4.6 here.

**Physics is where zerank-2 helps least** — +4.6 against its +13.2 ten-dataset
average (Finance gets +20.8). The §3 prediction band of +8/+14 was built on that
average and was never right for this dataset; +5.08 is the expected magnitude.

The remaining caveat is the corpus: 49.23 is the `vidore_page` arm, ViDoRe's own
text. The chandra2 arm — our parsing — starts at 42.90 and is being reranked
separately.

**The cheapest untried lever is depth.** Our depth-20 oracle ceiling is 71.68 and
depth-50 is 83.88; we are at 49.23 with depth 20 only because the free Voyage tier
cannot fit a larger request. A paid tier would test depth 50 for a few dollars —
that is the next experiment, and it needs a billing decision, not engineering.

### What this changes in the pipeline: almost nothing, deliberately

Three findings each argue against a code change, and only one argued for one:

- **`analyzer=french` ships, but no default moves.** It is registered in
  `ANALYZERS` and tested. It is worth +2.71 on a BM25-only arm and **−0.13 on the
  α=0.7 fusion**, so promoting it — or teaching `resolve_analyzer` to pick it by
  language — would contradict our own measurement. It is available, not default.
- **α stays at 0.7.** E0b found the curve flat from 0.4 to 0.7 and the
  fold-selected α swinging 0.45 → 0.75. Moving a weakly-identified parameter to
  another point on a flat plateau is churn, not improvement.
- **The reranker is not wired into `src/retrieval/`.** This is the one real gain
  (+5.08), and it is deliberately left as a documented recommendation. Every
  reranker in `src/retrieval/` reaches its model through the model-service
  gateway; the arm we measured calls `api.voyageai.com` directly, because the
  gateway's `cohere_compatible` adapter reads `payload["results"]` and sends
  `top_n` while Voyage returns `data` and takes `top_k`. **The productionization
  path is a `VoyageAdapter` in the gateway** (platform repo), after which
  `src/retrieval/` needs no new code at all — only
  `RETRIEVAL_RERANKER=gateway` with a Voyage alias.

Worth knowing before anyone plans that work: **`src/retrieval/search.py` and
`app.py` are still `NotImplementedError` skeletons** (13 and 11 stubs). There is
no serving path today that a rerank setting would flow through, which is the other
reason nothing was wired now.

**Recommended serving config once the gateway adapter exists**, stated so it is
not re-derived: `RETRIEVAL_RERANKER=gateway` against a Voyage `rerank-2.5` alias,
`RETRIEVAL_K3=20` (unchanged), `RETRIEVAL_ALPHA=0.7` (unchanged),
`analyzer=plain` for fused serving. Expected +5.08 NDCG@10 on physics, with the
truncation and depth caveats in [vidore_v3_results.md](vidore_v3_results.md) §1b-iii.

---

## 5. Visual retrieval: text experiments first, and here is why

**Recommendation: run E0–E3 first. Visual retrieval comes after, and it enters as
a pool contributor and a generation-context change, not as a replacement first
stage.** This is not a scheduling preference — the paper's own numbers argue it.

### The paper's verdict is that reranked text beats visual

| System | avg NDCG@10 (10 datasets) |
|---|---:|
| **Jina-v4 (text) + zerank-2** | **63.6** — highest overall |
| ColEmbed-3B-v2 (best visual retriever) | 59.8 |
| visual retriever + jina-reranker-m0 | +0.2 over base, degrades on 4/10 |

The textual reranker is what produces the benchmark's best retrieval result. The
visual reranker is nearly inert. Adopting visual retrieval *instead of* reranking
would be adopting the weaker of the two levers.

On physics specifically, visual retrievers score 47.0 (ColEmbed-3B-v2) and 48.3
(ColNomic-7B) against our 44.15. **That is a +3 to +4 first-stage lever.** Our
rerank headroom on the pool we already have is +39.7 (oracle) with +13.2 realized
in the paper. The ordering of these two bets is not close.

### It attacks a bottleneck we measured and do not have

§1 established `R@100 = 84.2`, `any-gold@100 = 98.3`. A visual retriever is a
*first-stage* change — it improves candidate generation. Our candidate generation
is not what is broken. Swapping it would raise a number that is already high while
leaving the ordering failure untouched.

The right mental model: **a visual leg feeds the pool; the reranker is what
converts pool quality into NDCG@10.** Build the converter first, or the extra
recall a visual leg buys has no mechanism to reach the top-10. Sequenced properly,
visual is a third fusion leg (BM25 + dense + visual → rerank) and its recall gain
compounds with the reranker instead of competing with it.

### Where visual genuinely wins — and it is the QA leg, not retrieval

This is the part worth being excited about, and it inverts our current picture:

- Image context beats text context for **generation** by **+2.4 to +2.8pp** on the
  hard subset.
- **Hybrid** (image + text) context is best on challenging queries — 54.7%.
- Page image + OCR text as VLM input is ~**+6.5%** over image alone.

§2 argued our QA gain from retrieval is capped at ~3.6pp credited by the oracle
gap. **Visual context is not subject to that cap**, because it changes what the
generator can see rather than which pages it sees — it raises the oracle itself.
If the QA number is the goal, a VLM generator over page images is a bigger lever
than any retrieval work in this document.

The caveat is that it requires a VLM generator (Gemini 3 / Qwen3-VL class), which
is a real cost line versus deepseek-v4-flash, and our judge stability was measured
on text answers only.

### Two prerequisites that are not free — both verified

1. **We do not have the page images.** `data/benchmark/vidore_v3/physics/corpus.parquet`
   is **380 KB** — text columns only. `CORPUS_COLUMNS` excludes `image` on purpose
   ("~12 GB of page renders no text arm reads"). Visual work needs a fresh
   **442 MB – 2.2 GB per subset** download. Physics first; do not pull all 8.
2. **Our index cannot hold multi-vector embeddings.** `LocalIndex.vectors` is a
   strict 2-D `np.ndarray` with `shape[0] == len(records)` and a `dim` assertion,
   scored by a single dot product (`index.vectors @ vector`). ColPali-family late
   interaction needs *ragged multi-vector storage plus MaxSim* — roughly 1,030
   patch vectors × 128 dim per page. **This is an architectural change to the
   index contract, not a new embedder plugin.** Budget it as such; it is the real
   cost of a visual arm, not the GPU time.

GPU access itself is solved — the existing `Chandra_serving_de.ipynb` /
`KDL_serving_de.ipynb` Colab pattern applies, and 1,674 physics pages is small.

### The honest counter-argument

Visual retrievers lead the physics column outright (48.3 vs our 44.15), and the
paper's textual pipeline is Jina-v4, a much stronger first stage than ours. If our
reranked result lands below ~48, the visual first stage becomes the better bet
sooner than this section implies. **E1's result should decide it** — that is
another reason to run it first. It is a $0.12 decision input for a multi-week
engineering commitment.

---

## 6. Reading list for the visual leg

Ordered for someone picking this up cold.

**Foundations**
1. **ColBERT** (Khattab & Zaharia, 2020) and **ColBERTv2** (2022) — where late
   interaction and MaxSim come from. Read first; every model below is this idea on
   a vision backbone. ColBERTv2's residual compression is directly relevant to
   the storage problem in §5.
2. **ColPali: Efficient Document Retrieval with Vision Language Models**
   (Faysse et al., ICLR 2025, [arXiv 2407.01449](https://arxiv.org/abs/2407.01449))
   — the founding paper of this line and of the ViDoRe benchmark itself. The
   single most important read.

**Current systems**
3. **Llama NemoRetriever ColEmbed** ([arXiv 2507.05513](https://arxiv.org/pdf/2507.05513))
   — this is ColEmbed-3B-v2, the 47.0 on our physics column. Closest thing to a
   spec for what we would be reproducing.
4. **ViDoRe V3** ([arXiv 2601.08620](https://arxiv.org/abs/2601.08620)) — reread
   **Table 2** specifically, the rerank ablation this whole document turns on.

**Frontier, for where to contribute rather than reproduce**
5. **Argus-Retriever** ([arXiv 2606.04300](https://arxiv.org/abs/2606.04300)) —
   query-conditioned late interaction on Qwen3.5-VL. Attacks the structural
   weakness that documents are embedded without seeing the query; Argus-2B (82.9)
   matches ColNomic-7B (81.5) at a quarter the size.
6. **Beyond Bag-of-Patches** ([arXiv 2605.08421](https://arxiv.org/html/2605.08421v1))
   — global layout via textual supervision. Relevant to us because our chandra2
   parse already produces block/layout structure that this line of work is trying
   to recover from pixels; we may be able to supervise with it.
7. **Spatially-Grounded Document Retrieval / patch-to-region relevance**
   ([arXiv 2512.02660](https://arxiv.org/pdf/2512.02660)) — visual grounding is
   the benchmark's weakest leg by far: **F1 0.065–0.089 against human agreement of
   0.602.** That is an order-of-magnitude gap and the most under-served metric in
   ViDoRe V3. If we want a research contribution rather than a leaderboard entry,
   this is where the room is.

**Reranking, for E1**
8. zerank-2 model card and the ZeroEntropy reranker notes — instruction-following
   and calibrated scores are both usable in our setting.

---

## 7. Picking this up in a new session

Read this document plus [vidore_v3_results.md](vidore_v3_results.md). Then:

**State as of this session**
- Diagnosis complete and evidenced: the bottleneck is **ordering, not recall**
  (`R@100 = 84.2`, `any-gold@100 = 98.3`, oracle reorder of served top-50 = 83.88
  vs served 44.15). Confirmed independently by the paper's +13.2 zerank-2 ablation.
- **E0a and E0b are done** (see §3). E0a shipped `analyzer=french` in
  `src/retrieval/sparse.py` (+2.71 BM25, but ~0 on the fusion). E0b showed α=0.7
  is *not* the in-sample optimum and costs ~0.8 out-of-sample.
- **E1 is done: 44.15 → 49.23 (+5.08, p=0.0001)**, Voyage `rerank-2.5` at depth 20
  — not zerank-2 at depth 50, see the two corrected premises in §3.
  `physics_rerank_voyage.py` is resumable and checkpoints per query; the free tier
  paces it at ~100 minutes.
- **E2 is blocked** (torch 2.2.2 vs transformers 5.14.1, no GPU) and **E3 was not
  run** — QA impact of reranking is unmeasured and bounded at ~+3.6pp credited.
- Three reproducible scripts exist and cost $0 to re-run (all read the embedding
  cache at `data/work/vidore_physics_emb`):
  `physics_recall_curve.py`, `physics_rerank_ceiling.py`, `physics_gold_buckets.py`.
  `physics_alpha_holdout.py` is also free and rebuilds the α curve.
- `data/benchmark/vidore_v3/results/physics_served_pool.json` holds the served
  top-100 α=0.7 candidates for all 302 physics queries — **rerank arms need no
  retrieval recompute.** `physics_rerank_voyage_scores.json` holds the per-query
  rerank scores already paid for; do not re-spend them.
- Python is `/usr/local/Caskroom/miniconda/base/envs/axiom-de-rd/bin/python`.

**Start here:** the depth-50 rerank, which needs a paid Voyage tier (§4). Failing
that, E3 at k=3 to put a QA number against the +5.08. E1's result is the decision
input for §5: at 49.23 we clear the un-reranked visual retrievers (47.0, 48.3), so
a visual *first stage* is less urgent than §5 feared — but §5's counter-argument
is not retired, because the paper's reranked physics number is unknown and likely
near 57.

**Do not:**
- Report the oracle-rerank numbers (60.49 / 83.88 / 90.27) as pipeline results.
  They are gold-label-informed ceilings and exist only to size the headroom.
- Compare our fusion row against the paper's single-retriever rows without the
  not-like-for-like caveat from §1.
- Compare our reranked arm against the paper's *un-reranked* rows (44.0, 45.8,
  47.0, 48.3). The reranked physics figures are Table 2: **48.2** textual+zerank-2,
  46.9 visual+jina-reranker-m0. Use those.
- Apply the **+13.2** zerank-2 average to physics. Physics is its weakest dataset
  at **+4.6**; the average is dragged up by Finance (+20.8) and C.S. (+17.8).
- Report the `vidore_page` arm as if it were our pipeline. It is ViDoRe's own text
  extraction. The chandra2 arm is ours.
- Tune rerank depth on the same 302 queries and report the best — that repeats the
  α=0.7-tuned-on-test problem. **This applies directly to the depth-50 run named
  above:** report it *alongside* depth 20, never instead of it. Depth 20 is the
  pre-registered number because it is `RETRIEVAL_K3`; if depth 50 wins and becomes
  the headline, it needs a held-out split like E0b's, or an explicit
  "selected on test" label.
- Raise `--depth` on `physics_rerank_voyage.py` without also raising `--rpm` and
  `--tpm` to the paid tier's limits. One request over the token cap can never
  satisfy the throttle, and it crashes rather than waiting.
- Measure QA at k=10 on `correct_only` and conclude reranking did not help; §3 E3
  explains why that experiment cannot detect anything.
