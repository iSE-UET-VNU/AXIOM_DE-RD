# Tight coupling plan

Approved 2026-11-02. For this track recall and coupling come before latency. Cold cost is reported, it does not decide.
Dropped: RLRF and PilotRAG (names not verified). Sources kept: RRPO, LoopRAG's Faith term, CRITIC-R1, BAR-RAG.
Agent Reader = the existing single reader call in `ondemand/qa.py`; no multi-turn agent.

## Baseline
`results/xk_evie_v2_stop10_214`: `gate_ppocrv5_ungated_xgbK214.json` + `visual:evie_cached:skip_blank+ppocrv5+stop:10`
+ `fixed:512:128` + hybrid + top 10. R@10 all 69.46, industrial 58.31, physics 58.13, OHR scan 100, OHR handwritten 96.67,
28.75 encoded pages/q. No QA run. Reference: `xk_evie` R@10 70.52, QA C 50.93, C+P 91.12.

## Constraints from earlier work
- Chunk & Rank cannot move recall: ~11.2 chunks reach a top-10 cut that drops 0 gold pages.
- QA headroom from retrieval is small and noisy: gold pages C 56.07; ~35 Correct flips per fresh draw.
- Closed: margin-based K shrinking (`stop:10` does it, adaptive K no gain over 59), PRF on 128 tokens, RRF, RankGPT,
  doc judges, query-level sufficiency gating (AUC 0.53-0.57).
- Every threshold needs an invariant, a product budget or unlabeled traffic as its source.

## Track A: patch-to-text diagnostic
Attribute by query token: score = sum_i max_j q_i.d_j, each query token picks one winning square.

Gates before patch work:
- Scorer: only 128-dim tokens are cached, ranking uses 2048. E0 measures the gap. On failure re-encode contrast pages at 2048.
- Alignment: cached rows = image squares + ~11 non-image tokens. Reuse notebook 11 `grid_of` / `blank_of`, assert counts per page.
- Square size: A4 at 1000 squares is ~22 pt, about 2 lines by 4-5 characters. Match text over a 3x3 neighbourhood.
- Tokens are from full EVIE, the baseline uses skip-blank: findings explain full EVIE.

Per-token log: query_id, group, page_id, contrast set, query token index and string, is_content, argmax square (row, col),
similarity, is_blank, is_non_image, words in square, words in 3x3 neighbourhood, match type (exact, stem, translation, none),
region type, font size vs page median, text-layer coverage of square, doc rank, light rank, EVIE rank.
Per page: score mass by category, spatial spread of argmax squares vs permutation null, entropy of s_i, word count,
language pair, near-copy flag.

| # | Hypothesis | Metric | Decides |
|---|---|---|---|
| H1 | Winning squares sit on the relevant text | score share near a matching word vs shuffled null; share on blank squares and non-image tokens | B dies if this fails |
| H2 | Visual matches meaning, not words | exact / stem / translation / none, V+L- vs V+L+ | physics, feeds B |
| H3 | Evidence is concentrated, text scoring dilutes it | spatial spread vs null, by word count | new feature |
| H4 | Layout or font emphasis | log-odds of region type vs area; font vs page median | chunking |
| H5 | Dense tables | winner share in table regions; light rank there | table handling |
| H6 | OCR noise | V+L- rate by text source | expected false on ViDoRe |
| H7 | Evidence has no text layer | winners on squares with no text-layer words | parser vs scorer gap |

A feature counts only if its AUC CI excludes 0.5 in leave-one-collection-out testing.

## Track B: prototypes

### A. Answer-conditioned MaxSim (reader to visual)
Reader drafts from light's context; EVIE rescores pages already encoded with query + draft as one text, so no fusion weight.
Signal: draft token matrix Q2.

E2 oracle, Q2 = query + gold answer, evaluated on three nested scopes:
- (a) only the pages actually encoded before `stop:10` (~28.75 pages/q): fixes ranking order within encoded pages?
- (b) the full xgbK pool, ignoring `stop:10`, and also `stop:10` re-run under Q2 scores: prevents halting too early?
- (c) all cached pages for the query: rescues pages outside xgbK?

If the oracle's CI does not clear 0, A is dead and no reader money is spent.
E2 also logs, per scope: share of top-10 slots in a gold document (baseline 61%), whether the #1 page is in a gold document,
distinct documents in the top 10. Q2 is evaluated under full EVIE and skip-blank scorers; skip-blank tokens are not cached.

### B. Patch back-projection (visual to text and reader)
Words under winning squares on EVIE's top pages feed BM25 in the page's own language; regions holding winning squares feed the reader.
Signal: map from query token to square, region and words. Needs H1.

Fusion rule: do not fuse BM25 and EVIE scores. `bm25(query + patch_terms, pool)` is a candidate proposer: its top-M
unencoded pages are injected into EVIE's encoding queue (couples B with C's scheduler) and EVIE scores them in native MaxSim.
M needs a non-benchmark source. Decision 2026-11-02: injection is a secondary add-on, capped by full-corpus EVIE.
Priority: Q_patch (`query + patch_terms` encoded as an EVIE query, rescoring existing pages) and region context selection.

### C. Document-belief scheduler (visual to itself)
After each encode, update per-document belief (max EVIE score seen) and encode the likeliest pages next; window rule unchanged.
Can only change which pages are encoded, so it cannot beat full EVIE. Free to simulate on `scores_v2_all.npy`.

E5 addition: test document belief on the final top 10 too, against the oracle-doc-in-pool ceiling (+10.93 on ViDoRe):
- gold documents per query; how often the #1 page or top-1 doc (max, top-2 score) is the right document;
- restrict or prioritize output to the top-D belief documents, and top-1 doc when its margin is high, with leave-one-collection-out.
D and any margin are free parameters and need a source. Earlier ColVec doc-first commit failed on fold 0 (physics 44.1 to 31.2).

Near-copy definition everywhere on this track: 0.6 shingle overlap within the document, as in `ondemand/routing_axes/e0.py`.

## E0 result (2026-11-02)
ViDoRe 154 q, deltas vs baseline: full-corpus EVIE +2.97 [+0.34,+6.03], full-corpus skip_blank +0.86 [-2.30,+4.19],
oracle doc in pool +10.93 [+7.84,+14.30], oracle doc all pages +15.16 [+11.30,+19.43].
EVIE-only proposers (B injection, C) are capped by the full-corpus rows. Larger gains need a changed scorer (A, Q_patch).
Recall split: hit 58.22, outranked while encoded 11.94, stop-halted 11.24, outside K in light 100 3.07, outside light 100 15.54.
128-dim gate: ViDoRe passes, industrial does not. xgbK is nested OOF; run fx59 alongside as a check.

## Roadmap
| Step | What | Cost | Go/no-go |
|---|---|---|---|
| E0 | Reproduce baseline, 128 vs 2048, confound table, doc-oracle ceiling, xgbK provenance vs fx59 | Local cache | 128 CI includes 0, otherwise re-encode |
| E1 | Grid alignment, H1-H7 | Small Colab job, then local | Row counts match on every page |
| E2 | A with gold answer, scopes a/b/c | Query encode on GPU, no LLM | Oracle CI clears 0 |
| E3 | A with real drafts | ~154 reader calls | CI lower bound above 0 vs baseline |
| E4 | B queue injection, Q_patch, then B context | Local after E1 | Needs H1. Context metric: tokens at equal gold coverage |
| E5 | C simulation | Free | R@10 against pages encoded |
| E6 | Ablation: baseline, +A, +B, +C, then remove one at a time | Mostly cache | Paired vs baseline |
| E7 | QA on finalists and baseline | API, repeated draws | Agreed before the run |

Statistics: ViDoRe 154 q plus per-collection, R@10 and NDCG@10, 10k paired bootstrap seed 0, leave-one-collection-out for any parameter.
