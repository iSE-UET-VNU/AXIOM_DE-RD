# STATE — carry-over for the next session

Updated 2026-11-05. Branch `adaptive`. This file beats memory; cache and git beat it.
Full history (every earlier entry and the pre-rewrite STATE.md, verbatim): `.plans/state-archive-2026-10-27.md`.
`P=/usr/local/Caskroom/miniconda/base/envs/axiom-de-rd/bin/python`. Work dir `data/work/ondemand_v2/ef7f11811803/`.
Framing: recall first, cost second (user 2026-10-28). Cold timing, no cross-query cache. No rule fitted to this benchmark;
every number from an invariant, a product budget, or deployment traffic.

## START HERE
- PROXIMITY CHECK 2026-11-05 (user approved, cache only, printed only, no API or GPU): are pages missed by `xk_evie_v2_stop10_214` near its top 10? ViDoRe 154 q, gap `|p-s|` in the same document, near-copy = 3-word shingles at 0.6 of the smaller set, null = non-gold non-seed pages of the same document, per-query mean, 10k paired bootstrap seed 0. Self-check 58.22 / 88.96; 637 missed gold pages.
  Nearest seed: 596/637 = 93.6% have a seed in their document (87 q). Median gap 6 vs 16 null; gap==1 23.34 vs 14.60, +8.74 [+3.60,+14.27] (pooled 19.46 vs 8.48); gap<=2 37.53 vs 25.10, +12.43 [+6.63,+18.64] (pooled 31.54 vs 15.12); near-copy 6.94 vs 0.57, physics 11.80 vs 0.95, industrial 0.39 vs 0.06 n.s.
  Nearest found gold (labels): 533/637 = 83.7% (80 q). Median gap 8 vs 25; gap==1 +6.56 [+1.99,+11.55]; gap<=2 +11.56 [+6.02,+17.44]. gap==1 by gold bucket: 2-5 +6.35 [-6.73,+22.87], 6-10 +6.40 [-0.75,+13.72], >10 +6.83 [+2.36,+11.69]; gap<=2: +16.07 [-1.28,+36.06], +9.63 [+3.25,+15.96], +10.61 [+4.29,+17.12]. Document-clustered CIs on the 4 overall gap rows stay above 0 (separate recomputation matches).
  Gold sets are mostly scattered: adjacent consecutive gold vs random, 2-5 15.88 vs 5.06, 6-10 26.17 vs 12.99, >10 29.94 vs 23.09; mean run length 1.12, 1.28, 1.40; gold pages in runs of 3+ 7.0%, 14.5%, 22.5%.
  Seed neighbours: gap==1 14.4 pages/q, gold 5.22%, gap<=2 26.1 pages/q, 4.68%; other non-seed pages of seed documents 0.91%, 0.80%; R@10 if all added, no slot limit, +7.61, +12.03. E8's replaced slots were gold 15-16%, cross-run comparison, no swap tested.
  Gate passes, signal is weak. Scripts `ondemand/routing_axes/gold_proximity.py`, `gold_proximity_check.py` (`-m`, from repo root), summary in plan; log not kept, rerun to regenerate. Propagation test: see E10.
- E10 PROPAGATION 2026-11-05: intra-document score diffusion, `ondemand/routing_axes/e10_propagation.py`, print only, no API. 154 ViDoRe, ENC V2 scores, W = exp(-gap/l) * te3s cosine, a in {0.05,0.1,0.3,0.5,0.7,0.9}, l in {1,2,4,8,16,32}, 5-fold CV. Out-of-fold R@10 57.88 vs 58.22, -0.34 [-1.68,+0.97], FAIL. Reference a=0.5, label-free l: 58.74, +0.53 [-0.43,+1.64]. Closed as negative.
- FULL V2 CACHE 2026-11-05: Kaggle nb 18 (`kaggle/18_evie_v2_patch_cache.ipynb`, builder `ondemand/routing_axes/v2_patch_cache_make_notebook.py`) encoded all 5091 pages with V2 (blank squares skipped), T4 x2, 0.74 h, left 0. Output `outputs/k18_out_v2.zip` (752 MB; Claude cannot unzip into data/work, user unzips to `routing_axes/k18/`). Holds `scores_v2.npy` (220 x 5091, pages in `pool_page_ids.json` order; equals `k13/scores_v2_all.npy`, max diff 0, corr 1.0), `win_row.npy`/`win_sim.npy` (5091 x 7171 query tokens, 2048-dim winning token row in the full sequence and its score; query tokens in `q_index.json`, 220 queries, matches e2 on the 214), `vectors_128.f16` + `positions.i16` + `vectors_index.jsonl` (kept tokens at 128 dim, row positions, per-page offsets). Keep-all check passed (cos >= 0.99999, 128 head too). Rows are positions in the full token sequence, same indexing as `e1_grids/grids.jsonl` (`row - image_start` = square).
- VIEWER 2026-11-05: `$P -m ondemand.routing_axes.evie_miss_viewer --variant v2_stop|full` (reads `outputs/k18_out_v2.zip` directly), output `routing_axes/evie_viewer/{v2_stop,full}/index.html`, chooser at `evie_viewer/index.html`, images shared in `evie_viewer/img`. v2_stop = real baseline (V2 ranks, V2 patch winners, encoded/halted flag from a replay of the stop rule on V2 scores (matches the logged `cost.pages` on all 214; `branch_notes.scores` holds only the top 10)); full = full-EVIE reference (128-dim cache winners). Finding: query token 1 always wins a non-image token (score ~0.95, attention sink), 1906/1906 shown pages.
- Cleanup pending (deny rule blocks Claude in data/work): `rm -r routing_axes/k1/k1_input routing_axes/k1/k1_input.zip routing_axes/k1/k1_input_disagree routing_axes/k1/k1_input_disagree.zip` (3.3 GB, only k1_prep.py writes them), plus `routing_axes/tl/`, `routing_axes/strategies/sq_input*`.
- E8 REOPENED THE TRACK 2026-11-04 (user) for one zero-extra-encoding baseline coupling check. `ondemand/routing_axes/e8_coupled_router.py`, result in plan, outputs to `routing_axes/e8/` (user copies from session 7cac0e6a scratchpad). The QA row is now E9, not run. Gate FAIL on all 7 primary rows: M1a k=1,2,3 -1.15, -1.35, -1.66 (CIs include 0); M1b foreign tau + lexical -0.71, -1.46, -1.59 (CIs below 0); M2 -5.82 [-9.59,-2.24]. New Light picks gold 9-10%, replaced baseline slots 15-16%. Whether the track closes again is the user's call.
- TRACK CLOSED 2026-11-04 (user): tight coupling E0 to E7 complete, no retrieval gate passed by a non-oracle method; QA (now E9) not run. Commit hold lifted; user commits, Claude stages only. Diagnostic report for the professors (Vietnamese): https://claude.ai/artifact/4HofRZqdzBNiWFA1AmwnEe. Outputs now in `routing_axes/e1` to `e7` (foreign-null log in `e6/`, whole-page log in `e4/`); scratchpads were wiped by /tmp cleanup, so older scratch paths below are dead. E7, `e5_foreign_null.py` and `e4_whole_page.py` were rerun 2026-11-04 and match their recorded results. Next direction open: see chat brainstorm, nothing approved.
- TIGHT COUPLING (user 2026-11-02, plan `.plans/tight-coupling-plan.md`): for this track recall and coupling beat latency.
  Baseline `xk_evie_v2_stop10_214`. E0 DONE (cache, printed only).
  Baseline rebuilt from scores: 214/214 top-10 lists identical. ViDoRe 154 R@10 58.22: outranked while encoded 11.94,
  stop-halted 11.24, outside K in light 100 3.07, outside light 100 15.54 (cap 92.09; 7.91 unreachable, overlaps those).
  Halted is not lost: V2 no stop -0.31 [-2.74,+2.12]. Vs baseline, ViDoRe: full-corpus EVIE +2.97 [+0.34,+6.03], full-corpus
  skip_blank +0.86 [-2.30,+4.19], oracle doc in pool +10.93 [+7.84,+14.30], oracle doc all pages +15.16 [+11.30,+19.43].
  128 vs 2048: ViDoRe dR -0.87 [-2.27,+0.42] gate passes, industrial -1.91 [-4.08,-0.22] does not; V+L- 91 vs 92 of 111.
  V+L- pages: right doc already in light's top 10 98%, light rank median 17, same word count as V+L+, overlap .232 vs .386.
  xgbK = nested OOF per query (matches gate), but folds are not grouped by doc or collection; vs fx59 ViDoRe -0.17 [-0.45,0.00].
  Skip-blank vs full EVIE, paired: full corpus ViDoRe -2.11 [-4.32,-0.19], industrial -4.45 [-8.75,-0.80], physics +0.12;
  inside the oracle doc ~0 (ViDoRe -0.20 [-1.34,+0.90]), so its loss is between documents. `ondemand/routing_axes/e0.py` committed.
  Skip-blank tokens are not cached (K13 saved scores only). Settings for E2, E1 confirmed by user 2026-11-02.
  E5 DONE (`ondemand/routing_axes/e5.py`, printed only): top-1 doc right 74.0%; D=1 restrict -8.10, prioritize -3.58; D=2 -0.25; margin AUC 0.832;
  margin gate LOCO +1.45 [+0.24,+2.98], with that doc's pool pages +2.30 [+0.62,+4.17] at 38.18 pages/q; tau has no non-benchmark source yet.
  Kaggle 15 and 16 done 2026-11-03, outputs extracted to `routing_axes/e2/`, `routing_axes/e1/` by the user (Claude's writes under `data/work/` were denied). Grids match cache on all 5,091 pages.
  E2 stage 1 scored (`routing_axes/e2/run.log`, ~29 min local), result in plan. ViDoRe Q2 vs Q: (a) +1.53 [-0.66,+3.74], (b1) +2.63 [+0.01,+5.29], (b2) +1.34, (c) +5.56 [+2.34,+9.02]. Gate prints GO on b1, c.
  Stage 2 (skip-blank tokens, ~1.33 T4 GPU-h) not built; held by user until E3 reports. E1 done 2026-11-03 (`routing_axes/e1/run.log`, result in plan): H1 passes but winners match words on hard negatives too; `report`'s feature bar and H4 lines fixed 2026-11-03, run.log regenerated.
  E1: `e1_patch_text.py --translate` (79 reader calls, approved) then `e1_patch_text.py`. Scope (c) at 128 is 58.78 vs 61.18 at 2048 on ViDoRe.
  2026-11-03 follow-ups, cache only, ViDoRe, 128 dims, deltas vs same arm on the xgbK pool (scripts and logs in session 8b58e634 scratchpad, not in repo). User: commit nothing until a real (non-oracle) method shows a result. Stage 2 held. B = region context only, Q_patch dropped. Mixed translations not re-run.
  Light with BM25 tokens q+gold and dense q, union K (87.4 pages/q): pulls 92/278 pool-missed gold pages; scorer q +0.55 [-0.55,+1.74], scorer q2 +2.88 [+0.83,+5.32] = 63.68 vs full-corpus q2 64.34. Same with cached xk_light drafts: pulls 64/278, scorer q -0.19 [-1.83,+1.04]; this does not test drafts, q+gold is also n.s. under scorer q, the gain needs the answer in the EVIE query. Pure bm25(q+gold) replacing light loses (-7.43 at K).
  Siblings of top-1 doc on scope (a): q +1.85 [-0.06,+4.04], q2 +4.00 [+1.73,+6.42], about 130 added pages/q (mean; median 57). Image-only MaxSim: no effect (q scope c +0.05 [-0.44,+0.53], #1 gold doc 72.1 to 70.1); closed.
  E1 `report` fixed in repo, `routing_axes/e1/run.log` regenerated, per_page and per_token byte-identical. Plan updated. `bm25_oracle.py`, `image_only.py`, `tight_coupling_common.py` moved into `ondemand/routing_axes/`, logs identical to the scratch runs. Scratch `draft_oracle.py` not moved (session 8b58e634).
  NEXT = E3, first non-oracle test. Settings below CONFIRMED by user 2026-11-03: control = q on xgbK (not the 2048 stop-rule baseline, that delta is diagnostic), 0 reader calls, refusal-style drafts kept, e3_score.py replaces scratch draft_oracle.py. Nothing committed and no commit commands: user commits only after a non-oracle method shows a result.
  E3 status: `e3_make_notebook.py` and `kaggle/17_e3_draft_query_tokens.ipynb` built, not run. `ondemand/routing_axes/e3_score.py` in repo; `--self-test` passed 2026-11-03 (every delta 0, 214/214 identical, oracle rows +5.51, +2.40, +7.76, +9.79 reproduced, 79 s). E3 SCORED 2026-11-03: NO-GO, ViDoRe 58.17 -> 56.29, -1.88 [-4.41,+0.44] vs oracle +5.51 [+2.12,+8.99]; Correct +0.43, Partially Correct -2.73, Incorrect -12.08 (n=9) vs oracle +2.40, +7.76, +9.79. Result in plan. Input `outputs/e3_out.zip`; outputs (`scores_full_qd.npy`, `report_e3.json`, `run.log`) only in session d0146a9a scratchpad `e3_result/`, user to copy to `routing_axes/e3/`. User decisions 2026-11-03: A with light drafts closed as negative, no tuning on the 154 q; Stage 2 dropped permanently; next = E4 region context selection. Commit 6115ded was undone at user request; E3 files uncommitted again.
  E4 settings CONFIRMED by user 2026-11-03: baseline top-10 pages, 214 q; content-token winners, full EVIE 128 (E1 definition); keep read regions whose box overlaps a winner, no N or threshold; rotated pages keep all regions; Paddle-VL text (`k14_out_paddlevl(1).zip` + `_rest.zip`); o200k tokens; coverage = share of gold answer enfr words in context text from gold pages; no API calls; ViDoRe 154 q, 10k paired bootstrap seed 0. PRIMARY GO only if tokens CI upper < 0 and coverage CI upper >= 0, vs windows (3,511.9 tokens on ViDoRe; 3,964 is ungated_evie_m10, not baseline). Diagnostics: EVIE coverage vs windows, all regions, random and BM25 at same count; per collection; OHR separately.
  E4 SCORED 2026-11-03: NO-GO, ViDoRe vs windows tokens 3511.9 -> 1982.4, -1529.55 [-1687.31,-1360.31], coverage 39.81 -> 35.07, -4.74 [-7.37,-2.35]. Selection vs all regions coverage -7.32 [-9.78,-5.21]; vs BM25 same count +0.70 [-0.49,+1.90]. Result in plan. `ondemand/routing_axes/e4_region_context.py` in repo, uncommitted, self-test passed; outputs only in session d0146a9a scratchpad `e4_result/`. User decides next step; nothing committed.
  User decisions 2026-11-03: E4 closed as negative, roadmap row updated; A and B closed so E6 is unnecessary; only live lead = E5 margin gate, needs tau from a non-benchmark source.
  Whole-page PP-OCRv5 arm run (`ondemand/routing_axes/e4_whole_page.py routing_axes/e4/per_query_e4.json`, log `routing_axes/e4/e4_whole_page.log`), result in plan: ViDoRe coverage windows 39.81, whole 41.09, all regions 42.39.
  Tau sources checked (`tau_sources.py`, same scratchpad, print only, no ViDoRe gold, no gain computed). Encoded-set margins. OHR invalid: top-1 doc right 59/60, 1 negative. No traffic: score matrix holds benchmark queries only, no query logs found. Otsu not supported: raw 0.72-0.74 fires only on one-doc queries (ViDoRe 20/154, margin set to 1.0; no-op on encoded pages, the doc-pool variant can still add pages); on log margins 0.0295 ViDoRe, 0.1205 industrial, 0.0259 physics, 2-vs-1 Gaussian BIC gain +0.3, -3.7, +1.2. Budget: none, and budget knob declined. Candidate left: foreign-query null on the target collection's pages; settings not confirmed, not run. If it is declined or fails, wrap up the track as a diagnostic report.
  E6 COUPLED RETRIEVAL SCORED 2026-11-04, gate FAIL, result table in plan. `ondemand/routing_axes/e6_coupled_retrieval.py`; outputs in `routing_axes/e6/`. Non-ASCII token bug fixed in `query_word_of_tokens` (byte-level tokens, 13 of 214 queries changed, 0 unmapped), E1 rerun to scratchpad `e1_after/` (10.6 min, no API calls), copied over `routing_axes/e1/` by the user, old files in `e1/before_token_fix/`. Entropy now clears E1's feature bar, post hoc. User declined a 2048 GPU run for S_comp. Light sibling picks: 58.82 at 83.6 pages/q, vs S +0.50 [-1.41,+2.24], vs baseline +0.60 [-2.14,+3.34], FAIL.
  E5 foreign-query null run 2026-11-04 (one shot, settings from session ef4d5ce1): NO-GO, ViDoRe +0.32 [+0.00,+0.80], fires 30.5%, right 95.7%. Script `ondemand/routing_axes/e5_foreign_null.py`, log `routing_axes/e6/e5_foreign_null.log`. Per STATE's earlier rule the track now wraps up as a diagnostic report; all local experiments closed.
  E7 PATCH VERIFIER SCORED 2026-11-04: negative, no gate set, closed by user; the plan's QA row is now E9. `ondemand/routing_axes/e7_patch_verifier.py`; outputs in `routing_axes/e7/`. ViDoRe, 60 rows (S_trap, S_pileup, S_e1 x all/content tokens x full, top15, top20, ranksum15, ranksum20 x a, b1): R@10 or NDCG lower bound > 0 vs S or baseline on 0 rows; R@10 upper bound < 0 vs S on 19. Best b1 S_trap x H all tokens 58.68, vs S +0.51 [-0.31,+1.49], vs baseline +0.47 [-1.52,+2.71], NDCG vs S -0.33 [-1.63,+0.79]. Winners recomputed from evie_pages_128.npy; features match E1 per_page. E1 entropy is over content-token scores, not spatial; H mean 0.979, p5 0.955.
  E3 settings:
  - Files: `ondemand/routing_axes/e3_make_notebook.py` -> `kaggle/17_e3_draft_query_tokens.ipynb` -> `e3_out.zip` -> unzip to `routing_axes/e3/`; `ondemand/routing_axes/e3_score.py`.
  - Notebook: copy of 15. Kaggle T4 x1, Internet ON, no dataset, EVIE-4.5B repo pin 70b8a2be, transformers 5.13.1, fp16, sdpa, 128-dim, batch 1. Arms `q` (stop unless min cosine vs cached >= 0.999) and `qd` = query + " " + `generated_answer`, verbatim, no refusal filter. 214 q. Prints truncated_queries. No reader or judge calls.
  - Drafts: `results/xk_light/qa.jsonl` (deepseek-v4-flash, temp 0, seed 0, 512 tokens, light top-10 chunks, judge gpt-4o-mini).
  - PRIMARY, ViDoRe 154 q, full EVIE 128: `qd` on xgbK + light (te3s q, BM25 q+draft) top K vs `q` on xgbK, R@10, 10k paired bootstrap seed 0, GO only if CI lower bound > 0. Everything else is diagnostic. Oracle same shape +5.51 [+2.12,+8.99] (industrial +8.50, physics +2.67 [-1.05,+6.44]); q+gold pool 63.68 vs q pool 58.17.
  - Diagnostics: same delta by draft grade; scopes (a), (b1), (c) qd vs q; NDCG, #1 gold doc, slots, pages/q; per collection; OHR separately.
  - Grades on ViDoRe are 3-way: Correct 68, Partially Correct 77, Incorrect 9 (so 44.2% Correct does not mean 55.8% hallucinated; Incorrect stratum too small). Oracle by the same grades: +2.40 [-1.58,+6.56], +7.76 [+2.61,+12.99], +9.79 [-7.67,+35.19], so grade is confounded with query difficulty: read each qd stratum against its oracle row, not against 0.
  - Limits: full EVIE only (skip-blank tokens not cached), 128 dims, no stop rule at ~86 pages/q (baseline 32.9 at 2048), draft grades use gold labels (diagnostic only).
- TARGET (user 2026-10-28): the gap is latency, not recall. Aim = oracle routing {light, EVIE @1024}: visual only
  where visual beats light -> fx59 71.35 R@10 / 23.8 s/q vs always-EVIE 70.32 / 94.5 (old 1.6 s/page basis);
  Jev->full 66.28 / 39.4. Next: literature pass on routing / selective escalation to close it.
- STOP-EARLY (2026-10-28, cache only, scratch evie_stop_early{,_subset,_v2}.py): EVIE encodes pool pages from the
  documents of light's top 10 first (light order), then the rest; stops after m=10 (= output size) pages in a row fail
  to enter its top 10. fx59, ViDoRe 154 q: full scores 58.11 (-0.64 [-2.41,+0.78]) at 57% of pages; + V2 58.05
  (-0.70 [-2.63,+0.97]) at 40% of full encode cost (V2 0.701 Colab ratio). All 214: V2+stop 69.81 (-0.50), 41%.
  Light order without doc grouping is worse (ViDoRe -0.81 at 60%). OHR loses 0; quote ViDoRe-only. Encode cost only;
  forces sequential encoding (no batching).
  WIRED 2026-10-28 (wrap 5) as the "window rule", m = window: `visual.py` branch spec `...+ppocrv5+stop:10`
  (VisualBranch.stop_early; staged HEAD+stop only, the older dedupe edits stay unstaged). Runs: `fx59_evie_v2_stop10`
  69.81 (= scratch); `xk_evie_v2_stop10_214` (adaptive-K) 69.46 vs xk_evie 70.52 (-1.06 [-2.86,+0.45]; ViDoRe 58.22,
  -0.82 [-2.86,+1.01], 32.9 vs 73.7 pages/q = ~31% encode cost); 1 OHR handwritten loss (K 100, gold light rank 21 in a
  doc outside light's top-10 docs, stopped at 66 pages). Doc-by-doc order (all pages of doc 1 first) CLOSED: -2 to -3
  R@10 at every m (m=10 67.84 vs 69.86). Chart (user-approved style: one line, m at each dot, light/visual/random/
  Jev/oracle baselines, x "% visual"): scratch `window_baselines.py` + `tradeoff_data_all.py`. Jev point 66.28 @ 41.7%
  is from this file (no fx59_jev run folder). Open: batched version (m pages per GPU batch, stop if none enters top 10),
  real GPU timing, unit test for stop_early.
- K14 PADDLE RESULT (2026-10-29): outputs/k14_out_paddlevl.zip = 5580/15775 regions in 10.52 h on T4 x2 (meta pages_done 1339 is layout only;
  left 10195, 174 tables). 494/1334 pages complete (OHR 251, industrial 243, physics 0). Tables 57% of time; s/token flat in length,
  grows with crop area -> likely no KV cache. User: no setting change unless output stays the same. Built `--notebook cache_parity` ->
  `kaggle/14d_cache_parity.ipynb` (50 done regions re-read with use_cache=True, same everything else; PASS >=96% identical, NED<0.02,
  x1.5 faster; NO GAIN -> resume unchanged, est 20 GPU-h). Reader stage passes use_cache only if cfg sets it; 14c has USE_CACHE=None,
  not regenerated.
  PARITY PASS (outputs/k14_cache_parity.zip): 49/50 identical, 1 table NED 0.0045 (393->395 tok), x28.4 total (text med 1.48->0.40 s,
  table 114->5.4 s); model default use_cache False (generation + text_config). 14c_reader_paddlevl regenerated with USE_CACHE=True
  (paddlevl only), TIME_BUDGET_H 5.0 (user has 6 h Kaggle left). Resume est 3 GPU-h = ~1.5 h on T4 x2 (~0.8 h L4). Next: user runs
  14c-paddlevl with the paddle zip as the only dataset, then I score all 1334 pages.
- K14 FULL RUN (2026-10-28, user's call, no smoke): one reader per Kaggle session, all 1339 pages, pages split over
  the 2 GPUs: `kaggle/14_layout_regions_then_ocr_paddlevl.ipynb` -> `k14_out_paddlevl.zip`,
  `kaggle/14_layout_regions_then_ocr_xiaomi.ipynb` -> `k14_out_xiaomi.zip`. Colab dropped (needs browser open).
  Cut by user: PP-OCRv5-on-crops (flat PP-OCRv5 stays the baseline), fp32 check, smoke mode, Colab/shards.
  Layout runs in both sessions; scorer prints pages whose kept boxes differ. First attempt died at install (Paddle in
  Kaggle's Python broke its torch) -> Paddle in its own torch-free uv venv.
  Score: `$P ondemand/routing_axes/layout_ocr_score_against_ground_truth.py <d>/k14_out_paddlevl <d>/k14_out_xiaomi`.
  Xiaomi session log (2026-10-28): orient ~580 s, layout ~290 s, 15775 kept regions; 69 pages turned, incl. a wrong
  vidore_industrial 02ca p10 at 270 (0.928). Reader runs on slow reference kernels (no flash-linear-attention /
  causal_conv1d), so its timing is not Xiaomi's real speed. User: if it can't finish in 11 h, drop it.
  STAGED RUN BUILT 2026-10-28 (wrap 6), user running it: `layout_ocr_make_kaggle_notebook.py --notebook prep|wheels|reader
  [--reader X]` -> `kaggle/14a_prep.ipynb` (T4 x1 -> k14_prep.zip = dataset k14-regions), `14b_wheels.ipynb` (fla 0.6.0 +
  causal-conv1d 1.7.0 source build, kernel call, Xiaomi fused vs reference on 5 text/3 table/2 formula crops ->
  k14_wheels.zip = dataset k14-wheels), `14c_reader_{paddlevl,xiaomi}.ipynb` (T4 x2, crops from /kaggle/input, probe 5+50
  crops/GPU, abort if ETA > 95% of time left or xiaomi kernels not loaded, progress every 100, PYTHONUNBUFFERED).
  Order: A -> user pastes A cell 4 (turned + disagree list) -> C-paddlevl || B -> I hand-check B section 5 (kernels True
  fused / False ref, fused clearly faster, texts same) -> C-xiaomi -> score. Both old sessions stopped (user, wrap 6).
  Orientation rule: any text-layer char -> text direction (char-weighted line dir + page /Rotate, pymupdf 1.28.0 pinned);
  else PP-LCNet; PP-LCNet runs on all pages, both stored. CPU check on 1339: text 0 1223, text 270 38, lcnet 78
  (78 = no text layer, not 148; scans, still OCR'd); 02ca p10 -> 0; synthetic /Rotate 0/90/180/270 all upright.
  Prep zip keeps pages/ (size unmeasured). Old `kaggle/14_layout_regions_then_ocr_{paddlevl,xiaomi}.ipynb` no longer
  regenerable: user to decide keep/delete. Hidden OCR text layers follow the old engine's angle (only guard: disagree list).
- K14 layout + readers (`.plans/k14-layout-readers.md`): BUILT 2026-10-28. Next = user runs the Kaggle smoke
  (see K14 FULL RUN above), then I score it with
  `$P ondemand/routing_axes/layout_ocr_score_against_ground_truth.py <unzipped>/k14_out` and review loops / orientation / fp16 before the full run.
  Files: `routing_axes/layout_ocr_build_pages_and_ground_truth.py` (page list + references), `layout_ocr_score_against_ground_truth.py` (+ `otsl_to_html`), `layout_ocr_score_against_ground_truth_test.py`,
  `layout_ocr_make_kaggle_notebook.py` (generates the notebook). References `routing_axes/k14/reference.jsonl` (work dir):
  OHR = OHR-Bench_v2.parquet gt_text (v1 covers 1/65 docs), ViDoRe = corpus markdown; 0 missing; index shift 0 wins
  1323/1339 (no off-by-one). Tables: OHR LaTeX tabular 67 pages, ViDoRe md 222. Flat PP-OCRv5 vs refs (CPU, in memory):
  NED 0.287, word P 0.832 R 0.846 (handwritten R 0.449); loop flag fires on 2.5% of flat pages (legit repeats) = floor.
  Confirmed settings: 144 dpi; PP-LCNet orientation rotate only; drop image/chart; greedy, no guard, max new 1024/2048/512,
  loop = 8-gram x4; transformers fp16 sdpa + fp32 check 5 text + 5 table crops; HTML tables (OTSL converted);
  T4 x2 (PaddleOCR-VL GPU0 || Xiaomi GPU1); pins PP-DocLayoutV3_safetensors e4f489dc, PaddleOCR-VL-1.6 c5630aba,
  Xiaomi-OCR-0 6e75d0cb; normalization NFKC + strip md/HTML/LaTeX + lowercase; NED = Levenshtein/max len; word P/R
  multiset; TEDS (apted, best match per ref table); reading order OmniDocBench style (blank-line paras, Jaccard >= 0.3);
  ViDoRe = markdown agreement on all + hand-check 20 ind + 20 phys (viewer not built; ask first).
  rapidfuzz 3.14.6 + apted installed into axiom-de-rd env (user approved).
  Scorer scores every reader on pages complete for ALL readers (rows == layout n_kept), prints excluded per reader;
  writes `routing_axes/k14/score_{smoke|full}.json` (or --tag). Full-run zip keeps crops/ (resume needs them).
  Defaults CONFIRMED by user 2026-10-28 (tweak after smoke); smoke dataset uploaded, run started by user: layout post-process default threshold, fp32, no extra NMS (overlap
  count printed); reader image size = processor defaults (no max_pixels); seal -> PaddleVL seal prompt / Xiaomi text;
  paddle 3.0.0 + paddleocr>=3,<4 unpinned, PP-LCNet/PP-OCRv5 weights unpinned; PP-OCRv5 on crops orientation off.
  Smoke risks (unverified): torch 2.14 wheel on T4 driver (TORCH_INDEX cu126; probe cell stops if no GPU);
  layout result order assumed = reading order (card prints "Order"); PIL rotate(angle) CCW assumed to match PaddleX;
  Paddle 3.0.0 vs paddleocr 3.x resolve; OTSL tags must survive decode (special tokens stripped except OTSL).
- K13 DONE 2026-10-28: merged, check pages 100 corr 1.000000 max diff 0 -> `routing_axes/k13/scores_v2_all.npy`
  (220 x 5091) + `meta_v2_all.json`. `evie_cached:skip_blank` wired in visual.py (uncommitted). Teammate zip
  `k13_v2_scores_cache.zip` (repo root, gitignored; unzip at repo root). Check run `results/xk_evie_v2_214` (--exclude-flawed): 214 q R@10 69.71 (ind 56.37, phys 59.36) = K11
  target exactly; full xk_evie 70.52. `results/xk_evie_v2` = same run on 220 q (70.38, do not quote; user may delete).
- Also open: user picks the next cost lever beyond V2 (V2 accepted, ~30% encode saved, "not good enough"):
  (a) engineering speed-ups on top of V2 (batching / kernels / int8; declined as P3 on 10-27, recall-neutral),
  (b) accept more recall loss for fewer tokens (e.g. V2 @256), (c) stop at V2 and write it up.
  Everything else is used up or closed: crop (subset of V2), V4 (8%), V5 (fails), font-size resolution (closed 10-27),
  routing (Jev -4.2), duplicates (0).
- User owes:
  - approve scorer diff: `evie_blankdrop12.py --timing <dir>` + v1/v2_again rows + V5 shift print (shown in chat, not applied).
  - commit K12 files (nothing staged): builder, unit test, kaggle/12 and colab/12 notebooks; older uncommitted
    dedupe/Jev edits in ondemand/adaptive/*, routing_axes/, kaggle 08-11, two crc json.
  - keep the deleted report files deleted? (orig.docx, v4.pdf, 3 Bao_cao files, Mau bao cao .doc, nhat ky .md,
    baseline/improved png+svg)
  - delete by hand `routing_axes/tl/` and `routing_axes/strategies/sq_input*` (permission rule blocks it).
  - Chandra settings in services/chandra.py (192 dpi, ocr_layout, 12384 tokens); QA reader provider pinning.
  - xgb sheet row 57.7 / 87.3 not reproduced by default: ask haidang241 which cell.
  - amend Decision 2026-10-16 "CRC fails safe"? (LOCO @256 physics eps 0.05 -> 100% skip, loss 3x eps).
  - `k12_sizing.py` reads sys.argv[1] at import (cannot be imported cleanly): move into __main__?
- Unexplained, unverified: K12 Kaggle V2 0.982 s/page vs K11 0.810 and Colab 0.939 (K12 outlier); V1 -20% (Colab) vs
  -6% (K11); render 0.07 vs 0.342 s/page between K11 and k10 bases. Quote ratios, not K12 absolute s/page.

## Current settings
- Queries: 214, 6 flawed excluded (`ondemand/bench.py` FLAWED_QIDS). 2026-10-15. Never quote 220-query numbers.
- Pipeline `$P -m ondemand.adaptive run`: gate -> k (`kpolicy.py`) -> router -> branch -> `chunk_rank.py` (512 words /
  128 overlap, hybrid alpha 0.7, top 10, reading order) -> QA. 2026-10-15.
- Pool default `gate_ppocrv5_ungated.json` (`ondemand/adaptive/__main__.py --gate`), top-100. Light top-20, visual top-50.
  Cost studies use adaptive-K pools `kaggle/k11_pools_xgbK214.json` (K mean 59.3) or fixed K=59 (`fx59_*`).
- Text ppocrv5 only (`light_prep/pages_ppocrv5.jsonl`; native PDF text, PP-OCRv5 where it is missing or weak, no Tesseract;
  the old Tesseract cascade is `pages_ocr_ppocr.jsonl`, not used). Visual EVIE-4.5B 70b8a2be,
  fp16, sdpa, 144 dpi, MVT 1024, 2048-dim MaxSim, transformers 5.13.1.
- Visual cost method V2, accepted 2026-10-28; wired into `ondemand/adaptive` as scorer `evie_cached:skip_blank` (cached scores from Kaggle nb 11/13; no live encoder). Variants:
  V1 blank squares dropped before LLM only; V2 blanks skip ViT+LLM; V4 = V2 + filler tokens; V5 = reuse repeated squares.
- Router configs `adaptive/crc_stage0_{fx59,xgbK}.json` via `--router crc:<file>`; dedupe `--dedupe 0.6` (uncommitted).
- QA harness locked 2026-10-15 (`ondemand/qa.py`, `ondemand/adaptive/qa.py`): reader deepseek-v4-flash, judge
  gpt-4o-mini, temp 0, seed 0, 512 tokens; ~35 Correct flips per fresh draw. QA context top 10.
- Timing: cold, T4 batch 1. Latest basis Colab T4 2026-10-28 (`outputs/k12_out_timing.zip`, 100 pairs, shuffled order).
  L4 is the deployment GPU: spec estimate /1.86 until measured.

## Key numbers (214 q)
- Blank-drop family (R@10, adaptive-K pools; R@10 from Kaggle 11/12, s/page ratio from Colab 2026-10-28):
  full 70.52 / 1.339 s/page; V2 69.71 (-0.82 [-2.28,+0.38], industrial -2.28) / -30%; V4 69.93 (-0.59 [-1.47,+0.09]) / -8%;
  V1 68.78 (-1.75) / -20%; V3 67.73 (-2.79); V5 67.28 (-3.24 [-5.80,-0.94], industrial -9.23) / -14.5%.
  Results: `outputs/k11_out_*.zip`, `outputs/k12_out_*.zip`; plans `.plans/k11-evie-blankdrop.md`, `.plans/k12-evie-blankdrop-next.md`.
- Squares kept (CPU count, scratch crop_count.py): V2 69.9% (ohr 64.4, ind 47.9, phys 94.9); crop alone 89.6%;
  crop + V2 69.9% (= V2). V5 reuses 53.9% of squares on 85% of pairs.
- Cost frontier, fx59 page R@10 / cold EVIE s/q (`$P -m ondemand.routing_axes.evie_pareto`, old 1.6 s/page basis):
  light 61.43 / 0; @256 68.29 / 43.5; @1024 70.32 / 94.5; oracle {light,@1024} 71.35 / 23.8; Jev->full 66.28 / 39.4.
- Closed speed-ups 2026-10-27: cascade @256 -> top 20 @1024 69.39 (-0.93); small-print resolution rule 69.44 / 73.2 s/q,
  on the @256-@1024 line (`evie_smallprint`); 0 exact duplicate pages.
- Routing: CRC does not transfer (LOCO); signals on light's output AUC 0.47-0.57; ingest cues AUC 0.45-0.53.
  The gap is EVIE's scoring (document choice), not visual-only pages.
- Retrieval chunk R@10 ungated: light 60.2, EVIE k=50 69.3; any-gold 92.5. QA C / C+P: ungated_evie_m10 52.34 / 88.79,
  light 49.53 / 88.79, gold-page ppocrv5 56.07 / 91.12, gt_text 60.75 / 96.26 (headroom almost all OHR).

## In progress
- K13 (2026-10-28, merged; see START HERE): V2 on the 1901 corpus pages K11 skipped + 100 check pages ->
  `kaggle/13_evie_skip_blank_squares_remaining_pages.ipynb` (copy of 11, V2 pass only, list built in-notebook from k3_input - k11-pools; no new dataset), `ondemand/routing_axes/evie_skip_blank_squares_merge_scores.py <k13_out>` merge ->
  `routing_axes/k13/scores_v2_all.npy` (220 x 5091, colvec_keys order). Then `evie_cached:skip_blank` in visual.py (diff not
  applied) and check run `xk_evie_v2` (target 69.71). Needs K11 zip unzipped to `routing_axes/k11/` first.
- Chunk/rank input = EVIE top-10 pages -> ~11.2 chunks/query (median 10), top-10 chunk cut drops 0 gold pages
  (any-gold 92.5 pages = 92.5 chunks). Ranking inside it is a no-op for recall.
  Tokens (o200k, 214 q, ungated_evie_m10): top-10 page text mean 4522 / median 4067 / p90 7510; QA context after cut
  mean 3964 (ohr 5200, ind 4750, phys 2280). Cut trims text in 97/214 queries; 34 of 433 gold-page appearances lose a
  chunk (24 queries); short-answer string match finds no answer loss (answers paraphrased, needs the oracle).
- K14 layout + readers (plan only, 2026-10-28): PP-DocLayoutV3 regions on the 1339 unique context pages
  (ind 612, phys 472, OHR 255) -> readers PP-OCRv5 / Xiaomi-OCR-0 0.8B / PaddleOCR-VL-1.6 0.9B, scored against
  ground-truth text (OHR gt_text trusted; ViDoRe markdown agreement only). Reserves TeleOCR 1.2B, OvisOCR2 0.8B.
  Region chunks feed a later `region` chunker plugin. `.plans/k14-layout-readers.md`.
- KDL cache `kdl/kdl_pages.jsonl` (5091 pages, nano, 144 dpi, L4): gold-page QA on 214 C 50.93 vs PP-OCRv5 56.07
  (OHR hand 53.3 vs 70.0, scan 80.0 vs 73.3, ind 33.3 vs 42.7, phys 55.7 vs 57.0). Handwritten losses are repetition
  loops; user thinks OHR tilted/flipped pages trigger them (unverified, check images in K14 smoke).
- Scorer `evie_blankdrop12.py`: `--timing` diff pending approval (see START HERE).
- Step 3 QA: Chandra re-parse of context pages; judge-free answer-in-text oracle on OHR short answers (ppocrv5 22/40),
  then QA vs 56.07 / 60.75. Not started; settings unconfirmed. Layout model joins here (Decision 2026-10-28).
- Ingest rule (approved idea, not coded): page with hidden text layer (render mode 3 / GlyphLessFont) -> PP-OCRv5.

## Decisions
- 2026-10-28 (wrap 4) Gap is latency: target the visual-only-where-it-gains oracle. Reopens light-vs-visual routing
  (replaces "routing stays off"); recall rule holds: a router must show ~no R@10 loss vs always-EVIE.
- 2026-10-28 Recall first, cost second: no method or router may give up R@10 beyond what is unavoidable.
- 2026-10-28 V2 accepted as the visual cost cut: -0.82 R@10 (n.s.) for ~30% encode.
- 2026-10-28 V5 closed: -3.24 R@10, copied squares shift scores unevenly. V4 not taken: 8% saving.
- 2026-10-28 Content crop closed: its squares are a subset of V2's; only an alternative to V2 (~10%), not an add-on.
- 2026-10-28 (superseded wrap 4) Light-vs-visual routing stays off under recall-first; keyword query-intent router dropped before testing.
- 2026-10-28 Layout model deferred to QA phase (reading order, tables, region chunks, at ingest with PP-OCRv5).
- 2026-10-28 (wrap 3) Chunk/rank input stays top-10 pages; no whole pages to the reader (cost, small target readers).
  Layout runs on QA context pages only, quality over speed, must host on Kaggle/Colab. Parsers judged against
  ground-truth text, not QA, until QA credits are known.
- 2026-10-27 CRC does not transfer across collections; exact-duplicate sharing closed; cascade and resolution rule
  closed; cold timing only; no budget knob, no ~250M arm, no P3 speed-ups (user).
- 2026-10-23 Adaptive K adds no measurable retrieval gain over fixed K=59.
- 2026-10-20 No more signal hunting on light's output; better parsing is not the retrieval lever.
- 2026-10-15 and earlier (ungated pool, EVIE as visual arm, closed list): archive.

## Test status
- K12 notebook unit test (repo root): `$P -m pip install -q "numpy<2" --target <dir>` then
  `PYTHONPATH=<dir> $P ondemand/routing_axes/k12_nb_test.py colab/12_evie_timing_colab.ipynb` ->
  last line `v4_plan / v4_assemble / Fillers ok; all-blank and no-blank pages ok`. 2026-10-28.
- V5 on 214 (repo root, scratch `v5_full.py`): `all 214 full 70.52 v5 67.28 d -3.24 [-5.80, -0.94]`.
- Colab timing: `unzip -p outputs/k12_out_timing.zip k12_out/timing.jsonl | $P -c "import sys,json,statistics as s;
  r=[json.loads(l) for l in sys.stdin]; [print(v, round(s.mean(x['encode_s_'+v] for x in r),3)) for v in
  ['full','v1','v2','v2_again','v4','v5']]"` -> `full 1.339 v1 1.069 v2 0.939 v2_again 0.941 v4 1.226 v5 1.145`.
- Crop count (~5 min): `$P <scratch>/crop_count.py unused` -> `all | 214 | 100.0 | 69.9 | 89.6 | 69.9 | 96.7 | 74.5`.
- Scratch (may be wiped): `/private/tmp/claude-501/-Users-khoatran-Desktop-ISE-Axiom-DE-RnD-AXIOM-DE-RD/8edac3fd-d971-44e9-a05a-6e2aae01bf50/scratchpad/`.
  QA commands and the `sufficiency.py` warning (never import; ~640 API calls): archive "Test status".

## Facts to keep
- EVIE right-document ceiling measured 2026-11-02 (E0): perfect document choice, all pages of gold docs, 214 q 80.84 full EVIE /
  80.70 skip_blank, ViDoRe 73.38 / 73.18. Uses gold labels: an oracle, not a reachable target. The older "~80" was a ColVec-era 3-arm oracle.
- OHR (60 q) ~100 on every visual arm; only ViDoRe separates arms. Physics: English queries over French pages.
- EVIE scores + token cache: `routing_axes/k3/k3_out_shard0`; @256 scores `k3/k10_out`. Kaggle T4 x2, Colab T4 x1.
- Colab input bundle `k12_colab_input.zip` (repo root, gitignored): k3_input + pools; `k3_input/pdfs` are symlinks.
- Kaggle sessions can come up with no GPU (nvidia-smi missing): set Accelerator T4 x2 and check quota before committing.
