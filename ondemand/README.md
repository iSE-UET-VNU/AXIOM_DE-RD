# On-demand retrieval: handoff guide

This folder holds the on-demand pipeline (light retrieval, then accurate operators on a small pool), the routing
experiments, and the adaptive pipeline scaffold in `adaptive/`. This guide is for whoever continues the work: how to
set up, how to reproduce every number below, what we learned, and what to do next.

The pipeline developer guide (services, GPU runs, flags) is in [adaptive/README.md](adaptive/README.md).

## 1. The task

- Benchmark: `0.1. BENCHMARK/`, 220 queries over 103 PDFs (5,091 pages), three sources: ViDoRe industrial, ViDoRe
  physics, OHR-Bench (scan and handwritten).
- We report on **214 queries**. Six are excluded as flawed (`FLAWED_QIDS` in `bench.py`). Never quote a 220-query number.
- Metric so far: page-level R@10. The pool is light retrieval's top 100 pages per query.
- Two arms: **light** (keep light retrieval's order) and **visual** (re-sort the pool with ColVec).

Goal for the next phase:

1. Find the tools each kind of query needs, to push R@10 toward 80.
2. Then design a router that is QA-facing: it applies operators step by step, checks whether the evidence is enough to
   answer, and stops as early as it can. Two targets: best recall and lowest cost. Worst case is "use every tool".

## 2. Setup

```bash
conda create -n axiom-de-rd python=3.11 && conda activate axiom-de-rd
pip install -e .
cp .env.example .env        # only OPENROUTER_API_KEY is needed, and only for runs with --allow-api
```

Then get the data bundle (`handoff_bundle.zip`, about 660 MB, shared outside git) and unzip it at the repo root:

```bash
unzip -o handoff_bundle.zip
python -m ondemand.adaptive doctor
```

The zip restores `0.1. BENCHMARK/` and `data/work/ondemand_v2/ef7f11811803/`. Both are git-ignored. The folder name
`ef7f11811803` is a fingerprint of the benchmark's `documents.jsonl`, `queries.jsonl` and `qrels.jsonl`; every cache is
keyed by it, and the code refuses caches from a different benchmark. The zip also carries a copy of `ondemand/` and
`pyproject.toml` from the commit it was built on, so `git status` should stay clean after unzipping.

To rebuild the zip: `python -m ondemand bundle --target handoff` (writes `data/work/ondemand_v2/<fp>/upload/`).

### What is in the bundle

| Path under `data/work/ondemand_v2/ef7f11811803/` | What it is |
|---|---|
| `light_prep/gate_ppocrv5_backfill.json` | light retrieval output: top 100 pages per query (`light_run_top100`), settings, gold ranks |
| `light_prep/pages_ppocrv5.jsonl` | text of every page (pdf-inspector, PP-OCRv5 on pages that need OCR) |
| `light_prep/qrels_page_level.jsonl` | page-level gold |
| `colvec/colvec_scores.npy` + keys, qids, meta | ColVec score of **every query against every page** (220 x 5,091) |
| `embedding_cache/`, `light_v2/`, `embeddings/kdl` | dense embedding caches, so light retrieval reruns without API calls |
| `kdl/` | KDL parses of the pages |
| `routing/labels_ladder_k100.jsonl` | per query: group, fold, R@10 of each arm, which arm wins |
| `routing/decisions/ladder_k100/` | decisions of every router we tested |
| `results/adaptive_regression`, `oracle_gold`, `oracle_image`, `legacy_fullcorpus`, `plain_bm25_colvec` | cached QA answers and verdicts |
| `qa_cache/answers`, `qa_cache/judge` | generator and judge API cache |

Because the ColVec matrix covers the whole corpus, re-sorting a pool of any size with ColVec costs nothing locally.
Rendered page images (`qa_cache/page_images`, 112 MB) are not in the bundle; the PDFs they come from are.

## 3. Reproduce the numbers

All of these read caches and call no API. The first three only print; the last also rewrites
`routing/scores_ladder_k100.jsonl`.

```bash
python -m ondemand.miss_breakdown                 # section 4.1 and 4.2
python -m ondemand.miss_breakdown --per-query     # plus one JSON row per query with a fixable miss
python -m ondemand.qa_vs_recall                   # section 4.3
python -m ondemand.routing_ladder score           # every router against random and fixed depth
```

Other entry points, each with `--help`:

| Command | Purpose |
|---|---|
| `python -m ondemand.adaptive run / query / call / prep / doctor` | the adaptive pipeline, see `adaptive/README.md` |
| `python -m ondemand light-retrieval` | rebuild a gate file (light retrieval) |
| `python -m ondemand.routing_ladder labels` | rebuild the label file from a gate and the ColVec cache |
| `python -m ondemand.routing_labels`, `routing_score`, `routing_random` | older two-arm labels, scoring, random curve |
| `python -m ondemand.routing_api` | baselines run from their own pinned repos: UniversalRAG, AutoMix |
| `python -m ondemand.routing_rerank`, `routing_docjudge` | RankGPT rerank and document judge (both failed, kept for reference) |
| `kaggle/05`, `06`, `07` | RetrievalRouter, T5Gemma and system-1 routers on a T4 |

Anything that calls an API refuses to run without `--allow-api`.

## 4. What we know

### 4.1 Baseline, R@10 on 214 queries, pool of 100

| Group | n | Light | Visual | Best of two | Ceiling | Gold in pool | ColVec, whole corpus |
|---|---|---|---|---|---|---|---|
| All | 214 | 61.20 | 67.77 | 69.51 | 94.30 | 87.50 | 68.60 |
| Industrial | 75 | 47.04 | 51.96 | 54.91 | 92.72 | 74.27 | 53.82 |
| Physics | 79 | 46.43 | 58.31 | 60.23 | 91.48 | 90.57 | 58.79 |
| OHR scan | 30 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| OHR handwritten | 30 | 96.67 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |

- "Best of two" picks the better arm per query with hindsight. It is the most any light-or-visual router can reach,
  so **routing between these two arms cannot get to 80**. New tools are needed.
- "Ceiling" is below 100 because 35 queries have 11 to 28 gold pages and only 10 slots.
- ColVec over the whole corpus is no better than ColVec over the pool of 100.

### 4.2 Failure cases

Measured against the better arm per query. The four columns of lost points add up to 100 minus "best of two".

| Case | Meaning | All | Industrial | Physics |
|---|---|---|---|---|
| 0. More than 10 gold pages | cannot be fixed under R@10 | 5.70 | 7.28 | 8.52 |
| 1. Not in the pool | gold page is outside light's top 100 | 11.01 | 22.96 | 8.03 |
| 2. Wrong document on top | gold is in the pool, other documents fill the top 10 | 6.23 | 5.64 | 11.52 |
| 3. Right document, wrong page | the top 10 is mostly the gold document, not the gold pages | 7.55 | 9.22 | 11.71 |

- 98 queries have a fixable miss (46 industrial, 52 physics, none in OHR). 19 have zero recall (14 industrial, 5 physics).
- Industrial is a pool problem. Whole-corpus ColVec finds only 25 of the 194 out-of-pool gold pages.
- Physics is a ranking problem: 221 of its 264 missed in-pool pages sit at ranks 11 to 50.
- Missing OCR text is not the cause: missed and found gold pages have the same text length.
- Cases 2 and 3 are split by a rule: if at least half of the non-gold pages in the top 10 come from a gold document,
  it is case 3, otherwise case 2. Near-duplicate pages are not separated yet.

### 4.3 Retrieval against answer quality

Cached QA run `adaptive_regression`: visual arm, pool of 20, top-10 chunks as context, generator
`deepseek/deepseek-v4-flash`, judge `openai/gpt-4o-mini`, temperature 0, seed 0.

| Gold pages in the context | Queries | Correct | Partially Correct | Incorrect |
|---|---|---|---|---|
| None | 32 | 6 | 16 | 10 |
| Some | 73 | 30 | 37 | 6 |
| All | 109 | 64 | 24 | 21 |
| Total | 214 | 100 (46.7%) | 77 | 37 |

- Having some gold versus all of it barely changes the answer. Having none versus some changes it a lot.
- Gold pages only, same text reader (`oracle_gold`): 50.9% Correct. Perfect retrieval adds about 4 points.
- Gold pages read as images (`oracle_image`): 55.6% Correct, 12 Incorrect. The reader limits QA more than retrieval.
- OHR finds all gold on 59 of 60 queries yet 16 answers are Incorrect; image reading halves that.
- The rows are not a clean comparison: queries with all gold found tend to have fewer gold pages.
- These QA files use a pool of 20, not 100. Do not mix them with the pool-of-100 retrieval numbers.

### 4.4 Routers tested (all on the two arms, R@10, 214 queries)

A router passes only if it beats random routing at the same visual rate (95th percentile over 1,000 seeds) and a
fixed-depth re-sort at the same number of pages.

- RetrievalRouter (released checkpoints and retrained), T5Gemma, AutoMix, UniversalRAG with GPT-5 (61.72 against a
  random bar of 62.42), score-based signals, RankGPT rerank, document judges: none passes.
- Of 29 signals available before calling ColVec, only one separates "visual helps" from the rest: dense WIG@5, and
  mostly on physics. A one-threshold router on it reaches 66.90 at 54% visual calls (always visual: 67.77).
- 148 of 214 queries are ties, 48 are visual wins, 18 are light wins. That is too few labels to train a router.

## 5. Plan

1. **Read the failure cases.** Build a small viewer with one tab per case in 4.2: query, gold pages (image and text),
   both top-10 lists, where each missed gold page ranked. Tag what the evidence is (text, table, chart, diagram).
   `miss_breakdown --per-query` already gives the rows.
2. **Pool recall against pool size.** The gate stores only the top 100 (`POOL` in `light_retrieval.py`). Raise it and
   rebuild the gate at 200, 500, 1,000 and the whole corpus, then rerun `miss_breakdown` to see how case 1 turns into
   cases 2 and 3. The settings of the current gate are recorded inside the gate file (`variant`, `file_k` 3,
   `w_dense` 0.7, cache `te3s_ppocr`); with the embedding cache this needs no API.
   A larger pool alone will not help, since whole-corpus ColVec scores 68.60. It only helps together with a tool that
   ranks differently.
3. **One tool per case, each with a cost unit** (pages scored, model calls, seconds):
   - Case 1: query expansion or rewriting, a text index of VLM page descriptions, larger pool.
   - Case 2: a document-identification step before page ranking.
   - Case 3: in-document page ranking, neighbour pages, near-duplicate removal.
   - Any case: a visual reranker over the top 20 to 50.
   - Reader: answer from page images when text reading fails.
   - Sufficiency judge: "is this evidence enough to answer?". It is an operator too and has a cost.
   These are candidates. None is tested yet.
4. **QA-facing router as a cascade.** Run a cheap operator, check sufficiency, stop or escalate. Evaluate it against
   three baselines before training anything: every operator on every query, a fixed-order cascade with the stop rule,
   and the same cascade stopping at random at the same rate.

### Open decisions

- Metric for the router: answer correctness with top-10 context against cost, with R@10 as a diagnostic, or R@10 alone.
  "At least one gold page in the top 10" matches 4.3 better than full R@10.
- Whether a large pool replaces the upstream adaptive top-k (mean about 88 pages) or sits after it.
- Judge model, prompt and which arm's pages feed the sufficiency check.
- The cost unit.

## 6. Things that will bite

- Use the `axiom-de-rd` environment. The base Python lacks the dependencies.
- torch 2.2.2 on the Mac blocks local rerankers and embedders; GPU work goes to Colab or Kaggle
  (`colab/04_adaptive_gpu_run.ipynb`, bundle target `adaptive`).
- `router.py` in `adaptive/` is a scaffold: `rule` and `fixed` only. The learned or cascade router is not written.
- Chandra (enrich branch) changes text for QA, not page order. It is not a retrieval arm.
- Baselines from other papers run their own code at pinned commits (`routing_api clone`); only the input wrapper is ours.
- A run that reports fallbacks did not run the configured pipeline. Check the end-of-run banner.
