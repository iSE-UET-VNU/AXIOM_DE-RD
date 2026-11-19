# Data discovery, Qwen setting (2026-11-14)

This folder holds the code, inputs, caches and current outputs of the **data discovery** stage of the AXIOM_DE-RD
retrieval pipeline. It is self-contained: it does not import anything else from the repo. Only the 5,091-page text
and page vectors are not in this folder; they are read from the repo's local data caches (see Quick start). Use it to:

- reproduce and check the reported numbers;
- start **accurate retrieval** from `outputs/data_discovery_output.jsonl`.

The data discovery stage has three parts:

```text
1. light preparation   1a data preparation  : page text (pdf-inspector + PP-OCRv5) -> BM25 index + Qwen3-Embedding-8B page vectors
                       1b query preparation : Qwen3-32B (DeepInfra, no reasoning) splits the question into SCOPE / TOPIC / ANCHOR / DISCARD
2. light retrieval     question -> top-5 files -> up to 100 ranked pages (BM25 on the concepts + dense on the full question)
3. adaptive retrieval  XGBoost picks K per question; the first K pages are passed on
   ---> accurate retrieval (next stage, not in this folder) -> answer generation / QA
```

**Official question set: 214 questions.** These are the 220 benchmark questions minus the 6 in
`inputs/excluded_questions.json`, which the user excluded on 2026-11-14.
- Outputs, metrics and the adaptive-K model are all computed on those 214 questions.
- `outputs/` holds exactly one result: the chosen setting (Qwen3-32B on DeepInfra, 214 questions).

QA and accurate retrieval (EVIE) are not part of this folder and are not reported here.

### Headline numbers (214 questions)

| Stage | Metric | Value |
|---|---|---:|
| Light retrieval (top-5 files, up to 100 pages) | Recall@10 / Recall@20 / Recall@100 | 67.36 / 78.11 / 93.16 |
| Light retrieval | nDCG@10 / nDCG@20 | 62.82 / 66.28 |
| Light retrieval | gold file in the 5 selected files | 97.66 |
| Adaptive K (nested OOF) | mean K / prefix recall@K | 58.86 / **90.82** |

Recall@100 (93.16) is the recall of the full 100-page list. The recall of the pages actually passed on to accurate
retrieval is the adaptive-K prefix recall, **90.82** at mean K 58.86. Do not mix the two.

## Quick start (no API key needed)

The 5,091-page corpus data is **not copied** into this folder, to keep it small (8.9 MB). Two files are read from the
local repo caches that already exist on the team machines (`data/work/` is git-ignored, so they are not in git):

| What | Default path (relative to the repo root; this folder sits in `<repo>/research/data_discovery_qwen/`) | Override |
|---|---|---|
| Page text, 11 MB | `data/work/adaptive_routers_handoff/adaptive_routers_handoff/data/work/ondemand_v2/ef7f11811803/light_prep/pages_ppocrv5.jsonl` | `--pages <file>` or env `DD_PAGES` |
| Page vectors, 83 MB (Qwen3-Embedding-8B, 5,091 × 4096) | `data/work/Cached-Ondemand-Eval/data/work/ondemand_v2/ef7f11811803/light_v2/dense_qwen8b_ppocr_pages.npy` | `--page-vectors <file>` or env `DD_PAGE_VECTORS` |

`cache/vectors_meta.json` holds the 5,091 page ids in order. The code checks that both files match them.

The repo root is found automatically: the first parent folder of this one that has `data/work/`. Set `DD_REPO=<repo root>`
when the data caches live in another checkout (for example a git worktree without `data/work/`).

```bash
cd research/data_discovery_qwen
pip install -r requirements.txt          # numpy scipy scikit-learn xgboost PyStemmer requests
python run_all.py                        # replays the chosen result (DeepInfra, 214 questions) from cache/, ~30 s, no API call
python run_all.py --no-ablations         # same, without the ablation rows of metrics.json (faster)
python run_all.py --pages <pages_ppocrv5.jsonl> --page-vectors <dense_qwen8b_ppocr_pages.npy>   # data files elsewhere
DD_REPO=<path to the repo that has data/work/> python run_all.py                                # data caches in another checkout
```

If the page text file is missing, the run stops with `page text not found: ...`. Ask for the two files (11 MB + 83 MB)
or point `--pages` / `--page-vectors` at your copy. The page text cannot be rebuilt by this folder; the page vectors can
(see Live mode).

The last line must be `reproduction check (DeepInfra, 214 questions): OK, all 10 numbers match`.
- The script rebuilds every file in `outputs/` from `inputs/`, `cache/` and the two repo files.
- It compares the main metrics with the chosen result of 2026-11-14. Those values are hard-coded in `run_all.py` as `EXPECTED`.

**Live mode** calls OpenRouter only for items missing from `cache/`, so use it for new questions or a new corpus:

```bash
OPENROUTER_API_KEY=... python run_all.py --live      # Qwen3-32B is always pinned to DeepInfra
```

- If the page vectors are missing, `--live` re-embeds the corpus: about 20 min of OpenRouter calls, saved to `cache/page_vectors_qwen3-embedding-8b.npy` (git-ignored, 83 MB). This needs `tiktoken`.
- Never commit an API key. The code reads `OPENROUTER_API_KEY` from the environment or from a `.env` file (git-ignored).

## Benchmark

| | |
|---|---|
| Questions | 214 official (220 minus 6 excluded): `vidore_physics` 79 (English questions on French course slides), `vidore_industrial` 75 (USAF technical manuals and checklists), `ohrbench` 60 (single-fact questions) |
| Corpus | 103 files, 5,091 pages |
| Page id | `<doc_id>#page=<zero-based page>` |
| Doc id | `<source>::file_<hash>`; `inputs/documents.jsonl` maps it to the PDF path. The PDFs are not copied; they live in the repo under `data/raw/0.1. BENCHMARK/data/` |
| Qrels (`inputs/qrels.jsonl`) | 1,083 (question, page) pairs over the 220 questions, graded: 2 = fully relevant, 1 = partly relevant (OHR pages are 1). Used **only** for labels and evaluation, never as an input to ranking or K |
| Excluded | `inputs/excluded_questions.json`: 5 OHR questions and `vidore_physics::326`. All 6 were retrieved well (gold page at rank 1 or 2), so excluding them lowers the averages slightly |

## Folder layout

```text
README.md                  this file
.gitignore                 keeps live-mode page vectors and .env out of git
run_all.py                 runs stages 1 -> 3, writes outputs/, checks the numbers
requirements.txt
data_discovery/
  common.py                paths, prompt, OpenRouter client, language detection, rank-matched hybrid, metrics
  text.py                  enfr analyzer (English + French stemming), PageBM25, real_text (copied from ondemand/text.py)
  light_preparation.py     stage 1: prepare_data() and prepare_queries()
  light_retrieval.py       stage 2
  adaptive_retrieval.py    stage 3
inputs/
  queries.jsonl            220 questions (query_id, source, query, answers = reference answers for later QA)
  qrels.jsonl              query_id, page_id, relevance
  documents.jsonl          103 files: doc_id, path, page_count, ...
  excluded_questions.json  the 6 questions left out of the official set
  (page text: NOT in the bundle, read from the repo, see Quick start)
cache/
  query_vectors_qwen3-embedding-8b.npy   220 x 4096, row order = vectors_meta.json "qids"
  vectors_meta.json                      model, page_ids (order of the page vectors), qids, how texts were built, timings
  llm_decomposition/DeepInfra/           220 raw Qwen3-32B responses + .meta.json (latency, provider, cost, tokens);
                                         file name = sha256("qwen/qwen3-32b\n" + prompt) + ".txt"
  (page vectors: NOT in the bundle, read from the repo, see Quick start)
outputs/                   the chosen result (DeepInfra, 214 questions); regenerated identically by run_all.py
  query_preparation.jsonl  per question: scope, topic, anchor, discard, answer_shape, bm25_text, fallback flags, LLM call info
  light_retrieval_top100.jsonl  qid, selected_files (top 5), n_candidates, ranked_pages, scores
  adaptive_k.jsonl         qid, k (nested OOF), k_final_model, target_k, n_candidates, oof_prediction_raw
  adaptive_model.json      XGBoost model fit on the 214 questions (xgboost JSON format)
  adaptive_calibration.json  features, params, scale / margin of K = round(pred*scale + margin), per-fold calibrations
  data_discovery_output.jsonl   <- input for accurate retrieval (schema below)
  metrics.json             all metrics per stage and source, ablations, reproduction check
  timings.json             timings of this run (decomposition latency taken from the cached call meta)
provenance/
  build_bundle.py          how inputs/ and cache/ were assembled from the repo (author's machine paths)
  original_run_timings.json  timings of the original API runs (preparation, embeddings, decomposition)
```

## Stage 1: light preparation

### 1a. Data preparation

- **Page text:** pdf-inspector native text for 4,210 pages. PP-OCRv5 text for 881 scanned or image pages:
  - 482 pages pdf-inspector + PP-OCRv5;
  - 399 pages PP-OCRv5 only.
- **Cleaning:** `real_text()` removes `[Image: ...]` placeholders before indexing.
- **BM25:** `PageBM25` with k1 = 1.2, b = 0.75, `enfr` analyzer (lower-case, stop words, English + French Snowball stems).
- **Dense:** Qwen3-Embedding-8B (`qwen/qwen3-embedding-8b` via OpenRouter), **one vector per page**, no 512-word chunks.
  - Page text is clipped to 6,000 cl100k tokens.
  - 154 empty pages (149 Industrial, 5 OHR) have a zero vector. None of them is a gold page.

| Step | Time |
|---|---|
| pdf-inspector, original run | 27.4 s for 5,091 pages (5.4 ms/page) |
| PP-OCRv5, original run | about 0.9 s/page (render + OCR) |
| Page embedding, original run | 1,199 s for the corpus (about 0.24 s/page, batch 32) |
| Replay: load pages + BM25 build + load vectors + language detection | 2.5 s |

### 1b. Query preparation

- **Model:** Qwen3-32B via OpenRouter, **pinned to DeepInfra** (`provider.order = ["DeepInfra"]`, no fallback), **reasoning off** (`reasoning.enabled = false`, 0 reasoning tokens), temperature 0, JSON output. The prompt is `PROMPT_V3` in `common.py`.
- **Output:** each question is split into pieces by asking *where would the document print this piece?*

| Piece | Where the document prints it | Examples |
|---|---|---|
| SCOPE | cover / title | document name, equipment family, course subject (`R-12 HSV`, `MCS 51 Family microcontroller`) |
| TOPIC | a section heading or slide title | `concurrent servicing`, `TC device` |
| ANCHOR | a detail inside a sentence of the answer page | a part, value, code, condition |
| DISCARD | nowhere | question form (`Describe the`, `List the`), abstract categories (`safety risks`), collection boilerplate (`as per the USAF technical manuals and orders`) |

- **How it is used:** BM25 uses `bm25_text = SCOPE + TOPIC + ANCHOR`. The dense vector **always uses the full question**, with the prefix `Instruct: Given a question, retrieve document pages that answer the question\nQuery: `.
- The gain comes mostly from removing the DISCARD words from BM25. Those words make BM25 match everywhere, especially in Industrial.

| Count (214 questions) | Value |
|---|---|
| Questions with SCOPE / TOPIC / ANCHOR | 107 / 176 / 170 |
| BM25 falls back to the full question | 4 |
| LLM failures | 0 |

**Time (DeepInfra, 220 calls):**
- Decomposition: median **1.88 s**/question, mean 2.04 s, p90 2.82 s, max 5.19 s; 220 questions in 29.6 s with 16 workers; **$0.022**.
- Query embedding: 0.19 s/question.
- Replay from cache: under 1 s for all questions. The reported latencies come from the `.meta.json` of each cached call.

## Stage 2: light retrieval (`light_retrieval.py`)

For each question, with all scores max-normalised over the corpus:

```text
bm25   = BM25(bm25_text)                           dense = cosine(page vector, full-question vector)
hybrid = 0.3*bm25 + 0.7*dense   with rank-matched language routing (below)
page   = max_norm(0.85*hybrid + 0.15*file_prior)   file_prior = best dense page of the page's file
file   = min-max over files of the mean of the file's 3 best page scores
final  = page + 0.2*file
keep only pages of the 5 best files, sort by final, cut at 100
```

**Rank-matched language routing ("arm D"):**
- *Why:* Physics questions are English but the slides are French, so BM25 is close to 0 on the right pages.
- *How:* a page whose detected language differs from the question's language borrows a BM25 value by rank. The r-th cross-language page by dense score gets the r-th highest BM25 value among same-language pages, with ranks scaled by the two group sizes. Both groups are then scored on the same scale.
- *Language detection:* French vs English function-word counts. "unknown" counts as same-language.
- See `common.rank_matched_hybrid`.

**Top-5 file cut:** 97.7% of the questions' gold files are inside the 5 selected files. 19 OHR questions get fewer than 100 pages because their files are short; the mean list length is 96.2 pages.

### Results (214 questions, % / nDCG × 100)

| Setting | R@10 | R@20 | R@100 | nDCG@10 | nDCG@20 | File recall@5 |
|---|---:|---:|---:|---:|---:|---:|
| **Chosen, all** | **67.36** | **78.11** | **93.16** | **62.82** | **66.28** | **97.66** |
| Chosen, Physics (79) | 59.91 | 77.27 | 97.76 | 55.84 | 61.89 | 100 |
| Chosen, Industrial (75) | 50.44 | 61.50 | 82.85 | 48.27 | 51.40 | 93.33 |
| Chosen, OHR (60) | 98.33 | 100 | 100 | 90.20 | 90.65 | 100 |
| Ablation: BM25 on the full question, whole corpus | 66.73 | 77.26 | 92.64 | 62.06 | 65.46 | – |
| Ablation: BM25 on the concepts, whole corpus | 67.34 | 78.11 | 93.39 | 62.78 | 66.26 | – |
| Ablation: BM25 on the full question, top-5 files (no LLM) | 66.80 | 77.32 | 92.54 | 62.13 | 65.52 | – |

All ablations keep the file score.

| Change | Gain | Scope |
|---|---|---|
| Qwen3-Embedding-8B, one vector per page (vs text-embedding-3-small on 512-word chunks) | +5.9 R@20 | research runs, 220 questions |
| Rank-matched routing | R@20 unchanged, R@10 +2.4 (Physics R@10 +6.6) | research runs, 220 questions |
| File score | +0.8 R@20, +1.7 R@100 | research runs, 220 questions |
| BM25 on the concepts (LLM) | +0.8 R@20 (Industrial +2.5) | this bundle, 214 questions |
| Top-5 file cut | no loss vs the whole corpus; later stages only handle 5 files | this bundle |

The previous light retrieval (te3s, saved `light_run_top100`) scored R@20 71.28 on 220 questions.

**Time:** 5 ms/question (BM25 0.4 ms, dense dot product 2.2 ms, fusion + file score + cut 1.4 ms; measured separately). Setup (index and vectors) is the 2.5 s of stage 1a.

## Stage 3: adaptive retrieval (`adaptive_retrieval.py`)

- **Features:** 21 features of the score curve of the light-retrieval list:
  - mean of the top 1/5/10/20/50/100 scores and of the tail;
  - standard deviation and range;
  - gaps at ranks 1/5/10/20/50, the maximum gap and its rank;
  - slopes over the top 10/20/50/100;
  - softmax entropy.
  - Lists shorter than 100 are padded with their last score.
- **Label:** rank of the last gold page in the list; 100 if a gold page is not in the list.
- **Model:** XGBRegressor (`PARAMS`).
- **Evaluation:** nested OOF on the 214 questions, 5 outer × 4 inner source-stratified folds.
- **Calibration:** the inner folds choose `K = clip(round(pred*scale + margin), 1, 100)` with the highest prefix recall under **mean K < 60**. K is then clipped to the list length.
- **Reported numbers use the OOF K.** Each question's K comes from a model that never saw its qrels. This is the `k` field.
- **Deployment model:** `adaptive_model.json`, fit on the 214 questions, with scale 1.50 and margin 15 chosen on the OOF predictions. Use it for new questions. Its in-sample K is `k_final_model` and is optimistic.

### Results (214 questions)

| Policy | Source | Mean K | Median K | p90 K | Prefix recall | Prefix precision | Prefix nDCG |
|---|---|---:|---:|---:|---:|---:|---:|
| **Adaptive K (OOF)** | **All** | **58.86** | 57.0 | 100 | **90.82** | 6.79 | **71.03** |
| Adaptive K (OOF) | Physics | 62.01 | 62.0 | 87 | 92.49 | 10.50 | 68.11 |
| Adaptive K (OOF) | Industrial | 77.03 | 84.0 | 100 | 81.71 | 5.55 | 58.42 |
| Adaptive K (OOF) | OHR | 31.98 | 29.5 | 45 | 100 | 3.46 | 90.65 |
| Fixed K = 50 | All | 49.34 | 50 | 50 | 87.37 | 7.41 | 69.78 |
| Fixed K = 100 | All | 96.20 | 100 | 100 | 93.16 | 4.41 | 71.81 |

Prefix nDCG is binary nDCG over the first K pages, so it is not the graded nDCG of stage 2. For reference, the previous te3s setting scored mean K 57.70 and prefix recall 86.92 on 220 questions.

**Time:** features 0.13 ms/question; prediction 0.2 ms for a single question (about 0.005 ms/question in a batch); nested-OOF training 1.5 s; final fit 0.07 s.

**Known weakness:** OHR uses about 32 pages although its single gold page is almost always rank 1.

## Online time per question (index already built)

| Step | Time |
|---|---:|
| Qwen3-32B decomposition (DeepInfra) | 1.88 s (median), 2.82 s (p90) |
| Query embedding | 0.19 s |
| Light retrieval | 0.005 s |
| Adaptive K | 0.0003 s |
| **Total** | **about 2.1 s** |

About 90% of the time is the LLM call. Without query preparation (BM25 on the full question, same file score and top-5 cut), the total is about 0.2 s/question. The ablation row "full question, top-5 files" shows the cost of dropping it on the 214 questions:

| | R@20 | nDCG@20 | R@100 |
|---|---:|---:|---:|
| All | −0.79 | −0.76 | −0.62 |
| Industrial | −2.47 | −2.42 | −1.82 |
| Physics, OHR | about 0 | about 0 | about 0 |

## Output for accurate retrieval: `outputs/data_discovery_output.jsonl`

One JSON object per question (214 rows), the chosen result:

```json
{"qid": "vidore_industrial::192", "source": "vidore_industrial", "query": "...",
 "query_preparation": {"scope": ["R-12 HSV"], "topic": ["concurrent servicing"], "anchor": ["key safety steps", "servicing procedures"],
                       "discard": [...], "bm25_text": "R-12 HSV ; concurrent servicing ; key safety steps ; servicing procedures"},
 "selected_files": ["<doc_id>", "... 5 files, best first"],
 "k": 71,                      "k_final_model": 68,
 "selected_pages": ["<page_id>", "... the first k pages of top100"],
 "top100": [{"page_id": "<doc_id>#page=<n>", "score": 1.2}, "... up to 100, best first"]}
```

- **Use `selected_pages`** (the first `k` pages, OOF K) as the candidate set for accurate retrieval. Accurate retrieval should rerank these pages and pass its top 10 to answer generation.
- To test another budget, cut `top100` yourself.
- Page text for those pages is in the repo page file (Quick start). Accurate retrieval may use its own parser (e.g. KDL) on the PDFs listed in `inputs/documents.jsonl`.
- Reference answers for QA are in `inputs/queries.jsonl` (`answers`).

## Reliability notes (read before quoting numbers)

- **Tuned on the same set:** these numbers were developed on this benchmark.
  - Chosen by 5-fold CV: the file-score weight 0.2 and the whole adaptive-K model.
  - Chosen after looking at results on the same set: the routing rule D, BM25 on SCOPE + TOPIC + ANCHOR, and top-5 files. These have almost no free parameters, but treat the light-retrieval numbers as slightly optimistic.
- **LLM stochasticity:** the decomposition is not deterministic. An earlier run with an unpinned provider differed on 53/220 questions, and retrieval moved by at most 0.6 points. Temperature 0 on API providers is not reproducible.
  - Replay uses the cached responses, so it is exact.
  - A live rerun can change a few questions' pieces.
- **Metric scope:** R@K counts all gold pages (grades 1 and 2). Questions often have several gold pages, so R@10 is not "found the answer page".
- **Generality:** Physics questions are English on French slides; Industrial manuals have near-duplicate sibling checklists. Expect different behaviour on other corpora.

## Tried and not kept

These were all measured on 220 questions with research scripts in `research/query_preparation/` of this repo; some of them are local and not
committed.

- Qwen3-32B with thinking: about 33 s/question, no quality gain.
- Dense on the concatenated concepts instead of the full question: R@20 −2.
- BM25 on ANCHOR only: R@20 −2.8.
- Hard cut to the top 3 files: R@20 −1, Physics −3.2.
- Table-of-contents or title scores for file selection: no gain on top of Qwen.
- A top-3 sufficiency check by Qwen3-32B before adaptive K: the model says YES on 73% of questions; Physics and Industrial lose most of their gold pages.
- Plain language routing (cross-language pages get dense only): it inflates French pages in OHR questions, so the rank-matched version D is used.

## Provenance

- This folder is a copy of the bundle `data/work/data_discovery_handoff_20261114/` that was shared with the team on
  2026-11-14. The only code change is the repo-root lookup in `common.py` (so the folder works at `<repo>/research/data_discovery_qwen/`);
  `python run_all.py` reproduces the same `outputs/` byte for byte (only timing fields differ).
- `provenance/build_bundle.py` shows where every input and cache file came from in the author's repo copy. It is a record,
  not something to run: its source files are local research artifacts.
- Research scripts used during development (220 questions, an earlier unpinned decomposition run; some are local, not committed):
  - `research/query_preparation/qwen_baseline.py`, `qwen_keyword_bm25.py`, `qwen_file_cut.py`, `qwen_export_top100.py`, `qwen_adaptive_k.py`;
  - `three_layer_decompose.py --version three_layer_v3 --model qwen/qwen3-32b --all-sources`;
  - summary: `research/query_preparation/results/qwen_setting/SUMMARY_2026-11-14.md`.
- The chosen result was produced with this bundle (`python run_all.py --live`, Qwen3-32B on DeepInfra, 2026-11-14).
- The adaptive-K protocol follows `research/experiments/evaluate_reranker_adaptive_k_20261026.py`.
