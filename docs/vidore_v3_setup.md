# ViDoRe V3: setup, adapter decisions, and acceptance run

ViDoRe V3 (ACL 2026, [aclanthology.org/2026.acl-long.755](https://aclanthology.org/2026.acl-long.755/))
is the first benchmark in our set that carries **retrieval, answer-generation and
localization labels over one corpus**, each with a published per-dataset
baseline. That makes it the only chance we have to check the generation leg
against a number somebody else produced.

## Say "8 public subsets", never "ViDoRe V3"

The paper's headline — 10 datasets, ~26,000 pages, 3,099 queries — counts two
private hold-outs (nuclear, telecom) that are not released. What we can reach:

| | Public 8 | Paper headline |
|---|---|---|
| Subsets | 8 | 10 |
| Pages | **19,252** | ~26,000 |
| Unique queries | **2,419** | 3,099 |
| Query rows (× 6 languages) | 14,514 | — |
| Relevance judgements | 74,016 | — |
| Documents | 189 | — |

Per subset:

| Subset | Docs | Pages | Queries (unique) | Qrels | Distractor pages |
|---|---|---|---|---|---|
| hr | 14 | 1,110 | 318 | 10,386 | 44.8% |
| energy | 41 | 2,225 | 308 | 6,618 | 67.6% |
| computer_science | 2 | 1,360 | 215 | 6,294 | 56.7% |
| physics | 42 | 1,674 | 302 | 13,068 | 42.5% |
| finance_en | 6 | 2,942 | 309 | 8,766 | 73.3% |
| finance_fr | 5 | 2,384 | 320 | 8,808 | 63.0% |
| industrial | 27 | 5,244 | 283 | 9,684 | 80.9% |
| pharmaceuticals | 52 | 2,313 | 364 | 10,392 | 59.1% |

"Distractor pages" bear no gold for any query — 66.2% overall.

## Download

Use the **native** configs, not `vidore_v3_*_mteb_format`. See the trap below.

```bash
mkdir -p data/benchmark/vidore_v3/hr
huggingface-cli download vidore/vidore_v3_hr --repo-type dataset \
    --local-dir data/benchmark/vidore_v3/hr
```

Expected layout per subset: `corpus.parquet`, `queries.parquet`, `qrels.parquet`,
`documents_metadata.parquet`.

| Config | Size | Needed for |
|---|---|---|
| `queries` / `qrels` / `documents_metadata` | **~700 kB per subset** | everything below |
| `corpus` | 442 MB – 2.2 GB per subset (~12 GB total) | the `markdown` column only |

The corpus config is large because `image` holds the page renders. **The adapter
never reads that column** — `CORPUS_COLUMNS` excludes it and parquet column
pruning means the bytes are never decoded. The retrieval and Oracle/Text
generation arms run entirely off `markdown`, the release's own page text, so
neither Chandra nor the chunker is involved.

`data/benchmark/vidore_v3/page_index.parquet` (46 kB, committed) carries
`subset · corpus_id · doc_id · page_number_in_doc · width · height` for all
19,252 pages, fetched via the datasets-server rows API. Regenerating it costs
~1,500 API calls; the page dimensions in it are what a future localization run
needs, and they are the expensive part.

## The mteb_format trap

`vidore/vidore_v3_hr_mteb_format` ships **all 10,386 qrels inside each language
config** but only that language's 318 queries. A naive join reports 32.7 gold
pages per query; the correct answer is 5.44. It also drops `content_type`,
`bounding_boxes`, `language`, and `answer`. The official loader uses the native
configs, and so do we.

## Language is required and has no default

BM25S, per the paper's own tables:

| Setting | Public subsets | BM25S NDCG@10 |
|---|---|---|
| English queries on English docs | 5 | **53.3** |
| French queries on French docs | 3 | **44.4** (reproduces the published average exactly) |
| All 6 languages | 8 | **19.1** |

Source documents are English and French only; the other four languages are
Qwen3-235B translations of the *queries*. So four of six are cross-lingual
retrieval by construction, and the multilingual average measures lexical
mismatch, not retrieval quality. Anyone glancing at 19.1 reads it as our
pipeline underperforming.

`ViDoreV3(language=...)` therefore accepts only one of the six named languages —
no default, no `"all"`. Language appears in every metric key. **Run English-only
first.** It is also what the community pipeline leaderboard requires
(`--language english`).

The paper's own macro-average of **20.3 is over all 10 datasets** and is *not* a
number we can reproduce. 19.1 is its public-8 equivalent, recomputed from the
per-dataset columns.

## Adapter decisions worth knowing

**Retrieval is open-corpus within a subset** — the opposite of MMDocIR. There is
no within-document restriction: 9.5% of queries have gold pages in more than one
document. `scope_for` returns the whole subset.

**Ids are per-subset and collide across subsets.** Across the 8 public subsets,
14,514 query rows collapse to 2,184 distinct `query_id` ints and 6,500
`corpus_id` ints to 2,799. Questions are namespaced `{subset}::{query_id}`.
Retrievable units are `{subset}::{doc_id}#page={n}` — keyed on `doc_id`, the only
globally unique key in the release (189/189 distinct), so `corpus_id` collision
is neutralized by the choice of key rather than by the prefix.

**Each language variant is its own query with its own id**, and one qrel table
holds all six. Selecting a language means filtering queries *and then dropping*
the orphaned qrels. Skip the drop and gold inflates 6×, recall rises, and nothing
errors. The stride between variants is not derivable arithmetically (hr: 0, 301,
602, 903, 1204, 1505 for 318 queries), and **no translation-group id ships** —
recovering groups by qrel signature over-merges (312 signatures for 318 groups in
hr). Treat each language as its own eval set.

**Relevance is graded 1/2 and stays graded.** `qrels()` returns
`{qid: {unit: gain}}` for `pytrec_eval.RelevanceEvaluator` with no translation.
`gold_docs()` is the binary view for recall-style metrics. The paper describes a
three-point scale ("Not Relevant / Critically Relevant / Fully Relevant") but the
release contains no 0 rows — **the description does not match the data.**

**One taxonomy**, unlike MMDocIR: `Text, Table, Infographic, Chart, Mixed, Image,
Other`. One residue — 96 qrels carry `content_type = "N/A (If relevance
score=0)"` while scoring 1 or 2. It is a leaked form option, dropped rather than
counted as a modality.

**`file_name` carries `.pdf`, `doc_id` does not** (0/189 equal, 189/189 equal on
the stem). Same shape as the MMDocIR trap, but harmless here because nothing
joins on `file_name`.

## Why we mirror the official loader instead of calling it

`illuin-tech/vidore-benchmark` at `a70f23af`, file
`src/vidore_benchmark/pipeline_evaluation/dataset_loader.py`:

- line 88 does `corpus_images = corpus_ds["image"]`, materializing all 19,252
  page renders as decoded PIL objects, with no flag to skip it
- line 61 downloads the whole corpus config first
- line 117 raises if that list is empty, so there is no text-only path through it
- importing the package pins `sentence-transformers<4.0.0` against our 5.6.0

Their loader is correct for visual retrievers, which is what their leaderboard is
for. `ViDoreV3._read_queries` mirrors lines 66–111 exactly — the `str()` casts on
both id columns, the filter-then-drop **order**, and the
`{query_id: {corpus_id: score}}` shape carrying the raw graded int — over
column-pruned reads. Diff against those lines before changing it.

**Their `pyproject.toml` is missing `pytrec_eval` entirely** although
`evaluator.py` imports it. Anyone reproducing from their repo hits this. We use
`pytrec_eval-terrier` (prebuilt wheel, imports as `pytrec_eval`, no toolchain);
PyPI `pytrec_eval` 0.5 ships only a source tarball.

We do **not** use their bundled `bm25` retriever: it is `rank_bm25.BM25Okapi`
with hardcoded English stopwords, whereas the paper used BM25S (Lù 2024). Exact
reproduction of 53.3 is therefore not guaranteed even with correct wiring —
**treat it as a plausibility band, not a target**, and say so in every table.

## Generation

The upstream repo has **no generation and no localization code**; it is
retrieval-only. This leg is built from the paper's published prompts.

| | |
|---|---|
| Metric | % correct final answers, pass@1 |
| Judge | GPT-5.2, medium reasoning effort (Appendix H) |
| Judge stability | 5 runs, mean 72.09%, σ 0.22%, Krippendorff α 0.91 |
| Our comparator | **Oracle/Text, Gemini 3 Pro: 70.6 global, 62.3 hard, 79.3 easy** |

Both prompts are pinned verbatim in `vidore_v3_judge.py` with a drift test. Two
mechanical changes only: Jinja `{{ query }}` → `str.format` `{query}`, and the
paper's `{{ true_answer} }` typo normalized.

**The three-way / binary ambiguity is measured, not guessed.** The rubric returns
Correct / Partially Correct / Incorrect; Appendix H says the judge returns a
binary label and never says how Partial folds in. The raw label is stored and
both aggregations come from one judge pass:

- `correct_only` — the headline, matching the paper's stated metric
- `correct_plus_partial` — the upper edge of the band their unstated choice occupies

Report both, always. If `correct_only` lands near 70.6 we have located their
choice empirically.

**Abstention is rejected, not bucketed.** The paper's prompt has no abstain
option; our house generator's does. A fourth bucket would make our denominator
differ from theirs while every number still rendered, so `score()` raises.

### The context budget bites, and the truncation rate is an output

`generate.MAX_CONTEXT_CHARS = 12000` is fixed across every iSE and MMDocIR arm so
that chunk size cannot become the treatment. Under ViDoRe's Oracle/Text it is
too small, measured over all 74,016 judgements:

| Subset | Oracle context p50 | p90 | max | Queries over cap | Gold pages lost |
|---|---|---|---|---|---|
| finance_en | 14,940 | 54,969 | 107,665 | **55.3%** | 5,118 |
| finance_fr | 11,543 | 47,251 | 137,980 | 48.1% | 4,752 |
| hr | 11,125 | 45,754 | 93,435 | 47.5% | 5,478 |
| computer_science | 9,208 | 25,869 | 60,275 | 39.1% | 1,956 |
| industrial | 6,687 | 33,526 | 89,249 | 37.1% | 3,726 |
| energy | 5,541 | 22,596 | 103,910 | 25.6% | 2,268 |
| physics | 3,806 | 12,158 | 24,165 | 10.6% | 762 |
| pharmaceuticals | 2,283 | 11,652 | 45,894 | 9.6% | 912 |
| **all** | | | | **33.5%** (4,866 / 14,514) | **24,972** |

Cap required to keep every gold page: 6,820 chars for half of queries, 16,349 for
75%, 33,134 for 90%, 137,980 for all.

**No structural abstains.** `pack_context` skips an oversized unit whole rather
than truncating it, and returns `ABSTAIN` if nothing fits — but only one page in
the release exceeds 12,000 chars on its own (15,465, in pharmaceuticals) and its
queries have smaller gold pages that do fit. So the abstain branch never fires
from packing. That was worth checking rather than assuming: under Oracle it would
have meant the answer was simply absent from the context.

**Report `context_over_cap_n` and `gold_pages_dropped_n` next to the score,
permanently**, the way `units_indexed` sits next to coverage. A third of Oracle
queries missing evidence is not a footnote.

If the cap is raised for this benchmark it is a **per-benchmark constant recorded
in the run manifest**, never a global edit — raising it globally would silently
change every iSE and MMDocIR number that has already been produced.

**Easy/hard stratification is derived, not shipped.** A 6-LLM panel answers each
query with no context; easy = at least one correct. 48.6% overall, published per
subset (CS 86.5, Fin_en 31.7, Phar 57.1, HR 36.5, Ind 38.9, Phys 86.4, Ener 32.5,
Fin_fr 30.0). Reproducing it is a second full generation pass over six models —
run unstratified first and compare to Avg. Global.

## Localization: built, deferred, not an acceptance check

Bounding boxes are a **localization target, not a retrievable unit** — the
release ships no layout table and no region text. `gold_regions()` returns them
with a geometric `region_id`.

The data blocker is gone: page dimensions come back from the rows API without
fetching a render, and **145,632 of 145,632 gold boxes fall inside their page's
reported (width, height)** across 8 subsets and 141 distinct page sizes. 84
degenerate boxes (`x2<=x1` or `y2<=y1`) are dropped; 30 qrels carry no box.
Median box covers 7.6% of a page (p90 46%), so these are genuinely region-sized.

What is *not* comparable: the paper scores a **VLM emitting boxes inline in a
generated answer**, zone-based (all of one annotator's boxes merged into a single
zone, pixel overlap, IoU and Dice F1, best-matching annotator kept, no
threshold). Human inter-annotator F1 is **0.602**; the best models reach **0.089**
(Qwen3-VL-30B-A3B) and **0.065** (Gemini 3 Pro). Recall is the bottleneck —
26–27% of human-annotated pages receive no model annotation at all.

Scoring a layout parser's regions against the same gold is a **different task
with no published baseline**. A parser plausibly beating 0.089 would mean
nothing. Build it, report the human-vs-model gap because a gap that size is worth
reporting, do not gate on it.

## Acceptance

Bar is **plausible, not good**.

| Arm | Comparator | Published |
|---|---|---|
| `bm25` English-only, public 5 | BM25S Table 9 | ~53.3 NDCG@10 |
| `bm25` French-only, public 3 | BM25S Table 10 | 44.4 NDCG@10 |
| `dense` / `rrf` | band between BM25S and Jina-v4 (56.7 EN / 50.7 FR) | band |
| `generate` gold pages, text | Oracle/Text Table 3 | 70.6 global |

If retrieval is wildly off, suspect the `corpus_id` join and per-subset
namespacing first. If generation is wildly off *with gold context*, suspect the
Partial collapse before the generator — that one decision can move the number ten
points.
