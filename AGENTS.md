# AXIOM_DE-RD: agent handoff and research context

This file is the short handoff document for a new coding/research session. Read
it before changing code or interpreting benchmark numbers. The repository may
contain uncommitted user work; preserve it unless the user explicitly asks for
a rewrite.

## Project in one paragraph

AXIOM_DE-RD is a document-engineering and retrieval research repository. The
main pipeline ingests PDFs, images, office files and other raw inputs, produces
structured/enriched artifacts, and exposes retrieval/evaluation code. The
current research question is **light document preparation plus light data
discovery**: obtain cheap page/file-level signals first, retrieve a small set
of likely evidence pages, and reserve expensive parsing, OCR, VLM/MLLM
understanding and answer generation for those pages.

The current experimental focus is ViDoRe V3 Physics and Industrial, not production API work.
The working baseline is fast PDF-inspector page text indexed with BM25. The
main complementary signal under study is V-SPLADE, a visual-document learned
sparse retriever over rendered page images.

## Repository map

- `src/`: reusable pipeline, retrieval and evaluation implementation.
  - `src/retrieval/`: BM25, dense retrieval, fusion, run records and index
    contracts. The retrieval HTTP service is still scaffolding in several
    modules; use the local library/benchmark runners for experiments.
  - `src/evaluation/`: benchmark adapters, retrieval metrics and answer
    generation/judging. It reads parser output; it does not parse documents.
- `research/data_discovery/`: experimental on-demand discovery pipeline.
  It starts with cheap page extraction/BM25 and can optionally perform accurate
  ingestion and downstream QA.
- `research/experiments/`: benchmark runners and analysis scripts still cited
  in this file (one-off scripts were moved to `archive/`).
- `research/query_preparation/`: query-side processing (keyword extraction,
  decomposition), its driver, the query review UI and git-ignored `results/`.
  Active research direction as of 2026-11-10; see its `README.md`.
- `research/baselines/hierarchical_light/`: the hierarchical light BM25
  baseline and the `ondemand` package (`enfr`, `PageBM25`) imported by the
  query-preparation and adaptive-K code (formerly `_repro_retrieval_baseline/`).
- Convention: new research outputs go to `research/<topic>/results/`
  (git-ignored); large shared caches stay in `data/work/` and must not be moved,
  because many scripts reference them by fixed path.
- `configs/`: pipeline and ViDoRe/Physics settings. Relevant files include
  `pipeline.data-discovery.yaml`, `pipeline.vidore-v3-physics-discovery-pages-global-two-phase.yaml`,
  `pipeline.vidore-v3-physics-kdl-pdf-inspector.yaml`, and `pipeline.yaml`.
- `scripts/run_pipeline.py`: main pipeline CLI.
- `data/raw/`: downloaded/raw benchmark inputs.
- `data/output/`, `data/ingested/`, `data/benchmark/`: generated parser,
  retrieval and evaluation artifacts. These are local research artifacts and
  are generally not committed.
- `experiments/`: cloned external experiments, including the V-SPLADE and
  ParseBench material.
- `archive/` (git-ignored, local only): 327 finished/one-off research scripts
  and dashboards moved out on 2026-11-10, mirroring their original paths
  (e.g. `archive/research/experiments/<name>.py`). If a script named in this
  file is missing from `research/`, look there first; `archive/MANIFEST.json`
  lists original paths and `archive/classification.csv` the reason. Scripts run
  from inside `archive/` need `Path(__file__).parents[N+1]` for the repo root.

Read the more detailed local docs when working in a subsystem:

- `README.md`
- `research/data_discovery/README.md`
- `src/evaluation/README.md`
- `src/retrieval/README.md`

## Current research direction

The intended architecture is a staged discovery pipeline:

```text
PDF/image
  -> cheap page/file signal (raw text, OCR, or visual sparse vector)
  -> light retrieval / candidate generation
  -> top files or pages
  -> expensive parsing or visual understanding only where needed
  -> optional QA / answer generation
```

The present research hypothesis is that PDF-inspector/BM25 and V-SPLADE carry
different information:

- PDF-inspector/BM25 is strong for exact French lexical terms, names, numbers,
  formulas, symbols, tables and text-heavy pages.
- V-SPLADE can expose visual/topic concepts and image/chart semantics that are
  absent from the extracted text. It may retrieve a page even when the useful
  concept is not literally present in the PDF text.
- V-SPLADE is not a free-form semantic vocabulary: its sparse output has a
  fixed 50,368-coordinate vocabulary. Coordinates are learned subword/token
  features and the continuous activation weights matter.
- The current comparison is not language-symmetric: BM25 uses French query
  text, while the cached V-SPLADE query vectors are English translations and
  are evaluated against the French Physics qrels. Keep this caveat in every
  interpretation.

The near-term goal is not to replace PDF-inspector with V-SPLADE. It is to
determine whether a cheap, explainable combination improves page retrieval and
file discovery enough to justify a richer research idea.

## Physics benchmark protocol

The canonical current experiment is:

- Dataset: `vidore_v3/physics`.
- Corpus: 42 files, 1,674 pages.
- Questions: 302 French queries.
- Query IDs: French evaluation IDs are `physics::0` through `physics::301`.
  The English translation set is a separate ID range; pair French and English
  queries by sorted ordinal, never by using the same raw qid.
- Page unit: `physics::<doc_id>#page=<page_number>`.
- Page metrics: top-10 page `nDCG`, `page_hit`, `page_recall` and
  `page_precision`.
- File metric selected for current research: take the first 100 page
  candidates, deduplicate to the first 3 unique files, and compute
  `file_recall@3` against all gold evidence files. This is a metric derived
  from page retrieval, not an independently indexed file retriever.
- A file is relevant when it contains at least one gold evidence page.
- Many questions have multiple evidence pages; do not interpret page hit as
  full evidence recall.

When comparing two arms, keep the corpus, qrels, query set, page depth and file
derivation fixed. The current visual arm uses rendered page images and cached
V-SPLADE vectors; the current lexical arm uses KDL + PDF-inspector page text.

## Important existing artifacts

The following are already present locally and should be reused before running
expensive rendering or inference:

- KDL + PDF-inspector parsed run:
  `data/output/vidore-v3-physics-kdl-pdf-inspector/32c32a45a92c45bb/`
- Cached V-SPLADE Physics page vectors and metadata:
  `data/output/vsplade/vidore_v3_physics_48q/`
  (`page_vectors.npz` is 1,674 x 50,368.)
- Cached V-SPLADE English query vectors:
  `data/output/vsplade/vidore_v3_physics_english_302q/`
- Existing paired BM25/V-SPLADE runs and diagnostics:
  `data/benchmark/vidore_v3/results/physics_vsplade_bm25_fusion/`
- Tesseract + BM25 run:
  `data/benchmark/vidore_v3/results/physics_tesseract_bm25/`
- Earlier visual baselines, when present:
  `data/output/visual_retrieval/vidore_v3_physics_jina_clip_v2/`
  and `data/output/visual_retrieval/vidore_v3_physics_colsmol/`.
- ParseBench V-SPLADE diagnostic artifacts:
  `data/output/vsplade/parsebench_subtest100/`.

Before a new run, check that these paths exist and that the run contains all
302 qids and 100 page candidates per qid. Do not re-render/re-encode merely to
reproduce an offline fusion result.

## Recent experiments and results

### Handoff update: 2026-09-06

The older fusion screening below is historical. Hierarchical and legacy
second-retrieval experiments now exist; see `research/AGENTS.md` for the
updated comparison and artifact paths. No new benchmark was executed during
this handoff audit.

- Physics full cascade Kf=3: page recall@10 44.28%, nDCG@10 40.71,
  derived file recall@3 85.60%. The no-V-SPLADE ablation is 42.26%,
  39.80, and 83.28%, respectively. These are fixed/full-set experiment rows,
  not a newly confirmed promotion gate.
- Industrial cached page BM25: page recall@10 47.71%; hard hierarchical
  BM25 Kf=3: 45.54%; soft legacy/file10 plus global BM25: 49.85% in the
  full-set screening, but broad five-fold method selection gives 48.67%.
  Industrial runs listed here do not use V-SPLADE.
- Industrial V-SPLADE indexing notebook:
  `research/experiments/kaggle_index_vsplade_industrial.ipynb`.
  It now downloads `vidore/vidore_v3_industrial` corpus parquet shards
  automatically; page images are already embedded, so no new PDF rendering
  is required. Model: `naver/v-splade-efficient`; page indexing only.
  Expected output: 5,244 x 50,368 CSR plus aligned `page_metadata.json`.
  Import destination: `data/output/vsplade/vidore_v3_industrial_page_index/`.
  No completed Industrial index was found locally on this audit; the older
  `vidore_v3_industrial_english_283q/` directory is empty. Do not report an
  Industrial visual-fusion result until artifacts have been imported and evaluated.
- Physics E2E reports now exist, superseding the earlier failed-attempt-only
  status below, but remain incomplete: BM25 has 3 errors, weighted fusion 2,
  full cascade 2. Retry only failed queries; preserve successful answers.
  Source: `data/benchmark/vidore_v3/results/physics_retrieval_e2e/report.md`.

### Direct BM25 versus V-SPLADE

The paired comparison is implemented in:

- `research/experiments/analyze_physics_bm25_vs_vsplade.py`
- `research/experiments/inspect_physics_vsplade_tokens.py`

The readable per-query comparison is:

- `data/benchmark/vidore_v3/results/physics_vsplade_bm25_fusion/per_query_file_recall_at3/report.md`
- `data/benchmark/vidore_v3/results/physics_vsplade_bm25_fusion/per_query_file_recall_at3/per_query.md`

The high-level observation is that the arms are complementary but asymmetric:
V-SPLADE has page-level wins where visual/topic evidence helps, while BM25 is
more reliable overall for this French, text-heavy Physics corpus. Earlier
analysis found 29 V-SPLADE-only page-hit queries, 61 BM25-only page-hit
queries, 172 where both hit and 40 where both missed.

### Offline fusion-option screening

The latest screening is implemented in:

- `research/experiments/evaluate_physics_fusion_options.py`

It reuses cached artifacts and tests:

- BM25 and V-SPLADE baselines;
- score-normalised weighted fusion with several BM25 weights;
- RRF with several rank constants;
- BM25 over learned sparse-coordinate tokens;
- separate two-field text/token fusion;
- direct joint PDF text + learned-token representation;
- bilingual French + English lexical query expansion;
- file-level max pooling and sum-of-best-two pooling;
- a fixed formula/symbol router;
- oracle page/file routing and candidate-union ceilings.

Latest key numbers from 302 French queries:

| Arm | nDCG@10 | Page recall@10 | File recall@3 |
|---|---:|---:|---:|
| PDF-inspector + BM25 | 36.77 | 38.85% | 81.46% |
| V-SPLADE, English query | 25.98 | 29.66% | 71.52% |
| Weighted fusion, BM25 alpha 0.70 | 38.38 | 41.28% | 85.60% |
| Weighted fusion, BM25 alpha 0.80 | 38.48 | 41.24% | 84.11% |
| File sum2 aggregation on weighted 0.70 | 35.06 | 35.00% | 86.26% |
| RRF, constant 20 | 35.46 | 39.18% | 82.95% |
| Joint PDF text + V tokens | 38.45 | 40.11% | 83.44% |

Interpretation:

- Weighted fusion around alpha 0.70 is the best practical page-level result in
  this screening and improves over BM25 by +2.43 percentage points in page
  recall and +4.14 points in file recall@3.
- File `sum2` aggregation is promising for a file-discovery stage, but it
  groups pages by file and therefore hurts page-level ranking. It should not be
  used as the final page ranking without a second page-ranking step.
- RRF is robust and explainable but did not beat calibrated weighted fusion on
  this corpus.
- Treating sparse coordinates as unweighted lexical tokens is substantially
  worse than using the original continuous V-SPLADE dot-product score. The
  learned vocabulary is useful, but its activation weights and cross-modal
  query/document scoring are part of the signal.
- Adding an English translation directly to French BM25 degraded results; it
  is not equivalent to semantic query expansion.
- The formula/symbol router is only a quick heuristic and degraded overall.
  A learned router must use a held-out split; do not train and report it on
  the same 302 qrels without clearly labeling the result as overfit.
- The union ceiling shows useful headroom: either baseline hits a gold page
  in its top 10 for 86.75% of queries, versus 80.79% page hit for the best
  practical weighted run. The candidate pool is therefore not the entire
  problem; selecting and ranking the complementary candidates is.

Full outputs:

- `data/benchmark/vidore_v3/results/physics_vsplade_bm25_fusion/fusion_options_screening/report.md`
- `data/benchmark/vidore_v3/results/physics_vsplade_bm25_fusion/fusion_options_screening/report.json`
- `data/benchmark/vidore_v3/results/physics_vsplade_bm25_fusion/fusion_options_screening/per_query.jsonl`

## Current implementation status and caveats

- The fusion screening is retrieval-only and offline. It does not call an
  MLLM, OpenRouter, VLLM or a new OCR model.
- Earlier E2E attempts failed with HTTP 401 and local model timeouts.
  Later partial comparison artifacts exist (see the dated handoff above);
  do not present their accuracy as a completed, error-free evaluation.
- Native visual E2E, AgenticOCR and a fair modality-aware router have not yet
  been completed.
- Using benchmark evidence modality labels as routing features would leak
  gold information. A fair router must infer features from the query and
  available cheap page/file signals only.
- `file_recall@3` is intentionally a discovery diagnostic. For a production
  file index, implement and evaluate an actual file representation rather than
  deriving files from page ranks.
- V-SPLADE query language remains a confound. A future controlled experiment
  should compare French V-SPLADE queries, English V-SPLADE queries and a
  multilingual visual retriever under the same evaluation language.

## Recommended next research steps

1. Freeze a held-out protocol or cross-validation split before tuning alpha,
   file pooling or any router.
2. Keep two separate fields per page: PDF-inspector text and visual sparse
   activations. Start with calibrated weighted fusion; do not concatenate them
   blindly.
3. For data discovery, evaluate a two-stage system explicitly:
   file score/aggregation -> top files -> page score within selected files.
   Report file recall@1/@3 and page recall after the file budget.
4. Add query-time explanations: BM25 matched terms, V-SPLADE top shared
   coordinates/tokens, component scores, and why a page/file was promoted.
5. Only after the lightweight baselines are stable, test selective OCR,
   AgenticOCR or MiniCPM-V on retrieved pages and measure total cost/time.

## Useful commands

Run the latest offline fusion screening:

```powershell
python research/experiments/evaluate_physics_fusion_options.py
```

Recompute the paired BM25/V-SPLADE per-query comparison:

```powershell
python research/experiments/analyze_physics_bm25_vs_vsplade.py
```

Evaluate derived file metrics for cached runs:

```powershell
python research/experiments/evaluate_physics_file_level.py
```

Run the general pipeline only when a new parser artifact is genuinely needed:

```powershell
python scripts/run_pipeline.py --config configs/pipeline.yaml
```

For any new experiment, record dataset/language, page depth, file derivation,
query language, parser/text source, model/checkpoint, timing, and output path.

## Working rules for future sessions

- Inspect existing artifacts before downloading, rendering or encoding again.
- Preserve current uncommitted changes and unrelated user files.
- Keep generated outputs under `data/`; do not commit `.env`, credentials,
  model caches or large generated artifacts.
- Use exact qid pairing and verify counts before evaluating.
- Keep page retrieval, file discovery and end-to-end QA as separate metrics;
  one should not be used as a proxy for another.
- When a result is tuned on the same benchmark, label it exploratory and avoid
  presenting it as a generalizable final baseline.

## Latest merged-benchmark handoff (2026-09-22)

This section supersedes the older MPDocVQA-based 220-query notes for the
current comparison table. The current benchmark is:

- 80 Physics queries;
- 75 Industrial queries;
- 65 OHR-Bench queries;
- 220 queries total;
- 103 files and 5,091 pages in the merged retrieval corpus.
- V-SPLADE is disabled for this merged table; the comparison uses cached
  PDF-inspector/Tesseract text, embedded-image OCR text, PyMuPDF text, dense
  text embeddings and the Cohere reranker.

The OHR replacement is intentional: the previous 65 MPDocVQA queries were
removed from the official comparison because many of them are document-local
questions (for example, asking for a page number at the top of a particular
document), which is not a fair lake-level file/page discovery task. Do not
combine old MPDocVQA numbers with the current OHR table without explicitly
labeling the benchmark change.

### Canonical inputs and qrels

On this machine the merged benchmark input is under
`data/raw/0.1. BENCHMARK/` and contains `documents.jsonl`, `queries.jsonl`
and `qrels.jsonl`. The exact dataset root can be changed when reproducing,
but the source labels and counts must remain 80/75/65.

The merged evaluator uses:

- the old Physics/Industrial page-level qrels from
  `data/work/light_debug_bundle_benchmark_default_bm25_tesseract_kf3_top20/bundle_debug_light-retrieval/input/qrels_page_level.jsonl`,
  filtered to `vidore_physics::` and `vidore_industrial::` (1,018 qrel rows,
  155 queries);
- the 65 explicit OHR page qrels from
  `data/work/ohr_replacement_20260920/inputs/qrels.jsonl`;
- canonical page IDs of the form `<doc_id>#page=<zero-based_page>`.

The qrel audit found zero unmapped pages against the merged metadata. Before
trusting a new run, verify 220 query IDs, 103 file IDs, 5,091 unique page IDs,
and that every qrel page maps to an existing file/page. Qrels are used only
for normalization/evaluation; they are not used to build an index, select
files, or rank pages.

### Current protocol and metric definitions

The merged all-arm protocol is fixed as follows:

- file stage: select `Kf=3` files;
- page candidate depth: up to 200 pages from those files;
- final metric depth: 10 and 20;
- saved final run depth: 20;
- no qrel-guided ranking;
- Cohere concurrency: 32 requests;
- if a selected file set contains fewer than 200/20 pages, the run contains
  fewer than that depth; this is expected and is reported as “up to K”.

Metrics are macro-averages over queries:

- `page_recall@K`: number of gold pages in the first K divided by the number
  of gold pages for that query, then averaged over queries;
- `page_hit@K`: whether at least one gold page appears in the first K, then
  averaged;
- `nDCG@K`: graded binary page nDCG, reported on a 0--100 scale in the merged
  report;
- `file_recall@3`: fraction of the query's gold files present in the three
  selected files, then averaged;
- `candidate_pool_recall@200`: fraction of gold pages present in the up-to-200
  page pool after file selection, then averaged. This is a ceiling for page
  ranking, not final recall.

### Reproduced merged results

The authoritative table is
`data/work/ohr_replacement_20260920/retrieval/merged_220_all_arms/report.md`
and the machine-readable source is the adjacent `report.json`.

| Arm | Recall@10 | nDCG@10 | Recall@20 | nDCG@20 | File recall@3 | Candidate recall@200 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline BM25 | 33.49% | 31.51 | 36.52% | 32.47 | 49.17% | 45.88% |
| Embedded-image OCR + BM25 | 36.92% | 35.02 | 39.66% | 35.97 | 54.17% | 49.87% |
| PyMuPDF + BM25 (fallback arm) | 35.91% | 33.77 | 39.59% | 35.07 | 53.03% | 50.11% |
| Text embeddings | 50.35% | 44.93 | 59.16% | 47.84 | 81.44% | 79.51% |
| Hybrid BM25 + embeddings | 52.37% | 48.90 | 60.73% | 51.61 | 80.53% | 77.99% |
| Hybrid + Cohere reranker | **55.15%** | **51.22** | **63.28%** | **54.04** | 80.53% | 77.99% |

The Cohere arm cannot recover pages absent from the hybrid up-to-200
candidate pool, so its file and candidate-pool recall are identical to the
hybrid arm by design. Its gain is page ordering only.

These numbers are not directly comparable to the old full benchmark baseline
(`37.16%` recall@10 and `40.78%` recall@20), because that older result used
the previous 80/75/65 composition with MPDocVQA and a different corpus/qrel
set. Always state the benchmark composition beside a metric.

### What each current arm actually does

#### 1. Baseline BM25

Uses the cached page text from PDF-inspector plus selective full-page
Tesseract for pages classified as scanned/empty. It performs hierarchical
BM25: page/file signals are computed over all pages, three files are selected,
then pages inside those files are ranked. The effective score in
`run_ohr_replacement_all_arms.py` is:

```text
page_base  = 0.70 * normalized(page_BM25)
file_score = 0.50 * normalized(direct_file_BM25)
             + 0.50 * normalized(max_page_BM25)
page_final = 0.85 * page_base + 0.15 * normalized(file_score)
```

This is the current merged implementation's BM25 arm; it is not the old
V-SPLADE cascade.

#### 2. Embedded-image OCR + BM25

Adds Tesseract text extracted from embedded PDF images to the page text and
runs the same hierarchical BM25 logic. The OHR portion uses the
`size_filter_only` image policy; the cached old Physics/Industrial portion
uses the earlier embedded-image OCR cache. Therefore this is an effective
benchmark arm, but not a perfectly uniform OCR filtering ablation across all
three sources.

#### 3. Text embeddings

Uses cached OpenRouter `openai/text-embedding-3-small` vectors (dimension
1536). Important naming caveat: in the current merged run, page text includes
embedded-image OCR for the cached text corpus, and OHR uses cleaned/size-filtered
OCR text. Therefore “Text embeddings” means OCR-augmented dense text
embeddings, not pure PDF-inspector-only embeddings.

File selection is by the best dense page score per file, followed by dense page
ranking within the selected three files. The old page vectors are reused; OHR
vectors are cached in the merged embedding artifact.

#### 4. Hybrid BM25 + embeddings

This is the cached language-routed hybrid from
`research/experiments/run_ohr_replacement_hybrid_cohere.py`:

- same-language file/page text: `0.70 * normalized BM25 + 0.30 * normalized
  dense score`;
- cross-language file/page: dense score only;
- direct file score combines the direct file score and best page score;
- final page score combines the page score with the normalized file score using
  `0.85/0.15`.

The original query is used for dense embeddings. Language routing is inferred
from the available file/page metadata and must not be interpreted as knowing
the gold file. The cached hybrid run is reused by the merged evaluator rather
than recomputed from qrels.

#### 5. Hybrid + Cohere reranker

Takes the hybrid candidate pages (up to 200), sends their text and the query
to OpenRouter's rerank endpoint, using model `cohere/rerank-v3.5`, then saves
the top 20. It is a reranker, not an embedding index and not a new candidate
generator. The request cache is per query, so interrupted runs can resume
completed queries from:
`data/work/ohr_replacement_20260920/retrieval/merged_220_all_arms/cohere/cohere_cache.jsonl`.

The API key must be supplied through `OPENROUTER_API_KEY`; never put it in
this handoff file or commit it.

#### 6. PyMuPDF ablation

The merged report's `pymupdf` arm is a fallback arm: it uses native PyMuPDF
text when available and the cached baseline page text otherwise. It is not a
strict native-only comparison. The strict native-only one-off measurement was
not persisted as the official merged arm and was much weaker (recall@10
about 30.39%). Do not call the official merged row “PyMuPDF without OCR”.

### Preparation and OCR artifacts

The current OHR-only preparation is in:
`data/work/ohr_replacement_20260920/ohr_preparation/`.

It contains `config.json`, `parsing/pages.jsonl`, `summary.json`,
`timing.json`, `logs/events.jsonl` and `bundle/ohr_preparation.zip`.
The OHR preparation counts are 65 documents, 409 pages, 386 nonempty pages
and 23 empty pages. Selective Tesseract was requested for 262 pages: 239
successful nonempty results, 23 no-text results and zero OCR errors. The
reported Tesseract wall time was 376.333 seconds (about 6m16s) with 32
workers; the parse-stage wall time was 392.163 seconds. Per-task CPU/render
sums are not wall time and must not be added as if they were elapsed time.

Embedded-image OCR for the OHR pages is in:
`data/work/ohr_replacement_20260920/embedded_image_ocr/`.
It processed 317 image occurrences with 48 workers, produced 280 nonempty OCR
records and took 270.879 seconds wall time. Filtered variants are in
`embedded_ocr_filtered/`; the merged 220 arm uses
`embedded_ocr_filtered/size_filter_only/pages.jsonl` for OHR. The filter
keeps images with display-area ratio at least 1% or sufficient pixel area,
while retaining unknown-geometry images for safety.

The old Physics/Industrial preparation is reused from:
`data/work/light_debug_bundle_benchmark_default_bm25_tesseract_kf3_top20/bundle_debug_light-retrieval/`.
Relevant subpaths are:

- `preparation/baseline_pdf_inspector_tesseract/pages.jsonl`;
- `preparation/embedded_image_ocr/pages_with_embedded_ocr.jsonl`;
- `input/documents.jsonl`, `input/queries.jsonl`,
  `input/qrels_page_level.jsonl`;
- `runs/` and `reports/` for cached ranking/embedding diagnostics.

### Dense and hybrid caches

The merged dense artifacts are under:
`data/work/ohr_replacement_20260920/retrieval/combined_220_base/`.
Important files are `embedding_metadata.json` (under the embedding cache),
`metadata.json`, `query_page_dense.npy`, `page_bm25_raw.npy`,
`file_bm25_raw.npy` and `combined_top20.jsonl`.

The dense cache uses `openai/text-embedding-3-small` through OpenRouter,
dimension 1536. Old page vectors and old query vectors are reused; only the
65 OHR queries/pages needed new API calls. The cache metadata records the
cleaning policy, vector shape and query order. Before reusing it, verify the
query order matches the 220 merged IDs and the matrix shape is `(220, 5091)`.

The final merged artifacts are:

```text
data/work/ohr_replacement_20260920/retrieval/merged_220_all_arms/
  report.json
  report.md
  runs/baseline_bm25.jsonl
  runs/embedded_image_ocr_bm25.jsonl
  runs/pymupdf.jsonl
  runs/text_embeddings.jsonl
  runs/hybrid_bm25_embeddings.jsonl
  cohere/cohere_cache.jsonl
  cohere/cohere_top20.jsonl
```

Each ordinary run has 220 qid rows and unique candidate/returned pages. The
candidate list is scoped to the selected three files, so it can contain fewer
than 200 pages when those files are short. The same “up to K” rule applies to
the returned top-20 list.

### Reproduction commands

Materialize the OHR-only input view (this does not parse the old Physics or
Industrial corpus):

```powershell
python research/experiments/prepare_ohr_replacement_inputs.py `
  --dataset-root "data/raw/0.1. BENCHMARK" `
  --output-dir "data/work/ohr_replacement_20260920/inputs"
```

For the cached OHR preparation, use the recorded configuration in
`data/work/ohr_replacement_20260920/ohr_preparation/config.json`. The generic
benchmark runner used for that preparation is the
`benchmark-hierarchical` branch/Colab runner; it must be configured with
`visual_mode=disabled`, `device=cpu`, `ocr_mode=selective`,
`ocr_language=eng`, `ocr_workers=32`, `ocr_timeout=300`, `file_k=3`,
`metric_page_k=10` and `saved_page_k=20`. Do not enable V-SPLADE for this
run. The embedded-image OCR stage is resumable and is run with:

```powershell
python research/experiments/ocr_benchmark_embedded_images.py `
  --pages "<baseline-or-ohr-pages.jsonl>" `
  --output-dir "<output-dir>" `
  --language eng --workers 48
```

To rebuild the merged offline arms from the caches:

```powershell
python research/experiments/run_ohr_replacement_all_arms.py `
  --output-dir "data/work/ohr_replacement_20260920/retrieval/merged_220_all_arms" `
  --skip-cohere
```

To run the Cohere arm after setting `OPENROUTER_API_KEY`:

```powershell
python research/experiments/run_ohr_replacement_all_arms.py `
  --output-dir "data/work/ohr_replacement_20260920/retrieval/merged_220_all_arms" `
  --cohere-workers 32
```

The runner reuses complete run JSONL files, dense caches and completed Cohere
query entries. A changed corpus, query order, OCR policy, embedding model,
candidate depth or scoring formula requires a new output directory or an
explicit cache invalidation; never mix rows from different protocols.

### Timing interpretation and fairness caveats

The merged report's `timing_seconds.total` (about 34.86 seconds in the
cached run) is retrieval/evaluation/reranking orchestration time with
preparation, OCR and embeddings already cached. It is not end-to-end first-run
latency. Use the preparation `timing.json`, embedded-image OCR `summary.json`
and embedding metadata for first-run costs.

The current numbers are internally consistent and passed the following
checks: 220 qids per run, no duplicate page IDs in a ranked list, qrels map to
the 5,091-page metadata, no qrels used in ranking, and Cohere preserves the
hybrid candidate pool while changing only ordering. The main comparability
caveats are the mixed historical preparation caches, the OCR policy difference
between old and OHR pages, the fallback nature of the PyMuPDF arm, and the
language-routed hybrid. Any future report must state these caveats rather than
claiming a pure apples-to-apples parser ablation.

### Next-session checklist

1. Read this section and the merged `report.md` before running anything.
2. Confirm the benchmark composition is 80/75/65 and not the old MPDocVQA
   composition.
3. Verify qrel mapping and the `(220, 5091)` dense matrix before ranking.
4. Reuse caches; do not rerender or re-embed unless the config/model/checksum
   changed.
5. Keep file discovery, page candidate recall, final page ranking and QA as
   separate metrics.
6. Treat all same-benchmark arm selection/tuning as exploratory unless a
   held-out protocol is added.

## PyMuPDF native-text audit (2026-09-22)

The merged PyMuPDF row is a fallback arm, not a strict native-only arm. The
audit script is:
`research/experiments/evaluate_merged_pymupdf_native_only.py`.
Its outputs are under:
`data/work/ohr_replacement_20260920/retrieval/pymupdf_native_only/`.

On the current 5,091-page merged corpus:

- PyMuPDF `Page.get_text("text")` returned non-empty native text for 4,806
  pages (94.40%).
- 285 pages were native-empty (149 Industrial and 136 OHR); Physics had no
  native-empty pages in the selected 1,024-page Physics subset.
- The official mixed arm keeps cached preparation text for those 285 pages.
  Its 220-query metrics are Recall@10 35.91%, Recall@20 39.59%, nDCG@10
  33.77 and file recall@3 53.03%.
- Strict native-only leaves those 285 pages empty and scores Recall@10
  30.39%, Recall@20 33.83%, nDCG@10 28.63 and file recall@3 47.12%.
- Baseline BM25 is Recall@10 33.49%, Recall@20 36.52%, nDCG@10 31.51 and
  file recall@3 49.17%.

Therefore the apparent PyMuPDF gain comes from the combination of native text
on the 4,806 non-empty pages and preservation of fallback text on the 285
pages where the native layer is absent. Native extraction alone is worse than
the baseline; fallback OCR/text is essential. The official mixed arm is
`+2.42` points Recall@10 and `+3.07` points Recall@20 over baseline, but this
does not prove PyMuPDF is semantically better. It changes tokenization,
reading order and noise: PDF-inspector may emit image markers and structured
layout text, while the current PyMuPDF call returns plain native text.

The current code calls `Page.get_text("text")` without `sort=True`, without
PyMuPDF OCR, and without table extraction. It therefore reads the PDF's
native text layer in creator order. PyMuPDF can additionally expose `blocks`,
`words`, `dict`/`rawdict`, HTML/XML, image metadata, links, OCR via
`get_textpage_ocr()` (Tesseract), and tables via `find_tables()`, but those
features were not part of this arm. If testing reading-order or layout as a
new arm, use a new output directory and record the exact extraction mode.

Reproduce the audit with:

```powershell
python research/experiments/evaluate_merged_pymupdf_native_only.py
```

The script writes page-level native/fallback status, a strict native-only
top-20 run, and `report.json`/`report.md`; it does not alter the official
merged run and was not committed.

## PDF-inspector native-text-only audit (2026-09-22)

The cached merged preparation was also evaluated with only pages whose
`text_source` is `pdf_inspector`. All full-page Tesseract text, empty parser
pages and embedded-image OCR were removed from the page text before rebuilding
the same hierarchical BM25 arm. The script is
`research/experiments/evaluate_merged_pdf_inspector_only.py`; its outputs are
under `data/work/ohr_replacement_20260920/retrieval/pdf_inspector_only/`.

The run used the same 220 queries, 103 files, 5,091 pages, Kf=3, candidate
depth 200 and saved/evaluated depths 20/10. It retained 4,632 PDF-inspector
pages and excluded 459 pages (287 Tesseract-source and 172 empty). Results:

| Arm | Recall@10 | nDCG@10 | Recall@20 | nDCG@20 | file recall@3 | candidate recall@200 |
|---|---:|---:|---:|---:|---:|---:|
| PDF-inspector only | 24.64% | 22.78 | 28.15% | 23.88 | 41.74% | 38.63% |
| Baseline BM25 + selective Tesseract | 33.49% | 31.51 | 36.52% | 32.47 | 49.17% | 45.88% |
| Embedded-image OCR + BM25 | 36.92% | 35.02 | 39.66% | 35.97 | 54.17% | 49.87% |

This is a cached-text ablation, not a second parser run. It shows that the
Tesseract fallback pages are materially important on the merged benchmark;
the embedded-image OCR arm adds a further gain. Do not confuse this row with
the separate PyMuPDF native-only audit, which uses a different extractor.

## PDF-inspector versus PyMuPDF page audit (2026-09-22)

The direct page-level comparison is implemented in
`research/experiments/compare_merged_pdf_inspector_pymupdf.py`. Outputs are in
`data/work/ohr_replacement_20260920/retrieval/parser_comparison/`:
`report.md`, `report.json` and `per_page.jsonl`.

This compares the same 5,091 pages using PDF-inspector text only when
`text_source=pdf_inspector` and native `PyMuPDF Page.get_text("text")`. It does
not use qrels to construct text. PDF-inspector has 4,632 non-empty pages while
PyMuPDF has 4,806. There are 174 PyMuPDF-only non-empty pages and no
PDF-inspector-only non-empty pages; 172 of the PyMuPDF-only pages were marked
as Tesseract-source by the preparation cache and two were marked empty.

On pages where both are non-empty, median token Jaccard is 0.907. Median
PDF-inspector token coverage inside PyMuPDF is 0.934, while the inverse is
0.986. PDF-inspector emits image placeholders on 3,162 pages (10,012 markers),
which is useful layout evidence but can also add retrieval noise.

Strict native retrieval metrics by dataset are:

| Dataset | Extractor | Recall@10 | Recall@20 | nDCG@10 | file recall@3 |
|---|---|---:|---:|---:|---:|
| Physics | PDF-inspector | 12.41% | 16.39% | 10.93 | 23.12% |
| Physics | PyMuPDF | 10.01% | 13.07% | 10.00 | 19.38% |
| Industrial | PDF-inspector | 45.71% | 50.41% | 41.83 | 79.11% |
| Industrial | PyMuPDF | 47.79% | 54.62% | 43.47 | 85.56% |
| OHR | PDF-inspector | 15.38% | 16.92% | 15.38 | 21.54% |
| OHR | PyMuPDF | 35.38% | 35.38% | 34.44 | 36.92% |
| All 220 | PDF-inspector | 24.64% | 28.15% | 22.78 | 41.74% |
| All 220 | PyMuPDF | 30.39% | 33.83% | 28.63 | 47.12% |

The conclusion is dataset-dependent: PDF-inspector's layout-oriented output
helps Physics despite lower native coverage, PyMuPDF is slightly better on
Industrial, and PyMuPDF is much better on OHR mainly because it exposes native
text on pages that the preparation cache sent to Tesseract/empty. Therefore
the overall PyMuPDF advantage is not evidence that its text is universally
better; it combines better native coverage with different tokenization and
reading order.

The direct fallback replacement test is in
`research/experiments/evaluate_pymupdf_fallback_vs_tesseract.py`, with outputs
in `data/work/ohr_replacement_20260920/retrieval/parser_comparison/`.
Keeping PDF-inspector text on successful pages and replacing only the fallback
pages gives:

| Arm | Recall@10 | nDCG@10 | Recall@20 | nDCG@20 | file recall@3 |
|---|---:|---:|---:|---:|---:|
| PDF-inspector + Tesseract fallback | 33.49% | 31.51 | 36.52% | 32.47 | 49.17% |
| PDF-inspector + PyMuPDF fallback | 27.14% | 25.74 | 30.07% | 26.66 | 42.65% |

Among the 459 non-PDF-inspector pages, Tesseract produced non-empty text on
287 pages while PyMuPDF native extraction was non-empty on 174. Both were
non-empty on 172 pages; Tesseract-only was 115 pages; PyMuPDF-only was two;
both were empty on 170. On the 172 shared pages, median token Jaccard was
0.857. Therefore native PyMuPDF is useful as a cheap first fallback, but it is
not a replacement for OCR on pages whose useful content exists only in the
rendered image.

Image-marker cleanup was also tested in
`research/experiments/evaluate_pdf_inspector_marker_cleanup.py`. Removing
10,012 `[Image: ...]` markers from the PDF-inspector text changed the merged
baseline from Recall@10 33.49% to 33.03% and Recall@20 36.52% to 36.07%; it did
not improve the current BM25 cascade. Keep markers as separate metadata or
test them as a separate field before removing them from the final index.

For a clean two-extractor comparison, use
`research/experiments/evaluate_trusted_pdf_inspector_vs_pymupdf.py`. It keeps
only 4,632 pages with non-empty `text_source=pdf_inspector` and
`needs_ocr=false`, gives both extractors exactly those same pages, and leaves
all other pages blank in both arms. The eligible set contains 1,034/1,083
gold page records (95.48%); 171/220 queries have all gold pages in the set.
The controlled all-query results are PDF-inspector Recall@10 24.64%, Recall@20
28.15%, nDCG@10 22.78 and file recall@3 41.74%, versus PyMuPDF Recall@10
27.36%, Recall@20 30.93%, nDCG@10 25.51 and file recall@3 45.30%. By dataset,
PDF-inspector remains better on Physics (12.41% vs 10.64% Recall@10), while
PyMuPDF is better on Industrial (47.57% vs 45.71%) and OHR (24.62% vs
15.38%). This is the fairest parser-only retrieval comparison; it excludes
fallback coverage as a confounding factor.

The proposed “keep all PDF-inspector native text, then OCR only flagged pages”
test is implemented in
`research/experiments/evaluate_pdi_native_preserve_ocr.py`. On the 5,091-page
corpus, 431 pages had `needs_ocr=true`; cached Tesseract text was available on
259 of them. PDF-inspector's full-page native extraction on those flagged pages
was empty on 149 pages, short/marker-only on 243 pages, and substantive on only
39 pages. The native-only arm therefore stayed at Recall@10 24.64%, Recall@20
28.15%, nDCG@10 22.78 and file recall@3 41.74%. Keeping that native output and
adding selective cached OCR reproduced the current baseline exactly: Recall@10
33.49%, Recall@20 36.52%, nDCG@10 31.51 and file recall@3 49.17%. This means
preserving PDF-inspector output is safe, but it does not add measurable signal
on this cache because most flagged-page output is empty or only an image marker.

## Oracle QA with all gold pages (2026-09-29)

The latest QA experiment is an explicit oracle upper-bound experiment. It is
not a retrieval result: page qrels are used to select the context. For every
query, the generator receives the KDL-parsed text of **all** page-level gold
evidence pages, not the light-retrieval top-k pages. This isolates the
generation/parsing ceiling from retrieval loss.

The current benchmark and context contract are:

- 220 queries: 80 Physics, 75 Industrial and 65 OHR-Bench;
- 103 files and 5,091 pages;
- 1,083 deduplicated gold-page references, mean 4.92 pages/query;
- all 1,083 gold pages map to the cached KDL page export;
- KDL cache: `data/work/Cached-Ondemand-Eval/data/work/ondemand_v2/ef7f11811803/kdl/kdl_pages.jsonl`;
- Physics/Industrial page qrels: the filtered canonical qrels in
  `data/work/light_debug_bundle_benchmark_default_bm25_tesseract_kf3_top20/bundle_debug_light-retrieval/input/qrels_page_level.jsonl`;
- OHR page qrels: `data/work/ohr_replacement_20260920/inputs/qrels.jsonl`;
- context budget: 200,000 characters, so the largest gold context (99,112
  characters) is not truncated;
- qrels are intentionally leaked into context selection; never compare this
  directly with a retrieval arm as if it were a fair end-to-end score.

The reproducible runner is
`research/experiments/evaluate_merged_oracle_kdl_qa.py`. It writes the oracle
context, per-query answers/judgments, input manifest and report, and supports
checkpoint/resume. It uses `OPENROUTER_API_KEY` from the environment or repo
`.env`; do not write the key into this handoff file or commit it.

The existing KDL-text oracle runs are:

| Generator | Correct-only | Correct + partial | Valid queries | Provider errors | Mean generation time |
|---|---:|---:|---:|---:|---:|
| `deepseek/deepseek-v4.1-flash` | 58.99%* | 93.09%* | 217/220 | 3 | 10.11 s/query |
| `deepseek/deepseek-v4-flash` (V4.0) | **49.09%** | **87.73%** | 220/220 | 0 | **5.95 s/query** |

`*` V4.1 percentages are over the 217 valid queries. If its three provider
errors are counted as incorrect over all 220 queries, the values are 58.18%
correct-only and 91.82% correct-plus-partial. V4.1 had a few empty
`message.content` responses because the model consumed its 2,048-token output
budget in reasoning; one manually rescued query required 8,192 tokens. V4.0
completed all 220 queries without provider errors.

Per-dataset oracle results:

| Dataset | V4.1 Correct | V4.1 Correct + partial | V4.0 Correct | V4.0 Correct + partial |
|---|---:|---:|---:|---:|
| Physics | 66.25% | 98.75% | 55.00% | 97.50% |
| Industrial | 42.67% | 97.33% | 33.33% | 90.67% |
| OHR-Bench | 69.35%* | 80.65%* | 60.00% | 72.31% |

Interpretation: even with perfect page selection, KDL text plus the generator
does not make every answer correct. V4.1 is materially stronger than V4.0,
especially on exact correctness, but it is slower and less stable on a small
number of OHR requests. The oracle's high correct-plus-partial score confirms
that retrieval/context selection remains a major bottleneck for the ordinary
pipeline; the remaining oracle failures are useful parsing/generation cases,
especially pages whose KDL text is only an image marker or contains very little
text.

Artifacts:

- V4.1: `data/work/ohr_replacement_20260920/e2e_oracle_kdl_gold_v2048/`;
- V4.0: `data/work/ohr_replacement_20260920/e2e_oracle_kdl_gold_v40/`;
- each run contains `contexts/oracle_gold_kdl_pages.jsonl`,
  `qa/oracle_gold_kdl_pages.jsonl`, `qa/oracle_gold_kdl_pages.report.json` and
  `oracle_input_manifest.json`;
- the V4.1 run also has a human-readable
  `qa/oracle_gold_kdl_pages.report.md`.

Reproduce V4.0:

```powershell
python research/experiments/evaluate_merged_oracle_kdl_qa.py `
  --generator deepseek/deepseek-v4-flash `
  --judge openai/gpt-4o-mini `
  --workers 8 `
  --max-output-tokens 2048 `
  --output-dir data/work/ohr_replacement_20260920/e2e_oracle_kdl_gold_v40
```

For V4.1 replace the generator id and output directory with
`deepseek/deepseek-v4.1-flash` and
`e2e_oracle_kdl_gold_v2048`. Do not mix output directories or compare an
oracle row with a light-retrieval/accurate-retrieval row without labeling the
qrel leakage.

## Adaptive retrieval and reranking handoff (2026-11-02)

This section records the latest merged-benchmark adaptive-K and reranking
results for the next research session. It supplements the data-discovery
handoff above; it does not replace the Physics/V-SPLADE protocols.

### Shared benchmark and QA contract

- Merged benchmark: 220 queries (OHR-Bench 65, Industrial 75, Physics 80),
  5,091 pages, 103 files. Keep this composition explicit; these are not the
  older 220-query MPDocVQA results.
- Adaptive/router candidate pipeline freezes up to 100 discovery-ranked pages
  per query. Adaptive K is learned on nested, source-stratified 5-fold OOF
  splits; qrels for an outer test fold must not affect its model/calibration.
- For the adaptive accurate-QA runs, EVIE reorders only the chosen prefix and
  sends up to 10 accurately ranked pages to QA. Context is the first 512 words
  of each KDL page; there is no chunk-embedding retrieval after those pages.
  Qrels are evaluation-only. Reference answers are given only to the judge.
- Distinguish candidate-prefix recall (gold coverage in first K) from accurate
  Recall@10 and nDCG@10 after EVIE. QA Correct is strict answer/reference
  agreement; Correct+Partial is reported separately. High page recall does not
  guarantee higher QA accuracy, and the judge does not check grounding.

### Adaptive K: score-only XGBoost

The current score-only baseline is a continuous-K XGBoost regressor over the
retrieval score-curve features. It predicts K from 1 to 100; it does not use
source ID or neighbor gold-page IDs. At nested-OOF calibration budget 60:

| Policy / evaluation | Mean K | Prefix recall | Precision | F1 | nDCG | EVIE Recall@10 | EVIE nDCG@10 | QA Correct | QA Correct+Partial |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Score-only XGB, budget 60 | 57.70 | 86.92% | 9.19% | 14.43% | 66.78 | 71.64% | 68.97 | 48.18% | 88.18% |

The QA row uses DeepSeek V4 Flash and GPT-4o-mini, with one first-512-word
excerpt per EVIE-ranked page. There were 220 queries, no final answer/judge
failures, and reported spend was $0.1670 (failed token-limited attempts from
the earlier retries were not retained in that figure). The earlier Qwen3-VL QA
run with the same XGB policy scored 52.27% Correct / 88.18% Correct+Partial;
do not mix its generator result with the DeepSeek row.

Primary artifacts:

- OOF predictions/features and calibration: `research/adaptive/llm/results/xgb_llm_features_nested_oof_20261021/`
  and `research/adaptive/llm/results/adaptive_accurate_top10_qa_xgb_only_budget60_evie_deepseek_v4_flash_20261029/`.
- The 220-query retrieval/QA handoff bundle: `data/work/adaptive_xgb_evie_qa_handoff_20261029/`.
- Runner: `research/adaptive/llm/scripts/run_adaptive_accurate_qa.py`.

### XGBoost with LLM trajectory features

This is distinct from using an LLM answer as a feature. An iterative LLM stop
run produces per-query trajectory signals; these are added to XGBoost inputs,
and XGBoost still predicts the deployed K. The nested-OOF budget-50 result is:

| Policy | Mean K | Prefix recall | EVIE Recall@10 | EVIE nDCG@10 | QA Correct | QA Correct+Partial |
|---|---:|---:|---:|---:|---:|---:|
| XGB + LLM trajectory features, budget 50 | 48.28 | 85.65% | 71.22% | 68.72 | 48.18% | 87.27% |

This QA evaluation used Qwen3-VL and GPT-4o-mini. At nearly the same Mean K as
score-only budget 50 (48.51), it raised prefix recall from 83.82% to 85.65%;
the EVIE Recall@10 gain was smaller (+1.20 pp), and strict QA Correct did not
improve. The score-only and trajectory-feature QA comparison is in
`research/adaptive/llm/results/adaptive_accurate_top10_qa_xgb_only_20261023/comparison_with_xgb_llm.md`;
OOF feature results are in `research/adaptive/llm/results/xgb_llm_features_nested_oof_20261021/report.md`.

Direct LLM STOP with XGB fallback is a separate policy, not the trajectory-
feature model. In the evaluated b1/cap20 version, it scored Mean K 39.86,
prefix recall 75.91%, EVIE Recall@10 63.02%, EVIE nDCG@10 62.47, QA Correct
45.45% and Correct+Partial 85.45% with Qwen3-VL. Lower page budget did reduce
coverage, so this policy should not be described as outperforming score-only
XGBoost on QA or recall. Details are in
`research/adaptive/llm/results/adaptive_accurate_top10_qa_20261021/report.md`.

### Reranking before adaptive K

For the reranker experiments, all systems rerank the frozen top-100 discovery
candidates. A separate XGBoost K model is trained for each reranked ordering;
feeding changed reranker scores to the original score-only model would be
out-of-distribution. EVIE then reorders only the predicted prefix and its top
10 KDL excerpts go to QA.

| Top-100 reranker -> XGB K -> EVIE top-10 | Mean K | Prefix recall | EVIE Recall@10 | EVIE nDCG@10 | QA Correct* | QA Correct+Partial* |
|---|---:|---:|---:|---:|---:|---:|
| Original discovery scores | 55.96 | 85.94% | 70.77% | 68.19 | 50.91% | 87.73% |
| Cohere Rerank 4 Pro | 58.05 | 88.37% | 71.64% | 68.90 | 52.73% | 88.64% |
| Qwen3 Reranker 8B | 57.06 | 88.27% | 71.83% | 69.03 | 51.82% | 87.73% |
| BGE-reranker-v2-m3 | 57.03 | 85.83% | 70.40% | 67.90 | 52.27% | 87.73% |

`*` These original reranker QA figures used Qwen3-VL as answer generator and
GPT-4o-mini as judge. They are not DeepSeek QA metrics. Cohere and Qwen are
API rerankers; BGE-M3 is an open model run locally. Cohere/Qwen improved
candidate-prefix coverage, but that gain narrowed after EVIE top-10; QA gains
were small and not a reliable consequence of the retrieval gain.

The exact Qwen3-reranker arm was subsequently re-run with DeepSeek V4 Flash
and GPT-4o-mini on all 220 queries, holding the ranking, adaptive K, EVIE top
10, and 512-word/page QA context fixed:

| Mean K | Prefix recall | EVIE Recall@10 | EVIE nDCG@10 | DeepSeek QA Correct | Correct+Partial | API cost |
|---:|---:|---:|---:|---:|---:|---:|
| 57.06 | 88.27% | 71.83% | 69.03 | 50.45% | 86.36% | $0.1251 |

All 220 answers and judgments completed. This is the preferred same-generator
comparison to the DeepSeek score-only baseline above; do not substitute the
older Qwen-generator QA scores. The run and script are
`data/work/reranker_adaptive_k_qwen3_deepseek_qa_20261030/` and
`research/experiments/evaluate_qwen3_reranker_adaptive_qa_deepseek_20261030.py`.

The full 2026-10-26 reranker report includes fixed-50 arms, per-query
bootstrap and timing/cost notes:
`data/work/reranker_adaptive_k_20261026/report.md`.
Top-100 rerank latency recorded there is Cohere 2.22 s/query ($1.035 total),
Qwen3 10.37 s/query ($1.938 total; 28 HTTP 503 failures at 32 workers, then
retried at 8), and local BGE-M3 3.04 s/query ($0 API). XGBoost K inference
was 0.55 ms/query. These exclude EVIE score inference because only its cached
matrix was available; the reported local EVIE re-sort time is not cold model
latency.

### MetaRank-style learning-to-rank experiments

There are two related but distinct result sets; neither has end-to-end QA yet.

- Non-reranker feature ablation uses no Cohere/Qwen/BGE scores. The strongest
  shown set, `discovery_plus_dense_lexical`, gives Recall@20 76.38% and
  Recall@50 83.35%; adding page-shape gives 76.00% and 83.69%. This is the
  feature-ablation report at
  `data/work/metarank_nonreranker_feature_families_20261028/report.md`.
- MetaRank-style pairwise LTR plus adaptive K, nested source-stratified OOF,
  budget below mean K 58: `score_rank` reached Mean K 57.71, prefix recall
  88.52%, precision 7.61%, F1 13.04%, nDCG 74.34. The report is
  `data/work/metarank_adaptive_k_nested_20261027/report.md`.
- A paired EVIE comparison at a stricter budget 54 gives MetaRank LTR + K
  Mean K 51.88, prefix recall 88.12%, EVIE Recall@10 71.62%, nDCG@10 68.87.
  Against Cohere-score XGB at the same budget it uses 2.86 fewer pages on
  average while retrieval metrics are nearly equal. No QA experiment has
  validated whether this translates into better answers. See
  `data/work/metarank_adaptive_k_nested_20261027/budget_54/comparison_to_xgb_policies/report.md`.

Feature-ablation candidates and pairwise smoke scripts are under
`research/experiments/`; the MetaRank-specific reports and implementations
are listed above. Do not interpret the LTR nDCG values as the EVIE top-10
nDCG scale unless the report explicitly names EVIE nDCG@10.

### Next-session checklist for this work

1. Use the 220-query 65/75/80 merged composition and 5,091-page corpus; verify
   qid count and reranker cache before running.
2. Keep XGB score-only, XGB+LLM trajectory, reranker-conditioned XGB, and
   MetaRank LTR as separate policies. Keep their generator/judge pair and
   retrieval ranker visible in every comparison.
3. Reuse cached rankings, EVIE scores, OOF predictions and QA responses. Do
   not rerun paid calls just to recover already reported metrics.
4. Treat benchmark-wide OOF and feature selection results as exploratory on
   this dataset, even with nested folds. Report paired per-query uncertainty
   for future promotions.
5. Separate three questions: did gold pages survive the adaptive prefix, did
   EVIE rank them in the top 10, and did the generator answer correctly?

## Top-3 LLM stopping and QA handoff (2026-11-09)

This is the latest adaptive-retrieval result and should be read before
continuing the LLM stopping work. It evaluates a one-shot evidence-sufficiency
check over the first three discovery-ranked pages, followed by the frozen
nested-OOF XGBoost K when the LLM does not accept STOP.

### Policy and implementation

- Discovery ranking: `adaptive_retrieval_20261020_router_te3s_top100`, using
  `openai/text-embedding-3-small`, with at most 100 pages per query.
- Controller: `deepseek/deepseek-v4-flash`. One request receives the query and
  the first three PP-OCRv5 page texts. It does not receive qrels, reference
  answers or the XGBoost prediction.
- Accept STOP at K=3 only when the response says `STOP`, confidence is at least
  0.85, the provisional answer is non-empty, at least one cited page is in the
  three pages shown, and `missing_information` is empty. Invalid responses or
  rejected STOPs fall back to the query's nested-OOF calibrated XGBoost K.
- After choosing K, cached EVIE scores rerank only that prefix. Up to ten pages,
  represented by the first 512 KDL words per page, go to
  `deepseek/deepseek-v4-flash`; `openai/gpt-4o-mini` judges the answer against
  the reference. The judge does not verify whether individual answer claims
  are supported by the supplied pages.

The controller runner is
`research/adaptive/llm/scripts/run_ohr_top3_pagewise_vs_oneshot.py` and the QA
runner is `research/adaptive/llm/scripts/run_adaptive_accurate_qa.py`. The
controller runner now supports `--sources`, `--arms` and `--oneshot-k`; the QA
runner supports `llm_top3_xgb_fallback` and `fixed_k100` policies.

### Full 220-query comparison

All three rows below use the same cached EVIE ranker, DeepSeek V4 Flash answer
generator and GPT-4o-mini judge. `Prefix recall` is gold-page recall within the
chosen discovery prefix; `EVIE Recall@10` is recall after accurate reranking.

| Policy | Mean K | Prefix recall | EVIE Recall@10 | EVIE nDCG@10 | QA Correct | QA Correct+Partial | Pages sent to QA |
|---|---:|---:|---:|---:|---:|---:|---:|
| No adaptive, fixed K=100 | 100.00 | 89.13% | 71.72% | 68.93 | 42.73% | 85.91% | 10.00 |
| Nested-OOF score-only XGBoost | 57.70 | 86.92% | 71.64% | 68.97 | 48.18% | 88.18% | 9.65 |
| Top-3 LLM STOP, otherwise XGBoost | 44.98 | 80.86% | 66.89% | 65.81 | 46.82% | 87.27% | 7.47 |

The Top-3 policy saves 12.72 pages/query relative to XGBoost, but loses 6.06
points of prefix recall, 4.75 points of EVIE Recall@10 and 1.36 points of
strict QA Correct. Its Correct+Partial score is only 0.91 points lower. It is
therefore a useful cost/coverage point, not a retrieval-quality replacement
for XGBoost.

Per-source results are:

| Dataset | Policy | Mean K | Prefix recall | EVIE Recall@10 | EVIE nDCG@10 | QA Correct | Correct+Partial |
|---|---|---:|---:|---:|---:|---:|---:|
| OHR (65) | XGBoost | 22.46 | 100.00% | 100.00% | 100.00 | 63.08% | 72.31% |
| OHR (65) | Top-3 LLM + fallback | 11.22 | 100.00% | 100.00% | 100.00 | 61.54% | 70.77% |
| Physics (80) | XGBoost | 64.94 | 84.35% | 60.55% | 55.80 | 51.25% | 98.75% |
| Physics (80) | Top-3 LLM + fallback | 56.50 | 74.85% | 53.01% | 50.59 | 48.75% | 98.75% |
| Industrial (75) | XGBoost | 80.52 | 78.33% | 58.89% | 56.12 | 32.00% | 90.67% |
| Industrial (75) | Top-3 LLM + fallback | 61.95 | 70.67% | 52.98% | 52.40 | 32.00% | 89.33% |

The OHR reduction is safe with respect to the available qrels: all 40 accepted
OHR STOPs retained all annotated gold pages. Physics/Industrial are different:
36 queries were stopped, but only 13 were safe relative to XGBoost and 23 cut
at least one annotated gold page. The one-shot controller is consequently
overconfident outside OHR under the current prompt and threshold.

### Do all annotated gold pages need to survive?

The LLM accepted STOP for 76/220 queries. On those same 76 queries:

| Context policy | Correct | Partial | Incorrect | Correct+Partial |
|---|---:|---:|---:|---:|
| Top-3 context | 53 (69.74%) | 15 | 8 | 68/76 (89.47%) |
| XGBoost context | 56 (73.68%) | 14 | 6 | 70/76 (92.11%) |

The most important diagnostic subset is the 23 `qrel-unsafe` STOPs: K=3 cut
one or more gold pages that XGBoost would have retained. Top-3 QA produced 14
Correct, 8 Partial and 1 Incorrect, or 22/23 (95.65%) credited. XGBoost on the
same 23 produced 15 Correct, 7 Partial and 1 Incorrect, also 22/23 credited.
Thus removing annotated gold changed one strict Correct answer to Partial but
did not add another fully Incorrect answer in this small exploratory subset.

This supports the hypothesis that full annotated-gold recall is often not
necessary to answer the benchmark question. It does not prove that the answer
was grounded in the top-three pages: an alternative relevant page may be
unannotated, DeepSeek may use parametric knowledge, and GPT-4o-mini currently
checks answer correctness rather than citation-level entailment. Five unsafe
STOPs had zero annotated gold in the top three yet four were judged Correct,
which makes a no-context/shuffled-context control and an evidence-aware judge
the highest-priority follow-up.

The only fully Incorrect unsafe STOP was `vidore_industrial::112`: the question
asks for three reporting categories of corrosion. The reference is Uniform
Surface, Galvanic and Pitting Corrosion; the generated answer was Light,
Moderate and Severe Corrosion. Its gold ranks were 35, 38, 50 and 58, so the
top-three prefix contained none of the annotated evidence.

### Cost, logs and artifacts

- OHR controller comparison (pagewise 1-to-3 and one-shot top-3):
  `research/adaptive/llm/results/ohr_top3_pagewise_vs_oneshot_te3s_20261103/`.
  One-shot top-3 made 65 calls, cost $0.019666 and averaged 2.51 s/request;
  pagewise made 118 calls, cost $0.019766 and averaged 2.42 s/request.
- Physics/Industrial one-shot top-3:
  `research/adaptive/llm/results/top3_oneshot_physics_industrial_te3s_20261106/`.
  It made 155 calls, cost $0.029082 and averaged 4.69 s/request. Per-query chat
  histories, console output, request cache and trajectories are preserved.
- Physics/Industrial one-shot top-1 diagnostic:
  `research/adaptive/llm/results/top1_oneshot_physics_industrial_te3s_20261106/`.
  It cost $0.020160 but had even lower safe-stop precision (30.30%); do not
  promote it merely because it uses less context.
- Full Top-3 fallback QA:
  `research/adaptive/llm/results/top3_llm_xgb_evie_deepseek_qa_20261106/`.
  Read `comparison_vs_xgb.md`, `comparison_with_fixed_k100.md`,
  `stopped_and_qa_good.md`, `stopped_and_qa_good.csv` and `per_query.jsonl`.
- Fixed-K=100 QA comparator:
  `research/adaptive/llm/results/fixed_k100_evie_deepseek_qa_20261106/`.
- Human-readable 220-query review workbook:
  `archive/outputs/adaptive-llm-query-review-20261106/adaptive_llm_query_review_220.xlsx`
  (the former top-level `outputs/` folder was moved to `archive/outputs/` on 2026-11-10).
  The canonical query source is `data/raw/0.1. BENCHMARK/queries.jsonl`.

The final QA validation reused cached prompts where possible: 144 answer
prompts and 179 judge prompts came from the frozen XGBoost run. It required 76
new generator calls and 41 new judge calls, with provider-reported incremental
cost $0.018786. The full Top-3 QA artifact reports $0.133023 across its unique
cache entries, but that is not the incremental price of this comparison.

### Recommended next-session work

1. Do not rerun the 220 controller or QA calls; reuse trajectories and caches.
2. Audit the 23 qrel-unsafe STOPs first, especially the five with zero gold in
   top three, using an evidence-entailment judge that sees the answer and pages.
3. Add no-context and shuffled-page controls to estimate how much answer
   correctness comes from model knowledge rather than retrieved evidence.
4. Calibrate a source-agnostic safety gate on observable query/retrieval
   signals; source ID and qrels must remain unavailable at inference time.
5. Report strict Correct, Correct+Partial, prefix recall and EVIE Recall@10
   together. A cheaper policy should not be called better solely from QA credit.

## Query preparation / pain-point decomposition screen (2026-11-10)

The recent query-preparation experiments focus on Industrial queries with long
or noisy wording. The goal is to preserve discriminative entities and answer
facets while reducing broad lexical matches. These are exploratory retrieval
screens on the existing PP-OCRv5 page corpus; they do not change the canonical
retrieval pipeline or benchmark artifacts.

### Artifacts and reproducibility status

Outputs are under `data/work/query_preparation/`:

- `industrial_all_keyword_fusion/`: full 75-query keyword extraction and
  retrieval comparison. Read `report.md`; `keyword_queries.jsonl` stores the
  generated pain-point phrases and `per_query.jsonl` stores per-query ranks and
  metrics. `metrics.json` has aggregate metrics. The `extractor_cache/` holds
  cached LLM extraction responses. A looser first-prompt pilot is preserved in
  `loose_extraction_pilot/`; use the parent report/final artifacts for the
  strict experiment.
- `industrial_192_painpoint_fusion/`: focused analysis for
  `vidore_industrial::192`, with `report.md`, `experiment.json` and a compact
  embedding cache.
- `industrial_low_recall_decomposition/strict_prompt_v2/`: the final
  low-recall decomposition screen. Read `report.md`; `decompositions.jsonl`
  records each keep/decompose decision and subqueries, `per_query.jsonl`
  records retrieval/reranking details, `metrics.json` has aggregates, and
  `decision_cache/` preserves LLM responses. The earlier, weaker prompt is
  retained in `initial_prompt_v1/` for audit; do not treat it as the final
  decomposition result.

The original one-off drivers were lost; they were rebuilt on 2026-11-10 as
`research/query_preparation/query_preparation_screen.py` with subcommands
`keywords`, `decompose`, `focus` and `verify`. In the default
`--llm-mode replay`, the saved LLM outputs (`keyword_queries.jsonl`,
`decompositions.jsonl`) are reused, with no API calls. The rebuild reproduces
all five original artifacts exactly (every per-query top-20/EVIE top-10, gold
rank and macro metric): strict keywords, loose pilot, q192 focus, decomposition
v1 and strict v2. Rebuilt outputs are under `research/query_preparation/results/rebuild/`.
Details that had to be recovered empirically:
BM25 and chunking use `real_text()` (strips `[Image: ...]`); dense uses
**cosine** (not raw dot) max-pooled per chunk; the saved hybrid is
`light_run_top100` in `gate_ppocrv5_ungated.json`; EVIE indices come from
`colvec_qids.json`/`colvec_keys.json`. The original prompts were not preserved:
`--llm-mode live` uses reconstructed prompts (`reconstructed_*_v1`) and must
write to a new output directory and be labelled as a new prompt version. The
original strict-v2 `metrics.json` has a literal trailing `\n` after the JSON
object; `verify` reads it leniently. `tmp/query_review/update_query_metrics.mjs`
is a workbook updater, not an experiment driver. No production code was changed.

Reproduce and check:

```powershell
python research/query_preparation/query_preparation_screen.py keywords
python research/query_preparation/query_preparation_screen.py decompose
python research/query_preparation/query_preparation_screen.py focus --qid vidore_industrial::192
python research/query_preparation/query_preparation_screen.py verify `
  --original data/work/query_preparation/industrial_low_recall_decomposition/strict_prompt_v2 `
  --rebuilt research/query_preparation/results/rebuild/decompose
```

### Full Industrial keyword screen

- 75 Industrial questions; 5,091 PP-OCRv5 pages.
- GPT-4o-mini selected pain-point phrases from query text only; qrels and
  reference answers were not sent to the extractor. Extracted phrases were
  validated against terms in the original question.
- BM25 used the frozen `enfr` analyzer and PageBM25 implementation.
- The exploratory fusion combines compact-query BM25 with cached
  `text-embedding-3-small` vectors for the original full query: fixed weights
  0.30/0.70, per-signal min-max normalization over each signal's top 100,
  dense chunk scores max-pooled by page. This page-level arm omits the saved
  hierarchical baseline's parent/file boost.

| Arm | Recall@10 | Recall@50 | Recall@100 |
|---|---:|---:|---:|
| BM25, full query | 46.14% | 64.21% | 69.88% |
| BM25, extracted pain-point phrases | 45.20% | 63.76% | 70.90% |
| Pain-point BM25 + original-query embedding | 46.94% | 67.64% | 78.45% |
| Saved original hierarchical hybrid | 48.53% | 69.11% | 79.20% |

At Recall@100, keyword BM25 beats/ties/loses to full-query BM25 on 7/64/4
queries. Adding original-query embeddings to keyword BM25 beats/ties/loses on
21/49/5. Initial signal: compact lexical terms help reduce noise at deeper
recall when combined with semantics, but the exploratory fusion has not beaten
the saved hierarchical hybrid and is not an apples-to-apples replacement.

### Focused Industrial query 192

Query: “Describe the key safety steps and servicing procedures for concurrent
servicing of the R-12 HSV as per the USAF technical manuals and orders.” The
two annotated gold pages were the table-of-contents page (zero-based page 4)
and the concurrent-operations procedure page (zero-based page 77).

| Arm | Gold ranks (TOC, procedure) | Recall@10 | Recall@50 | Recall@100 |
|---|---|---:|---:|---:|
| BM25, full query | 99, >100 | 0% | 0% | 50% |
| BM25, pain-point terms | 20, 24 | 0% | 100% | 100% |
| Saved original hybrid | >100, 13 | 0% | 50% | 50% |
| Pain BM25 + full-query embedding | 60, 6 | 50% | 50% | 100% |
| Pain BM25 + compact-sentence embedding | 62, 8 | 50% | 50% | 100% |

Pain-point terms were `safety servicing procedures concurrent servicing R-12
HSV`; the compact semantic query was “Safety and servicing procedures for
concurrent servicing of the R-12 HSV.” The two new fusion rows are page-level
and omit the hierarchical file-parent boost. Treat this as a single-query
diagnostic, not an aggregate result. The surrounding ranks also suggest some
qrel incompleteness: a neighboring page titled “Concurrent Servicing Operations
Procedures” is relevant-looking but is not one of the annotated gold pages.

### Low-recall decomposition screen

- Selected 18 of the 75 Industrial questions whose saved original-hybrid
  Discovery Recall@100 was below 0.60. This is a qrel-selected exploratory
  subset; its metrics are not an unbiased estimate for all queries.
- GPT-4o-mini made the keep/decompose decision using only query text, with 18
  concurrent requests and the strict facet prompt v2. It decomposed 12 queries
  and kept 6 as one concise query. Subquery counts were: 1: 6, 2: 1, 3: 9,
  4: 2.
- Each subquery used frozen `enfr` BM25 over the same 5,091 pages. Results were
  deduplicated by page ID. Candidate-page and gold-file recall are measured
  before reranking. EVIE Recall@10 uses cached EVIE-4.5B scores for the original
  query over each candidate pool; no new EVIE inference was run and the
  reranker was not made subquery-aware.

| Arm | Mean candidates | Candidate page recall | Gold-file recall | EVIE Recall@10 |
|---|---:|---:|---:|---:|
| Saved hybrid top-100 | 100.0 | 31.56% | 91.67% | 9.49% |
| BM25 full query top-100 | 100.0 | 25.07% | 88.89% | 5.10% |
| BM25 pain-point query top-100 | 100.0 | 26.87% | 88.89% | 8.07% |
| Subquery BM25 union, top-5 each | 11.3 | 5.88% | 63.89% | 4.49% |
| Subquery BM25 union, top-10 each | 22.6 | 6.71% | 72.22% | 5.32% |
| Subquery BM25 union, top-20 each | 44.0 | 13.38% | 80.56% | 8.61% |

The practical candidate-augmentation comparison preserves saved hybrid top-20
and adds subquery BM25 results before reranking:

| Arm | Mean candidates | Candidate page recall | Gold-file recall | EVIE Recall@10 |
|---|---:|---:|---:|---:|
| Saved hybrid top-20 | 20.0 | 10.08% | 86.11% | 8.01% |
| Hybrid top-20 + subquery top-5 each | 27.3 | 12.13% | 86.11% | 8.03% |
| Hybrid top-20 + subquery top-10 each | 36.3 | 12.32% | 88.89% | 7.63% |
| Hybrid top-20 + subquery top-20 each | 53.9 | 15.61% | 88.89% | 7.63% |

Interpretation: subquery-only retrieval is a weak candidate generator here. A
small top-5-per-subquery union modestly increases candidate page recall while
keeping EVIE Recall@10 effectively flat. Top-10 finds more gold files but
slightly lowers EVIE Recall@10; top-20 expands the pool without improving
gold-file recall or final EVIE Recall@10. This does not establish that
decomposition improves final ranking. A subquery/facet-aware reranker and a
non-qrel-selected evaluation are needed before making that claim.

### Recommended follow-up

1. Preserve the original full-query retrieval as a safety branch; only enable
   decomposition when the query has distinct answer facets or a diagnosed
   low-recall/noisy-match signal.
2. Test a small per-subquery candidate budget (top-5) unioned with the original
   hybrid top-20, then compare against the exact same baseline on a held-out or
   cross-validated split.
3. Make reranking facet-aware: check whether candidates cover the distinct
   required facets instead of reranking every page only against the full
   question. Keep original-query scoring as a control.
4. Inspect query-by-query wins/losses and possible qrel omissions, especially
   where a relevant-looking page is adjacent to an annotated page.
5. Report candidate page recall, gold-file recall and post-rerank EVIE
   Recall@10 separately. Do not describe candidate-pool gains as answer or
   final-ranking gains.
