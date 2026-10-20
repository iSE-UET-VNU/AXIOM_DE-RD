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
- `research/experiments/`: one-off benchmark runners and analysis scripts.
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
