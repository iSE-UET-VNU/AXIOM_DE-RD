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

## Latest handoff: merged benchmark light retrieval and KDL run (2026-09-17)

The active branch is `benchmark-hierarchical`. The current merged benchmark is
`data/raw/BENCHMARK` (the phrase “220-file benchmark” is incorrect):

- 101 PDFs, 5,334 pages and 220 queries.
- Source composition: ViDoRe Physics 22 PDFs / 80 queries; ViDoRe Industrial
  16 PDFs / 75 queries; MPDocVQA 63 PDFs / 65 queries.
- Evaluation is page-level with 1,083 normalized page-qrel records. Qrels are
  used only for normalization/evaluation, not parsing or ranking.

### Completed light baseline on the new benchmark

Artifact root: `data/work/benchmark_hierarchical_local/`

Run: `runs/hierarchical_kf3_top20.jsonl`

Report: `reports/report.json`

Protocol:

```text
PDF/image input
  -> PDF-inspector page extraction/classification
  -> selective Tesseract for scanned/empty/image-placeholder pages
  -> page/file BM25 indexes
  -> hierarchical file selection Kf=3
  -> page ranking; save top-20 and evaluate top-10
```

V-SPLADE was explicitly disabled in this completed run:

| Metric | Result |
|---|---:|
| Page recall@10 | 36.72% |
| nDCG@10 | 32.59 |
| Page recall@20 | 40.64% |
| nDCG@20 | 33.68 |
| File candidate/derived recall@3 | 53.79% |

Preparation diagnostics: 821 pages requested OCR, 652 returned text, 169
returned no text, and 0 OCR errors. The cached run used `ocr_language=eng`
(see `config.json`); it must not be reported as the later planned `fra+eng`
Colab run. Timing artifacts are in `timing.json`: qrels normalization
206.97 s, BM25 build 12.08 s, retrieval 4.16 s; OCR wall time was 581.01 s
with 32 workers.

### Current implementation and Colab/KDL path

Commit `1a21c36` is pushed to `origin/benchmark-hierarchical`. It adds
selectable `--visual-mode disabled`, selective/all/disabled Tesseract OCR,
OCR language/worker/timeout options, resume-compatible preparation artifacts,
and tests.

The KDL notebook is a local user artifact at
`C:/Users/admin/Downloads/KDL_serving_de_full.ipynb`. Its intended flow is:

1. Start `KDLAI/KDL-Frontier-Parser-nano` through vLLM on the same Colab GPU,
   bound to `127.0.0.1:8000`.
2. Use `KDL_API_BASE=http://127.0.0.1:8000/v1`; ngrok is not needed when light
   preparation and KDL run in the same notebook.
3. Clone/pull `benchmark-hierarchical`, install Tesseract `eng+fra`, and run
   light retrieval with `Kf=3`, metric depth 10, saved depth 20, BM25 only,
   selective Tesseract and `fra+eng`.
4. Use `research/experiments/run_benchmark_kdl_second.py` for KDL second
   retrieval over the saved top-20 pages, retrieving top-10 pages or chunks.
   This second-retrieval run has not yet been completed on the new benchmark.

The notebook source is statically validated. The runner accepts all OCR
options, the benchmark tests pass (`8 passed`), and a real CPU smoke test
completed every stage on one PDF and two queries. A full Colab result still
requires starting vLLM successfully on an L4; do not claim KDL metrics until
its `reports/report.json` exists.
