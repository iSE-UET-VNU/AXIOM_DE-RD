# Research handoff: abstraction-instance fusion

This file is the handoff for the research branch that studies how to combine
cheap lexical evidence from PDF-inspector with visual-semantic evidence from
V-SPLADE. It supplements the repository-level AGENTS.md. Preserve unrelated
uncommitted work and keep new experiments in separate scripts and output
directories.

## Research question

The central hypothesis is not that V-SPLADE should replace PDF-inspector.
The two systems expose different evidence:

- PDF-inspector plus BM25 is an instance-level lexical signal. It is precise
  for names, French terms, numbers, units, equations, symbols, tables and
  wording that is actually present in extracted page text.
- V-SPLADE is an abstraction-level visual signal. It can retrieve topic,
  layout, figure, chart or other visual concepts that are not represented
  faithfully in the extracted text. Its sparse coordinates are a fixed
  50,368-dimensional learned vocabulary and the continuous activation weights
  are important; treating coordinates as unweighted words performed poorly.

The intended product direction is light document preparation and light data
discovery:

    cheap page/file signals
        -> candidate generation
        -> file discovery or page retrieval
        -> expensive OCR/VLM/MLLM parsing only for selected evidence

The current fusion research asks whether abstraction should act as a prior or
candidate-recall mechanism, while lexical evidence verifies concrete page
instances. A single global score should not be assumed to be the final design.

## Frozen Physics protocol

- Dataset: vidore_v3/physics.
- Corpus: 42 files and 1,674 pages.
- Queries: 302 French queries.
- Page ID: physics::<doc_id>#page=<page_number>.
- Main page metric: page_recall@10.
- Secondary page metrics: nDCG@10, page_hit@10 and page_precision@10.
- Main discovery metric: file_recall@3.
- File recall protocol: use the first 100 page candidates, deduplicate them
  into the first 3 unique files, and compare with all gold evidence files.
  This is a derived discovery diagnostic, not an independent file index.
- V-SPLADE query vectors currently use English translations while evaluation
  qrels are French. This language mismatch is a permanent caveat until a
  controlled French/multilingual visual run is available.

Never pair English and French queries by their raw qid. Pair by sorted query
ordinal. Before a run, verify 302 qids, 100 candidates per input run and
reachable qrel pages.

## Cached inputs

- Parsed page text:
  data/output/vidore-v3-physics-kdl-pdf-inspector/32c32a45a92c45bb/
- V-SPLADE page vectors:
  data/output/vsplade/vidore_v3_physics_48q/
- V-SPLADE English query vectors:
  data/output/vsplade/vidore_v3_physics_english_302q/
- Paired page runs:
  data/benchmark/vidore_v3/results/physics_vsplade_bm25_fusion/

Do not render or encode again for an offline fusion experiment.

## Results so far

### Hierarchical and legacy experiments: audited 2026-09-06

These results supersede the historical weighted-fusion-only discussion below.
This update inspected existing artifacts and did not execute new benchmarks.
Metrics are listed as page recall@10 / nDCG@10 / derived file recall@3.

| Dataset and arm | Page recall@10 | nDCG@10 | File recall@3 |
|---|---:|---:|---:|
| Physics full cascade Kf=3 | 44.28% | 40.71 | 85.60% |
| Physics full cascade Kf=3 without V-SPLADE | 42.26% | 39.80 | 83.28% |
| Physics no-V-SPLADE + legacy second | 41.90% | 39.96 | 83.28% |
| Physics full cascade + legacy second | 44.16% | 41.28 | 85.60% |
| Industrial cached page BM25 | 47.71% | 44.36 | 82.27% |
| Industrial rebuilt page BM25 on KDL cache | 46.36% | 42.54 | 82.45% |
| Industrial hard BM25 cascade Kf=3 | 45.54% | 42.01 | 79.51% |
| Industrial hard BM25 cascade Kf=10 | 47.47% | 43.34 | 82.86% |
| Industrial soft legacy file10, gamma .1, interleave 3:1 | 49.85% | 44.86 | 84.45% |

The table contains fixed/full-set screening rows. Industrial has 283 English
queries, 27 files and 5,244 pages; its parsed cache has text on 5,039 pages.
All Industrial rows above are without V-SPLADE. Keep cached and rebuilt BM25
baselines distinct; their scores differ and are not interchangeable.

Primary broad method-selection OOF results from the soft-cascade reports:

- Physics: page recall 43.17%, nDCG 40.76, file recall 85.60%; page gain
  +4.31 pp vs BM25, bootstrap CI [2.15, 6.58] pp.
- Industrial: page recall 48.67%, nDCG 44.41, file recall 83.98%; page gain
  +0.95 pp, CI [-0.71, 2.58] pp. The restricted two-arm check also reports
  49.85%, but was narrowed after screening and is not an independent holdout
  confirmation of the chosen configuration.

Sources under `data/benchmark/vidore_v3/results/`:
`physics_soft_cascade_feedback/report.md`,
`physics_cascade_no_vsplade/report.md`,
`industrial_hierarchical_retrieval/report.md`,
`industrial_soft_cascade_feedback/report.md`.

Kf means the number of retained candidate files. Distinguish file-stage
candidate recall@Kf from file recall@3 derived from final page ranks. In the
Industrial report the column labeled file candidate recall@3 stays at a
fixed cutoff of 3 even for other Kf rows; do not quote it as recall@Kf.
Legacy retrieval cannot recover evidence excluded from its candidate scope.

### Timing and E2E status

Standalone PDF-inspector parsing measurements: Industrial 11.039024 s for
5,244 pages; Physics 12.454784 s for 1,674 pages. Physics BM25-only Kf=3
hierarchy plus page/file BM25 setup: 14.433101 s; retrieval for 302 queries:
11.041189 s (36.560229 ms/query); dense embeddings: zero. Setup is not a
measurement of BM25 indexing alone. The standalone parser timings and the
retrieval runs using cached KDL + PDF-inspector text are separate measurements,
not a single end-to-end timed run.

Sources: `data/work/pdf_inspector_benchmark/industrial_stdout_4.json`,
`physics_stdout.json`, `physics_hierarchical_no_v_stdout.json` in that directory.
Parser stdout files contain a warning line before the JSON record.

`data/benchmark/vidore_v3/results/physics_retrieval_e2e/report.md` reports
correct/302 of 46.03% (BM25), 48.34% (weighted), 45.70% (full cascade),
with respectively 3, 2, 2 errors. These are incomplete E2E results, not
evidence that higher retrieval recall necessarily improves answer accuracy.
Retry only failed query records and reuse successful generation/judging cache.

### Pending Industrial V-SPLADE handoff

Notebook: `research/experiments/kaggle_index_vsplade_industrial.ipynb`.
Default `ATTACHED_INPUT_ROOT=None` downloads only `corpus/*.parquet` from
`vidore/vidore_v3_industrial`; these include rendered page images. Kaggle
needs Internet and GPU. It uses `naver/v-splade-efficient`, batch 3, with
per-source-shard checkpoints and exports a ZIP of page vectors, metadata,
metrics and index contract. It does not encode queries or evaluate retrieval.
Notebook structure was checked locally; Kaggle inference completion has not
been verified. Checkpoints resume only when the prior shard files are available.

Expected local import: `data/output/vsplade/vidore_v3_industrial_page_index/`.
Validate 5,244 unique `industrial::<doc_id>#page=<p>` IDs, 27 files,
50,368 dimensions and exact agreement with local corpus IDs before scoring.
The old `vidore_v3_industrial_english_283q/` directory is empty on this audit.
Next: import the user's Kaggle output, encode the 283 English queries with
the matching checkpoint, then compare BM25-only, global fusion, and
hierarchical fusion at matched file/page budgets, with optional legacy second
retrieval. Do not launch another full page-encoding run merely to check status.

The direct and weighted screening was implemented in
research/experiments/evaluate_physics_fusion_options.py. The paired per-query
analysis is in research/experiments/analyze_physics_bm25_vs_vsplade.py.

The strongest practical page-level result in that earlier screening was weighted fusion:

| Method | nDCG@10 | Page recall@10 | File recall@3 |
|---|---:|---:|---:|
| PDF-inspector + BM25 | 36.77 | 38.85% | 81.46% |
| V-SPLADE, English query | 25.98 | 29.66% | 71.52% |
| Weighted fusion, BM25 alpha 0.70 | 38.38 | 41.28% | 85.60% |
| Weighted fusion, BM25 alpha 0.80 | 38.48 | 41.24% | 84.11% |
| RRF, constant 20 | 35.46 | 39.18% | 82.95% |
| Joint PDF text plus learned-token field | 38.45 | 40.11% | 83.44% |

The weighted score is:

    score = 0.7 * normalized_BM25 + 0.3 * normalized_V-SPLADE

The two input systems are complementary but asymmetric. Earlier per-query
analysis found 61 BM25-only page-hit queries, 29 V-SPLADE-only page-hit
queries, 172 where both hit and 40 where both missed.

## Adaptive evidence-aware fusion experiment

The new independent experiment is in
research/experiments/physics_adaptive_evidence_fusion.py.

It constructs the union of top-100 BM25 and top-100 V-SPLADE candidates. For
each query-page pair it records:

- raw and normalized scores, ranks, top-10/top-100 membership;
- lexical term coverage, IDF mass, matched term count and anchor coverage;
- numbers, units and formula/table/image/chart marker signals;
- visual percentile, shared sparse dimensions and activation contribution;
- agreement, conflict, semantic-rescue and lexical-guard signals.

It compares the fixed weighted baseline with two fold-fitted methods:

1. Adaptive rule: an explainable branch that increases lexical weight for
   strong anchors and visual weight for high-confidence low-coverage pages.
2. Linear evidence ranker: NumPy pairwise logistic ranking with training-fold
   standardization and a fixed random seed.

The evaluation is five-fold query cross-validation. Test-fold qrels are not
used to select thresholds or train weights. Output is isolated at:

data/benchmark/vidore_v3/results/physics_adaptive_evidence_fusion/

Latest OOF result:

| Method | nDCG@10 | Page hit@10 | Page recall@10 | File recall@3 |
|---|---:|---:|---:|---:|
| Weighted alpha 0.70 | 38.38 | 80.79% | 41.28% | 85.60% |
| Adaptive rule | 38.24 | 80.79% | 41.08% | 85.26% |
| Linear evidence ranker | 38.06 | 82.45% | 42.34% | 82.95% |

Relative to the weighted baseline:

- Adaptive rule: -0.20 percentage points page recall.
- Linear ranker: +1.05 percentage points page recall.
- Neither reached the target of +5 percentage points.
- Linear ranker improved page hit and page recall but lost nDCG and
  file_recall@3. It found more queries with at least one gold page while
  making the top-3 file discovery ranking worse.
- The rule marked 232 semantic-rescue pages in adaptive top-10, but only 5
  queries had a successful rescue and 89 queries had a wrong rescue. A high
  visual percentile is therefore not enough evidence for promotion.
- The top-10 union page-hit ceiling is 86.75%, so ranking is not the only
  bottleneck; candidate generation and language mismatch still matter.

Interpret the +1.05 point result as evidence that feature-level learned
reranking can exploit some complementarity, not as a new SOTA result. The
bootstrap interval crosses zero and file discovery regresses.

## Why simple score averaging is limited

Weighted averaging treats every query and every candidate as if both signals
were equally calibrated. It does not distinguish:

- exact anchors from generic lexical overlap;
- a visual score that agrees with text from a visual score that is isolated;
- a visual candidate that adds new evidence from one that merely duplicates
  BM25;
- file-level topic relevance from page-level evidence relevance.

The core design principle for the next experiments is:

    visual abstraction proposes or broadens the evidence set
    lexical signal verifies concrete page identity
    page-level evidence decides final ranking

This suggests a cascade or structured reranker, rather than concatenating all
scores into one undifferentiated vector.

## Surveyed research directions

### 1. Calibrated convex fusion

Bruch, Gai and Ingber analyze convex combination and RRF for hybrid retrieval.
Their results argue that a calibrated convex combination is often stronger and
more stable than RRF, while RRF can be sensitive to its parameters. This
supports keeping weighted fusion as the honest baseline, but tuning alpha only
inside training folds.

Reference:
https://arxiv.org/abs/2210.11934

Action here: calibrate each score by a query-local percentile or a held-out
reliability curve before any more complex model. Do not assume raw BM25 and
V-SPLADE scores are commensurate.

### 2. Query-adaptive fusion

Query-adaptive multimodal search has a long history: the key idea is that the
best modality depends on the query. A query-adaptive late-fusion paper
estimates feature usefulness from the score curve instead of using a fixed
weight. Recent query-adaptive hybrid search work predicts the lexical/semantic
mix from the query and explicitly targets low-latency deployment.

References:
https://research.google/pubs/query-adaptive-fusion-for-multimodal-search/
https://openaccess.thecvf.com/content_cvpr_2015/html/Zheng_Query-Adaptive_Late_Fusion_2015_CVPR_paper.html
https://www.mdpi.com/2504-4990/8/4/91

Action here: learn one query-level gate alpha(q), not a page-level rescue
threshold. Useful inputs are query length, number of anchors, fraction of
unmatched terms, BM25 score-curve concentration, V-SPLADE score-curve
concentration, and the disagreement between the two top-10 lists. This is
cheap, interpretable and less vulnerable to promoting many unrelated visual
pages.

### 3. Hybrid-candidate training and learned reranking

HYRR shows that a reranker trained on candidates collected from hybrid
retrievers can be more robust to the first-stage retriever. The important
lesson is not only to train a ranker, but to expose it to positives and hard
negatives from both retrieval paths. The candidate distribution itself is
part of the training problem.

Reference:
https://aclanthology.org/2024.lrec-main.748/

Action here: keep BM25-only hard negatives, V-SPLADE-only hard negatives,
agreement negatives and near-tie negatives separate. Train a small listwise
or pairwise ranker with interaction terms such as lexical coverage times
visual percentile and agreement times anchor strength. The current linear
ranker is a first probe, not the final reranker.

### 4. Late interaction and multi-vector matching

ColBERT and ColBERTv2 preserve token-level document representations and use
late interaction rather than compressing a whole document into one vector.
ColPali transfers this idea to document images using multiple visual patch
vectors and MaxSim. This is a principled way to preserve local instances
while retaining visual abstraction.

References:
https://arxiv.org/abs/2004.12832
https://aclanthology.org/2022.naacl-main.272/
https://arxiv.org/abs/2407.01449

Action here: this is a longer-term model branch. A direct fusion could use a
text late-interaction score for extracted terms and a visual MaxSim score for
patches, with a query-conditioned gate. It is not a drop-in replacement for
the current V-SPLADE dot product and would require new embeddings/models.

### 5. Guided query refinement

Guided Query Refinement is especially relevant to this project because it
studies hybrid retrieval for visual-document models and argues that coarse
score/rank fusion misses interactions inside the representation spaces. It
uses a complementary text retriever to guide test-time refinement of the
visual query representation.

Reference:
https://proceedings.iclr.cc/paper_files/paper/2026/hash/015d6a156390d23c19d17651d60b3db6-Abstract-Conference.html

Action here: use the idea conceptually before implementing it. The closest
offline approximation is to let lexical evidence identify query anchors or
hard negatives, then adjust the visual candidate score only on the residual
set. A full GQR-style test-time optimization requires a compatible visual
retriever, so it is a later branch.

### 6. Hierarchical file-to-page retrieval

M3DocRAG and related visual-document systems explicitly separate document
embedding, page retrieval and multimodal answering. This matches the data
discovery objective better than treating every page as an independent final
answer unit. Existing local ablations already show that pooling page scores
by file can improve file recall while hurting final page ranking.

References:
https://arxiv.org/abs/2411.04952
https://aclanthology.org/2025.emnlp-main.1576/

Action here: build two separate objectives:

    file_score(file) = max or logsumexp over page-level abstract scores
    page_score(page | selected file) = lexical evidence + visual residual

Report file_recall@1/@3 independently, then page_recall after a fixed file
budget. Do not use a file-pooled score directly as the final page rank.

### 7. Visual-first RAG is adjacent, not the same problem

VisRAG and M3DocRAG retrieve page images and preserve visual information for
multimodal generation. They are important references for the eventual QA
stage, but they do not by themselves answer the current fusion question:
how to combine a cheap concrete text representation with a visual prior for
data discovery.

Reference:
https://arxiv.org/abs/2410.10594

## Recommended experiment order

1. Query-level adaptive gate. Use the current cached runs, five-fold
   validation, no new model. Compare a learned alpha(q) with fixed alpha=0.7.
2. Residual novelty fusion. Keep BM25 as the lexical backbone; add a
   V-SPLADE page only when it contributes a visual candidate not already
   explained by lexical evidence. Evaluate both page recall and file recall.
3. Hierarchical file-to-page fusion. Optimize file recall@3 first, then
   rerank pages within selected files. This is the most aligned with data
   discovery.
4. Stronger feature-level reranker. Use structured hard negatives and
   separate page-recall and file-recall diagnostics. A new dependency is not
   needed for a NumPy prototype.
5. Language-controlled visual run. Compare French and English V-SPLADE
   query vectors, then a multilingual visual retriever if resources permit.
6. Only after the lightweight methods are stable, evaluate selective OCR,
   AgenticOCR, native visual reranking or an MLLM on the selected pages.

## Evaluation guardrails

- Keep the 302 French queries, qrels, page depth and file derivation fixed.
- Fit alpha, gates, thresholds and ranker weights only on training folds.
- Do not use gold modality labels as scoring features.
- Separate candidate recall, page ranking, file discovery and end-to-end QA.
- Keep the English V-SPLADE query confound visible in every report.
- Report bootstrap intervals and per-query gains/losses; do not claim a
  method is better from a same-set sweep alone.
- Every new experiment must be a new script and a new output directory.
