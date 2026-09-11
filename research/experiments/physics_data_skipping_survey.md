# Data skipping for AXIOM_DE-RD light preparation

Research note for the ViDoRe V3 Physics light-retrieval experiment. The
current benchmark protocol is 302 French queries, 42 files, 1,674 pages and
full page IDs of the form `physics::<doc>#page=<n>`. The cached V-SPLADE query
vectors used by the historical fusion runs are English translations; this is
kept as an explicit confound.

## Mechanism taxonomy and transfer

| Lakehouse mechanism | Typical granularity | Transfer to documents | Recommended role here |
|---|---|---|---|
| Explicit/hidden partition pruning | table partition / file set | Partition by source file, collection, tenant, date or other stable file metadata | Safe file inventory reduction before parser work; not a relevance ranker |
| Manifest-list and manifest pruning | manifest -> data file | A small metadata manifest over file IDs, page count, text availability, section ranges and index versions | First light-preparation gate; preserve snapshot/version and make stale metadata fail closed |
| File/column statistics | file, row group | Min/max/null/count and token/length/selectivity summaries for page zones | Candidate planning and cost estimation; only prune when recall is measured |
| Zone maps / row-group statistics | row group / block | Contiguous page or section zones with text and visual score summaries | Soft page prior or OCR/VLM work-unit selection; hard skip only after high candidate recall |
| Parquet page/column indexes | column chunk / data page | Exact page/block locator for selective numeric, symbolic or text fields | Useful for selective expensive parsing; ordinary page BM25 already supplies an inverted lexical index |
| Dictionary, set and Bloom filters | row group / column | Typed exact field for numbers, units, identifiers and formula variables | Guard/skip for expensive work; should not be expected to improve semantic relevance |
| Bitmap/record/secondary indexes | value / record | Exact file/page lookup for stable IDs or extracted entities | Useful after entity extraction; outside the first light baseline |
| Expression indexes | transformed column | Normalized forms such as accent-folded terms, units or formula canonicalization | Promising next preparation feature if a deterministic transform is defined; avoid gold-derived transforms |
| Z-order / liquid or sort clustering | data layout | Co-locate pages with source/section/entity/visual locality so zones are selective | Preparation-time organization; page-neighbor expansion is a ranking hypothesis, not a safe filter |
| Multi-index union | several metadata indexes | Lexical and visual file indexes each propose candidates; union before exact page verification | Safer than one hard index, but union size must be budgeted |

The primary references are the Apache and Delta specifications/docs:

- [Delta data skipping and statistics](https://docs.delta.io/optimizations-oss/)
- [Delta file skipping through scan filters](https://docs.delta.io/delta-kernel/)
- [Delta Z-ordering and data locality](https://docs.delta.io/optimizations-oss/)
- [Apache Iceberg metadata filtering and column bounds](https://iceberg.apache.org/docs/latest/performance/)
- [Apache Iceberg hidden partitioning](https://iceberg.apache.org/docs/latest/partitioning/)
- [Apache Hudi metadata, column statistics and partition statistics](https://hudi.apache.org/docs/next/metadata)
- [Parquet page indexes](https://parquet.apache.org/docs/file-format/pageindex/)
- [Parquet Bloom filters](https://parquet.apache.org/docs/file-format/bloomfilter/)

## Proposed hierarchy

```text
file manifest
  -> file/partition stats
  -> section or page zone
  -> exact page BM25 + visual sparse verification
  -> block/paragraph OCR or VLM only for selected evidence
```

The important invariant is the same as in a lakehouse scan planner: a false
negative at a parent level cannot be recovered at a child level. Therefore
the retrieval version of a skip index must expose `candidate_recall` and not
only final latency or page recall.

## Cache-only baseline results

All values below use the repository's full page-unit metric and are held
against the existing Physics references.

| Experiment | Best / OOF page recall@10 | File recall@3 | Interpretation |
|---|---:|---:|---|
| Existing hierarchical OOF | 44.28% | 85.60% | Current handoff reference |
| Fixed 4-page row-group soft prior | 44.10% / 42.98% OOF selection | 86.59% / 85.26% | Coarse pooling does not improve ranking |
| 4-page hard 16-page/file | 40.90% | 86.59% | Page candidate recall only 59.42%; unsafe |
| Section-zone soft prior | 44.28% / 43.91% | 86.59% / 86.42% | Headings do not add enough page discrimination |
| Typed exact anchor field | 44.20% | — | Mostly duplicates BM25; typed-only is 1.61% |
| Adaptive Kf by file-score gap | 44.76% | — | Fixed gap rule does not justify widening |
| Lexical/visual file union | 44.42% | — | Candidate file union dilutes page ranking |
| Multi-stream RRF | 45.24% OOF | 85.60% | Weak streams pollute consensus |
| Multi-stream rank LTR | 44.41% OOF | 86.42% | Fold-safe rank metadata is not enough |
| Existing cross-Kf OOF | 45.39% | — | Best cached structural reference found |

The first successful verifier experiment combines the skipping-style
candidate union with a small multilingual cross-encoder. Reranking only the
c014/cross-Kf top-30 union reaches **47.43% OOF** by itself; a fixed RRF-20
ensemble with the independent dual-encoder OOF run reaches **47.84% OOF** and
therefore clears the requested 47.47% target by **0.37 percentage points**.
The CE full-set value is 47.68%, but it is not used as the success claim; the
held-out RRF result is the primary result. The final artifact is
`data/benchmark/vidore_v3/results/physics_ce_dual_rrf/`.

The candidate-ceiling audit is the key diagnosis: union of existing cache-only
top-10 streams covers **53.26%** of gold page recall, and union of their
top-100 streams covers **93.72%**. The missing performance is therefore in
query-conditioned page ranking, not in adding another hard skip layer.

Additional independent-index checks reached the same conclusion. Under the
canonical full-page-unit evaluator, the cached Tesseract OCR run reaches
38.24%; the 44.70% value in its old metrics file is a legacy page-number
metric. Fixed Tesseract fusion/gates do not beat the 45.39% cross-Kf control.
A fold-safe ranker over seven stream ranks, scores, lexical coverage and cheap
layout markers reaches 43.94% OOF. Thus a larger union is not by itself a
better page ranker.

## Protocol warning

The old `src/evaluation/run_retrieval.py` page metric compares page numbers
without the document ID. That can produce a different number when different
documents have the same page number. The local hierarchical reports use the
full page unit and must remain the canonical comparison. In the current cache,
the old weighted English-visual report shows `47.43%` under the page-number
metric, while the full-unit derived metric is `41.28%` for the same weighted
run. These are not interchangeable baselines.

## Decision

Keep data skipping as a light-preparation/cost-control layer:

1. materialize versioned file/page metadata;
2. use soft file/zone priors and candidate-union coverage checks;
3. enable hard skipping only with an explicit held-out candidate-recall gate;
4. spend the next relevance effort on a controlled page ranker or a fair
   French/multilingual visual model, not additional fixed zone sizes or Bloom
   filter variants.

The skipping layer is useful because it preserves a high-recall union while
limiting the verifier to a small page pool; it is not sufficient as a
standalone relevance ranker. The accepted light-retrieval result is the
two-stage candidate union plus cheap verifier, with fixed RRF-20 ensemble:
**47.84% OOF page recall@10**, above 47.47%. This result remains subject to
the explicit French-query/English-V-SPLADE confound in the structural input
runs and should be rechecked with a French visual query model later.
