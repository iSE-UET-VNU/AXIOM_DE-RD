# Independent adaptive retrieval module

This package predicts the page budget `K` for a frozen lightweight retrieval
ranking. It sits beside `ondemand/adaptive/` and does not call or modify the
processor router, branches, data-discovery pipeline, or QA code.

## Current transfer experiment

The first experiment matches the adaptive-router light-retrieval configuration
from the local handoff artifact:

- corpus: merged 220-query benchmark, 5,091 pages;
- ranking cap: top 100;
- dense model: `openai/text-embedding-3-small` (1536 dimensions);
- score fusion: BM25 0.30 + dense 0.70, all-language signals;
- file selection: dense pool over all 103 files, with the router's 0.15 parent
  score contribution;
- cached PP-OCRv5 page text and the existing TE3-small embedding cache; the
  run refuses to make embedding API calls;
- features: 21 score-curve statistics from the final router scores only;
- model: continuous-K XGBoost regressor, source-stratified 5-fold outer OOF,
  with scale/margin calibration on 4-fold inner OOF predictions;
- label: minimum K containing every gold page in the top-100 candidate list;
  if any gold page is outside top-100, target K is 100 and an unreachable flag
  is retained.

Gold ranks are never passed into the feature extractor. The runner rebuilds the
top-100 scores and uses the frozen router order. It requires the top-100 page
set to match; a score-order discrepancy is tolerated only for a numerical tie
whose score inversion is at most `1e-6`, and is written to the reproduction
report. This occurred once in the current bundle: the recomputed scores differ
by `1.19e-7` for an adjacent tie. Any larger discrepancy stops the run before
training.

## Run

From the repository root:

```powershell
pip install -e ".[adaptive-retrieval]"
python -m ondemand.adaptive_retrieval
```

The default run is offline and writes under
`data/work/adaptive_retrieval_20261020_router_te3s_top100/`. Alternative input
or output locations can be passed with `--artifact-root`, `--benchmark-root`,
and `--output-dir`.

Inspectable outputs include `feature_matrix.csv` (features and labels),
`training_table.csv` (features, labels, and outer-OOF predictions in one
spreadsheet-friendly table), `retrieval_scores.jsonl` (the 100 page IDs and
final score values per query), `oof_predictions.jsonl`, fold calibration and
train/test error diagnostics, `report.md`, and a final model/policy artifact.
These outputs are standalone research artifacts; no integration into the
adaptive router is performed yet.

For a future router integration, load the exported all-data model and predict
K directly from a query's already-computed, final fused score curve:

```python
from ondemand.adaptive_retrieval.inference import AdaptiveKPredictor

allocator = AdaptiveKPredictor.load("data/work/adaptive_retrieval_20261022_router_te3s_top100")
ranked_pages = subdata.pages[:100]
ranked_final_scores = [subdata.scores[page_id] for page_id in ranked_pages]
chosen_k = allocator.predict_k(ranked_final_scores)
```

`ranked_final_scores` must use the same retriever configuration as the model:
PP-OCRv5 text, `text-embedding-3-small`, BM25/dense fusion 0.30/0.70, and the
router's top-100 ranking. The wrapper does not inspect query identity, qrels,
gold pages, or labels. `report.json`'s headline metrics are nested outer-OOF;
the exported all-data deployment model is trained on all 220 benchmark rows and
must be validated on unseen queries before production use.

## Deliberate first-version exclusions

- no neighbor/query-cluster features;
- no router/processor selection;
- no live embedding requests;
- no loss reweighting, synthetic augmentation, or model tuning;
- no use of gold-page IDs as model inputs.

This keeps the first result attributable to one change: the continuous-K
regressor is retrained and calibrated on score curves from the router's own
top-100 / TE3-small ranking.
