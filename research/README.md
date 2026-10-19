# Adaptive retrieval research runners

The XGBoost Adaptive-K experiment is intentionally separate from the live
`ondemand.adaptive` runtime:

- `experiments/adaptive_k_optimization_core.py` rebuilds the frozen global
  PP-OCRv5/Qwen3 score matrices and feature families.
- `experiments/run_xgboost_adaptive_k_optimization.py` performs nested
  five-fold OOF feature ablation, calibration, loss/classification/neighbor
  screens and writes the dashboard/report.
- `../ondemand/adaptive/xgboost_score_modality_oof_20261016.json` is the small
  benchmark-only policy artifact used to connect the result to the live
  pipeline without adding XGBoost to production inference.

Install the research-only dependencies and point the runner at the frozen
reproduction bundle if it is not located at `_repro_retrieval_baseline/`:

```powershell
pip install -e ".[adaptive-k-research]"
$env:AXIOM_ADAPTIVE_BASELINE_ROOT = "D:\path\to\_repro_retrieval_baseline"
python research/experiments/run_xgboost_adaptive_k_optimization.py `
  --stage features `
  --output-dir data/work/xgboost_adaptive_k_optimization_20261016
```

The currently persisted `features/score_modality_oof.csv` reports Mean K
59.43 and macro recall 85.23%. It is nested OOF on the merged 220-query
benchmark, not a model validated on unseen corpora. An earlier table row of
59.23 / 85.37 was not found among saved OOF artifacts, so the committed policy
does not claim that number.
