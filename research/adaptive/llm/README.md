# Adaptive retrieval with LLM stopping

This area contains the LLM-based adaptive retrieval work.

```text
research/adaptive/llm/
  scripts/   experiment runners and tests
  results/   dated experiment outputs and API checkpoints
```

## Current experiment

- Runner: `scripts/run_adaptive_llm_stopping.py`
- Tests: `scripts/test_adaptive_llm_stopping.py`
- Smoke results: `results/adaptive_llm_stopping_20261019/`

Run the existing smoke configuration:

```powershell
python research/adaptive/llm/scripts/run_adaptive_llm_stopping.py `
  --smoke --resume
```

The runner uses the frozen PP-OCRv5/Qwen top-200 ranking. It reads
`OPENROUTER_API_KEY` from the repository `.env` and checkpoints every API
response so interrupted runs can resume safely.

## XGBoost-hard rescue batch comparison

`scripts/run_xgb_hard_rescue_batch_comparison.py` compares iterative batches
of 3 and 10 pages on a source-balanced pilot drawn only from the 134 queries
routed hard by the XGBoost OOF model. It inspects at most 50 pages; an accepted
LLM stop uses the current K, otherwise the original hard action K=100 is kept.

```powershell
python research/adaptive/llm/scripts/run_xgb_hard_rescue_batch_comparison.py `
  --resume
```

To run every OHR query routed as hard, comparing batch sizes 3 and 10 with up
to four query trajectories in parallel:

```powershell
python research/adaptive/llm/scripts/run_xgb_hard_rescue_batch_comparison.py `
  --source ohrbench --workers 4 --resume `
  --output-dir research/adaptive/llm/results/ohr_xgb_hard_iterative_b3_b10_20261019
```
