# Adaptive retrieval research

Adaptive-retrieval experiments are grouped by decision mechanism rather than
mixed into the general `research/experiments` directory.

- `llm/`: LLM-based evidence sufficiency and adaptive stopping.
- `ondemand/adaptive_retrieval/`: independent XGBoost continuous-K module
  trained on the adaptive-router's frozen top-100 retrieval score curve. Its
  current local run artifacts are under `data/work/adaptive_retrieval_20261020_router_te3s_top100/`.

