# GPU stages

Each script runs one stage online on a GPU and leaves its output in the cache folder that `--mode offline` reads.
Run from the repo root on the GPU machine, after copying the bundle (`python -m ondemand bundle --target adaptive`).

| script | stage | GPU | output to bring back |
|---|---|---|---|
| `setup_gpu.sh` | installs, clones EVIE at the pinned commit | any | `export EVIE_DIR=...` |
| `qwen3_embed_pages.sh` | Qwen3-Embedding-8B pages + queries via local vLLM, then cosine parity against the pinned Nebius matrix | 24 GB+ (L4) | `light_v2/<stem>_*` |
| `evie_encode.sh` | EVIE V2 skip-blank encoder: keep-all/parity check, then the pipeline with `visual_encoder=online` | 16 GB+ (T4/L4) | `evie/live/` |
| `serve_qwen3_vl.sh` | Qwen3-VL-32B reader via vLLM (answers only; judge runs locally) | FP8 ~40 GB, bf16 ~80 GB (A100/H100) | `qa_cache/answers_qwen3_vl_32b_endpoint/` |

Determinism: every server starts with `--seed 0 --max-num-seqs 1` (no cross-request batching); clients send
temperature 0, top_p 1, seed 0. The vLLM flags and the Qwen3-VL model id are unverified on a real GPU yet.
