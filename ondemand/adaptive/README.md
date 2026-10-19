# On-Demand Adaptive Multimodal Retrieval & QA — Developer & Tool Guide

```
Raw data ──► Light Prep ──► Light Retrieval ──► SubData ──► Router ──► Branch ──► Chunk&Rank ──► QA Reader
             (PP-OCRv5)     (BM25 + Dense Qwen/TE3) (top-k)     (Rule/    ├─ 1. Light (CPU OCR)
                                                                 Fixed)   ├─ 2. Enrich (Chandra vLLM / OCR)
                                                                          └─ 3. Visual (ColVec cache [default] / ColVec live + KDL)
```

This module provides a modular, reliable tool-calling architecture for document retrieval and question answering across heterogeneous formats (native text, scanned contracts, engineering drawings, tables, and figures).

---

## 1. Quick Setup & Health Check

Teammates can check their environment, endpoints, and local cache artifacts in 1 second:

```bash
conda activate axiom-de-rd
python -m ondemand.adaptive doctor
```

The `doctor` command inspects:
- **Endpoints**: `CHANDRA_ENDPOINT` / `VLLM_API_BASE`, `COLVEC_ENDPOINT`, etc.
- **API Keys**: `OPENROUTER_API_KEY` / `OPENAI_API_KEY`
- **Local Artifacts**: gates, PP-OCRv5 text, ColVec scores, KDL parses, embedding caches
- **Tool Readiness**: reports whether branches are running in Live/Cached mode or Graceful Fallback mode.

### Setting Up `.env`

Copy `.env.example` to `.env`:

```bash
# Chandra2 / Layout OCR endpoint (vLLM or OpenAI-compatible vision completion)
CHANDRA_ENDPOINT=http://localhost:8000/v1
# Or standard teammate vLLM base:
# VLLM_API_BASE=http://localhost:8000/v1
CHANDRA_MODEL=datalab-to/chandra-ocr-2

# Optional: only needed to override the default ColVec cache with a live
# ColVec reranking endpoint for the visual branch.
# COLVEC_ENDPOINT=http://localhost:8001/v1

# LLM / Vision QA API keys
OPENROUTER_API_KEY=your_key_here
# OPENAI_API_KEY=your_openai_key_here
```

---

## 2. Interactive Tool Calling & Single Query CLI

Teammates can run on-demand retrieval for any single query or benchmark query ID directly from the CLI:

```bash
# Query with adaptive routing (clean digital -> Light, scanned/diagrams -> Visual, tables -> Enrich)
python -m ondemand.adaptive query "What is the maximum playback time for an NV-T60 tape?" --ranker page_order

# Benchmark query ID lookup
python -m ondemand.adaptive query "ohrbench::803adf07-c9f4-4910-8c12-277ff7aaac7e" --ranker page_order
```

Output:
```
============================================================
Query:  What is the maximum record/playback time for a video cassette in the SP mode when using an NV-T60 tape?
Branch: light (k=20)
Route:  {'reason': 'clean_digital_text', 'score_gap': 0.0539, 'ocr_candidates': 0}
Ranked Pages (20):
  #1: ohrbench::file_65c13bc2ecc7d7ea#page=4
...
============================================================
```

---

## 3. Python SDK / Programmatic Tool Usage

Teammates can import and call individual tools or the complete pipeline in scripts and notebooks:

### Full Pipeline Query
```python
from ondemand.adaptive import AdaptivePipeline

pipe = AdaptivePipeline(
    gate_name="gate_ppocrv5_all.json",
    router="rule",
    pages_name="pages_ppocrv5.jsonl",
    colvec_mode="local", kdl_mode="vllm",   # optional: host the models in this process
)

result = pipe.query("What is the invoice amount for project X?")
print("Chosen Branch:", result["branch"])
print("Routing Reason:", result["route_notes"])
for chunk in result["chunks"][:3]:
    print(f"[{chunk['page_id']}] {chunk['text'][:80]}...")
```

### Direct Tool Calling: one model service

Each model is a service with the same interface (`prepare`, `parse`/`score`/`ocr`, `close`) and a mode:

| Service | Modes | Hosted how |
|---|---|---|
| `colvec` | `cached`, `local`, `endpoint` | in-process HF model (8b, bf16, 1792 visual tokens, 144 dpi) |
| `kdl` | `cached`, `vllm` | `vllm serve KDLAI/KDL-Frontier-Parser-nano` started and stopped by the service; the selected pages are cut into a sub-PDF |
| `chandra` | `cached`, `vllm`, `local` | `vllm serve datalab-to/chandra-ocr-2`, or the `chandra` HF backend; prompts and rendering from the `chandra-ocr` package |
| `ppocr` | `cached`, `local`, `endpoint` | PP-OCRv5 in a worker process (`PPOCR_PYTHON` may point at a separate venv) |

```python
from ondemand.adaptive.services import KDL, ColVec

with ColVec("local") as colvec:
    scores, missing = colvec.score("q1", "What does port 2 emit?", ["ohrbench::file_d7c46ad196d20c27#page=3"])

with KDL("vllm") as kdl:
    text, missing = kdl.parse(["ohrbench::file_d7c46ad196d20c27#page=3"])
```

```bash
python -m ondemand.adaptive call colvec --colvec local --qid <query_id> --pages <page_id> ...
python -m ondemand.adaptive call colvec --colvec local --verify 20     # reproduction gate vs the corpus matrix
python -m ondemand.adaptive call kdl --kdl vllm --pages <page_id> ... --out kdl.json
python -m ondemand.adaptive call chandra --chandra vllm --chandra-endpoint http://host:8001/v1 --pages <page_id>
python -m ondemand.adaptive call ppocr --ppocr local --pages <page_id>
```

Every service writes, under `data/work/ondemand_v2/<fp>/<service>/live/<settings-hash>/`:
`settings.json`, an append-only per-page cache (`pages.jsonl`, or `scores.jsonl` for ColVec), `timings.jsonl`, and
logs to `data/work/ondemand_v2/<fp>/logs/<service>.log` (vLLM server output in `logs/vllm_<service>.log`).
A finished page is never recomputed; changing a setting starts a new settings hash.

---

## 4. Benchmark run with hosted models

```bash
# cache only (any machine)
python -m ondemand.adaptive run --tag rule_cached --router rule --exclude-flawed

# GPU: ColVec in-process, KDL and Chandra on self-hosted vLLM
python -m ondemand.adaptive run --tag rule_gpu --router rule --exclude-flawed \
  --colvec local --kdl vllm --chandra vllm --allow-api

# after a crash or disconnect: same command plus --resume
```

A run routes every query first, then runs one model at a time over all the pages it needs (ColVec, then KDL, then
Chandra, then PP-OCR), releasing the GPU between models, then assembles each query from the caches. `per_query.jsonl`
is appended one query at a time; `--resume` skips finished queries and refuses to continue with different settings.
`results/<tag>/` holds `manifest.json` (settings, service modes, GPU, git sha), `run.log`, `per_query.jsonl` and
`metrics.json`. Each record's `cost` carries `<service>_model_seconds`: the model time its pages took, counted as if the
query ran alone. `--keep-loaded` keeps every model resident (large GPUs only).

Light prep (pdf-inspector, PP-OCRv5, gate) is a separate one-time step:

```bash
python -m ondemand.adaptive prep --allow-api      # writes pages_ppocrv5_live.jsonl and gate_ppocrv5_live.json
```

On Colab or Kaggle: `python -m ondemand bundle --target adaptive`, upload the zip, run `colab/04_adaptive_gpu_run.ipynb`.

| Flag | Default | Description |
|---|---|---|
| `--colvec/--kdl/--chandra/--ppocr MODE` | `.env`, else `cached` | service modes, see the table above |
| `--colvec-model`, `--colvec-8bit` | 8b, off | smaller GPUs; scores then differ from the cache |
| `--kdl-endpoint`, `--chandra-endpoint` | none | use a running vLLM server instead of starting one |
| `--all-local-gpu` | off | colvec local, kdl vllm, chandra vllm, ppocr local |
| `--resume` | off | continue `results/<tag>` |
| `--router`, `--k`, `--refine-k`, `--ranker`, `--top-chunks` | `rule`, `fixed:20`, `fixed:10`, `hybrid`, 10 | as before |
| `--qa`, `--allow-api` | off | QA judging; live embedding/LLM calls |

Environment: `VLLM_BIN` (a `vllm` binary in its own venv), `PPOCR_PYTHON`, `COLVEC_MODE`/`KDL_MODE`/..., `KDL_ENDPOINT`,
`CHANDRA_ENDPOINT`, `COLVEC_ENDPOINT`, `PPOCR_ENDPOINT`, `OPENROUTER_API_KEY`.

### Importing an offline XGBoost Adaptive-K policy

The live pipeline does not train or import XGBoost. Training remains an
offline research step and exports one integer `chosen_k` per query. Convert
the nested-OOF CSV into a versioned policy, then attach it to a compatible
retrieval gate:

```bash
python scripts/adaptive_k_policy.py export \
  --predictions path/to/score_modality_oof.csv \
  --output path/to/xgboost_topk_policy.json \
  --name xgboost_score_modality_oof \
  --source-run xgboost_adaptive_k_optimization_20261016 \
  --metrics '{"mean_k": 59.4318, "macro_recall": 0.852258}'

python scripts/adaptive_k_policy.py apply \
  --policy ondemand/adaptive/xgboost_score_modality_oof_20261016.json \
  --gate data/work/ondemand_v2/<fingerprint>/light_prep/gate_ppocrv5.json \
  --output data/work/ondemand_v2/<fingerprint>/light_prep/gate_ppocrv5_xgb_topk.json

python -m ondemand.adaptive run --tag xgb_topk \
  --gate gate_ppocrv5_xgb_topk.json --k adaptive --router rule
```

The importer requires the policy qids to match the gate and refuses to
silently cap K. The continuous policy can request up to 200 pages, so rebuild
the global PP-OCRv5/Qwen gate after this change to materialise
`light_run_top200`. `--allow-cap` is available only for an explicitly labelled
diagnostic run. OOF qid lookups are benchmark evaluation artifacts, not a
policy for unseen user queries; unseen queries continue to use the existing
score-gap fallback.

---

## 5. Fallbacks

A page a service could not produce (no cache entry in cached mode, a failed parse in a live mode) falls back to the
light OCR text (KDL, Chandra) or the light-retrieval score (ColVec). Every fallback prints a loud warning, is counted per
query in `per_query.jsonl`, logged in `run.log`, and summarised in `metrics.json` and the end-of-run banner. A run with
fallbacks is not the configured pipeline.
