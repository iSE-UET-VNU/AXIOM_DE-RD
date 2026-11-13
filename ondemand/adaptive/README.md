# Adaptive retrieval + QA: end-to-end run guide

```
query ─► query_prep ─► light_retriever ─────────► adaptive_selector ─► top3_check ─► visual_encoder ─► chunk_rank ─► qa_reader ─► qa_judge
         (none)        BM25 enfr + Qwen3-Emb-8B    K from XGB (oof_k)   (none)        EVIE V2 skip-blank   512/128 words   Qwen3-VL     gpt-4o-mini
                       lang rule, blended file     top-K of light top 100              + stop:10 -> top 10   hybrid, top 10                structured
```

Every stage is a registry component (`registry.py`), chosen by a CLI flag, and runs in one of two modes:

- **offline**: read the cached phase output; stop with the path and the online command if anything is missing.
- **online**: run the model / API, write the result into the same cache, so the next offline run reads it.

Set `--mode` for all stages and override single stages with `--stage-mode STAGE=MODE`.
Heavy packages (`torch`, `transformers`, `vllm`, `colpali_engine`, `xgboost`, `sklearn`, `tiktoken`, `tokenizers`,
`fitz`) are imported only inside the component that needs them, so an offline run needs none of the GPU stack.

All commands run from the repo root with the project env:

```
cd /Users/khoatran/Desktop/ISE/Axiom-DE-RnD/AXIOM_DE-RD
P=/usr/local/Caskroom/miniconda/base/envs/axiom-de-rd/bin/python
```

## 1. Stages

| stage | flag | default | other registered | offline reads | online does |
|---|---|---|---|---|---|
| query_prep | `--query-prep` | `none` (raw query) | — (hook for vuhaidang's prep) | — | — |
| embedder | `--embedder` | `qwen3_8b` (page level, pinned Nebius bf16 matrix) | `qwen3_8b:<matrix stem>`, `te3s` (chunk-max, legacy) | `light_v2/<stem>_{pages,queries}.npy` | embeds pages + queries, one text per request, pinned provider |
| light_retriever | `--light` | `hier` | `gate:<gate json>` (legacy precomputed ranking) | pages file + embedder | — (CPU) |
| ↳ options | `--signals --file-mode --w-dense --parent` | `lang`, `blended_pool`, 0.70, 0.15 | `all`, `dense_pool`, `bm25_blend` | | |
| adaptive_selector | `--selector` | `oof_k` (out-of-fold K per benchmark query) | `xgb_k` (live model, new queries), `gate_k`, `fixed:N`, `score_gap` | `adaptive_k/<artifact>/` | — (CPU) |
| top3_check | `--top3` | `none` | — (hook: LLM "enough evidence?" on top 3) | | |
| visual_encoder | `--branch-arg visual=visual:<scorer>+ppocrv5+stop:10` | scorer `evie:skip_blank` | `evie_cached:skip_blank` (legacy, falls back to light score), `colvec*` | `routing_axes/k13/scores_v2_all.npy` + `evie/live/<sig>/scores.jsonl` | EVIE-4.5B V2 encoder on GPU, scores cached per (query, page) |
| chunk_rank | `--chunk --ranker --top-chunks` | `fixed:512:128`, `hybrid`, 10 | `page_order` | te3s chunk cache | embeds new chunks (te3s) |
| qa_reader | `--reader` | `qwen3_vl_32b` (OpenRouter pinned, or `:endpoint` vLLM) | `deepseek_v4_flash` (legacy) | `qa_cache/answers_<reader>/` | calls the model |
| qa_judge | `--judge` | `gpt4o_mini_structured` (9-field JSON prompt) | `legacy_4omini` | `qa_cache/judge_<judge>/` | calls the model |

The visual stage keeps the window rule: light's top-10 documents first, then the rest of the top K, scored one
page at a time; stop after 10 pages in a row fail to enter the current top 10; the top 10 by EVIE score is the output.

## 2. Determinism

| where | setting |
|---|---|
| all runs | `seed_everything(0)`: Python, NumPy, `PYTHONHASHSEED`, torch (deterministic algorithms) when loaded; ties broken by page id |
| Qwen3-Embedding-8B | OpenRouter `{"order":["nebius"],"quantizations":["bf16"],"allow_fallbacks":false}`, one text per request, served provider checked on every reply; query `Instruct: Given a question, retrieve document pages that answer the question\nQuery:{q}` (model card format); pages cut at 8192 Qwen tokens with Qwen's tokenizer |
| LLM calls | temperature 0, top_p 1, seed 0, one pinned provider, fallbacks off, `require_parameters` on |
| judge | `openai/gpt-4o-mini-2024-07-18` on provider `openai`, `response_format` json_object, strict 9-field parse |
| vLLM (GPU) | `--seed 0 --max-num-seqs 1` (no cross-request batching) |

A hosted model can still vary between calls. Measure it, don't assume it:

```
PYTHONPATH=. $P -m ondemand.adaptive.determinism_check --model <openrouter id> --provider <slug> --quantization bf16
```
Pass: `"identical_prompts": 20` of 20 (3 runs each, real top-10 QA prompts) and one provider in `served_by`.

## 3. Runs

**3.1 Offline benchmark run, new stack** (no GPU, no API; needs the cached matrix, K artifact, EVIE matrix):
```
PYTHONPATH=. $P -m ondemand.adaptive run --tag new_stack_214 --exclude-flawed
```
Writes `data/work/ondemand_v2/<fp>/results/new_stack_214/` (`per_query.jsonl`, `metrics.json` with R@10 per group,
`manifest.json` with every stage setting and mode). `--out-dir <dir>` writes elsewhere, `--limit N` for a smoke run.

**3.2 Legacy reproduction** (te3s gate, precomputed K, EVIE cached; should give R@10 69.46 on the 214):
```
PYTHONPATH=. $P -m ondemand.adaptive run --tag legacy_check --exclude-flawed \
  --light gate:gate_ppocrv5_ungated_xgbK214.json --selector gate_k \
  --branch-arg visual=visual:evie_cached:skip_blank+ppocrv5+stop:10
```

**3.3 QA on a finished run** (reader and judge are independent of retrieval; outputs are named per pair, so
`qa_<reader>__<judge>.{jsonl,csv}` + `qa_summary_<reader>__<judge>.json` never overwrite another pair):
```
PYTHONPATH=. $P -m ondemand.adaptive qa --tag new_stack_214 --reader qwen3_vl_32b:endpoint --judge gpt4o_mini_structured \
  --stage-mode qa_reader=online --stage-mode qa_judge=online
PYTHONPATH=. $P -m ondemand.adaptive qa --tag legacy_check --reader deepseek_v4_flash --judge legacy_4omini
```
The second line is offline and reproduces the cached baseline (Correct 50.93, C+P 89.72) with zero API calls.
`--qa --reader ... --judge ...` on `run` does the same right after retrieval.

**3.4 Rebuild the Qwen page matrix** (online, ~2.3M tokens, ~$0.02), then measure repeatability:
```
PYTHONPATH=. $P -c "
from ondemand.adaptive.embedders import Qwen3Embedding8B
from ondemand.bench import load_jsonl, work
rows = load_jsonl(work('light_prep') / 'pages_ppocrv5.jsonl')
e = Qwen3Embedding8B('<new stem>', online=True); e.prepare(rows, 'pages_ppocrv5.jsonl'); print(e.repeat_check(rows))"
```

**3.5 Retrain adaptive K for a light ranking** (CPU; teammate XGB parameters and calibration, nested 5-fold OOF on
the 214, deploy model on the 214):
```
PYTHONPATH=. $P -m ondemand.adaptive.retrain_k --embedder qwen3_8b --signals lang --file-mode blended_pool
```
Writes `adaptive_k/qwen8b_lang_blended_214/` (`oof_k.json` for benchmark runs, `xgboost_adaptive_k.json` +
`deployment_policy.json` for `xgb_k`, `dataset_manifest.json` with the ranking signature). `oof_k` refuses, and `xgb_k`
warns, when the run's light ranking differs from the one the artifact was trained on.

**3.6 GPU stages**: see `colab/README.md` (`setup_gpu.sh`, `qwen3_embed_pages.sh`, `evie_encode.sh`,
`serve_qwen3_vl.sh`). Bundle the repo + caches first: `PYTHONPATH=. $P -m ondemand bundle --target adaptive`.

**3.7 Tests** (offline, temp output only):
```
$P -m pytest ondemand/adaptive/modes_test.py -q
```
Checks: imports with every heavy package blocked; mode parsing; judge prompt filling and parsing; 3-query offline
smoke of the new stack; te3s `hier` ranking equals the legacy gate's top 100; legacy and new EVIE scorers give the
saved run's top 10; legacy QA labels come back from cache only.

## 4. Adding a component

```python
from ondemand.adaptive.registry import register

@register("qa_reader", "my_reader")
class MyReader:
    def __init__(self, argument=None, bench=None, **_): ...
    def settings(self): return {"reader": "my_reader", ...}
    def read(self, query, chunks, online): return {"answer": "...", "gen_seconds": 0.0}
```
Import the module in `pipeline.py` (or `qa.py` for QA parts), then `--reader my_reader[:argument]`. Interfaces:

| kind | methods |
|---|---|
| query_prep | `prepare(query, meta) -> PreparedQuery(raw, text, language, translations, sub_queries, meta)` |
| embedder | `prepare(rows, pages_name)`, `scores(text, qid, raw) -> np.array over pages` |
| light_retriever | `prepare(qids)`, `retrieve(qid, prepared) -> (SubData, notes)` |
| adaptive_selector | `bind(light_settings)`, `choose(subdata) -> (k, notes)` |
| top3_check | `check(prepared, subdata, online) -> {"action": "stop" or "continue"}` |
| scorer (visual) | `score(qid, pages, subdata) -> {page: score}` |
| qa_reader / qa_judge | `read(query, chunks, online)` / `judge(query, answers, answer, online)`, plus `settings()` |

## 5. Known limits

- `qwen/qwen3-vl-32b-instruct` has no OpenRouter provider (checked 2026-11-14); run it with vLLM
  (`colab/serve_qwen3_vl.sh`) or pick another model after `determinism_check` passes.
- Online EVIE, the local Qwen embedding script and vLLM serving are ported but not yet run on a GPU.
- `oof_k` K is calibrated with the teammate's budget (mean K < 60), not tuned here.
- 155 pages have no text (151 blank, 2 near-blank, 2 black); they get a zero dense vector.
- Legacy experiments (routing studies, Kaggle notebooks, plans) are archived outside the repo in
  `../axiom-de-rd-legacy-experiments-2026-11-14.tar.gz` (file list next to it).
