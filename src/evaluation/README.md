# src/evaluation

Scores retrieval and answer quality against a benchmark. Reads parser output;
parses nothing itself. Previously `research/harness/`.

## Setup

    pip install -e .
    python research/experiments/fetch_corpus.py                                  # queries + qrels


`.env` needs `OPENROUTER_API_KEY` for any dense arm.

## Run

Parse first — the existing pipeline, unchanged:

    python scripts/run_pipeline.py \
        --config configs/pipeline.vidore-v3-chandra2.yaml \
        --local-raw <dir-of-pdfs>
    # writes data/output/benchmarks/<run>/<run_id>/documents/*.json

### Parsing on Colab

chandra2 and KDL need a GPU with vLLM on localhost, so the parse runs where the
GPU is. `Chandra_serving_de.ipynb` and `KDL_serving_de.ipynb` start vLLM, patch
the config at runtime, and call the same `scripts/run_pipeline.py` above.

1. Open the notebook in Colab, GPU runtime.
2. Set `VLLM_API_KEY` in Colab **Secrets** — the cells read it via
   `userdata.get("VLLM_API_KEY")`, never a literal.
3. Run the cells in order. vLLM must report ready before the pipeline cell.
4. Download **both** `data/output/benchmarks/<run>/<run_id>/` and the matching
   `data/ingested/...` back to your machine, keeping the same relative layout,
   then point `--parsed-run` at the output one. Parser settings live in the
   ingested stage; with only `output/` the run is refused for missing parser
   metadata, because every corpus would otherwise hash to the same identity.

Both configs set `include_extensions: [".pdf"]`, so anything else in the input
directory is skipped without a warning. Neither notebook is tracked in this
repo — ask for them.

Then retrieve. The benchmark supplies questions and gold; `--parsed-run` only
changes where the text comes from.

    # baseline: the benchmark's own text
    python -m src.evaluation.run_retrieval \
        --benchmark vidore_v3 --subset physics --language french \
        --analyzer plain --arms bm25,dense,rrf --embedder openrouter_te3s --k 10

    # your parse: same command, one extra flag
    python -m src.evaluation.run_retrieval \
        --benchmark vidore_v3 --subset physics --language french \
        --parsed-run data/output/benchmarks/<run>/<run_id> \
        --analyzer plain --arms bm25,dense,rrf --embedder openrouter_te3s --k 10

Compare the arms:

    python -m src.evaluation.compare_arms data/benchmark/runs/*.report.json

Then answer and judge. `--run` is a JSONL written by the retrieval step above;
the benchmark supplies its own prompt and label set, so nothing here is edited
per subset.

    python -m src.evaluation.run_answer \
        --benchmark vidore_v3 --subset physics --language french \
        --arm chandra2_alpha0.7 \
        --run data/benchmark/runs/<index_id>/alpha0.7__<hash>.jsonl \
        --generator deepseek/deepseek-v4-flash --judge openai/gpt-4o

Add `--arms oracle` to the retrieval step for the gold-context ceiling; it emits
an ordinary run, so the same command scores it.

Reports `accuracy` (Correct) and `accuracy_credited` (Correct + Partially
Correct). Quote both — the only significant end-to-end effect we have measured
sits in the gap between them. Retrieval caches, generation does not: a failed
arm re-spends on the next attempt.

## Flags that matter

| Flag | Why |
|---|---|
| `--language` | required, no default — a multilingual average is not a number to report by accident |
| `--parsed-run DIR` | index a pipeline run instead of the benchmark's text |
| `--granularity page\|content` | `page` joins block text; `content` keeps table and list HTML |
| `--analyzer plain` | **pin it when comparing corpora** — `auto` picks per corpus by CJK presence, so two parsers can get different tokenizers |
| `--chunker`, `--prefix`, `--rerank` | optional index variations; each enters the run cache key |
| `--arms oracle` | gold pages as a perfect ranking — the ceiling retrieved arms are read against |
| `--generator`, `--judge` | a gateway alias, or a provider id like `deepseek/deepseek-v4-flash` to call OpenRouter directly; they must differ |

## Check it works

    python -c "from src.evaluation.benchmarks import _ADAPTERS; import importlib; \
      [importlib.import_module(f'src.evaluation.benchmarks.{n}') for n in _ADAPTERS]; print('OK')"

Adapters load by string, so a broken import path fails at run time, not import.

The test suite is not tracked here — ask for `tests/` before changing anything in
this module.
