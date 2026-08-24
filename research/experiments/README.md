# ViDoRe V3 experiment drivers

Run from the repo root. `PY` is the conda env binary:

    PY=/usr/local/Caskroom/miniconda/base/envs/axiom-de-rd/bin/python

Every script pins its configuration as module constants rather than taking flags,
so a run is reproducible from the file alone. Outputs land in
`data/benchmark/vidore_v3/results/`, embeddings cache in
`data/work/vidore_physics_emb/`.

## 0. Preconditions

    $PY research/experiments/verify_models.py      # every alias resolves, none is a mock
    $PY -m src.evaluation.build_corpus --help    # corpus path is an explicit argument

`.env` must carry `OPENROUTER_API_KEY`; `AXIOM_MODEL_SERVICE_URL` must point at a
running gateway for the LLM arms.

## 1. Fetch the benchmark (once)

    $PY research/experiments/fetch_corpus.py       # writes data/benchmark/vidore_v3/<subset>/*.parquet
    $PY research/experiments/to_csv.py             # human-readable queries.csv + pages.csv

## 2. Validate the wiring before trusting any number

    $PY research/experiments/bm25s_check.py

Runs the paper's own `bm25s` over our loader, queries and qrels. Must bracket the
published 39.8 on physics French. If it doesn't, the data path is wrong and every
downstream number is void.

    pip install bm25s PyStemmer   # not a repo dependency; only this check needs it

## 3. Check an external parse before designing against it

    $PY research/experiments/physics_format_report.py

Document-name join, page coverage, block-type inventory, unmapped types, parse
status and `table_refinement` config parity. Run this on any new parser output
*before* indexing it -- a coverage gap reads as a quality difference otherwise.

## 4. Retrieval ladder

    $PY research/experiments/physics_ladder.py         # 8 indexes x 4 retrievers
    $PY research/experiments/physics_significance.py   # paired permutation, 10k resamples

Writes `physics_retrieval_ladder.json` with per-query NDCG@10, which the
significance script consumes. ~$0.06 in embeddings; re-runs are free once cached.

## 5. End-to-end generation and judging

    $PY research/experiments/physics_e2e.py

Four arms: `oracle`, `oracle_chandra2`, `retrieved_vidore`, `retrieved_chandra2`.
Checkpointed per arm -- delete an arm's JSON to redo just that one. ~$0.30.

For a replicate, copy the file, point `OUT` at a new directory, and disable the
retrieval loop; run-to-run noise measured this way was +/-1pp.

## 6. Persist and export

    $PY research/experiments/persist_results.py    # re-derives retrieved ids, reconciles gold_hit
    $PY research/experiments/export_csv.py         # 7 CSVs under results/csv/

## 7. Inspect individual rows

    $PY research/experiments/browse.py retrieval computer_science bm25 --n 5 --only miss --text 200
    $PY research/experiments/browse.py generation computer_science --n 5 --only "Partially Correct"

## Harness entry point (the general path, not experiment-specific)

    $PY -m src.evaluation.run_retrieval \
        --benchmark vidore_v3 --subset physics --language french \
        --arms bm25,dense,rrf,alpha0.7 --embedder openrouter_te3s --k 10 \
        [--chunker fixed_overlap --chunk-param n_words=512 --chunk-param overlap=128] \
        [--prefix] [--rerank llm --rerank-model llm-rerank --rerank-depth 20]

    $PY -m src.evaluation.compare_arms data/benchmark/runs/*.report.json

`run_retrieval` writes a per-arm JSONL keyed by index identity, so two configs
never share a run cache. `compare_arms` refuses ragged arm sets and reports
coverage separately from accuracy.
