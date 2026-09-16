# Generic benchmark hierarchical retrieval

`run_benchmark_hierarchical.py` is the benchmark-independent light retrieval
runner used by `benchmark_hierarchical_colab.ipynb`.

It accepts a raw benchmark directory containing `documents.jsonl`,
`queries.jsonl`, and optionally `qrels.jsonl`. If `documents.jsonl` is absent,
PDF and image files are discovered recursively and assigned deterministic IDs.
PDF pages are parsed with PDF-inspector; images are represented as visual-only
pages with empty BM25 text. PDF pages are rendered in memory at 144 DPI and
encoded with `naver/v-splade-efficient`.

The fixed cascade is:

```text
page_base  = 0.70 * normalized(page BM25) + 0.30 * normalized(V-SPLADE)
file_score = 0.50 * normalized(file BM25) + 0.50 * normalized(max page_base)
page_final = 0.85 * page_base + 0.15 * normalized(file_score)
```

The default run selects `Kf=3` files, evaluates top-10 pages, and saves top-20
pages. Saved rows contain IDs, relative source path, page indices/numbers,
scores, and component scores; full page text remains in the parsing cache.

## Local command

```powershell
python research/experiments/run_benchmark_hierarchical.py `
  --dataset-root data/raw/BENCHMARK `
  --output-dir data/work/benchmark_hierarchical/run_id `
  --device cuda `
  --file-k 3 --metric-page-k 10 --saved-page-k 20 `
  --stage all
```

Stages can be run independently with `--stage`. A matching stage marker is
reused only when its config/input signature and required artifacts still match.
Use `--force-stage STAGE` to invalidate one stage. `--limit-documents`,
`--limit-queries`, and `--no-qrels` are intended for smoke tests only.

Physics and Industrial qrels use `source_corpus_id`. The runner first uses a
local `data/benchmark/vidore_v3/<subset>/corpus.parquet` when available and
otherwise downloads only `corpus_id`, `doc_id`, and `page_number_in_doc` from
the Hugging Face datasets-server. MPDocVQA uses `pdf_page_number - 1`.
Unmapped documents/pages fail before retrieval; qrels never enter parsing,
index construction, or ranking.

## Output contract

The run root contains validation inventory and manifest, normalized page qrels,
per-file parsing checkpoints, per-document V-SPLADE checkpoints, merged sparse
vectors/metadata, BM25 indexes, events/errors, per-query retrieval timing,
`reports/report.{json,md}`, `reports/per_query.jsonl`, and a `bundle/<run>.zip`.
The zip intentionally excludes raw PDFs and rendered image caches.

For Colab/Drive execution, use `research/experiments/benchmark_hierarchical_colab.ipynb`.
