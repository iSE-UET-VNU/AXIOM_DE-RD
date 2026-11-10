# Query preparation

Workspace for query-side processing before retrieval: pain-point keyword
extraction, keep/decompose decisions, subquery retrieval and per-query pattern
review. The context, protocol and results are in the repository `AGENTS.md`,
section "Query preparation / pain-point decomposition screen (2026-11-10)".

| File | Purpose |
|---|---|
| `query_preparation_screen.py` | Experiment driver: `keywords`, `decompose`, `focus`, `verify` |
| `query_evidence_review_app.py` | Streamlit UI to search queries and inspect gold files/pages and retriever scores |
| `retrieval_scores.py` | Per-page BM25 / dense / file-prior breakdown of the saved retriever (`python research/query_preparation/retrieval_scores.py verify`) |
| `results/` | Outputs (git-ignored). `results/rebuild/` holds exact reproductions of the original screens |

## Review UI (`streamlit run research/query_preparation/query_evidence_review_app.py`)

Sidebar: dataset filter (default Physics + Industrial), text/ID search, previous/next query,
page zoom and retriever depth (top-N, 5–100).

- **Gold evidence tab**: for each gold file, a page reader (previous/next, jump to any page,
  `★ p.N` shortcuts to the gold pages) showing the rendered page next to the retriever text;
  a PDF containing only the gold pages (viewable, copyable, downloadable); optional full PDF.
  Neighbouring pages are one click away and show their retriever rank.
- **Retriever top-N tab**: the saved light-retriever run (`light_run_top100`, BM25 + `text-embedding-3-small`
  over PP-OCRv5 text). Each page is labelled gold page / gold file but other page / other file, with the
  query terms that matched or are missing (BM25 analyzer, `enfr`). Includes a per-file breakdown, gold pages
  that missed the top-N (with their rank), a page inspector with "next noise" / "next gold" jumps, and an
  optional thumbnail gallery. Query words are highlighted in the page text.

Scores: `final = 0.85 * (0.3 * BM25_norm + 0.7 * dense_norm) + 0.15 * file_prior`, where `dense` is the best
512-word-chunk cosine of the page (text-embedding-3-small), norms divide by the corpus maximum and `file_prior`
is the best dense page of the file. The UI shows the raw values, each signal's rank over all 5,091 pages, the share
of the final score from each part (stacked chart + `Nguồn điểm` column), per-term BM25 contributions with IDF and
document frequency, and the chunk that sets the dense score. `retrieval_scores.py verify` re-computes all 220 saved
rankings (219 exact, 1 tie-only difference, max score error 3e-7).

Caveats: the top-N is the full-corpus hybrid run before adaptive K and EVIE. This variant applies no language
routing, so English Physics queries only match French pages through shared word stems (e.g. `conditions`, `dynam`).

## Inputs (frozen, not modified here)

- Benchmark queries: `data/raw/0.1. BENCHMARK/queries.jsonl` (Industrial = 75 queries).
- Page corpus: 5,091 PP-OCRv5 pages, saved hybrid run and embedding cache under
  `data/work/adaptive_routers_handoff/.../ondemand_v2/ef7f11811803/`.
- EVIE scores: `routing_axes/k3/k3_out_shard0/evie_scores_2048.npy` (220 x 5,091).
- Original screen artifacts and LLM caches: `data/work/query_preparation/`.
- BM25 (`enfr`, `PageBM25`): `research/baselines/hierarchical_light/ondemand/text.py`.

## Run

```powershell
python research/query_preparation/query_preparation_screen.py keywords
python research/query_preparation/query_preparation_screen.py decompose            # 18 low-recall queries
python research/query_preparation/query_preparation_screen.py decompose --subset all
python research/query_preparation/query_preparation_screen.py focus --qid vidore_industrial::192
python research/query_preparation/query_preparation_screen.py verify `
  --original data/work/query_preparation/industrial_low_recall_decomposition/strict_prompt_v2 `
  --rebuilt research/query_preparation/results/rebuild/decompose
streamlit run research/query_preparation/query_evidence_review_app.py
```

`--llm-mode replay` (default) reuses the saved LLM outputs and makes no API call.
`--llm-mode live` uses reconstructed prompts (the originals were lost); always pass
a new `--output-dir` and label it as a new prompt version.

## Guardrails

- Qrels are used only to select subsets and evaluate. They are never sent to an LLM.
- Subsets selected by benchmark recall (e.g. the 18 low-recall queries) are exploratory.
- Report candidate-pool recall, gold-file recall and post-rerank EVIE Recall@10
  separately. A gain in the candidate pool is not a gain in final ranking or QA.
