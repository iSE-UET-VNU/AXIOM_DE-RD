# ViDoRe V3 — Physics — revised results table

All retrieval numbers re-measured **2026-09-04** on the corrected parse
(**pdf-inspector + KDL**, run `32c32a45a92c45bb`), graded **NDCG@10** (the MTEB
ViDoRe V3 leaderboard metric — verified: our ColVec-8b visual-only reproduces
webAI's published 51.50 to within 0.13, ledger §26). French, 302 queries,
1,674-page physics corpus. Paired permutation tests vs baseline, 10k resamples.

Reproduce every row:
`python research/experiments/physics_kdl_slate.py --parse pdf-inspector --visual-dir data/work/vidore_physics_colvec --visual-name colvec --wv 0.8`

---

## Retrieval + latency table

| Method | NDCG@10 | Recall@10 | Correct_only | Correct+partial | Parsing | Chunk&Embed&Index | Retrieval (offline) | Online / query | Total runtime |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **Baseline Legacy** | **43.45** | 46.36 | 50.17 | 91.03 | 46m 39s | 37m 13s | 27s | — | 84m 19s |
| + Structural Rescoring (SEP) | 45.52 | 48.24 | 51.83 | 89.70 | — | — | +0.3 ms/q (≈27s) | — | 84m 19s |
| + Nemotron Rerank VL (free cross-encoder, top-20) | 47.42 | 48.95 | *not measured* | *not measured* | — | — | ~0.7 s/q API (~3.5 min / 302) | +0.7 s | 88m |
| + ColQwen2-2B visual fusion (w=0.7) | 47.00 | 49.50 | *not measured* | *not measured* | — | — | 15m 17s GPU index (1×) | — | 99m 36s |
| + SEP + ColQwen2-2B (w=0.7) | 47.73 | 50.71 | *not measured* | *not measured* | — | — | 15m 17s (1×) | — | 99m 36s |
| + **webAI-ColVec1.1-8b** visual fusion (w=0.8) | 51.73 | 56.16 | *not measured* | *not measured* | — | — | ~35 min GPU index (1×)† | — | ~2h |
| **+ SEP + webAI-ColVec1.1-8b (w=0.8)** | **53.15** | **55.91** | *not measured* | *not measured* | — | — | ~35 min (1×)† | — | ~2h |
| + Voyage rerank-2.5 (top-20) | *not measured on this pool* ‡ | | | | — | — | ~100 min API (free tier) | +20 s | — |

**Δ NDCG@10 vs Baseline** (all p < 0.005 except where noted): SEP +2.08 · Nemotron
+3.98 · ColQwen2 fusion +3.55 · SEP+ColQwen2 +4.28 · ColVec fusion +8.28 ·
**SEP+ColVec +9.70**.

† ColVec-8b visual index time not logged on the original Colab run. The estimate
scales ColQwen2-2B's 15m 17s by model size on comparable hardware; replace with
the real figure from the notebook's timing cell (`webAI_ColVec_visual_arm_physics_kaggle.ipynb`, cell 5).

‡ Voyage was measured only on the earlier `vidore_page` pool (baseline 44.15 →
49.23, +5.08). Not re-run on the pdf-inspector pool: a trained cross-encoder
*degrades* a strong visual-fused ranking (ColVec 51.73 → Nemotron 48.31), so it
is low priority. The CSV's current `48.35 / 50.63` for this row is a copy-paste
of the ColQwen2+SEP row and should be cleared.

### What moved from the old CSV numbers

| row | old CSV | revised | why |
|---|--:|--:|---|
| Baseline Legacy | 44.2 / 47.47 | 43.45 / 46.36 | corrected parse (pdf-inspector+KDL, not plain-KDL) + graded NDCG; ~0.75 lower |
| + Structural Rescoring | 46.27 / 48.88 | 45.52 / 48.24 | same |
| + ColQwen2 fusion | 47.47 / 49.77 | 47.00 / 49.50 | same |
| + ColQwen2 + Structural | 48.35 / 50.63 | 47.73 / 50.71 | same |

The shifts are all ≈ 0.5–0.8 and every conclusion is unchanged. The new rows
(Nemotron, ColVec, SEP+ColVec) are additions.

### Reference points (NDCG@10, MTEB ViDoRe V3, 6-language mean)

- webAI-ColVec1.1-8b, Physics: **51.50** — our ColVec-8b visual-only alone: 51.63 (French, full corpus). Match.
- nemotron-colembed-vl-8b-v2, Physics: 50.84
- Our SEP+ColVec **pipeline** (53.15) is modestly above these — but it is a pipeline
  (text pool + structural prior + visual fusion), French-only. "At the frontier
  with a small pipeline edge," not "past SOTA."

---

## Method descriptions

### Baseline Legacy

**Parse.** Each PDF goes through pdf-inspector (detects the layout regions on
every page — text blocks, tables, figures, captions) and KDL (reads the content
inside each region: running text, table structure, formulas). Output: clean
reading-order text per page for all 1,674 pages.

**Chunk.** Each page's text is split into windows of **512 words with 128-word
overlap** (`fixed_overlap`). One page produces one or more chunks; every chunk
records its source page.

**Embed & index.** Each chunk is embedded with **text-embedding-3-small**
(1,536-dim, via OpenRouter) and also added to a **BM25** keyword index
(`plain` analyzer). The query is embedded and analyzed the same way.

**Retrieve — two lists.** BM25 returns chunks ranked by keyword overlap; the
dense index returns chunks ranked by cosine similarity to the query vector.

**MaxP — chunks to pages.** Each page takes the score of its single
best-scoring chunk, collapsing both chunk lists into page rankings.

**α-fuse.** Each page ranking is min-max normalised over the fused candidate set,
then combined as

> **score(page) = 0.30 · BM25_norm(page) + 0.70 · dense_norm(page)**  (α = 0.7 on the dense side)

Keep the top **100** pages. This pool is what every arm below re-orders — recall
into the top 100 is ~98%, so the downstream problem is ordering, not coverage.

### + Structural Rescoring (SEP — Structural Evidence Propagation)

Post-hoc re-scoring of the α=0.7 pool. No model, no API, no GPU. For each
candidate page *c* in the pool:

> **s′(c) = 0.5 · s(c) + 0.5 · [ 0.75 · A_file(c) + 0.25 · N(c) ]**

- **s(c)** — the page's own fused score (min-max normalised within the query's pool).
- **A_file(c)** — the **top-3 mean** of the pool scores of pages belonging to the
  same document as *c* (a document that has several strong pages is itself likely
  relevant; mean-of-top-3 rather than sum, so long documents aren't rewarded for
  length).
- **N(c)** — distance-decayed evidence from *c*'s neighbours: for each page at
  distance *d* ∈ {−2, −1, +1, +2} in the same document that is also in the pool,
  add 0.5^|d| × its score (answers cluster — a page next to a strong page is
  itself more likely relevant).

Document and page numbers are read straight from the page id
(`physics::<file>#page=N`) — no parser or LLM. Recall is unchanged by
construction (it only re-orders the existing pool); cost is ≈ 0.3 ms/query.

### + Nemotron Rerank VL (free cross-encoder)

Take the **top 20** of the α=0.7 pool. A ~1.7B multimodal cross-encoder
(`nvidia/llama-nemotron-rerank-vl-1b-v2`, free via the OpenRouter `/rerank`
endpoint, no rate cap) reads **[query, page-text] as one joined sequence** and
outputs a single relevance score per pair — full cross-attention between query
and document, unlike the bi-encoder in the baseline. The 20 candidates are
re-ordered by that score; ranks 21–100 are left as they were. ~0.7 s/query,
≈ 3.5 min for all 302.

### + ColQwen2-2B / webAI-ColVec1.1-8b visual fusion

**Render.** Every page is rasterised to a **144-DPI image** straight from the PDF
(PyMuPDF) — no OCR, no parser involved.

**Encode.** A late-interaction vision-language retriever encodes each page image
into a **multi-vector** (one vector per image patch) and each query into a
multi-vector (one per token):

- **ColQwen2-v1.0** — Qwen2-VL-2B backbone, 128-dim token vectors.
- **webAI-ColVec1.1-8b** — Qwen3.5-VL ~9B backbone, fully bidirectional
  attention, 640-dim token vectors, up to 1,792 visual tokens/page. #1 on the
  ViDoRe V3 MTEB leaderboard.

**Score — MaxSim.** For a (query, page) pair:

> **MaxSim = Σ over query tokens  ( max over page patches  ⟨q_token, p_patch⟩ )**

i.e. each query token matches its single best-matching patch, and those matches
are summed. Embeddings are L2-normalised, so each term is a cosine.

**Fuse with text.** The visual MaxSim scores and the α=0.7 text-pool scores are
each min-max normalised per query, then combined as

> **score(page) = (1 − w) · text_norm(page)  +  w · visual_norm(page)**

with **w = 0.7 for ColQwen2**, **w = 0.8 for ColVec** (swept; the visual arm
carries most of the weight because it is the stronger signal here). Pages the
visual arm scores that were not in the text pool are added at *w* × their visual
score. Building the page index is a one-time GPU pass; at query time it is one
embedding lookup + a MaxSim, no image processing.

### + SEP + visual (the best free stack)

SEP is applied to the α=0.7 text pool first (Structural Rescoring above), and the
SEP-rescored scores are the `text_norm` term in the visual fusion. The two
levers address different failures — SEP fixes same-document / neighbour-page
ordering, the visual arm fixes cases the parsed text misses entirely (equations,
figures, tables) — so they compose: 43.45 → 45.52 (SEP) → **53.15** (+ ColVec).

### + Voyage rerank-2.5

Same top-20 cross-encoder rerank as Nemotron, using Voyage `rerank-2.5`
(proprietary; page text clipped to 1,200 characters by the free-tier token
budget). Measured only on the earlier `vidore_page` pool: +5.08 → 49.23. Not
re-run on the pdf-inspector pool — a trained cross-encoder degrades a strong
visual-fused ranking (§26), and the free tier caps depth at 20 and throughput at
~3 requests/min (~100 min for 302 queries).
