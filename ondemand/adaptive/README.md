# On-Demand Adaptive Multimodal Retrieval & QA — Developer & Tool Guide

```
Raw data ──► Light Prep ──► Light Retrieval ──► SubData ──► Router ──► Branch ──► Chunk&Rank ──► QA Reader
             (PP-OCRv5)     (BM25 + Dense Qwen/TE3) (top-k)     (Rule/    ├─ 1. Light (CPU OCR)
                                                                 Fixed)   ├─ 2. Enrich (Chandra vLLM / OCR)
                                                                          └─ 3. Visual (ColVec cache [default] / ColPali live + KDL)
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
- **Endpoints**: `CHANDRA_ENDPOINT` / `VLLM_API_BASE`, `COLPALI_ENDPOINT`, etc.
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
# ColPali reranking endpoint for the visual branch.
# COLPALI_ENDPOINT=http://localhost:8001/v1

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
    chandra_endpoint="http://remote-gpu:8000/v1"  # optional override
)

result = pipe.query("What is the invoice amount for project X?")
print("Chosen Branch:", result["branch"])
print("Routing Reason:", result["route_notes"])
for chunk in result["chunks"][:3]:
    print(f"[{chunk['page_id']}] {chunk['text'][:80]}...")
```

### Direct Tool Calling: Chandra Enricher
```python
from ondemand.adaptive import ChandraEnricher

enricher = ChandraEnricher(endpoint="http://localhost:8000/v1")
# Automatically renders PDF page image via PyMuPDF and returns structured Markdown
parsed = enricher.parse(["ohrbench::file_ce1d4cef3196514b#page=18"])
print(parsed["ohrbench::file_ce1d4cef3196514b#page=18"])
```

### Direct Tool Calling: Rule Router
```python
from ondemand.adaptive import RuleRouter
from ondemand.adaptive.subdata import SubData

router = RuleRouter()
# SubData holds query, candidate pages, and gate scores
subdata = SubData(qid="q1", query="Show the breakdown table of costs", pages=["doc#page=0"], scores={"doc#page=0": 0.8}, k=20)
branch, notes = router.route(subdata)
print("Route:", branch, notes)
```

---

## 4. Benchmark Evaluation CLI

```bash
# Run rule router with PP-OCRv5 gate and exclude the 6 flawed benchmark reference questions:
python -m ondemand.adaptive run --tag adaptive_rule_v5 \
  --gate gate_ppocrv5_all.json --pages pages_ppocrv5.jsonl \
  --router rule --k adaptive --ranker page_order --exclude-flawed

# Full run with QA judging:
python -m ondemand.adaptive run --tag adaptive_full_qa \
  --gate gate_ppocrv5_all.json --pages pages_ppocrv5.jsonl \
  --router rule --k fixed:20 --ranker page_order --qa
```

### CLI Flags Reference

| Flag | Default | Description |
|---|---|---|
| `--tag` | required | Output experiment name under `data/work/<fp>/results/<tag>/` |
| `--gate` | `gate_ppocrv5_all.json` | Gate index built via `python -m ondemand light-retrieval` |
| `--pages` | `pages_ppocrv5.jsonl` | OCR corpus text file |
| `--router` | `rule` | Routing policy: `rule`, `fixed:light`, `fixed:visual`, `fixed:enrich` |
| `--k` | `fixed:20` | Candidate page budget: `fixed:N` or `adaptive` (dynamic score-gap) |
| `--ranker` | `hybrid` | Chunk ranker: `hybrid` (dense+BM25), `page_order`, or `lexical` |
| `--store` | `kdl` | Precomputed chunk vectors (e.g. `kdl`) to avoid embedding calls |
| `--exclude-flawed` | `off` | Filter out the 2 confirmed flawed benchmark ground truth questions ($N=218$) |
| `--chandra-endpoint` | `None` | Override Chandra OCR / layout vLLM endpoint |
| `--colpali-endpoint` | `None` | Live ColPali endpoint; only used if no `colvec_scores.npy` cache is found. The visual branch's **default scorer is the ColVec cache**, not a live endpoint. |
| `--qa` | `off` | Run end-to-end VLM generation and judging |
| `--allow-api` | `off` | Allow live OpenAI/OpenRouter embedding API calls if uncached |

---

## 5. Built-in Resilience & Graceful Fallbacks

The architecture is built so teammates will **never experience hard crashes** when running across different environments:
1. **Chandra Enricher**: If no remote endpoint is provided or reachable, automatically falls back to local PaddleOCR text with an informative log.
2. **Visual Branch**: scores pages with the cached `colvec_scores.npy` matrix by default — no live endpoint needed. If that cache is missing (or stale for the current benchmark bundle) and `--colpali-endpoint` is set, it calls that endpoint instead. If neither is available, it falls back to the plain dense gate scores. Separately, if `kdl_pages.jsonl` is missing, page text falls back to OCR text. Every fallback prints a loud warning to stderr (see `fallback.py`).
3. **Adaptive K**: Dynamically saves 15-20% page budget when the top candidate is clearly dominant, while preserving 100% recall.
