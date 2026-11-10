# Hierarchical light preparation + light retrieval

This bundle reproduces the hierarchical light retrieval configuration:

```text
PDF
  -> PDF-inspector page text
  -> selective Tesseract OCR for sparse/image pages
  -> file BM25 retrieval, top 3 files
  -> page BM25 retrieval inside those 3 files
  -> top 20 pages
```

The current implementation is BM25-only. It does not use Qwen embeddings,
Cohere, ColVec, KDL or qrel-guided ranking.

The default config uses PDF-inspector plus selective Tesseract. Set
`preparation.ppocr.enabled` to `true` and provide the uploaded
`light_ocr_results_comparison_btw_ppocr.zip` at the configured path if the
PP-OCRv5 preparation variant is required. PP-OCRv5 is an optional preparation
variant; it does not change the hierarchical retrieval formula.

## Configuration

`hierarchical_light_config.json` fixes:

```text
file_k              = 3
page_k              = 20
page_bm25_weight    = 0.70
file_direct_weight  = 0.50
parent_file_weight  = 0.15
```

The page score is:

```text
page_base  = 0.70 * normalized(page_BM25)
file_score = 0.50 * normalized(direct_file_BM25)
           + 0.50 * normalized(best_page_BM25_per_file)
page_final = 0.85 * page_base + 0.15 * normalized(file_score)
```

File selection uses `file_score`; page ranking is then performed only inside
the selected three files.

## Run

From the repository root:

```powershell
python run_hierarchical_pipeline.py --config hierarchical_light_config.json
```

To reuse a previously created preparation artifact explicitly:

```powershell
python run_hierarchical_pipeline.py --skip-prep
```

The first preparation run requires `pdf-inspector` and the Tesseract binary.
The default preparation setting is `fra+eng`, 200 DPI, PSM 3. OCR is applied
only to sparse, empty or image-heavy pages. The `workers` value controls the
Tesseract preparation pool; it does not change retrieval.

Outputs are written below:

```text
data/work/ondemand_v2/<bundle_fingerprint>/
  light_prep/pages_ocr_sparse.jsonl
  results/hierarchical_light_bm25/metrics.json
  results/hierarchical_light_bm25/run_manifest.json
```

The run manifest records the actual preparation artifact and all retrieval
weights so that a cached preparation cannot be silently mixed with another
retrieval protocol.
