# Hierarchical retrieval survey for AXIOM_DE-RD

This note scopes the Physics experiment around a strict, explainable cascade:

```text
file -> page -> block/paragraph -> sentence or structural evidence atom
```

The first implementation uses existing KDL + pdf-inspector output and cached
V-SPLADE page vectors. It does not add OCR, VLM, LLM summarisation or online
query-time model calls.

## Literature matrix

| Direction | Core idea | Useful transfer to this repository | Cost/risk | Status |
|---|---|---|---|---|
| [DHR](https://aclanthology.org/2021.findings-emnlp.19/) | Retrieve documents first, retrieve passages inside them, then calibrate passage scores with document relevance. | Parent file score as a prior for page ranking; test hard scope versus soft parent prior. | Requires careful file-stage recall budget. | v1 |
| [FunnelRAG](https://aclanthology.org/2025.findings-naacl.165/) | Progressively reduce candidate count and increase retriever capacity from coarse to fine. | Use BM25/V-SPLADE for file/page and fine BM25 only inside candidate pages. | A missed parent cannot be recovered by a child stage. | v1 |
| [Dense X Retrieval](https://aclanthology.org/2024.emnlp-main.845/) | Compare document, passage, sentence and proposition retrieval units; fine units can concentrate relevant evidence. | Add paragraph, sentence-group and atomic structural units while keeping page parent IDs. | Benchmark qrels are page-level, so sentence scores need a page-backed proxy. | v1 |
| [SPLADE](https://arxiv.org/abs/2107.05720) | Learned sparse expansion keeps inverted-index efficiency while reducing vocabulary mismatch. | Use the existing lexical BM25 and V-SPLADE as separate fields/signals. | Sparse activation weights must be preserved; token IDs are not ordinary terms. | existing |
| [ColBERTv2](https://arxiv.org/abs/2112.01488) + [PLAID](https://arxiv.org/abs/2205.09707) | Token-level late interaction with efficient pruning and compressed representations. | Candidate-only fine reranker after top-50/top-100 pages if v1 misses the target. | New model/index and larger serving footprint. | fallback |
| [V-SPLADE](https://arxiv.org/abs/2605.30917) | Inference-free visual sparse representation for visual-document retrieval. | Page-level visual prior and file pooling without new rendering or inference. | Current Physics comparison uses English query vectors against French qrels. | existing |
| [HiKEY](https://aclanthology.org/2026.acl-long.818/) | Makes document hierarchy a first-class multimodal signal and routes from global to fine-grained evidence. | Structural parent-child IDs, section-aware evidence and multimodal score fusion. | Full document hierarchical parsing is beyond light v1. | reference |
| [RAPTOR](https://arxiv.org/abs/2401.18059) | Builds recursive embedding/cluster/summary trees for multi-level retrieval. | Possible future long-document abstraction layer. | Summary generation and embedding are not light preparation. | deferred |

## Testable hypotheses

1. File routing reduces page-level distractors only when `Kf` is large enough to
   preserve nearly all gold files; test `Kf=3,5,10` and report candidate recall.
2. A page score combining PDF-inspector BM25, V-SPLADE and parent file prior is
   stronger than applying file aggregation as the final page ranking.
3. Paragraph/sentence-group retrieval can improve page recall by concentrating
   lexical evidence, while structural atoms protect formulas, tables and figures.
4. Neighbor-page expansion should improve multi-page evidence coverage, but it
   must be evaluated as candidate/context expansion rather than blindly adding
   adjacent pages to the top rank.
5. If candidate union coverage is high but final recall remains low, the next
   bottleneck is fine ranking, not more corpus preparation.

## Evaluation guardrails

- Keep French Physics qids `physics::0` through `physics::301` and the existing
  page unit unchanged.
- Keep page qrels unchanged. Sentence results are reported as page-backed
  proxies unless a reliable block/bounding-box mapping exists.
- Never use evidence modality labels as a router feature.
- Treat the English V-SPLADE query cache as a documented confound, not as a
  fair multilingual comparison.
- Report candidate coverage, page/file recall, latency and index size together.
