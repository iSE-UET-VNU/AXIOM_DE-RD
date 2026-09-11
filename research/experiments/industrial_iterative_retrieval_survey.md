# Industrial hierarchical and iterative retrieval survey

This note records the second research pass after the initial `file -> page`
cascade. The target protocol is ViDoRe V3 Industrial: 283 English queries,
5,244 pages and 27 files. The cached page-BM25 baseline is 47.71% page
recall@10; the earlier hard `Kf=10` cascade is 47.47%. All text retrieval
uses KDL + PDF-inspector extracted text. No raw PDF is indexed directly.

## Literature-to-repository mapping

| Direction | Main lesson | Adaptation here | Risk |
|---|---|---|---|
| [NPRF](https://aclanthology.org/D18-1478/) | Top-ranked documents can supply weighted expansion terms to reduce vocabulary mismatch. | Use high-IDF terms from first-pass pages/chunks, but keep the original BM25 score in the final rank. | Pseudo-relevance is unsafe when the first pass is noisy; expansion can drift. |
| [Unsupervised iterative evidence retrieval](https://aclanthology.org/2020.acl-main.414/) | Refine the next query around terms not covered by already retrieved justifications and stop when coverage is sufficient. | Test residual query terms and one additional page/chunk pass; use coverage only as a diagnostic, not as a gold-aware stop rule. | Physics/Industrial qrels are page-level and do not provide answer/justification terms for a reliable stopping rule. |
| [ColBERT-PRF](https://arxiv.org/abs/2106.11251) | Feedback representations should be representative and discriminative; IDF helps suppress common feedback. | Approximate with cached legacy chunk embeddings and small Rocchio-style/query-local feedback experiments. | Current cache has single-vector chunks, not token-level late interaction. |
| [Hybrid Hierarchical Retrieval](https://aclanthology.org/2023.findings-acl.679.pdf) | Sparse and dense signals can be used at both document and passage stages, while passage retrieval remains scoped by parent documents. | Keep file selection and page ranking separate; use legacy BM25+dense only inside a file candidate scope. | Hard parent filtering can irreversibly remove a relevant page. |
| [Natural Logic-guided autoregressive retrieval](https://aclanthology.org/2022.emnlp-main.411/) | Later retrieval can be conditioned on earlier evidence, with explicit stopping/consistency logic. | Use a deterministic protected candidate union as a light substitute for a learned controller. | No reliable answer-consistency signal is available in retrieval-only evaluation. |
| [Bridge phrase query expansion](https://aclanthology.org/2022.lrec-1.485/) | Bridge phrases can connect otherwise weakly related evidence chains. | Future option: extract section/entity bridges from selected pages, then run a second page query. | Entity/phrase extraction must not use qrels and may add parser noise. |

## What was tested

### 1. Unweighted/IDF PRF and residual retrieval

`industrial_iterative_retrieval.py` tested page-level IDF expansion, residual
terms, and a soft file prior. The variants were below the cached BM25 baseline
on Industrial. This is consistent with the PRF assumption being violated: the
first pages are not reliably pseudo-relevant, and their long extracted text
contains many unrelated technical terms.

### 2. Fine-grained second retrieval

`industrial_fine_grained_retrieval.py` tested block, paragraph and sentence
group pooling. Naive max/sum pooling also decreased page recall. The likely
failure mode is provenance noise: legacy chunks often cross page boundaries,
and a high-scoring fine unit is not necessarily the best page-level evidence.

### 3. Protected soft cascade

`industrial_soft_cascade_feedback.py` uses the strongest practical second pass:

```text
global page BM25
  -> top-10-file scope
  -> cached legacy chunk BM25 + dense retrieval
  -> max-pool chunks to pages
  -> interleave 3 legacy pages : 1 global-BM25 page
```

The interleaving is a cheap feedback/iteration mechanism. It does not let the
file stage erase the original global candidates, so it addresses the main
failure mode of a hard cascade without adding a model or rebuilding the cache.

## Current evidence

| Arm | Page recall@10 | File recall@3 | Interpretation |
|---|---:|---:|---|
| Cached page BM25 | 47.71% | 82.27% | Fixed baseline |
| File10 + legacy second, max, gamma=0.12 | 48.20% | 82.74% | Positive full-set sensitivity row |
| Protected soft cascade, 3:1 | 49.85% | 84.45% | Best full-set row; exploratory until pre-registered |
| OOF arm selection over tested variants | 48.67% | 83.98% | Positive but bootstrap CI still crosses zero |
| Restricted OOF: baseline vs protected 3:1 | 49.85% | 84.45% | 3:1 selected on all 5 training folds; CI is positive |

The 49.85% row has a paired bootstrap page-recall CI of approximately
`[+0.51, +3.87]` percentage points against the cached baseline. However, the
configuration was screened on the same benchmark. The OOF selection result is
therefore the more conservative result: +0.95 points, with CI approximately
`[-0.71, +2.58]`.

When the choice is restricted to only the baseline and the protected 3:1 arm,
the training-fold selector chooses the protected arm in all five folds and the
held-out aggregate is again 49.85% (+2.14 points; paired bootstrap CI
`[+0.51, +3.87]`, p=0.005). This is stronger evidence than the many-arm sweep,
but the 3:1 ratio was discovered during this benchmark campaign, so a fresh
query set is still required for a final generalisation claim.

## Next experiment order

1. Freeze `file10 + legacy max gamma=0.1 + protected 3:1` before any further
   tuning and evaluate it on a new query split or a second ViDoRe subset.
2. Add a two-arm query-local gate using only retrieval observables: overlap of
   the first and second top-10 pages, legacy score concentration and file-score
   margin. The gate should choose between protected 3:1 and protected 1:1,
   with thresholds fitted only on training folds.
3. Implement controlled bridge/section feedback: add only terms that co-occur
   with an original query anchor in the same block, rather than all terms from
   a feedback page.
4. If the candidate union is still high but ranking remains weak, test a
   candidate-only late-interaction reranker. Keep this separate from the
   light-preparation latency budget.

## Guardrails

- No qrel or modality label is used to choose pages/files at retrieval time.
- The file metric remains the repository's page-derived `file_recall@3`, not an
  independent production file index metric.
- Full-set screening and OOF selection are reported separately.
- Legacy chunks crossing page boundaries are mapped to every touched page; this
  is useful for recall analysis but is a provenance caveat.
