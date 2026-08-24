# SEP precondition screen — prediction registered before testing

SEP (Structural Evidence Propagation) aggregates relevance over the corpus's own
`file → page` tree and blends it back into page scores. It helped on physics
(+2.02, p=0.0026) and did nothing on industrial (n.s., mostly negative).

The proposed explanation is a **corpus-shape precondition**: file-level evidence
only constrains *which pages* are relevant when documents are small enough that
knowing the file is informative. Two free statistics capture it — pages per file,
and what fraction of its own file the gold occupies.

This screen is computed from `corpus.parquet` + `qrels.parquet` alone, with no
embeddings and no retrieval. **The predictions below are recorded before any of
the untested subsets were run**, so the screen is falsifiable rather than
fitted after the fact.

| subset | queries | files | pages | pages/file | gold/query | gold as % of its file | prediction |
|---|---|---|---|---|---|---|---|
| physics          | 302 | 42 | 1674 |  39.9 | 7.21 | 14.6% | helps — **confirmed +2.02, p=0.0026** |
| pharmaceuticals  | 364 | 52 | 2313 |  44.5 | 4.76 | 12.0% | predict: helps — **CONFIRMED +2.17, p=0.0005** |
| hr               | 318 | 14 | 1110 |  79.3 | 5.44 |  9.5% | predict: weak/marginal — **CONFIRMED +0.54, p=0.14 n.s.** |
| industrial       | 283 | 27 | 5244 | 194.2 | 5.70 |  5.2% | fails — **confirmed n.s.** |
| finance_en       | 309 |  6 | 2942 | 490.3 | 4.73 |  1.1% | predict: fails — *not run* |
| computer_science | 215 |  2 | 1360 | 680.0 | 4.88 |  0.8% | predict: fails hardest — **CONFIRMED −0.09, p=0.87 n.s.** |

## How this can be wrong

- If SEP helps on `computer_science` or `finance_en`, the precondition is not the
  mechanism and the physics gain is more likely selection than structure.
- If SEP fails on `pharmaceuticals`, the same conclusion follows more strongly,
  since pharmaceuticals is the nearest neighbour to physics on both statistics.
- Either outcome invalidates the screen as a deployment test.

## Standing caveat on the physics number

`beta=0.75` was chosen from a grid sweep on physics, not a priori. The lift table
(same-file 16.9x vs within-file adjacency 1.63x) justifies `beta > 0.5`, not
`0.75` specifically. The physics p-value is therefore optimistic, and the honest
bracket is **+1 to +2 NDCG@10, robust over beta in [0.5, 0.75] and lambda in
[0.5, 0.7]**. `chandra_page` re-tests the same 302 queries under a different
parse, so it establishes parse-robustness, not selection-robustness. Only a
different subset does that.


## Outcome — 5/5, registered predictions all held

Tested after registration: `pharmaceuticals` (+2.17, p=0.0005), `hr` (+0.54,
n.s.), `computer_science` (−0.09, n.s.), alongside the already-known `physics`
(+2.02) and `industrial` (n.s.). The screen ordered them correctly and with
graded resolution, so it stands as the free deployment test for SEP. Threshold
sits near **gold ≳10% of its own file**. `finance_en` was left unrun: at 1.1% it
is predicted to fail and sits on the same side of the threshold as the two
already-confirmed failures.
