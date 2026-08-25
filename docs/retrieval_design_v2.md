# Retrieval design v2 — holistic, adversarial

Written after five refutations. The point of this document is to stop proposing
things the data has already killed, and to be explicit about what would have to
be true for each surviving option to be worth its cost.

All numbers are ViDoRe V3, KDL parse (the one we ship), NDCG@10, paired
permutation over 10,000 resamples.

---

## 1. The measured case taxonomy

**By failure shape** (physics, α=0.7 pool):

| case | share | what is wrong |
|---|---|---|
| A | 7.0% | gold file absent from top-10 |
| B | **68.9%** | gold file present, wrong pages within it |
| C | 24.2% | all gold already in top-10 — nothing to fix |

**By corpus shape** (this is the axis that decides whether structure helps):

| subset | pages/file | gold as % of its file | SEP |
|---|---|---|---|
| physics | 39.9 | 14.6% | **+1.73** |
| pharmaceuticals | 44.5 | 12.0% | **+1.41** |
| hr | 79.3 | 9.5% | +0.5 n.s. |
| industrial | 194.2 | 5.2% | n.s. |
| finance_en | 490.3 | 1.1% | untested, predicted fail |
| computer_science | 680.0 | 0.8% | n.s. |

**By ceiling.** Oracle reorder of the served top-100: top-10 → 60.49, top-20 →
71.68, top-50 → 83.88. Recall is not the constraint (`any-gold@100 = 98.3%`,
R@100 83.5 on KDL). **Ordering is the entire problem.**

---

## 2. What has been eliminated, with numbers

| idea | direction | result | why it failed |
|---|---|---|---|
| DCW — remove document centroid | localise | −5.73 / −9.50 | doc-topic component is *signal*; queries are topical |
| sub-page units + MaxSim | localise | −1.18 to −11.16 | ColEmbed's MaxSim needs a model *trained* for it |
| type-aware `boost_fields` | localise | no basis | no block type enriched in gold (ratios 0.80–1.03) |
| PRF / Rocchio | aggregate | −0.1 to −1.8 | improves R@100 (+0.6) but worsens ordering |
| French analyzer | lexical | −0.13 on fusion | +2.71 on BM25 alone, does not survive fusion |
| chunking variants | localise | −0.39 to −2.36 | fragmentation, same mechanism as MaxSim |
| SEP on large-doc corpora | aggregate | n.s. | file node too coarse at 194+ pages/file |

**Governing principle, earned rather than assumed:** on this benchmark
*localising interventions lose and aggregating ones win*. 962 of 1,674 physics
pages (57%) are gold for some query; 7.21 gold pages per query, dispersed not
contiguous. These are topical queries over small topical corpora — evidence is
diffuse. Methods built for needle-in-a-haystack extraction lose here.

The single exception is the cross-encoder, which localises and wins (+5.08). The
distinction is that it is **trained** for query–document interaction rather than
assembled at inference from a pooled embedder. That is the line: *learned*
localisation works, *arithmetic* localisation does not.

PRF is the informative near-miss — it is aggregating and still lost, because it
helps the axis we are not short on (recall) and hurts the one we are (ordering).
So the principle needs the sharper form: **aggregate evidence for ordering, not
for candidate generation.**

---

## 3. The design

Three tiers, ordered by evidence per unit cost. Nothing here is proposed without
a measured basis or an explicit falsification test.

### Tier 1 — free, ship now

**SEP, gated by the precondition screen.** +1.73 physics, +1.41 pharmaceuticals
on KDL. Zero marginal cost, parse-agnostic, reorders the pool only so recall is
unchanged by construction. Run the screen (`pages/file`, `gold % of file`) on any
new corpus and enable SEP only above ~10%.

*Open, must be measured before claiming it stacks:* does SEP compose with the
reranker, or do they fix the same queries? SEP changes which 20 pages reach the
reranker, and cross-file candidates are gold 0.46% of the time vs 7.83% same-file
— so SEP should hand the reranker a denser candidate set at identical API cost.
**This is the single highest-value untested experiment and it is nearly free.**

### Tier 2 — paid, known magnitude

**Rerank depth.** +5.08 at depth 20 is the largest measured single gain, and it
is capped by the free Voyage tier, not by engineering. Depth-20 oracle ceiling is
71.68; depth-50 is 83.88. This is a billing decision.

*Adversarial caveat:* physics is where reranking helps least — the paper's
zerank-2 gets +4.6 here against a +13.2 ten-dataset average. Do not extrapolate
+5 to other subsets in either direction without measuring.

*Do not* pursue best-window truncation selection (the earlier Part C plan). It is
a localising intervention and the granularity result predicts it loses. If
truncation is the issue, the fix consistent with the evidence is a
longer-context reranker, not a cleverer 1,200-char window.

### Tier 3 — the actual ceiling

**A visual arm.** Text-only caps this work. Visual retrievers run 43.2–48.5 on
physics standalone; ColEmbed reaches 91.0 / 63.5 on ViDoRe V1 / V2. No reranker
recovers what the parse discarded.

The affordable form is ColEmbed's own Table 6: a **single-vector visual
bi-encoder + reranker** reaches 0.9064 vs 0.9106 for full late interaction, at
**3.8 GB vs 10,311 GB per 1M pages**. Single-vector needs no index-contract
change — `LocalIndex.vectors` stays 2-D. Costs: page images are not downloaded
(442 MB–2.2 GB per subset) and the encoder needs the Colab GPU path.

---

## 4. Adversary: attack the whole programme

**"Even if all of this lands, is it worth it?"** Partly not, and this must be
said. The end-to-end QA budget is capped at roughly **−3.6pp credited** by the
oracle-vs-retrieved gap. Stacking SEP (+1.7) and rerank (+5) moves NDCG@10 from
43.03 to ~50 but cannot move QA accuracy by more than a few points, because gold
is redundant (7.21 pages available, ~2 needed). **The real QA ceiling is the
generator** — our oracle run scores 55.6% with gpt-4o-mini against the paper's
71.2% with Gemini 3 Pro. A 15-point discount no retrieval work closes.

*Implication:* if the goal is the end-to-end number in the CSV, upgrading the
generator is worth more than everything in Tier 1 and 2 combined, and is a
one-line change.

**"SEP works on 2 of 6 subsets."** True, and it is oversold if described as a
retrieval improvement. It is a small-document technique with a free applicability
test. On our two KDL-parsed subsets it applies; on the rest of ViDoRe it does not.

**"β=0.75 was selected on physics."** True. The honest bracket is +1 to +2, and
the pharmaceuticals transfer (+1.41, config untouched) is the only part of the
claim not exposed to that.

**"Large-document corpora have no method at all."** Correct, and unresolved. The
untested idea is a *section*-level node from KDL's 1,130 `SectionHeader` blocks
— an intermediate tier between file and page, which is precisely what
DISRetrieval contributes and what SEP's file→page tree lacks. It could not be
tested because **industrial and computer_science have no KDL parse**; only
physics and pharmaceuticals are parsed, and both are already small-document. Note
sections failed as *retrieval units* (§9), but that does not test them as
*aggregation nodes* — a different role, still open.

**"The text pipeline may be the wrong bet."** Possibly. Tier 3 exists for that
reason, and the honest sequencing is to establish whether the visual arm clears
the text arm on one subset before investing further in text-side ordering.

---

## 5. What would change the design

- SEP + rerank turn out to fix the *same* queries → SEP's production value drops
  to near zero and Tier 1 is not worth shipping.
- Section-level nodes rescue a large-document corpus → SEP generalises and the
  screen's threshold moves.
- A visual arm beats the text arm on one subset → Tiers 1–2 become a footnote.
- Generator upgrade closes most of the QA gap → retrieval work is deprioritised
  in favour of generation.
