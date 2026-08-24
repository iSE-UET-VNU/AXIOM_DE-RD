# corpus-service keyword search: analyzer audit

**Audience:** Platform (corpus-service), AXIOM_DE-RD.
**Why:** `corpus_service_hybrid` is the do-nothing baseline for our retrieval
comparison. If its lexical leg mis-tokenizes, every arm we measure looks good
against a broken reference and the improvements we report are partly artifact.

**Verdict:** one **high-severity** defect — queries in NFD lose **54.2% of their
terms** (measured over 3,115 terms in 111 real questions), affecting every script
with precomposed characters, not just Vietnamese — plus two low-severity
mismatches. CJK is equally limited on both sides, so it is a shared ceiling
rather than an asymmetry.

---

## The two sides

Keyword search compares query terms against `document_embeddings.lexical.tf`.
Those two sides are produced by **different code owned by different teams**:

| Side | Owner | Function |
|---|---|---|
| Index — writes `tf` | AXIOM_DE-RD pipeline | `chunking_embedding/lexical.analyze` → NFC + casefold + `[^\W_]+` |
| Query — reads `tf` | corpus-service | `_keyword_tokens` → `re.findall(r"\b\w+\b", …)` + `.lower()` |

`corpus_retrieval_repository.py:687`. There is **no Unicode normalization
anywhere** in corpus-service or Methods-Hub — the only `normalize` functions
(`_normalize_content`) reshape JSON, not text. The query reaches the tokenizer
exactly as the caller typed it.

Any divergence between those two functions silently drops terms: BM25 scores a
term that is absent from every posting list, it contributes nothing, and the
endpoint returns `200` with a ranking built from whatever terms happened to
survive. No error, no log, no signal that half the query went missing.

---

## Finding 1 — NFD queries lose half their terms (HIGH)

```
stored  (NFC, our index):  ['đại', 'học', 'công', 'nghệ']
queried (NFD, unchanged):  ['đa', 'i', 'ho', 'c', 'co', 'ng', 'nghe']
overlap:                   ∅
```

**Measured on all 111 real evaluation questions**, comparing our stored NFC terms
against what corpus-service would tokenize from the same question in NFD:

| | |
|---|---|
| Query terms still matchable under NFD | **1,427 / 3,115 (45.8%)** |
| Term-level recall loss | **54.2%** |
| Questions losing ≥1 term | **62 / 111 (55.9%)** |
| Questions losing *all* terms | 0 / 111 |

An isolated Vietnamese phrase loses everything, but a real question also carries
digits, Latin words and unaccented syllables that survive, so BM25 degrades
severely rather than going silent. **This corrects an earlier, stronger claim of
"returns zero results"** — the accurate statement is that a majority of query
terms become unmatchable and scoring proceeds on the surviving minority.

**Mechanism.** In NFD, `đại` is `đ` + `a` + U+0323 + `i`, and the combining mark
has Unicode category `Mn`. Python's `\w` does not match `Mn`, so `\b\w+\b` breaks
the token at every diacritic. Verified:

```python
unicodedata.category('̣')      # 'Mn'
re.match(r"\w", '̣')           # None
re.findall(r"\b\w+\b", NFD("đại"))  # ['đa', 'i']
```

Our index side is immune because `analyze` NFC-normalizes first. The query side
does not, so **the two sides only agree when the caller happens to send NFC**.

**Why this is not hypothetical.** macOS stores filenames in NFD, and text copied
from Finder, some IMEs, and several PDF extractors carries NFD through. That is
simply how a Vietnamese user on a Mac produces a query. The failure is invisible:
no error, no log, HTTP 200, and a plausible-looking ranking built from a minority
of the query's terms.

**Fix (one line, corpus-service):**

```python
def _keyword_tokens(text: str, *, case_sensitive: bool) -> list[str]:
    text = unicodedata.normalize("NFC", text)          # <-- add
    tokens = re.findall(r"\b\w+\b", text, flags=re.UNICODE)
    ...
```

Ideally paired with `.casefold()` rather than `.lower()`, to match the index side.

---

## Finding 2 — underscore (LOW)

`\w` includes `_`; our `[^\W_]+` treats it as a separator.

```
"invoice_number 2026"   stored ['invoice','number','2026']   queried ['invoice_number','2026']
```

Identifier-like terms are unreachable. Affects all languages. Common in the
table-heavy portion of our corpus (column headers).

## Finding 3 — apostrophes (LOW)

```
"the company's report"  stored ['the',"company's",'report']  queried ['the','company','s','report']
```

`company's` is unreachable; a stray `s` token is queried instead.

## Finding 4 — CJK is a shared ceiling, not an asymmetry

Both sides collapse a Chinese phrase into one token, so they agree — but agree on
a token that only matches an exact full-phrase query.

```
"建筑消防设施故障维修记录表"   stored = queried = one token
```

This is not a corpus-service bug; it is a limitation both sides inherit. It is
also the gap our own index closes with index-and-query-time bigrams.

## Finding 5 — our own analyzer shatters already-decomposed scripts (OURS, MEDIUM)

Found while establishing the blast radius. **This one is on our side of the
boundary, not Platform's.** `\w` matches no combining mark at all — neither `Mn`
nor `Mc` — so any script whose NFC form already contains separate marks is
tokenized into single letters by *both* analyzers:

```
analyze("مَكْتَبَة الجامِعَة")  ->  ['م','ك','ت','ب','ة','الجام','ع','ة']
analyze("विद्यालय की जानकारी")  ->  ['व','द','य','लय','क','ज','नक','र']
analyze("סֵפֶר")                ->  ['ס','פ','ר']
analyze("เอกสารสำคัญ")          ->  ['เอกสารสำค','ญ']
```

Because both sides shatter identically they *agree*, so this is a shared ceiling
like CJK rather than a silent asymmetry — but the tokens are meaningless and BM25
over single letters is noise. The lake contains one Arabic document
(`صدام حسين.md`, evidence for q72), so this is live in our own evaluation set,
not hypothetical.

**Status: known limitation, deliberately not fixed.** The fix belongs in
`chunking_embedding/lexical.analyze` -- the token pattern would need to admit
combining marks (a `\p{L}\p{M}*` equivalent, via the `regex` package). That
rewrites every stored `tf` and forces a full re-index, which is not worth the
critical path for one document and one question. Recorded here so the analysis
exists if a reviewer asks.

**Affected evidence:** q72 (`صدام حسين.md`). Any result involving that question's
lexical leg carries this limitation.

---

## Vietnamese word segmentation: measured, not assumed

Vietnamese compounds span whitespace (`công nghệ thông tin` is one term across
four tokens), so syllable tokenization loses compound cohesion on both sides.
The open question was whether that costs recall. It does not, at n=49:

| Metric (BM25, `fixed_512_ol`) | plain | pyvi-segmented | Δ |
|---|---|---|---|
| recall@10 | 0.6122 | 0.6122 | **0.0000** |
| ndcg@10 | 0.5649 | 0.5267 | −0.0382 |
| mrr@10 | 0.5714 | 0.5151 | −0.0563 |

Segmentation buys nothing on recall and slightly degrades ranking. Plausible
reason: BM25 already rewards co-occurrence of all syllables of a compound, so
binding them into one term adds no discrimination while making the vocabulary
sparser.

**How far this can be pushed.** At n=49 the minimum detectable effect is ~15pp,
so the recall Δ of exactly 0.0000 is a strong null, while the ndcg Δ of −0.038
and mrr Δ of −0.056 sit well inside the noise and are **directional at best**.
Report the null; do not report the degradation as an effect.

**Only the BM25 row is a clean comparison.** The archived segmented run used
`prefix: False` while the plain run used `prefix: True`, and that prefix is a
dense-side setting — so the dense/rrf/alpha deltas in those files confound the
analyzer with the embedder prefix and must not be attributed to segmentation.

**Conclusion:** compound cohesion is not the Vietnamese problem. **Normalization
is** (Finding 1).

---

## Blast radius: this is not a Vietnamese quirk

`\w` matches **no** combining mark — verified for `Mn` (Vietnamese dot-below,
acute, Arabic fatha, Thai sara-i) and `Mc` (Devanagari matra). So `\b\w+\b`
breaks at every mark in every script. Which failure that produces depends on
whether the script has *precomposed* forms:

| Script | NFC == NFD | Failure mode |
|---|---|---|
| Vietnamese, Spanish, German, Korean | **No** | **Asymmetry** — index NFC, query NFD, terms unmatchable (Finding 1) |
| Arabic, Hebrew, Devanagari, Thai | **Yes** | **Shared shattering** — both sides tokenize to single letters (Finding 5) |
| English, CJK | Yes | Unaffected / shared ceiling |

Measured term loss under NFD on one sample each: Vietnamese 100%, Spanish 100%,
German 100%, Korean 100%.

The generalization worth stating: **any lexical retrieval system that tokenizes
with `\w` and normalizes on only one side of the index boundary silently loses
most of a language.** It applies to most of Latin-script Southeast Asia and
Europe for the asymmetry mode, and to Indic, Arabic, Hebrew and Thai for the
shattering mode.

## Why neither test suite could see it

Both sides are internally consistent, and the invariant only exists across the
boundary:

- **Our tests pass** because we normalize on *both* sides — index and query both
  go through `analyze`, so round-trip tests find their own terms.
- **Their tests pass** because they normalize on *neither* side — an English
  fixture tokenizes identically with or without normalization.

The bug lives in the **composition**, which is exactly the surface no unit test
on either side covers. This is an argument for contract tests at service
boundaries that assert an end-to-end known-true fact — "index this Vietnamese
document, query a term from it, get it back" — rather than for more unit tests on
either side.

The provenance chain is ordinary, not exotic: macOS stores filenames in NFD, and
Finder copies, several IMEs, and some PDF extractors carry it through. That is
simply how a Vietnamese user on a Mac produces a query.

## Impact on our benchmark

Until Finding 1 is fixed, `corpus_service_hybrid` is an **artificially weak
baseline** for any Vietnamese query that arrives in NFD. Every table reporting it
carries an asterisk. Two mitigations, both applied:

1. Our sweep sends NFC-normalized queries to every arm, so the baseline is
   measured at its best rather than at its worst. The defect is reported here
   rather than harvested as a free win.
2. A separate NFD arm quantifies the production-realistic case, reported as a
   robustness result rather than folded into the headline comparison.

---

## Asks for Platform

1. **NFC-normalize the query** in `_keyword_tokens` (Finding 1). One line, and
   worth doing before our sweep rather than after — every day it is unfixed is
   production Vietnamese queries losing half their terms.

   **Please also confirm which normalization form the ingestion path stores.**
   Our pipeline emits NFC, but if anything else writes `document_contents` or
   `lexical.tf` without normalizing, the fix has to normalize *both* directions
   rather than only queries — and normalizing queries alone would then move the
   mismatch instead of removing it.
2. Consider `casefold()` over `lower()`, and aligning the token pattern with the
   index side (`[^\W_]+`) for Findings 2 and 3.
3. Longer term the analyzer should be **declared once and shared**, not
   implemented twice. The index side already stamps
   `lexical.analyzer = "unicode_word_v1"` into every row; corpus-service could
   read that field and refuse to score rows whose analyzer it does not implement.
   That converts this whole class of bug from silent-zero into a startup error.
