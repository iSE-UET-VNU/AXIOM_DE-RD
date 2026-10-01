import json
from time import perf_counter

import numpy as np

from ...bench import BENCH, fingerprint, load_jsonl, work
from ..registry import build, register
from ..subdata import BranchOutput, Passage


class Scorer:
    name = "base"

    def score(self, qid, pages):
        raise NotImplementedError

    def describe(self):
        return self.name


@register("scorer", "colvec_cached")
class CachedColVec(Scorer):
    name = "colvec_cached"

    def __init__(self, argument, bench=BENCH, **_):
        folder = work("colvec", bench=bench)
        meta = folder / "colvec_meta.json"
        if not meta.exists():
            raise SystemExit(f"{folder} has no cached ColVec scores; run the ColVec notebook for this bundle")
        if json.loads(meta.read_text())["bundle_fingerprint"] != fingerprint(bench):
            raise SystemExit("cached ColVec scores are from a different bundle")
        self.matrix = np.load(folder / "colvec_scores.npy")
        self.col = {k: i for i, k in enumerate(json.loads((folder / "colvec_keys.json").read_text()))}
        self.row = {q: i for i, q in enumerate(json.loads((folder / "colvec_qids.json").read_text()))}

    def score(self, qid, pages):
        if qid not in self.row:
            raise SystemExit(f"{qid} is not in the cached ColVec matrix; it covers {len(self.row)} queries")
        missing = [p for p in pages if p not in self.col]
        if missing:
            raise SystemExit(f"{len(missing)} pages are not in the cached ColVec matrix, e.g. {missing[:3]}")
        row = self.matrix[self.row[qid]]
        return {p: float(row[self.col[p]]) for p in pages}


class Parser:
    name = "base"

    def parse(self, page_ids):
        raise NotImplementedError

    def describe(self):
        return self.name


@register("parser", "kdl_cached")
class CachedKDL(Parser):
    name = "kdl_cached"

    def __init__(self, argument, bench=BENCH, **_):
        source = work("kdl", bench=bench) / (argument or "kdl_pages.jsonl")
        if not source.exists():
            raise SystemExit(f"{source} is missing; run the KDL notebook for this bundle")
        self.source = source
        self.text = {r["page_id"]: r["text"] for r in load_jsonl(source)}

    def parse(self, page_ids):
        out, missing, empty = {}, [], []
        for page in page_ids:
            text = self.text.get(page)
            if text is None:
                missing.append(page)
                text = ""
            elif not text.strip():
                empty.append(page)
            out[page] = text
        return out, {"missing_pages": missing, "empty_pages": empty}

    def describe(self):
        return f"kdl_cached:{self.source.name}"


@register("branch", "visual")
class VisualBranch:
    name = "visual"

    def __init__(self, argument, scorer="colvec_cached", parser="kdl_cached", refine_k="fixed:10", **context):
        self.scorer = build("scorer", argument or scorer, **context)
        self.parser = build("parser", parser, **context)
        self.refine_k = build("k", refine_k, **context)

    def run(self, subdata):
        started = perf_counter()
        candidates = subdata.top()
        scores = self.scorer.score(subdata.qid, candidates)
        ranked = sorted(candidates, key=lambda p: (-scores[p], p))
        refined = subdata.__class__(subdata.qid, subdata.query, ranked, scores, len(ranked), source="visual")
        keep = ranked[:self.refine_k.choose(refined)]
        rerank_seconds = perf_counter() - started

        started = perf_counter()
        text, notes = self.parser.parse(keep)
        passages = [Passage(page, text[page], rank) for rank, page in enumerate(keep, 1)]
        return BranchOutput(self.name, passages,
                            cost={"seconds": rerank_seconds + (perf_counter() - started),
                                  "rerank_seconds": rerank_seconds, "pages": len(candidates),
                                  "parsed_pages": len(keep)},
                            notes={"scorer": self.scorer.describe(), "parser": self.parser.describe(),
                                   "refine_k": self.refine_k.describe(),
                                   "scores": {p: round(scores[p], 6) for p in keep}, **notes})

    def describe(self):
        return f"visual:{self.scorer.describe()}+{self.parser.describe()}"
