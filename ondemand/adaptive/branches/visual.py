import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import requests

from ...bench import BENCH, fingerprint, load_jsonl, work
from .. import fallback
from ..config import get_config
from ..registry import build, register
from ..subdata import BranchOutput, Passage


class Scorer:
    name = "base"

    def score(self, qid, pages, subdata=None):
        raise NotImplementedError

    def describe(self):
        return self.name


@register("scorer", "colvec_endpoint")
@register("scorer", "colpali")
class EndpointColVec(Scorer):
    name = "colpali_endpoint"

    def __init__(self, argument=None, bench=BENCH, **context):
        cfg = get_config()
        self.endpoint = (argument if argument and (argument.startswith("http://") or argument.startswith("https://"))
                         else context.get("colpali_endpoint") or cfg.colpali_endpoint)
        self.api_key = context.get("colpali_api_key") or cfg.colpali_api_key
        self.bench = bench
        self.cache_dir = work("colvec", "cache", bench=bench)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def score(self, qid, pages, subdata=None):
        query = subdata.query if subdata else ""
        cache_key = hashlib.sha256(f"{qid}:{','.join(pages)}".encode("utf-8")).hexdigest()
        cache_file = self.cache_dir / f"{cache_key}.json"

        if cache_file.exists():
            return json.loads(cache_file.read_text(encoding="utf-8"))

        if self.endpoint:
            url = self.endpoint.rstrip("/")
            if not url.endswith("/rerank"):
                url = f"{url}/rerank"
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            payload = {"query": query, "documents": pages, "qid": qid}
            try:
                res = requests.post(url, json=payload, headers=headers, timeout=60)
                res.raise_for_status()
                scores = res.json().get("scores", {})
                cache_file.write_text(json.dumps(scores), encoding="utf-8")
                return scores
            except Exception as err:
                fallback.warn("colpali_endpoint", f"the endpoint call failed ({err}); ranking by the light-retrieval "
                                                  f"scores instead of ColPali")
        else:
            fallback.warn("colpali_endpoint", "no ColPali endpoint is configured; ranking by the light-retrieval "
                                              "scores instead of ColPali")

        return {p: (subdata.scores.get(p, 0.0) if subdata else 0.0) for p in pages}

    def describe(self):
        return f"colpali_endpoint:{self.endpoint or 'fallback'}"


@register("scorer", "colvec_cached")
class CachedColVec(Scorer):
    name = "colvec_cached"

    def __init__(self, argument=None, bench=BENCH, **context):
        folder = work("colvec", bench=bench)
        meta = folder / "colvec_meta.json"
        scores_file = folder / "colvec_scores.npy"
        self.has_cache = scores_file.exists() and meta.exists()

        if self.has_cache:
            if json.loads(meta.read_text())["bundle_fingerprint"] != fingerprint(bench):
                fallback.warn("colvec_scores", "the cached ColVec matrix is from a different benchmark bundle, so "
                                               "it is being ignored")
                self.has_cache = False
            else:
                self.matrix = np.load(scores_file)
                self.col = {k: i for i, k in enumerate(json.loads((folder / "colvec_keys.json").read_text()))}
                self.row = {q: i for i, q in enumerate(json.loads((folder / "colvec_qids.json").read_text()))}

        self.endpoint_scorer = None
        cfg = get_config()
        if not self.has_cache and (cfg.colpali_endpoint or (argument and argument.startswith("http"))):
            self.endpoint_scorer = EndpointColVec(argument, bench=bench, **context)

    def score(self, qid, pages, subdata=None):
        if self.has_cache and qid in self.row:
            row = self.matrix[self.row[qid]]
            # If all pages in col, use cached matrix
            if all(p in self.col for p in pages):
                return {p: float(row[self.col[p]]) for p in pages}

        if self.endpoint_scorer:
            return self.endpoint_scorer.score(qid, pages, subdata=subdata)

        why = "no cached ColVec matrix" if not self.has_cache else f"{qid} is not in the cached matrix, or some of its pages are"
        fallback.warn("colvec_scores", f"{why}; ranking by the light-retrieval scores instead of ColVec")
        return {p: (subdata.scores.get(p, 0.0) if subdata else 0.0) for p in pages}

    def describe(self):
        if self.has_cache:
            return "colvec_cached"
        if self.endpoint_scorer:
            return self.endpoint_scorer.describe()
        return "colvec_cached:fallback_dense"


class Parser:
    name = "base"

    def parse(self, page_ids):
        raise NotImplementedError

    def describe(self):
        return self.name


@register("parser", "kdl_cached")
class CachedKDL(Parser):
    name = "kdl_cached"

    def __init__(self, argument=None, bench=BENCH, pages_name="pages_ppocrv5.jsonl", **_):
        source = work("kdl", bench=bench) / (argument or "kdl_pages.jsonl")
        self.has_kdl = source.exists()
        if self.has_kdl:
            self.source = source
            self.text = {r["page_id"]: r["text"] for r in load_jsonl(source)}
        else:
            # Fallback to OCR text
            lp_source = work("light_prep", bench=bench) / pages_name
            if not lp_source.exists():
                lp_source = work("light_prep", bench=bench) / "pages_ocr_ppocr.jsonl"
            self.source = lp_source
            self.text = {r["page_id"]: r["text"] for r in load_jsonl(lp_source)} if lp_source.exists() else {}

    def parse(self, page_ids):
        if not self.has_kdl:
            fallback.warn("kdl_pages", f"kdl_pages.jsonl is missing, so pages are read from {self.source.name} "
                                       f"(OCR text, not a KDL parse)")

        out, missing, empty = {}, [], []
        for page in page_ids:
            text = self.text.get(page)
            if text is None:
                missing.append(page)
                fallback.warn("kdl_page_missing", f"{page} is not in {self.source.name}; its text is empty")
                text = ""
            elif not text.strip():
                empty.append(page)
            out[page] = text
        return out, {"missing_pages": missing, "empty_pages": empty}

    def describe(self):
        prefix = "kdl_cached" if self.has_kdl else "kdl_cached:fallback_ocr"
        return f"{prefix}:{self.source.name if self.source else 'none'}"


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
        scores = self.scorer.score(subdata.qid, candidates, subdata=subdata)
        ranked = sorted(candidates, key=lambda p: (-scores.get(p, 0.0), p))
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
                                   "scores": {p: round(scores.get(p, 0.0), 6) for p in keep}, **notes})

    def describe(self):
        return f"visual:{self.scorer.describe()}+{self.parser.describe()}"
