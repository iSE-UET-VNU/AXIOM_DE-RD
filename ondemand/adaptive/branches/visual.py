from dataclasses import replace
from time import perf_counter

from ...bench import BENCH, load_jsonl, work
from .. import fallback, services
from ..config import get_config
from ..registry import build, register
from ..subdata import BranchOutput, Passage


def service_for(name, mode=None, bench=BENCH):
    if mode is None:
        return services.get(name, bench)
    return services.create(name, replace(get_config(), **{f"{name}_mode": mode}), bench)


class ColVecScorer:
    name = "colvec"
    mode = None

    def __init__(self, argument=None, bench=BENCH, **_):
        self.service = service_for("colvec", self.mode, bench)

    def prepare(self, items):
        self.service.prepare(items)
        services.release(self.service)

    def score(self, qid, pages, subdata=None, warn=True):
        query = subdata.query if subdata else ""
        scores, missing = self.service.score(qid, query, pages)
        if missing and warn:
            fallback.warn(f"colvec_{self.service.mode}", f"{len(missing)} of {len(pages)} pages of {qid} have no "
                                                         f"ColVec score ({self.service.mode} mode); ranking them by "
                                                         f"the light-retrieval score instead")
        for page in missing:
            scores[page] = subdata.scores.get(page, 0.0) if subdata else 0.0
        return scores

    def seconds(self, pages):
        return self.service.page_seconds(pages)

    def describe(self):
        return f"colvec:{self.service.mode}:{self.service.model_id}"


@register("scorer", "colvec")
class ConfiguredColVec(ColVecScorer):
    pass


@register("scorer", "colvec_cached")
class CachedColVec(ColVecScorer):
    mode = "cached"


@register("scorer", "colvec_local")
class LocalColVec(ColVecScorer):
    mode = "local"


@register("scorer", "colvec_endpoint")
class EndpointColVec(ColVecScorer):
    mode = "endpoint"


class KDLParser:
    name = "kdl"
    mode = None

    def __init__(self, argument=None, bench=BENCH, pages_name="pages_ppocrv5.jsonl", **_):
        self.service = service_for("kdl", self.mode, bench)
        self.bench, self.pages_name = bench, pages_name
        self._ocr = None

    def ocr_text(self):
        if self._ocr is None:
            source = work("light_prep", bench=self.bench) / self.pages_name
            self._ocr = {r["page_id"]: r["text"] for r in load_jsonl(source)} if source.exists() else {}
        return self._ocr

    def prepare(self, page_ids):
        self.service.prepare(page_ids)
        services.release(self.service)

    def parse(self, page_ids):
        text, missing = self.service.parse(page_ids)
        if missing:
            fallback.warn(f"kdl_{self.service.mode}", f"{len(missing)} pages have no KDL parse ({self.service.mode} "
                                                      f"mode); using the light OCR text for them")
            ocr = self.ocr_text()
            text.update({p: ocr.get(p, "") for p in missing})
        empty = [p for p in page_ids if not text[p].strip()]
        return {p: text[p] for p in page_ids}, {"missing_pages": missing, "empty_pages": empty}

    def seconds(self, pages):
        return self.service.page_seconds(pages)

    def describe(self):
        return f"kdl:{self.service.mode}:{self.service.provider_name}"


@register("parser", "kdl")
class ConfiguredKDL(KDLParser):
    pass


@register("parser", "kdl_cached")
class CachedKDL(KDLParser):
    mode = "cached"


@register("parser", "kdl_vllm")
class VLLMKDL(KDLParser):
    mode = "vllm"


@register("branch", "visual")
class VisualBranch:
    name = "visual"

    def __init__(self, argument=None, scorer="colvec", parser="kdl", refine_k="fixed:10", **context):
        self.scorer = build("scorer", argument or scorer, **context)
        self.parser = build("parser", parser, **context)
        self.refine_k = build("k", refine_k, **context)

    def rank(self, subdata, warn=True):
        candidates = subdata.top()
        scores = self.scorer.score(subdata.qid, candidates, subdata=subdata, warn=warn)
        ranked = sorted(candidates, key=lambda p: (-scores.get(p, 0.0), p))
        refined = subdata.__class__(subdata.qid, subdata.query, ranked, scores, len(ranked), source="visual")
        return candidates, scores, ranked[:self.refine_k.choose(refined)]

    def prepare(self, subdatas):
        self.scorer.prepare([(s.qid, s.query, s.top()) for s in subdatas])
        keep = [p for s in subdatas for p in self.rank(s, warn=False)[2]]
        self.parser.prepare(keep)

    def run(self, subdata):
        started = perf_counter()
        candidates, scores, keep = self.rank(subdata)
        rerank_seconds = perf_counter() - started
        started = perf_counter()
        text, notes = self.parser.parse(keep)
        passages = [Passage(page, text[page], rank) for rank, page in enumerate(keep, 1)]
        return BranchOutput(self.name, passages,
                            cost={"seconds": rerank_seconds + (perf_counter() - started),
                                  "rerank_seconds": rerank_seconds, "pages": len(candidates),
                                  "parsed_pages": len(keep), "colvec_model_seconds": self.scorer.seconds(candidates),
                                  "kdl_model_seconds": self.parser.seconds(keep)},
                            notes={"scorer": self.scorer.describe(), "parser": self.parser.describe(),
                                   "refine_k": self.refine_k.describe(),
                                   "scores": {p: round(scores.get(p, 0.0), 6) for p in keep}, **notes})

    def describe(self):
        return f"visual:{self.scorer.describe()}+{self.parser.describe()}"
