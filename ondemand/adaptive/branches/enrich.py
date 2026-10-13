from time import perf_counter

from ...bench import BENCH, load_jsonl, work
from .. import fallback, services
from ..registry import build, register
from ..subdata import BranchOutput, Passage
from .visual import service_for


class ChandraEnricher:
    name = "chandra"
    mode = None

    def __init__(self, argument=None, bench=BENCH, pages_name="pages_ppocrv5.jsonl", **_):
        self.service = service_for("chandra", self.mode, bench)
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
            fallback.warn(f"chandra_{self.service.mode}", f"{len(missing)} pages have no Chandra parse "
                                                          f"({self.service.mode} mode); using the light OCR text")
            ocr = self.ocr_text()
            text.update({p: ocr.get(p, "") for p in missing})
        return {p: text[p] for p in page_ids}

    def seconds(self, pages):
        return self.service.page_seconds(pages)

    def describe(self):
        return f"chandra:{self.service.mode}"


@register("enricher", "chandra")
class ConfiguredChandra(ChandraEnricher):
    pass


@register("enricher", "cached")
class CachedEnricher(ChandraEnricher):
    mode = "cached"


@register("enricher", "chandra_vllm")
class VLLMChandra(ChandraEnricher):
    mode = "vllm"


@register("enricher", "chandra_local")
class LocalChandraEnricher(ChandraEnricher):
    mode = "local"


@register("branch", "enrich")
class EnrichBranch:
    name = "enrich"

    def __init__(self, argument=None, enricher="chandra", **context):
        self.enricher = build("enricher", argument or enricher, **context)

    def prepare(self, subdatas):
        self.enricher.prepare([p for s in subdatas for p in s.top()])

    def run(self, subdata):
        started = perf_counter()
        pages = subdata.top()
        text = self.enricher.parse(pages)
        passages = [Passage(page, text.get(page, ""), rank) for rank, page in enumerate(pages, 1)]
        return BranchOutput(self.name, passages,
                            cost={"seconds": perf_counter() - started, "pages": len(passages),
                                  "parsed_pages": len(passages), "chandra_model_seconds": self.enricher.seconds(pages)},
                            notes={"enricher": self.enricher.describe()})

    def describe(self):
        return f"enrich:{self.enricher.describe()}"
