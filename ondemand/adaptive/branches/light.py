from time import perf_counter

from ...bench import BENCH, load_jsonl, work
from ...light_ocr import merge_row
from ...light_prep import wants_ocr
from .. import fallback, services
from ..config import get_config
from ..registry import register
from ..subdata import BranchOutput, Passage

ENGINE = "ppocrv5_doc"


@register("branch", "light")
class LightBranch:
    name = "light"

    def __init__(self, argument=None, bench=BENCH, pages_name="pages_ppocrv5.jsonl", **_):
        folder = work("light_prep", bench=bench)
        source = folder / (argument or pages_name)
        if not source.exists():
            raise SystemExit(f"{source} is missing; run `python -m ondemand.adaptive prep` first")
        self.source = source
        self.text = {r["page_id"]: r["text"] for r in load_jsonl(source)}
        self.service = None
        if get_config().ppocr_mode != "cached":
            inspector = folder / "pages_inspector.jsonl"
            if not inspector.exists():
                raise SystemExit(f"live PP-OCRv5 re-reads pages from {inspector}; run "
                                 f"`python -m ondemand light-prep --ocr none` first")
            self.base = {r["page_id"]: r for r in load_jsonl(inspector)}
            self.service = services.get("ppocr", bench)

    def targets(self, pages):
        return [p for p in pages if p in self.base and wants_ocr(self.base[p])]

    def prepare(self, subdatas):
        if self.service is None:
            return
        self.service.prepare(self.targets([p for s in subdatas for p in s.top()]))
        services.release(self.service)

    def page_text(self, pages):
        if self.service is None:
            return {p: self.text.get(p, "") for p in pages}, 0.0
        targets = self.targets(pages)
        found, missing = self.service.ocr(targets)
        if missing:
            fallback.warn(f"ppocr_{self.service.mode}", f"{len(missing)} pages have no live PP-OCRv5 text; using "
                                                        f"{self.source.name} for them")
        rows = self.service.rows(found)
        out = {}
        for page in pages:
            if page in found:
                out[page] = merge_row(self.base[page], rows[page], rows[page].get("lines", []), ENGINE)["text"]
            elif page in self.base and page not in missing:
                out[page] = self.base[page]["text"]
            else:
                out[page] = self.text.get(page, "")
        return out, self.service.page_seconds(list(found))

    def run(self, subdata):
        started = perf_counter()
        pages = subdata.top()
        text, model_seconds = self.page_text(pages)
        passages = [Passage(page, text[page], rank) for rank, page in enumerate(pages, 1)]
        empty = sum(not p.text.strip() for p in passages)
        return BranchOutput(self.name, passages,
                            cost={"seconds": perf_counter() - started, "pages": len(passages), "parsed_pages": 0,
                                  "ppocr_model_seconds": model_seconds},
                            notes={"source": self.source.name, "empty_pages": empty,
                                   "ppocr": self.service.mode if self.service else "cached"})

    def describe(self):
        return f"light:{self.source.name}:{self.service.mode if self.service else 'cached'}"
