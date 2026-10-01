from time import perf_counter

from ...bench import BENCH, load_jsonl, work
from ..registry import register
from ..subdata import BranchOutput, Passage


@register("branch", "light")
class LightBranch:
    name = "light"

    def __init__(self, argument, bench=BENCH, pages_name="pages_ocr_ppocr.jsonl", **_):
        source = work("light_prep", bench=bench) / (argument or pages_name)
        if not source.exists():
            raise SystemExit(f"{source} is missing; run light-prep first")
        self.source = source
        self.text = {r["page_id"]: r["text"] for r in load_jsonl(source)}

    def run(self, subdata):
        started = perf_counter()
        passages = [Passage(page, self.text.get(page, ""), rank)
                    for rank, page in enumerate(subdata.top(), 1)]
        empty = sum(not p.text.strip() for p in passages)
        return BranchOutput(self.name, passages,
                            cost={"seconds": perf_counter() - started, "pages": len(passages), "parsed_pages": 0},
                            notes={"source": self.source.name, "empty_pages": empty})

    def describe(self):
        return f"light:{self.source.name}"
