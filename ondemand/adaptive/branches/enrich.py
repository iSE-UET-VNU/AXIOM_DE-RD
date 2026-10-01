from time import perf_counter

from ...bench import BENCH, load_jsonl, work
from ..registry import build, register
from ..subdata import BranchOutput, Passage


class Enricher:
    name = "base"

    def parse(self, page_ids):
        raise NotImplementedError

    def describe(self):
        return self.name


@register("enricher", "cached")
class CachedEnricher(Enricher):
    name = "cached"

    def __init__(self, argument, bench=BENCH, **_):
        source = work("enrich", bench=bench) / (argument or "enriched_pages.jsonl")
        if not source.exists():
            raise SystemExit(f"{source} is missing; no enriched parse is cached for this bundle")
        self.source = source
        self.text = {r["page_id"]: r["text"] for r in load_jsonl(source)}

    def parse(self, page_ids):
        return {p: self.text.get(p, "") for p in page_ids}

    def describe(self):
        return f"cached:{self.source.name}"


@register("enricher", "chandra")
class ChandraEnricher(Enricher):
    name = "chandra"

    def __init__(self, argument, **_):
        self.argument = argument

    def parse(self, page_ids):
        raise NotImplementedError(
            "Chandra is not wired up. It needs a serving endpoint and a cache under data/work/<fp>/enrich/; "
            "until then use --enricher cached with an exported parse.")


@register("branch", "enrich")
class EnrichBranch:
    name = "enrich"

    def __init__(self, argument, enricher="chandra", **context):
        self.enricher = build("enricher", argument or enricher, **context)

    def run(self, subdata):
        started = perf_counter()
        pages = subdata.top()
        text = self.enricher.parse(pages)
        passages = [Passage(page, text.get(page, ""), rank) for rank, page in enumerate(pages, 1)]
        return BranchOutput(self.name, passages,
                            cost={"seconds": perf_counter() - started, "pages": len(passages),
                                  "parsed_pages": len(passages)},
                            notes={"enricher": self.enricher.describe()})

    def describe(self):
        return f"enrich:{self.enricher.describe()}"
