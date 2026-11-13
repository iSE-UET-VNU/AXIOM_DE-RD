from dataclasses import asdict, dataclass, field

from ..bench import BENCH
from .registry import register


@dataclass
class PreparedQuery:
    raw: str
    text: str
    language: str = None
    translations: dict = field(default_factory=dict)
    sub_queries: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def public(self):
        return asdict(self)


@register("query_prep", "none")
class NoPrep:
    name = "none"

    def __init__(self, argument=None, bench=BENCH, online=False, **_):
        pass

    def prepare(self, query, meta=None):
        return PreparedQuery(raw=query, text=query, meta=dict(meta or {}))

    def settings(self):
        return {"query_prep": self.name}
