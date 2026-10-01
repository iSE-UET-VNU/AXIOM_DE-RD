from dataclasses import dataclass, field


@dataclass
class SubData:
    qid: str
    query: str
    pages: list
    scores: dict
    k: int
    source: str = "light"

    def top(self, k=None):
        return self.pages[:k if k is not None else self.k]


@dataclass
class Passage:
    page_id: str
    text: str
    rank: int


@dataclass
class BranchOutput:
    branch: str
    passages: list
    cost: dict = field(default_factory=dict)
    notes: dict = field(default_factory=dict)

    @property
    def pages(self):
        return [p.page_id for p in self.passages]
