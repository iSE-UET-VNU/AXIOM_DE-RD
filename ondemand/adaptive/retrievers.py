import json

from ..bench import BENCH, doc_of, fingerprint, load_jsonl, work
from ..light_retrieval import ANALYZERS, POLICIES, Hierarchy
from .registry import register
from .subdata import SubData

FILE_MODES = ("dense_pool", "blended_pool", "bm25_blend")


def load_gate(name, bench=BENCH):
    path = work("light_prep", bench=bench) / name
    if not path.exists():
        raise SystemExit(f"{path} is missing; run `python -m ondemand.adaptive prep` first")
    gate = json.loads(path.read_text())
    if gate["bundle_fingerprint"] != fingerprint(bench):
        raise SystemExit(f"{name} is from a different bundle")
    return gate


@register("light_retriever", "gate")
class GateRetriever:
    name = "gate"

    def __init__(self, argument=None, bench=BENCH, **_):
        self.gate_name = argument or "gate_ppocrv5_ungated_xgbK214.json"
        self.gate = load_gate(self.gate_name, bench)

    def prepare(self, qids):
        missing = [q for q in qids if q not in self.gate["gate"]]
        if missing:
            raise SystemExit(f"{self.gate_name} has no ranking for {len(missing)} queries, e.g. {missing[:2]}")

    def retrieve(self, qid, prepared):
        if prepared.text != prepared.raw:
            raise SystemExit("the gate retriever holds rankings of the raw benchmark queries only; use "
                             "--light hier with a query_prep that rewrites the query")
        pages = self.gate["gate"][qid]
        scores = dict(zip(pages, self.gate.get("gate_scores", {}).get(qid, [])))
        ranked = self.gate.get("light_run_top100", {}).get(qid, pages)
        return SubData(qid, prepared.text, ranked, scores, len(pages)), {"gate": self.gate_name}

    def settings(self):
        return {"light_retriever": self.name, "gate": self.gate_name, "gate_variant": self.gate["variant"],
                "scores_kept": "top 20 only"}


@register("light_retriever", "hier")
class HierRetriever:
    name = "hier"

    def __init__(self, argument=None, bench=BENCH, pages_name="pages_ppocrv5.jsonl", embedder=None, signals="lang",
                 file_mode="blended_pool", w_dense=0.70, parent=0.15, analyzer="enfr", **_):
        if embedder is None:
            raise SystemExit("the hier light retriever needs an embedder")
        if file_mode not in FILE_MODES:
            raise SystemExit(f"file mode {file_mode!r}; available: {', '.join(FILE_MODES)}")
        if signals not in POLICIES:
            raise SystemExit(f"signals {signals!r}; available: {', '.join(sorted(POLICIES))}")
        if analyzer not in ANALYZERS:
            raise SystemExit(f"analyzer {analyzer!r}; available: {', '.join(sorted(ANALYZERS))}")
        self.bench, self.pages_name, self.embedder = bench, pages_name, embedder
        self.options = {"signals": signals, "file_mode": file_mode, "w_dense": float(w_dense),
                        "parent": float(parent), "analyzer": analyzer}
        self.hierarchy = None

    def prepare(self, qids):
        if self.hierarchy is not None:
            return
        rows = load_jsonl(work("light_prep", bench=self.bench) / self.pages_name)
        self.n_files = len({doc_of(r["page_id"]) for r in rows})
        self.embedder.prepare(rows, self.pages_name)
        self.hierarchy = Hierarchy(rows, file_k=self.n_files, **self.options)

    def retrieve(self, qid, prepared):
        dense = self.embedder.scores(prepared.text, qid=qid, raw=prepared.raw)
        ranked, scores, notes = self.hierarchy.rank(prepared.text, dense)
        return SubData(qid, prepared.text, ranked, scores, len(ranked)), \
            {"query_language": notes["query_language"], "bm25_pages": notes["bm25_pages"]}

    def settings(self):
        return {"light_retriever": self.name, "pages_source": self.pages_name, **self.options,
                "file_k": "all files", "pool": 100, "embedder": self.embedder.settings()}
