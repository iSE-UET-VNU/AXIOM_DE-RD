import json
from importlib.metadata import PackageNotFoundError, version
from time import perf_counter

import numpy as np
import requests

from ...bench import BENCH, fingerprint, work
from .base import JsonlCache, Pages, Service, gpu_info, release_gpu

MODEL = "webAI-Official/webAI-ColVec1.1-8b"
MAX_VISUAL_TOKENS, DPI = 1792, 144
PINS = {"transformers": "5.14.1", "accelerate": "1.14.0"}


def maxsim(query, page):
    import torch
    sim = torch.einsum("ld,md->lm", query, page)
    return float(sim.max(dim=1).values.float().sum().item())


class ColVec(Service):
    name = "colvec"
    modes = ("cached", "local", "endpoint")

    def __init__(self, mode="cached", bench=BENCH, model=MODEL, endpoint=None, api_key="", load_8bit=False):
        super().__init__(mode, bench)
        self.model_id, self.endpoint, self.api_key, self.load_8bit = model, endpoint, api_key, load_8bit
        self.pages = Pages(bench)
        self.model = self.processor = None
        self.query_emb = {}
        folder = self.folder()
        self.scores = JsonlCache(folder / "scores.jsonl", lambda r: (r["qid"], r["page_id"]))
        self.encoded = JsonlCache(folder / "pages.jsonl", lambda r: r["page_id"])
        self.matrix = self._corpus_matrix() if mode == "cached" else None

    def settings(self):
        return {"model": self.model_id, "max_visual_tokens": MAX_VISUAL_TOKENS, "dpi": DPI,
                "load_8bit": self.load_8bit}

    def _corpus_matrix(self):
        folder = work("colvec", bench=self.bench)
        meta_path, scores_path = folder / "colvec_meta.json", folder / "colvec_scores.npy"
        if not (meta_path.exists() and scores_path.exists()):
            self.log.warning("no corpus ColVec matrix in %s; cached mode reads only the live score cache", folder)
            return None
        meta = json.loads(meta_path.read_text())
        if meta["bundle_fingerprint"] != fingerprint(self.bench):
            self.log.warning("the corpus ColVec matrix is from bundle %s, not this one; ignoring it",
                             meta["bundle_fingerprint"])
            return None
        if meta["model"] != self.model_id:
            self.log.warning("the corpus ColVec matrix was built with %s, not %s; ignoring it",
                             meta["model"], self.model_id)
            return None
        return {"scores": np.load(scores_path, mmap_mode="r"),
                "col": {k: i for i, k in enumerate(json.loads((folder / "colvec_keys.json").read_text()))},
                "row": {q: i for i, q in enumerate(json.loads((folder / "colvec_qids.json").read_text()))}}

    def load(self):
        if self.loaded or self.mode != "local":
            self.loaded = True
            return
        info = gpu_info()
        if not info["cuda"]:
            raise SystemExit("colvec local needs a CUDA GPU; use --colvec cached or --colvec endpoint here")
        import torch
        from transformers import AutoModel, AutoProcessor
        need = 20 if "8b" in self.model_id else 10
        if info["vram_gb"] < need and not self.load_8bit:
            raise SystemExit(f"{info['name']} has {info['vram_gb']} GiB; {self.model_id} needs about {need} GiB in "
                             f"16-bit. Use an L4/A100, --colvec-8bit, or --colvec-model "
                             f"webAI-Official/webAI-ColVec1.1-4b (different scores from the cache)")
        for package, pin in PINS.items():
            try:
                found = version(package)
            except PackageNotFoundError:
                found = None
            if found != pin:
                self.log.warning("%s %s is installed; the cached ColVec scores used %s", package, found, pin)
        kwargs = {"trust_remote_code": True, "attn_implementation": "sdpa", "device_map": "cuda:0"}
        if self.load_8bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            self.log.warning("loading ColVec in 8-bit; scores will not match the bf16 cache")
        else:
            kwargs["dtype"] = torch.bfloat16 if info["bf16"] else torch.float16
            if not info["bf16"]:
                self.log.warning("%s has no bf16; loading ColVec in float16, scores will differ slightly from "
                                 "the bf16 cache", info["name"])
        started = perf_counter()
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True,
                                                       max_num_visual_tokens=MAX_VISUAL_TOKENS)
        self.model = AutoModel.from_pretrained(self.model_id, **kwargs).eval()
        seconds = perf_counter() - started
        self.timings().record("colvec_model_load", "model", 1, seconds, model=self.model_id)
        self.log.info("loaded %s on %s in %.0fs", self.model_id, info["name"], seconds)
        self._load_query_embeddings()
        self.loaded = True

    def close(self):
        if self.model is not None:
            self.model = self.processor = None
            release_gpu()
        super().close()

    def _load_query_embeddings(self):
        import torch
        live = self.folder() / "query_embeddings.pt"
        if live.exists():
            self.query_emb.update(torch.load(live))
        corpus = work("colvec", bench=self.bench) / "query_embeddings.pt"
        meta = work("colvec", bench=self.bench) / "colvec_meta.json"
        if corpus.exists() and meta.exists() and json.loads(meta.read_text())["model"] == self.model_id:
            stored = torch.load(corpus)
            self.query_emb.update({q: e for q, e in stored.items() if q not in self.query_emb})
            self.log.info("reusing %d query embeddings from the corpus ColVec run", len(stored))

    def _embed(self, inputs):
        import torch
        with torch.inference_mode():
            out = self.model(**inputs.to(self.model.device))
        if hasattr(out, "embeddings"):
            return out.embeddings
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state
        return out[0]

    def query_embedding(self, qid, query):
        import torch
        if qid not in self.query_emb:
            started = perf_counter()
            self.query_emb[qid] = self._embed(self.processor.process_queries([query]))[0].to(torch.float16).cpu()
            self.timings().record("colvec_query_encode", "query", 1, perf_counter() - started, qid=qid)
            torch.save(self.query_emb, self.folder() / "query_embeddings.pt")
        return self.query_emb[qid].to(self.model.device)

    def encode_page(self, page_id):
        import torch
        started = perf_counter()
        image = self.pages.image(page_id, DPI)
        rendered = perf_counter()
        emb = self._embed(self.processor.process_images([image]))[0].to(torch.float16)
        torch.cuda.synchronize()
        encoded = perf_counter()
        return emb, {"render_seconds": round(rendered - started, 6), "encode_seconds": round(encoded - rendered, 6),
                     "tokens": int(emb.shape[0])}

    def lookup(self, qid, page_id):
        row = self.scores.get((qid, page_id))
        if row is not None:
            return row["score"]
        if self.matrix and qid in self.matrix["row"] and page_id in self.matrix["col"]:
            return float(self.matrix["scores"][self.matrix["row"][qid], self.matrix["col"][page_id]])
        return None

    def prepare(self, items):
        need = {}
        for qid, query, pages in items:
            for page in self.pending([(qid, p) for p in pages], lambda key: self.lookup(*key)):
                need.setdefault(page[1], []).append((qid, query))
        if not need or self.mode == "cached":
            return len(need)
        if self.mode == "endpoint":
            by_query = {}
            for page, owners in need.items():
                for qid, query in owners:
                    by_query.setdefault((qid, query), []).append(page)
            for (qid, query), pages in by_query.items():
                self._call_endpoint(qid, query, pages)
            return len(need)
        self.load()
        timings = self.timings()
        started = perf_counter()
        self.log.info("encoding %d pages for %d (query, page) pairs", len(need), sum(map(len, need.values())))
        for i, (page, owners) in enumerate(sorted(need.items()), 1):
            emb, notes = self.encode_page(page)
            score_started = perf_counter()
            for qid, query in owners:
                self.scores.put({"qid": qid, "page_id": page, "score": maxsim(self.query_embedding(qid, query), emb)})
            notes["score_seconds"] = round(perf_counter() - score_started, 6)
            if page not in self.encoded:
                self.encoded.put({"page_id": page, **notes})
            timings.record("colvec_page", "page", 1, notes["render_seconds"] + notes["encode_seconds"],
                           unit_id=page, **notes)
            if i % 25 == 0 or i == len(need):
                rate = (perf_counter() - started) / i
                self.log.info("colvec %d/%d pages, %.2fs/page, ETA %.0f min", i, len(need), rate,
                              (len(need) - i) * rate / 60)
        return len(need)

    def _call_endpoint(self, qid, query, pages):
        url = self.endpoint.rstrip("/")
        url = url if url.endswith("/rerank") else f"{url}/rerank"
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        started = perf_counter()
        res = requests.post(url, json={"qid": qid, "query": query, "documents": pages}, headers=headers, timeout=300)
        res.raise_for_status()
        scores = res.json()["scores"]
        for page in pages:
            if page in scores:
                self.scores.put({"qid": qid, "page_id": page, "score": float(scores[page])})
        self.timings().record("colvec_endpoint", "page", len(pages), perf_counter() - started, qid=qid)

    def score(self, qid, query, pages):
        self.prepare([(qid, query, pages)])
        found = {p: self.lookup(qid, p) for p in pages}
        return {p: s for p, s in found.items() if s is not None}, [p for p, s in found.items() if s is None]

    def page_seconds(self, pages):
        rows = [self.encoded.get(p) for p in pages]
        return round(sum(r["render_seconds"] + r["encode_seconds"] for r in rows if r), 6)

    def verify(self, items):
        folder = work("colvec", bench=self.bench)
        reference = np.load(folder / "colvec_scores.npy", mmap_mode="r")
        col = {k: i for i, k in enumerate(json.loads((folder / "colvec_keys.json").read_text()))}
        row = {q: i for i, q in enumerate(json.loads((folder / "colvec_qids.json").read_text()))}
        self.prepare(items)
        live, ref, top1 = [], [], []
        for qid, _, pages in items:
            pages = [p for p in pages if p in col and (qid, p) in self.scores]
            if qid not in row or not pages:
                continue
            a = [self.scores.get((qid, p))["score"] for p in pages]
            b = [float(reference[row[qid], col[p]]) for p in pages]
            live += a
            ref += b
            top1.append(int(np.argmax(a)) == int(np.argmax(b)))
        if not live:
            return {"n_pairs": 0}
        live, ref = np.array(live), np.array(ref)
        result = {"n_queries": len(top1), "n_pairs": len(live), "pearson": float(np.corrcoef(live, ref)[0, 1]),
                  "max_abs_diff": float(np.abs(live - ref).max()), "mean_abs_diff": float(np.abs(live - ref).mean()),
                  "score_scale": float(np.abs(ref).mean()), "top1_agreement": float(np.mean(top1))}
        result["reproduces"] = result["pearson"] >= 0.999 and result["top1_agreement"] >= 0.9
        self.log.info("verify vs corpus matrix: %s", json.dumps(result))
        return result
