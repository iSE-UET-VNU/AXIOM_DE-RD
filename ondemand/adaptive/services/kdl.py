import hashlib
import json
import os
from collections import defaultdict
from time import perf_counter

import requests

from ...bench import BENCH, load_jsonl, work
from .base import JsonlCache, Pages, Service, VLLMServer, gpu_info

HF_MODEL = "KDLAI/KDL-Frontier-Parser-nano"
SERVED = "kdl-frontier-parser-nano"
DPI = 144
TOKENS = {"layout_max_output_tokens": 6000, "text_max_output_tokens": 2048, "table_max_output_tokens": 5500,
          "picture_max_output_tokens": 4096, "formula_max_output_tokens": 128}


def tuning(name, vram_gb, cpus):
    if "H100" in name:
        return "256", "65536", "0.90", 64, 256, min(32, cpus)
    if "A100" in name and vram_gb >= 70:
        return "128", "32768", "0.90", 48, 128, min(24, cpus)
    if "A100" in name:
        return "64", "16384", "0.95", 32, 64, min(16, cpus)
    if "L4" in name:
        return "32", "8192", "0.92", 16, 32, min(8, cpus)
    return "8", "4096", "0.90", 8, 16, min(4, cpus)


class KDL(Service):
    name = "kdl"
    modes = ("cached", "vllm")

    def __init__(self, mode="cached", bench=BENCH, endpoint=None, provider="kdl_pdf_inspector", port=8000):
        super().__init__(mode, bench)
        self.endpoint, self.provider_name, self.port = endpoint, provider, int(port)
        self.pages = Pages(bench)
        self.provider = self.server = None
        self.served = SERVED
        self.parsed = JsonlCache(self.folder() / "pages.jsonl", lambda r: r["page_id"])
        self.corpus = {}
        source = work("kdl", bench=bench) / "kdl_pages.jsonl"
        if mode == "cached" and source.exists():
            self.corpus = {r["page_id"]: r["text"] for r in load_jsonl(source)}
        elif mode == "cached":
            self.log.warning("%s is missing; cached mode reads only the live page cache", source)

    def settings(self):
        return {"model": HF_MODEL, "provider": self.provider_name, "dpi": DPI, "scheduler": "parsebench_document",
                "unit": "selected pages as a sub-PDF", **TOKENS}

    def lookup(self, page_id):
        row = self.parsed.get(page_id)
        if row is not None:
            return row["text"]
        return self.corpus.get(page_id)

    def load(self):
        if self.loaded or self.mode != "vllm":
            self.loaded = True
            return
        from ...kdl_runner import make_provider
        info = gpu_info()
        cpus = os.cpu_count() or 1
        seqs, batched, util, workers, bbox, render = tuning(info["name"], info["vram_gb"], cpus)
        if self.endpoint:
            base = self.endpoint.rstrip("/")
            self.served = requests.get(f"{base}/models", timeout=30).json()["data"][0]["id"]
            self.log.info("using the KDL server at %s (%s)", base, self.served)
        else:
            if not info["cuda"]:
                raise SystemExit("kdl vllm needs a CUDA GPU, or --kdl-endpoint pointing at a running server")
            if not info["bf16"]:
                raise SystemExit(f"{info['name']} has no bf16; KDL is served in bfloat16 (use an L4/A100/H100)")
            if info["vram_gb"] < 20:
                raise SystemExit(f"{info['name']} has {info['vram_gb']} GiB; KDL needs about 20 GiB")
            args = ["--dtype", "bfloat16", "--max-model-len", "8192", "--max-num-seqs", seqs,
                    "--max-num-batched-tokens", batched, "--gpu-memory-utilization", util,
                    "--limit-mm-per-prompt", '{"image":1}', "--trust-remote-code", "--enable-chunked-prefill",
                    "--enable-prefix-caching", "--generation-config", "vllm"]
            self.server = VLLMServer("kdl", HF_MODEL, SERVED, args, self.port, self.log, self.bench)
            base = self.server.start()
        self.provider = make_provider(self.provider_name, {
            "endpoint_url": base, "model": self.served, "max_pages": 1024, "dpi": DPI,
            "scheduler": "parsebench_document", "continuous_page_queue": True, "max_workers": workers,
            "render_processes": render, "bbox_max_workers": bbox, "request_workers": 8, "request_batch_size": 1,
            "max_model_sequences": int(seqs), "request_timeout_seconds": 3600, "max_retries": 2, **TOKENS,
            "save_raw_outputs": True, "output_dir": str(self.folder() / "raw")})
        self.loaded = True

    def close(self):
        self.provider = None
        if self.server is not None:
            self.server.stop()
            self.server = None
        super().close()

    def _sub_pdf(self, doc_id, indices):
        import fitz
        digest = hashlib.sha1(",".join(map(str, indices)).encode()).hexdigest()[:10]
        object_id = doc_id.replace("::", "__").replace("/", "_") + "__" + digest
        path = self.folder() / "subpdf" / f"{object_id}.pdf"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with fitz.open(self.pages.path(f"{doc_id}#page=0")) as source, fitz.open() as target:
                for index in indices:
                    target.insert_pdf(source, from_page=index, to_page=index)
                target.save(path)
        return object_id, path

    def prepare(self, page_ids):
        todo = self.pending(page_ids, self.lookup)
        if not todo or self.mode == "cached":
            return len(todo)
        from ...kdl.models import DataObject
        self.load()
        by_doc = defaultdict(list)
        for page in todo:
            by_doc[page.partition("#page=")[0]].append(self.pages.index(page))
        jobs = []
        for doc_id, indices in by_doc.items():
            indices = sorted(indices)
            object_id, path = self._sub_pdf(doc_id, indices)
            jobs.append((doc_id, indices, object_id, path))
        self.log.info("parsing %d pages from %d documents with KDL", len(todo), len(jobs))
        timings = self.timings()
        started = perf_counter()
        failed = []

        def done(index, outcome):
            doc_id, indices, object_id, _ = jobs[index]
            if isinstance(outcome, Exception):
                failed.append(doc_id)
                self.log.error("KDL failed on %s pages %s: %s: %s", doc_id, indices, type(outcome).__name__,
                               str(outcome)[:300])
                return
            result = json.loads((self.folder() / "raw" / object_id / "result.json").read_text(encoding="utf-8"))
            text_of = {int(p["page_number"]) - 1: p.get("content") or "" for p in result.get("markdown_pages") or []}
            latency = float(outcome.metadata.get("latency_seconds") or 0.0)
            for position, original in enumerate(indices):
                text = text_of.get(position, "")
                self.parsed.put({"page_id": f"{doc_id}#page={original}", "doc_id": doc_id, "text": text,
                                 "kdl_status": "ok" if text.strip() else "empty",
                                 "seconds": round(latency / len(indices), 6)})
            timings.record("kdl_subdoc", "page", len(indices), latency, doc_id=doc_id)
            self.log.info("KDL %s: %d pages in %.1fs", doc_id, len(indices), latency)

        pairs = [(path, DataObject(object_id=object_id, uri=str(path), metadata={"format": "pdf"}))
                 for _, _, object_id, path in jobs]
        self.provider.parse_files_with_errors(pairs, on_document_complete=done)
        timings.record("kdl_prepare_wall", "page", len(todo), perf_counter() - started, failed_docs=len(failed))
        return len(todo)

    def parse(self, page_ids):
        self.prepare(page_ids)
        found = {p: self.lookup(p) for p in page_ids}
        return {p: t for p, t in found.items() if t is not None}, [p for p, t in found.items() if t is None]

    def page_seconds(self, pages):
        return round(sum(self.parsed.get(p)["seconds"] for p in pages if self.parsed.get(p)), 6)
