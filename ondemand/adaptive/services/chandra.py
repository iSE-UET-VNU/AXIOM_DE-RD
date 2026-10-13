import json
import math
from time import perf_counter

from ...bench import BENCH, load_jsonl, work
from .base import JsonlCache, Pages, Service, VLLMServer, gpu_info, release_gpu

HF_MODEL = "datalab-to/chandra-ocr-2"
SERVED = "chandra"
PROMPT_TYPE = "ocr_layout"
IMAGE_DPI, MIN_PDF_IMAGE_DIM, MAX_OUTPUT_TOKENS = 192, 1024, 12384


def server_tuning(vram_gb):
    ratio = max(vram_gb, 16) / 80
    batched = max(1024, 2 ** math.floor(math.log2(8192 * ratio)))
    seqs = max(8, (int(64 * ratio) // 8) * 8)
    return str(batched), str(seqs)


class Chandra(Service):
    name = "chandra"
    modes = ("cached", "vllm", "local")

    def __init__(self, mode="cached", bench=BENCH, endpoint=None, port=8001, batch=None):
        super().__init__(mode, bench)
        self.endpoint, self.port = endpoint, int(port)
        self.batch = batch or (32 if mode == "vllm" else 1)
        self.pages = Pages(bench)
        self.manager = self.server = None
        self.base = None
        self.parsed = JsonlCache(self.folder() / "pages.jsonl", lambda r: r["page_id"])
        source = work("enrich", bench=bench) / "enriched_pages.jsonl"
        self.corpus = ({r["page_id"]: r["text"] for r in load_jsonl(source)}
                       if mode == "cached" and source.exists() else {})

    def settings(self):
        return {"model": HF_MODEL, "prompt_type": PROMPT_TYPE, "renderer": "chandra.input.load_pdf_images",
                "image_dpi": IMAGE_DPI, "min_pdf_image_dim": MIN_PDF_IMAGE_DIM, "max_output_tokens": MAX_OUTPUT_TOKENS,
                "include_images": False, "include_headers_footers": False}

    def lookup(self, page_id):
        row = self.parsed.get(page_id)
        if row is not None and not row.get("error"):
            return row["text"]
        return self.corpus.get(page_id)

    def load(self):
        if self.loaded or self.mode == "cached":
            self.loaded = True
            return
        try:
            from chandra.model import InferenceManager
            from chandra.settings import settings as package
        except ImportError:
            raise SystemExit("chandra is not installed: pip install 'chandra-ocr[hf]' (local) or 'chandra-ocr' (vllm)")
        info = gpu_info()
        started = perf_counter()
        package.MODEL_CHECKPOINT = HF_MODEL
        if self.mode == "local":
            if not info["cuda"]:
                raise SystemExit("chandra local needs a CUDA GPU; use --chandra vllm with --chandra-endpoint")
            self.manager = InferenceManager(method="hf")
        else:
            if self.endpoint:
                self.base = self.endpoint.rstrip("/")
                self.log.info("using the Chandra server at %s", self.base)
            else:
                if not info["cuda"]:
                    raise SystemExit("chandra vllm needs a CUDA GPU, or --chandra-endpoint pointing at a server")
                dtype = "bfloat16" if info["bf16"] else "half"
                if dtype == "half":
                    self.log.warning("%s has no bf16; serving Chandra in float16", info["name"])
                batched, seqs = server_tuning(info["vram_gb"])
                args = ["--dtype", dtype, "--max-model-len", "18000", "--max-num-seqs", seqs,
                        "--max-num-batched-tokens", batched, "--gpu-memory-utilization", "0.85",
                        "--enable-prefix-caching", "--mm-processor-kwargs",
                        json.dumps({"min_pixels": 3136, "max_pixels": 6291456})]
                self.server = VLLMServer("chandra", HF_MODEL, SERVED, args, self.port, self.log, self.bench)
                self.base = self.server.start()
            package.VLLM_MODEL_NAME = SERVED if self.server else None
            self.manager = InferenceManager(method="vllm")
        self.timings().record("chandra_load", "model", 1, perf_counter() - started)
        self.loaded = True

    def close(self):
        self.manager = None
        if self.server is not None:
            self.server.stop()
            self.server = None
        release_gpu()
        super().close()

    def prepare(self, page_ids):
        todo = self.pending(page_ids, self.lookup)
        if not todo or self.mode == "cached":
            return len(todo)
        from chandra.input import load_pdf_images
        from chandra.model.schema import BatchInputItem
        self.load()
        timings = self.timings()
        self.log.info("Chandra on %d pages (batches of %d)", len(todo), self.batch)
        extra = {"vllm_api_base": self.base} if self.mode == "vllm" else {}
        for start in range(0, len(todo), self.batch):
            chunk = todo[start:start + self.batch]
            started = perf_counter()
            images = [load_pdf_images(str(self.pages.path(p)), [self.pages.index(p)], image_dpi=IMAGE_DPI,
                                      min_pdf_image_dim=MIN_PDF_IMAGE_DIM)[0] for p in chunk]
            rendered = perf_counter()
            outputs = self.manager.generate([BatchInputItem(image=i, prompt_type=PROMPT_TYPE) for i in images],
                                            max_output_tokens=MAX_OUTPUT_TOKENS, include_images=False,
                                            include_headers_footers=False, **extra)
            seconds = perf_counter() - started
            for page, out in zip(chunk, outputs):
                if out.error:
                    self.log.error("Chandra returned an error for %s", page)
                self.parsed.put({"page_id": page, "text": out.markdown, "raw": out.raw, "error": bool(out.error),
                                 "token_count": out.token_count, "seconds": round(seconds / len(chunk), 6)})
            timings.record("chandra_batch", "page", len(chunk), seconds,
                           render_seconds=round(rendered - started, 6))
            self.log.info("chandra %d/%d pages", min(start + self.batch, len(todo)), len(todo))
        return len(todo)

    def parse(self, page_ids):
        self.prepare(page_ids)
        found = {p: self.lookup(p) for p in page_ids}
        return {p: t for p, t in found.items() if t is not None}, [p for p, t in found.items() if t is None]

    def page_seconds(self, pages):
        return round(sum(self.parsed.get(p)["seconds"] for p in pages if self.parsed.get(p)), 6)
