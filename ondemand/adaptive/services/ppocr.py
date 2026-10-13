import base64
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from time import perf_counter

import requests

from ...bench import BENCH
from .base import JsonlCache, Pages, Service, gpu_info

WORKER = Path(__file__).resolve().parent / "ppocr_worker.py"


class PPOCR(Service):
    name = "ppocr"
    modes = ("cached", "local", "endpoint")

    def __init__(self, mode="cached", bench=BENCH, endpoint=None, api_key="", python=None):
        super().__init__(mode, bench)
        self.endpoint, self.api_key = endpoint, api_key
        self.python = python or os.environ.get("PPOCR_PYTHON") or sys.executable
        self.pages = Pages(bench)
        self.process = None
        self.parsed = JsonlCache(self.folder() / "pages.jsonl", lambda r: r["page_id"])

    def settings(self):
        return {"det": "PP-OCRv5_server_det", "rec": "PP-OCRv5_server_rec", "doc_orientation": True,
                "doc_unwarping": False, "textline_orientation": False, "dpi": 200, "max_side": 4000}

    def lookup(self, page_id):
        row = self.parsed.get(page_id)
        return None if row is None or row.get("error") else row["text"]

    def _drain(self, stream):
        for line in stream:
            if line.strip():
                self.log.info("worker: %s", line.rstrip()[:500])

    def load(self):
        if self.loaded or self.mode != "local":
            self.loaded = True
            return
        info = gpu_info()
        env = {**os.environ, "PPOCR_DEVICE": os.environ.get("PPOCR_DEVICE", "gpu:0" if info["cuda"] else "cpu")}
        self.log.info("starting the PP-OCRv5 worker with %s on %s", self.python, env["PPOCR_DEVICE"])
        self.process = subprocess.Popen([self.python, str(WORKER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, text=True, bufsize=1, env=env)
        threading.Thread(target=self._drain, args=(self.process.stderr,), daemon=True).start()
        hello = json.loads(self.process.stdout.readline() or '{"ready": false, "error": "worker exited"}')
        if not hello.get("ready"):
            self.close()
            raise SystemExit(f"the PP-OCRv5 worker failed to start: {hello.get('error')}. If paddle and torch clash "
                             f"in this env, point PPOCR_PYTHON at a separate venv with paddleocr installed")
        self.timings().record("ppocr_load", "model", 1, hello["load_seconds"], device=hello["device"])
        self.log.info("PP-OCRv5 ready on %s in %.1fs", hello["device"], hello["load_seconds"])
        self.loaded = True

    def close(self):
        if self.process is not None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
        super().close()

    def _local(self, page_id):
        job = {"page_id": page_id, "path": str(self.pages.path(page_id)), "index": self.pages.index(page_id)}
        self.process.stdin.write(json.dumps(job) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("the PP-OCRv5 worker exited; see logs/ppocr.log")
        return json.loads(line)

    def _remote(self, page_id):
        import fitz
        started = perf_counter()
        with fitz.open(self.pages.path(page_id)) as doc:
            page = doc[self.pages.index(page_id)]
            dpi = 200
            side = max(page.rect.width, page.rect.height) * dpi / 72
            if side > 4000:
                dpi = int(dpi * 4000 / side)
            image = page.get_pixmap(dpi=dpi).tobytes("png")
        url = self.endpoint.rstrip("/")
        url = url if url.endswith("/ocr") else f"{url}/ocr"
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        res = requests.post(url, json={"image": base64.b64encode(image).decode(), "page_id": page_id},
                            headers=headers, timeout=120)
        res.raise_for_status()
        body = res.json()
        return {"page_id": page_id, "text": body.get("text", ""), "lines": body.get("lines", []), "dpi": dpi,
                "seconds": round(perf_counter() - started, 6), "error": None}

    def prepare(self, page_ids):
        todo = self.pending(page_ids, self.lookup)
        if not todo or self.mode == "cached":
            return len(todo)
        self.load()
        timings = self.timings()
        started = perf_counter()
        failed = 0
        self.log.info("PP-OCRv5 on %d pages", len(todo))
        for i, page in enumerate(todo, 1):
            try:
                row = self._local(page) if self.mode == "local" else self._remote(page)
            except Exception as error:
                row = {"page_id": page, "text": "", "lines": [], "seconds": 0.0,
                       "error": f"{type(error).__name__}: {error}"[:300]}
            if row["error"]:
                failed += 1
                self.log.error("PP-OCRv5 failed on %s: %s", page, row["error"])
            self.parsed.put(row)
            timings.record("ppocr_page", "page", 1, row["seconds"], unit_id=page, failed=bool(row["error"]))
            if i % 50 == 0 or i == len(todo):
                rate = (perf_counter() - started) / i
                self.log.info("ppocr %d/%d pages, %.2fs/page, %d failed, ETA %.0f min", i, len(todo), rate, failed,
                              (len(todo) - i) * rate / 60)
        return len(todo)

    def ocr(self, page_ids):
        self.prepare(page_ids)
        found = {p: self.lookup(p) for p in page_ids}
        return {p: t for p, t in found.items() if t is not None}, [p for p, t in found.items() if t is None]

    def rows(self, page_ids):
        return {p: self.parsed.get(p) for p in page_ids if self.parsed.get(p)}

    def page_seconds(self, pages):
        return round(sum(self.parsed.get(p)["seconds"] for p in pages if self.parsed.get(p)), 6)
