import gc
import hashlib
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
import warnings
from pathlib import Path

import requests

from ...bench import BENCH, documents, load_jsonl, work
from ...timing import Timings

FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
ROOT_LOGGER = "ondemand"


def console(level=logging.INFO):
    root = logging.getLogger(ROOT_LOGGER)
    root.setLevel(logging.INFO)
    if not any(getattr(h, "_ondemand_console", False) for h in root.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s", "%H:%M:%S"))
        handler.setLevel(level)
        handler._ondemand_console = True
        root.addHandler(handler)
    return root


def attach(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(FORMAT))
    logging.getLogger(ROOT_LOGGER).addHandler(handler)
    return handler


def detach(handler):
    logging.getLogger(ROOT_LOGGER).removeHandler(handler)
    handler.close()


def service_logger(name, bench=BENCH):
    console()
    log = logging.getLogger(f"{ROOT_LOGGER}.{name}")
    if not log.handlers:
        handler = logging.FileHandler(work("logs", bench=bench) / f"{name}.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter(FORMAT))
        log.addHandler(handler)
    return log


def signature(settings):
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:10]


class JsonlCache:
    def __init__(self, path, key):
        self.path, self.key = Path(path), key
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows = {}
        if self.path.exists():
            for row in load_jsonl(self.path):
                self.rows[self.key(row)] = row

    def __contains__(self, key):
        return key in self.rows

    def __len__(self):
        return len(self.rows)

    def get(self, key, default=None):
        return self.rows.get(key, default)

    def put(self, row):
        with self.path.open("a", encoding="utf-8") as sink:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            sink.flush()
            os.fsync(sink.fileno())
        self.rows[self.key(row)] = row
        return row


class Pages:
    def __init__(self, bench=BENCH):
        self.bench = Path(bench)
        self.docs = {d["doc_id"]: d for d in documents(bench)}

    def path(self, page_id):
        doc_id, _, _ = page_id.partition("#page=")
        doc = self.docs.get(doc_id)
        if doc is None:
            raise KeyError(f"{doc_id} is not in documents.jsonl")
        return self.bench / doc["path"]

    def index(self, page_id):
        return int(page_id.partition("#page=")[2])

    def image(self, page_id, dpi):
        import fitz
        from PIL import Image
        with fitz.open(self.path(page_id)) as pdf:
            pix = pdf.load_page(self.index(page_id)).get_pixmap(dpi=dpi, colorspace=fitz.csRGB, alpha=False)
            return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def gpu_info():
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import torch
    except ImportError:
        return {"cuda": False, "name": "cpu (no torch)", "vram_gb": 0.0, "bf16": False}
    if not torch.cuda.is_available():
        return {"cuda": False, "name": "cpu", "vram_gb": 0.0, "bf16": False}
    props = torch.cuda.get_device_properties(0)
    return {"cuda": True, "name": props.name, "vram_gb": round(props.total_memory / 1024 ** 3, 1),
            "bf16": bool(torch.cuda.is_bf16_supported())}


def release_gpu():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except ImportError:
        pass


class Service:
    name = "service"
    modes = ()

    def __init__(self, mode, bench=BENCH):
        if mode not in self.modes:
            raise SystemExit(f"{self.name}: unknown mode {mode!r}; available: {', '.join(self.modes)}")
        self.mode, self.bench = mode, bench
        self.log = service_logger(self.name, bench)
        self.loaded = False
        self.attempted = set()

    def pending(self, keys, lookup):
        todo = [k for k in dict.fromkeys(keys) if lookup(k) is None and k not in self.attempted]
        if self.mode != "cached":
            self.attempted.update(todo)
        return todo

    def settings(self):
        return {}

    def folder(self):
        sig = signature(self.settings())
        path = work(self.name, "live", sig, bench=self.bench)
        meta = path / "settings.json"
        if not meta.exists():
            meta.write_text(json.dumps(self.settings(), indent=1), encoding="utf-8")
        return path

    def timings(self):
        return Timings(self.folder() / "timings.jsonl", service=self.name, mode=self.mode, gpu=gpu_info()["name"])

    def load(self):
        self.loaded = True

    def close(self):
        if self.loaded:
            self.log.info("releasing %s", self.name)
        self.loaded = False

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, *_):
        self.close()

    def describe(self):
        return {"service": self.name, "mode": self.mode, **self.settings()}


class VLLMServer:
    def __init__(self, name, model, served_name, args, port, log, bench=BENCH, startup_minutes=30):
        self.name, self.model, self.served_name = name, model, served_name
        self.args, self.port, self.log = list(args), int(port), log
        self.log_path = work("logs", bench=bench) / f"vllm_{name}.log"
        self.startup_seconds = startup_minutes * 60
        self.process = None

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.port}/v1"

    def up(self):
        try:
            return requests.get(f"{self.base_url}/models", timeout=5).status_code == 200
        except requests.RequestException:
            return False

    def command(self):
        binary = os.environ.get("VLLM_BIN") or shutil.which("vllm")
        if not binary:
            raise SystemExit("no vllm command: pip install vllm==0.19.0, or set VLLM_BIN to the vllm of a separate venv")
        return [binary, "serve", self.model, "--host", "127.0.0.1", "--port", str(self.port),
                "--served-model-name", self.served_name, *self.args]

    def check_memory(self):
        if "torch" not in sys.modules:
            return
        import torch
        if not torch.cuda.is_available():
            return
        free, total = torch.cuda.mem_get_info()
        wanted = float(self.args[self.args.index("--gpu-memory-utilization") + 1]) \
            if "--gpu-memory-utilization" in self.args else 0.9
        self.log.info("GPU before vLLM %s: %.1f of %.1f GiB free", self.name, free / 1024 ** 3, total / 1024 ** 3)
        if free / total < wanted:
            raise SystemExit(f"vLLM {self.name} wants {wanted:.0%} of the GPU but only {free / total:.0%} is free; "
                             f"something still holds GPU memory (another model or server)")

    def start(self):
        if self.up():
            self.log.info("a vLLM server already answers on :%s, reusing it", self.port)
            return self.base_url
        release_gpu()
        self.check_memory()
        cmd = self.command()
        self.log.info("starting vLLM: %s", " ".join(cmd))
        self.log.info("vLLM output -> %s", self.log_path)
        sink = self.log_path.open("a", encoding="utf-8")
        self.process = subprocess.Popen(cmd, stdout=sink, stderr=subprocess.STDOUT, start_new_session=True)
        started = time.time()
        while not self.up():
            if self.process.poll() is not None:
                tail = self.log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
                raise RuntimeError(f"vLLM ({self.name}) exited with code {self.process.returncode}:\n{tail}")
            if time.time() - started > self.startup_seconds:
                self.stop()
                raise RuntimeError(f"vLLM ({self.name}) was not ready after {self.startup_seconds // 60} min; "
                                   f"see {self.log_path}")
            time.sleep(5)
        self.log.info("vLLM %s ready after %.0fs", self.name, time.time() - started)
        return self.base_url

    def stop(self):
        if self.process is None or self.process.poll() is not None:
            self.process = None
            return
        os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
        try:
            self.process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
            self.process.wait()
        self.log.info("vLLM %s stopped", self.name)
        self.process = None
        release_gpu()
