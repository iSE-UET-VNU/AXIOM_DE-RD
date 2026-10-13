import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv

from ..bench import BENCH, ROOT, fingerprint, work

load_dotenv(ROOT / ".env")

COLVEC_MODEL = "webAI-Official/webAI-ColVec1.1-8b"
MODES = {"colvec": ("cached", "local", "endpoint"), "kdl": ("cached", "vllm"),
         "chandra": ("cached", "vllm", "local"), "ppocr": ("cached", "local", "endpoint")}


def _flag(name):
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes")


@dataclass
class AdaptiveConfig:
    colvec_mode: str = "cached"
    colvec_model: str = COLVEC_MODEL
    colvec_endpoint: Optional[str] = None
    colvec_api_key: str = ""
    colvec_8bit: bool = False

    kdl_mode: str = "cached"
    kdl_endpoint: Optional[str] = None
    kdl_provider: str = "kdl_pdf_inspector"
    kdl_port: int = 8000

    chandra_mode: str = "cached"
    chandra_endpoint: Optional[str] = None
    chandra_port: int = 8001

    ppocr_mode: str = "cached"
    ppocr_endpoint: Optional[str] = None
    ppocr_api_key: str = ""
    ppocr_python: Optional[str] = None

    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    qa_model: str = "deepseek/deepseek-v4-flash"
    judge_model: str = "openai/gpt-4o-mini"

    @property
    def colvec_local(self):
        return self.colvec_mode == "local"

    @property
    def chandra_local(self):
        return self.chandra_mode == "local"

    @property
    def ppocr_local(self):
        return self.ppocr_mode == "local"

    @classmethod
    def from_env(cls):
        cfg = cls(
            colvec_model=os.getenv("COLVEC_MODEL") or COLVEC_MODEL,
            colvec_endpoint=os.getenv("COLVEC_ENDPOINT") or None,
            colvec_api_key=os.getenv("COLVEC_API_KEY", ""),
            colvec_8bit=_flag("COLVEC_8BIT"),
            kdl_endpoint=os.getenv("KDL_ENDPOINT") or None,
            kdl_provider=os.getenv("KDL_PROVIDER") or "kdl_pdf_inspector",
            chandra_endpoint=os.getenv("CHANDRA_ENDPOINT") or None,
            ppocr_endpoint=os.getenv("PPOCR_ENDPOINT") or None,
            ppocr_api_key=os.getenv("PPOCR_API_KEY", ""),
            ppocr_python=os.getenv("PPOCR_PYTHON") or None,
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY") or "",
            openrouter_base_url=os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1",
            qa_model=os.getenv("QA_MODEL") or "deepseek/deepseek-v4-flash",
            judge_model=os.getenv("JUDGE_MODEL") or "openai/gpt-4o-mini")
        cfg.colvec_mode = ("local" if _flag("COLVEC_LOCAL") else "endpoint" if cfg.colvec_endpoint else "cached")
        cfg.kdl_mode = "vllm" if cfg.kdl_endpoint else "cached"
        cfg.chandra_mode = ("local" if _flag("CHANDRA_LOCAL") else "vllm" if cfg.chandra_endpoint else "cached")
        cfg.ppocr_mode = ("local" if _flag("PPOCR_LOCAL") else "endpoint" if cfg.ppocr_endpoint else "cached")
        for service in MODES:
            explicit = os.getenv(f"{service.upper()}_MODE")
            if explicit:
                setattr(cfg, f"{service}_mode", explicit)
        cfg.check()
        return cfg

    def check(self):
        for service, modes in MODES.items():
            mode = getattr(self, f"{service}_mode")
            if mode not in modes:
                raise SystemExit(f"{service} mode {mode!r} is not one of {', '.join(modes)}")
        if self.colvec_mode == "endpoint" and not self.colvec_endpoint:
            raise SystemExit("colvec endpoint mode needs --colvec-endpoint or COLVEC_ENDPOINT")
        if self.ppocr_mode == "endpoint" and not self.ppocr_endpoint:
            raise SystemExit("ppocr endpoint mode needs --ppocr-endpoint or PPOCR_ENDPOINT")

    def apply(self, **overrides):
        for key, value in overrides.items():
            if value is not None and value is not False and key in self.__dataclass_fields__:
                setattr(self, key, value)
        if overrides.get("all_local_gpu"):
            self.colvec_mode, self.kdl_mode, self.chandra_mode, self.ppocr_mode = "local", "vllm", "vllm", "local"
        if overrides.get("colvec_local"):
            self.colvec_mode = "local"
        if overrides.get("chandra_local"):
            self.chandra_mode = "local"
        if overrides.get("ppocr_local"):
            self.ppocr_mode = "local"
        if overrides.get("colvec_endpoint") and not overrides.get("colvec_mode") and self.colvec_mode == "cached":
            self.colvec_mode = "endpoint"
        if overrides.get("kdl_endpoint") and not overrides.get("kdl_mode"):
            self.kdl_mode = "vllm"
        if overrides.get("chandra_endpoint") and not overrides.get("chandra_mode") and self.chandra_mode == "cached":
            self.chandra_mode = "vllm"
        if overrides.get("ppocr_endpoint") and not overrides.get("ppocr_mode") and self.ppocr_mode == "cached":
            self.ppocr_mode = "endpoint"
        self.check()
        return self

    def services(self):
        return {s: {"mode": getattr(self, f"{s}_mode"),
                    "endpoint": getattr(self, f"{s}_endpoint", None)} for s in MODES}

    def public(self):
        return {k: v for k, v in asdict(self).items() if "api_key" not in k}


_GLOBAL_CONFIG: Optional[AdaptiveConfig] = None


def get_config() -> AdaptiveConfig:
    global _GLOBAL_CONFIG
    if _GLOBAL_CONFIG is None:
        _GLOBAL_CONFIG = AdaptiveConfig.from_env()
    return _GLOBAL_CONFIG


def set_config(config: AdaptiveConfig):
    global _GLOBAL_CONFIG
    _GLOBAL_CONFIG = config


def update_config(**kwargs) -> AdaptiveConfig:
    return get_config().apply(**kwargs)


def ping_endpoint(url: Optional[str], api_key: str = "", timeout: float = 3.0) -> Dict[str, Any]:
    if not url:
        return {"configured": False, "reachable": False, "status": "not_set"}
    base_url = url.rstrip("/")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    last_err = "no probe answered with a usable status code"
    for probe in (f"{base_url}/models", f"{base_url}/health", base_url):
        try:
            res = requests.get(probe, headers=headers, timeout=timeout)
            if res.status_code in (200, 401, 403, 404):
                return {"configured": True, "reachable": True, "status_code": res.status_code, "url": url,
                        "status": "online" if res.status_code == 200 else f"http_{res.status_code}"}
        except requests.RequestException as err:
            last_err = str(err)
    return {"configured": True, "reachable": False, "url": url, "status": "unreachable", "error": last_err[:120]}


def installed(module):
    from importlib.util import find_spec
    try:
        return find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def doctor(bench=BENCH, verbose=True) -> Dict[str, Any]:
    from .services.base import gpu_info
    cfg = get_config()
    hw = gpu_info()
    light = work("light_prep", bench=bench)
    artifacts_status = {
        "pages_ppocrv5.jsonl": (light / "pages_ppocrv5.jsonl").exists(),
        "gate_ppocrv5_all.json": (light / "gate_ppocrv5_all.json").exists(),
        "colvec corpus matrix": (work("colvec", bench=bench) / "colvec_scores.npy").exists(),
        "kdl corpus pages": (work("kdl", bench=bench) / "kdl_pages.jsonl").exists(),
        "te3s_ppocr cache": any(work("embedding_cache", "te3s_ppocr", bench=bench).iterdir()),
        "kdl chunk store": (work("embeddings", "kdl", bench=bench) / "chunk_vectors.npy").exists()}
    packages = {m: installed(m) for m in ("torch", "transformers", "vllm", "chandra", "paddleocr", "pdf_inspector",
                                           "fitz")}
    endpoints = {"colvec": ping_endpoint(cfg.colvec_endpoint, cfg.colvec_api_key),
                 "kdl": ping_endpoint(cfg.kdl_endpoint), "chandra": ping_endpoint(cfg.chandra_endpoint),
                 "ppocr": ping_endpoint(cfg.ppocr_endpoint, cfg.ppocr_api_key)}
    report = {"bundle_fingerprint": fingerprint(bench), "hardware": hw, "services": cfg.services(),
              "endpoints": endpoints, "packages": packages, "artifacts": artifacts_status,
              "openrouter_key": bool(cfg.openrouter_api_key), "colvec_model": cfg.colvec_model}
    if not verbose:
        return report
    need = {"colvec": {"local": ("transformers", 20 if "8b" in cfg.colvec_model else 10)},
            "kdl": {"vllm": ("vllm", 20)}, "chandra": {"vllm": ("vllm", 16), "local": ("transformers", 16)},
            "ppocr": {"local": ("paddleocr", 0)}}
    print("=" * 72)
    print(f"ondemand.adaptive doctor | bundle {report['bundle_fingerprint']}")
    print(f"GPU: {hw['name']} {hw['vram_gb']} GiB bf16={hw['bf16']}")
    print("-" * 72)
    for service, info in cfg.services().items():
        mode, note = info["mode"], ""
        if mode in need.get(service, {}):
            package, vram = need[service][mode]
            remote = (service == "kdl" and cfg.kdl_endpoint) or (service == "chandra" and cfg.chandra_endpoint)
            problems = []
            if not remote and hw["vram_gb"] < vram:
                problems.append(f"needs {vram} GiB")
            if not remote and not packages.get(package):
                problems.append(f"{package} not installed")
            if service == "chandra" and not packages["chandra"]:
                problems.append("chandra-ocr not installed")
            note = "READY" if not problems else "NOT READY: " + ", ".join(problems)
        elif mode == "endpoint" or (mode == "vllm" and info["endpoint"]):
            note = endpoints[service]["status"].upper()
        else:
            note = "cache only"
        print(f"  {service:8s} mode={mode:9s} {note}" + (f"  ({info['endpoint']})" if info["endpoint"] else ""))
    print("-" * 72)
    for name, present in artifacts_status.items():
        print(f"  {'✓' if present else '✗'} {name}")
    print(f"  {'✓' if report['openrouter_key'] else '✗'} OPENROUTER_API_KEY")
    print("=" * 72)
    return report
