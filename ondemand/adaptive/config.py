import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import requests
from dotenv import load_dotenv

from ..bench import BENCH, ROOT, fingerprint, work

# Automatically load environment variables from repo root
load_dotenv(ROOT / ".env")


@dataclass
class AdaptiveConfig:
    # Chandra OCR & Layout Enrichment
    chandra_endpoint: Optional[str] = None
    chandra_api_key: str = ""
    chandra_model: str = "datalab-to/chandra-ocr-2"
    chandra_local: bool = False

    # ColVec Visual Reranking
    colvec_endpoint: Optional[str] = None
    colvec_api_key: str = ""
    colvec_model: str = "webAI-Official/webAI-ColVec1.1-4b"
    colvec_local: bool = False

    # PP-OCRv5 Text & Layout
    ppocr_endpoint: Optional[str] = None
    ppocr_api_key: str = ""
    ppocr_local: bool = False

    # KDL / Layout Parsing
    kdl_endpoint: Optional[str] = None
    kdl_model: str = "kdl-frontier-parser-nano"

    # LLM / Vision QA
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    qa_model: str = "deepseek/deepseek-v4-flash"
    judge_model: str = "openai/gpt-4o-mini"

    # Fallback behavior
    allow_fallback: bool = True

    @classmethod
    def from_env(cls, **overrides) -> "AdaptiveConfig":
        chandra_ep = (overrides.get("chandra_endpoint")
                      or os.getenv("CHANDRA_ENDPOINT")
                      or os.getenv("CHANDRA_URL")
                      or os.getenv("VLLM_API_BASE"))
        chandra_key = (overrides.get("chandra_api_key")
                       or os.getenv("CHANDRA_API_KEY")
                       or os.getenv("VLLM_API_KEY")
                       or "")
        chandra_model = (overrides.get("chandra_model")
                         or os.getenv("CHANDRA_MODEL")
                         or os.getenv("VLLM_MODEL_NAME")
                         or "datalab-to/chandra-ocr-2")
        chandra_local = bool(overrides.get("chandra_local")
                             or os.getenv("CHANDRA_LOCAL") in ("1", "true", "True"))

        colvec_ep = (overrides.get("colvec_endpoint")
                     or os.getenv("COLVEC_ENDPOINT"))
        colvec_key = (overrides.get("colvec_api_key")
                      or os.getenv("COLVEC_API_KEY")
                      or "")
        colvec_model = (overrides.get("colvec_model")
                        or os.getenv("COLVEC_MODEL")
                        or "webAI-Official/webAI-ColVec1.1-4b")
        colvec_local = bool(overrides.get("colvec_local")
                            or os.getenv("COLVEC_LOCAL") in ("1", "true", "True"))

        ppocr_ep = (overrides.get("ppocr_endpoint")
                    or os.getenv("PPOCR_ENDPOINT"))
        ppocr_key = (overrides.get("ppocr_api_key")
                     or os.getenv("PPOCR_API_KEY")
                     or "")
        ppocr_local = bool(overrides.get("ppocr_local")
                           or os.getenv("PPOCR_LOCAL") in ("1", "true", "True"))

        kdl_ep = (overrides.get("kdl_endpoint")
                  or os.getenv("KDL_ENDPOINT")
                  or os.getenv("VLLM_API_BASE"))
        kdl_model = (overrides.get("kdl_model")
                     or os.getenv("KDL_MODEL")
                     or os.getenv("VLLM_MODEL_NAME")
                     or "kdl-frontier-parser-nano")

        or_key = (overrides.get("openrouter_api_key")
                  or os.getenv("OPENROUTER_API_KEY")
                  or os.getenv("OPENAI_API_KEY")
                  or "")
        or_url = (overrides.get("openrouter_base_url")
                  or os.getenv("OPENROUTER_BASE_URL")
                  or os.getenv("OPENAI_BASE_URL")
                  or "https://openrouter.ai/api/v1")

        qa_m = overrides.get("qa_model") or os.getenv("QA_MODEL") or "deepseek/deepseek-v4-flash"
        judge_m = overrides.get("judge_model") or os.getenv("JUDGE_MODEL") or "openai/gpt-4o-mini"
        fallback = overrides.get("allow_fallback", True)

        return cls(
            chandra_endpoint=chandra_ep,
            chandra_api_key=chandra_key,
            chandra_model=chandra_model,
            chandra_local=chandra_local,
            colvec_endpoint=colvec_ep,
            colvec_api_key=colvec_key,
            colvec_model=colvec_model,
            colvec_local=colvec_local,
            ppocr_endpoint=ppocr_ep,
            ppocr_api_key=ppocr_key,
            ppocr_local=ppocr_local,
            kdl_endpoint=kdl_ep,
            kdl_model=kdl_model,
            openrouter_api_key=or_key,
            openrouter_base_url=or_url,
            qa_model=qa_m,
            judge_model=judge_m,
            allow_fallback=fallback,
        )


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
    global _GLOBAL_CONFIG
    cfg = get_config()
    for k, v in kwargs.items():
        if hasattr(cfg, k) and v is not None:
            setattr(cfg, k, v)
    return cfg


def ping_endpoint(url: Optional[str], api_key: str = "", timeout: float = 3.0) -> Dict[str, Any]:
    if not url:
        return {"configured": False, "status": "not_set"}
    base_url = url.rstrip("/")
    if base_url.endswith("/chat/completions"):
        base_url = base_url[:-len("/chat/completions")]

    headers = {}
    last_err = "no probe answered with a usable status code"
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Try /models or /health or root
    for probe in (f"{base_url}/models", f"{base_url}/health", base_url):
        try:
            res = requests.get(probe, headers=headers, timeout=timeout)
            if res.status_code in (200, 401, 403, 404):
                return {
                    "configured": True,
                    "reachable": True,
                    "status_code": res.status_code,
                    "url": url,
                    "status": "online" if res.status_code == 200 else f"http_{res.status_code}"
                }
        except requests.RequestException as err:
            last_err = str(err)
    return {"configured": True, "reachable": False, "url": url, "status": "unreachable", "error": last_err[:120]}


def get_hardware_info() -> Dict[str, Any]:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            import torch
            if torch.cuda.is_available():
                dev = torch.cuda.get_device_properties(0)
                vram_gb = round(dev.total_memory / (1024**3), 1)
                bf16 = torch.cuda.is_bf16_supported() if hasattr(torch.cuda, "is_bf16_supported") else False
                return {"cuda": True, "device_name": dev.name, "vram_gb": vram_gb, "bf16": bf16}
            return {"cuda": False, "device_name": "CPU", "vram_gb": 0.0, "bf16": False}
        except Exception:
            return {"cuda": False, "device_name": "CPU (No torch)", "vram_gb": 0.0, "bf16": False}


def doctor(bench=BENCH, verbose=True) -> Dict[str, Any]:
    cfg = get_config()
    fp = fingerprint(bench)
    hw = get_hardware_info()

    # 1. Endpoints
    endpoints = {
        "chandra": ping_endpoint(cfg.chandra_endpoint, cfg.chandra_api_key),
        "colvec": ping_endpoint(cfg.colvec_endpoint, cfg.colvec_api_key),
        "ppocr": ping_endpoint(cfg.ppocr_endpoint, cfg.ppocr_api_key),
        "kdl": ping_endpoint(cfg.kdl_endpoint),
    }

    # 2. API Keys
    keys = {
        "openrouter_or_openai": bool(cfg.openrouter_api_key),
        "chandra_api_key": bool(cfg.chandra_api_key),
    }

    # 3. Local Artifacts
    artifacts_status = {
        "light_prep_ppocrv5": (work("light_prep", bench=bench) / "pages_ppocrv5.jsonl").exists(),
        "light_prep_ppocr": (work("light_prep", bench=bench) / "pages_ocr_ppocr.jsonl").exists(),
        "gate_ppocrv5": (work("light_prep", bench=bench) / "gate_ppocrv5_all.json").exists(),
        "gate_locked": (work("light_prep", bench=bench) / "gate_locked_all.json").exists(),
        "colvec_scores": (work("colvec", bench=bench) / "colvec_scores.npy").exists(),
        "kdl_pages": (work("kdl", bench=bench) / "kdl_pages.jsonl").exists(),
        "te3s_cache": (work("embedding_cache", "te3s_ppocr", bench=bench)).exists(),
        "qa_cache": (work("qa_cache", bench=bench)).exists(),
    }

    report = {
        "bundle_fingerprint": fp,
        "hardware": hw,
        "endpoints": endpoints,
        "api_keys": keys,
        "artifacts": artifacts_status,
        "active_models": {
            "chandra_model": cfg.chandra_model,
            "colvec_model": cfg.colvec_model,
            "qa_model": cfg.qa_model,
            "judge_model": cfg.judge_model
        }
    }

    if verbose:
        print("=" * 65)
        print("       AXIOM ADAPTIVE MODULE — SYSTEM & TOOL STATUS")
        print("=" * 65)
        print(f"Benchmark Fingerprint: {fp}\n")

        print("1. EXTERNAL / HOSTED ENDPOINTS (optional live microservices):")
        # Chandra
        ch_s = endpoints["chandra"]["status"]
        ch_url = cfg.chandra_endpoint or "None (default: local OCR fallback)"
        print(f"  • Chandra (Enrich): [{ch_s.upper():11s}] -> {ch_url}")

        # ColVec — live endpoint override; the default scorer is colvec_cached
        cv_s = endpoints["colvec"]["status"]
        cv_url = cfg.colvec_endpoint or "None (default: ColVec cached scores matrix)"
        print(f"  • ColVec live (Visual): [{cv_s.upper():11s}] -> {cv_url}")

        # PP-OCR
        pp_s = endpoints["ppocr"]["status"]
        pp_url = cfg.ppocr_endpoint or "None (default: pages_ppocrv5.jsonl corpus)"
        print(f"  • PP-OCRv5 (Text):  [{pp_s.upper():11s}] -> {pp_url}")

        # OpenRouter / OpenAI
        or_status = "CONFIGURED" if keys["openrouter_or_openai"] else "MISSING"
        print(f"  • LLM/VLM API Key:  [{or_status:11s}] -> {cfg.openrouter_base_url}")

        print("\n2. COMPUTE HARDWARE & LOCAL IN-PROCESS FLAGS:")
        if hw["cuda"]:
            print(f"  • Compute Device:    CUDA [{hw['device_name']}] ({hw['vram_gb']} GiB VRAM)")
            print(f"  • --chandra-local:   SUPPORTED (in-process Chandra2 on GPU)")
            print(f"  • --colvec-local:    SUPPORTED (in-process ColVec 4b on GPU)")
            print(f"  • --ppocr-local:     SUPPORTED (in-process PaddleOCR on GPU)")
        else:
            print(f"  • Compute Device:    CPU (No CUDA GPU detected)")
            print(f"  • Local GPU Flags:   DISABLED (safely runs via precomputed caches/endpoints)")

        print("\n3. LOCAL DATA ARTIFACTS (Default Offline Caches):")
        for name, present in artifacts_status.items():
            mark = "✓ PRESENT" if present else "✗ MISSING"
            print(f"  • {name:20s}: {mark}")

        print("\n4. TOOL READINESS:")
        print("  • Light Branch:   READY (Local CPU OCR text + BM25/Dense)")
        has_colvec = artifacts_status["colvec_scores"]
        vis_ready = has_colvec or endpoints["colvec"]["reachable"] or hw["cuda"]
        if has_colvec:
            vis_mode = "READY (ColVec cache, default — no live endpoint needed)"
        elif endpoints["colvec"]["reachable"]:
            vis_mode = "READY (live ColVec endpoint)"
        elif hw["cuda"]:
            vis_mode = "READY (--colvec-local on GPU)"
        else:
            vis_mode = "DEGRADED (no ColVec cache and no endpoint; will fallback to gate scores)"
        print(f"  • Visual Branch:  {vis_mode}")
        enr_ready = endpoints["chandra"]["reachable"] or (work("enrich", bench=bench) / "enriched_pages.jsonl").exists() or hw["cuda"]
        print(f"  • Enrich Branch:  {'READY (Live/Cache/GPU)' if enr_ready else 'DEGRADED (will fallback to PP-OCR text)'}")
        print("=" * 65)

    return report
