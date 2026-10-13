from ...bench import BENCH
from ..config import get_config
from .chandra import Chandra
from .colvec import ColVec
from .kdl import KDL
from .ppocr import PPOCR

SERVICES = {"colvec": ColVec, "kdl": KDL, "chandra": Chandra, "ppocr": PPOCR}
OPEN = {}
KEEP_LOADED = {"value": False}


def create(name, cfg=None, bench=BENCH):
    cfg = cfg or get_config()
    if name == "colvec":
        return ColVec(cfg.colvec_mode, bench, model=cfg.colvec_model, endpoint=cfg.colvec_endpoint,
                      api_key=cfg.colvec_api_key, load_8bit=cfg.colvec_8bit)
    if name == "kdl":
        return KDL(cfg.kdl_mode, bench, endpoint=cfg.kdl_endpoint, provider=cfg.kdl_provider, port=cfg.kdl_port)
    if name == "chandra":
        return Chandra(cfg.chandra_mode, bench, endpoint=cfg.chandra_endpoint, port=cfg.chandra_port)
    if name == "ppocr":
        return PPOCR(cfg.ppocr_mode, bench, endpoint=cfg.ppocr_endpoint, api_key=cfg.ppocr_api_key,
                     python=cfg.ppocr_python)
    raise SystemExit(f"unknown service {name!r}; available: {', '.join(SERVICES)}")


def get(name, bench=BENCH):
    key = (name, str(bench))
    if key not in OPEN:
        OPEN[key] = create(name, bench=bench)
    return OPEN[key]


def keep_loaded(value=True):
    KEEP_LOADED["value"] = bool(value)


def release(service):
    if not KEEP_LOADED["value"]:
        service.close()


def close_all():
    for service in OPEN.values():
        service.close()
    OPEN.clear()


__all__ = ["SERVICES", "ColVec", "KDL", "Chandra", "PPOCR", "create", "get", "release", "keep_loaded", "close_all"]
