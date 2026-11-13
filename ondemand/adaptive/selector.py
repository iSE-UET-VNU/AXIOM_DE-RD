import json
import logging
from pathlib import Path

from ..bench import BENCH, ROOT, work
from .kpolicy import AdaptiveK
from .registry import register
from .retrievers import load_gate

K_ARTIFACT = "qwen8b_lang_blended_214"
TEAMMATE_TE3S_ARTIFACT = ROOT / "data" / "work" / "adaptive_retrieval_20261022_check220"
COMPARED = ("dense_model", "signals", "analyzer", "file_mode", "w_dense", "parent_weight")
log = logging.getLogger("ondemand.selector")


def retriever_signature(light):
    if light.get("light_retriever") == "gate":
        return {"ranking_variant": light.get("gate_variant")}
    signature = {"dense_model": light["embedder"]["model"], "signals": light["signals"],
                 "analyzer": light["analyzer"], "file_mode": light["file_mode"], "w_dense": light["w_dense"],
                 "parent_weight": light["parent"]}
    if light["embedder"].get("matrix"):
        signature["dense_matrix"] = light["embedder"]["matrix"]
    return signature


@register("adaptive_selector", "gate_k")
class GateK:
    name = "gate_k"

    def __init__(self, argument=None, bench=BENCH, **_):
        self.gate_name = argument or "gate_ppocrv5_ungated_xgbK214.json"
        gate = load_gate(self.gate_name, bench)
        self.k = gate.get("adaptive_k") or {}
        self.rule = gate.get("adaptive_rule")
        if not self.k:
            raise SystemExit(f"{self.gate_name} holds no precomputed adaptive_k")

    def bind(self, light):
        pass

    def choose(self, subdata):
        if subdata.qid not in self.k:
            raise SystemExit(f"{self.gate_name} has no precomputed K for {subdata.qid}; use --selector xgb_k")
        return min(int(self.k[subdata.qid]), len(subdata.pages)), {}

    def settings(self):
        return {"adaptive_selector": self.name, "gate": self.gate_name, "rule": self.rule}


@register("adaptive_selector", "xgb_k")
class XGBK:
    name = "xgb_k"

    def __init__(self, argument=None, bench=BENCH, **_):
        from ..adaptive_retrieval.inference import AdaptiveKPredictor
        self.folder = Path(argument) if argument else work("adaptive_k", K_ARTIFACT, bench=bench)
        self.predictor = AdaptiveKPredictor.load(self.folder)
        manifest = self.folder / "dataset_manifest.json"
        self.trained = json.loads(manifest.read_text()) if manifest.exists() else {}
        self.policy = json.loads((self.folder / "deployment_policy.json").read_text())
        self.mismatch = {}

    def bind(self, light):
        trained = self.trained.get("settings", {})
        have = retriever_signature(light)
        self.mismatch = {k: {"trained": trained.get(k), "used": v} for k, v in have.items() if trained.get(k) != v}
        if self.mismatch:
            log.warning("xgb_k artifact %s was trained on a different light ranking: %s. Its K is out of "
                        "distribution here; retrain it on this retriever before quoting K or R@10",
                        self.folder.name, json.dumps(self.mismatch))

    def choose(self, subdata):
        curve = subdata.pages[:self.predictor.cap]
        missing = [p for p in curve if p not in subdata.scores]
        if missing:
            raise SystemExit(f"xgb_k needs the full top-{self.predictor.cap} score curve; {len(missing)} of "
                             f"{len(curve)} pages of {subdata.qid} have no score (the gate retriever keeps 20); "
                             f"use --light hier or --selector gate_k")
        return min(self.predictor.predict_k([subdata.scores[p] for p in curve]), len(subdata.pages)), {}

    def settings(self):
        return {"adaptive_selector": self.name, "artifact": str(self.folder.relative_to(ROOT))
                if self.folder.is_relative_to(ROOT) else str(self.folder),
                "trained_on": self.trained.get("settings"), "n_train_queries": self.trained.get("n_queries"),
                "policy_note": self.policy.get("note"), "retriever_mismatch": self.mismatch}


@register("adaptive_selector", "oof_k")
class OOFK:
    name = "oof_k"

    def __init__(self, argument=None, bench=BENCH, **_):
        path = Path(argument) if argument else work("adaptive_k", K_ARTIFACT, bench=bench)
        self.path = path / "oof_k.json" if path.is_dir() or not path.suffix else path
        if not self.path.exists():
            raise SystemExit(f"{self.path} is missing; run python -m ondemand.adaptive.retrain_k first")
        self.k = json.loads(self.path.read_text())["predictions"]
        manifest = self.path.parent / "dataset_manifest.json"
        self.trained = json.loads(manifest.read_text()) if manifest.exists() else {}
        self.mismatch = {}

    def bind(self, light):
        trained = self.trained.get("settings", {})
        have = retriever_signature(light)
        self.mismatch = {k: {"trained": trained.get(k), "used": v} for k, v in have.items() if trained.get(k) != v}
        if self.mismatch:
            raise SystemExit(f"{self.path} holds K for a different light ranking: {json.dumps(self.mismatch)}")

    def choose(self, subdata):
        if subdata.qid not in self.k:
            raise SystemExit(f"{self.path} has no out-of-fold K for {subdata.qid}; use --selector xgb_k for new queries")
        return min(int(self.k[subdata.qid]), len(subdata.pages)), {}

    def settings(self):
        return {"adaptive_selector": self.name, "file": str(self.path.relative_to(ROOT))
                if self.path.is_relative_to(ROOT) else str(self.path), "trained_on": self.trained.get("settings")}


@register("adaptive_selector", "fixed")
class FixedSelector:
    name = "fixed"

    def __init__(self, argument=None, **_):
        self.k = int(argument or 20)

    def bind(self, light):
        pass

    def choose(self, subdata):
        return min(self.k, len(subdata.pages)), {}

    def settings(self):
        return {"adaptive_selector": f"fixed:{self.k}"}


@register("adaptive_selector", "score_gap")
class ScoreGapSelector:
    name = "score_gap"

    def __init__(self, argument=None, **_):
        self.rule = AdaptiveK(argument)

    def bind(self, light):
        pass

    def choose(self, subdata):
        return self.rule.choose(subdata), {}

    def settings(self):
        return {"adaptive_selector": self.rule.describe()}


@register("top3_check", "none")
class NoTopCheck:
    name = "none"

    def __init__(self, argument=None, **_):
        pass

    def check(self, prepared, subdata, online):
        return {"action": "continue"}

    def settings(self):
        return {"top3_check": self.name}
