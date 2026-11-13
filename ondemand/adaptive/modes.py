import os
import random
import sys
from pathlib import Path

import numpy as np

MODES = ("offline", "online")
STAGES = ("query_prep", "embedder", "light_retriever", "adaptive_selector", "top3_check", "visual_encoder",
          "chunk_rank", "qa_reader", "qa_judge")
SEED = 0


class Modes:
    def __init__(self, default="offline", overrides=None):
        if default not in MODES:
            raise SystemExit(f"mode {default!r} is not one of {', '.join(MODES)}")
        self.default = default
        self.overrides = {}
        for stage, mode in (overrides or {}).items():
            if stage not in STAGES:
                raise SystemExit(f"unknown stage {stage!r}; stages: {', '.join(STAGES)}")
            if mode not in MODES:
                raise SystemExit(f"{stage}: mode {mode!r} is not one of {', '.join(MODES)}")
            self.overrides[stage] = mode

    def __call__(self, stage):
        if stage not in STAGES:
            raise SystemExit(f"unknown stage {stage!r}; stages: {', '.join(STAGES)}")
        return self.overrides.get(stage, self.default)

    def online(self, stage):
        return self(stage) == "online"

    def public(self):
        return {stage: self(stage) for stage in STAGES}


def parse_overrides(items):
    pairs = {}
    for item in items or ():
        stage, sep, mode = item.partition("=")
        if not sep:
            raise SystemExit(f"--stage-mode wants STAGE=MODE, got {item!r}")
        pairs[stage.strip()] = mode.strip()
    return pairs


def require(path, stage, hint):
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"{stage} is offline and {path} is missing; build it online with: {hint}")
    return path


def refuse(stage, what, hint):
    raise SystemExit(f"{stage} is offline and {what} is not cached; build it online with: {hint}")


def seed_everything(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if "torch" in sys.modules:
        torch = sys.modules["torch"]
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    return seed
