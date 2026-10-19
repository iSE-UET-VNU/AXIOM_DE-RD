from .registry import register


class KPolicy:
    name = "base"

    def choose(self, subdata):
        raise NotImplementedError

    def describe(self):
        return self.name


@register("k", "fixed")
class FixedK(KPolicy):
    name = "fixed"

    def __init__(self, argument, default=20, **_):
        self.k = int(argument) if argument else int(default)

    def choose(self, subdata):
        return min(self.k, len(subdata.pages))

    def describe(self):
        return f"fixed:{self.k}"


@register("k", "adaptive")
class AdaptiveK(KPolicy):
    name = "adaptive"

    def __init__(self, argument=None, k_confident=10, k_default=20, threshold=0.15, gate=None, **_):
        self.k_confident = k_confident
        self.k_default = k_default
        self.threshold = threshold
        self.gate_adaptive = (gate.get("adaptive_k") or {}) if isinstance(gate, dict) else {}
        if argument:
            parts = argument.split(":")
            if len(parts) >= 1 and parts[0].isdigit():
                self.k_confident = int(parts[0])
            if len(parts) >= 2 and parts[1].isdigit():
                self.k_default = int(parts[1])
            if len(parts) >= 3:
                try:
                    self.threshold = float(parts[2])
                except ValueError:
                    pass

    def choose(self, subdata):
        # 1. Gate-level precomputed confidence rule if available
        if subdata.qid in self.gate_adaptive:
            return min(self.gate_adaptive[subdata.qid], len(subdata.pages))

        # 2. Dynamic score-gap rule
        candidates = subdata.pages[:20]
        scores = [subdata.scores.get(p, 0.0) for p in candidates]
        if len(scores) >= 10:
            top_gap = scores[0] - scores[1]
            margin_10 = scores[0] - scores[9]
            # High confidence if top-1 margin is large or top-10 margin is dominant
            if top_gap >= self.threshold or margin_10 >= (self.threshold * 2):
                return min(self.k_confident, len(subdata.pages))

        return min(self.k_default, len(subdata.pages))

    def describe(self):
        return f"adaptive:{self.k_confident}:{self.k_default}:{self.threshold}"
