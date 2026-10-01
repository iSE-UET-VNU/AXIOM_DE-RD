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

    def __init__(self, argument, **_):
        self.argument = argument

    def choose(self, subdata):
        raise NotImplementedError(
            "adaptive k is not implemented. Score-gap, patience and CV-confidence rules were measured at "
            "10-17% page savings at equal quality; see docs before adding one here.")
