from .registry import register

BRANCHES = ("light", "enrich", "visual")


class Router:
    name = "base"

    def route(self, subdata):
        raise NotImplementedError

    def describe(self):
        return self.name


@register("router", "fixed")
class FixedRouter(Router):
    name = "fixed"

    def __init__(self, argument, **_):
        self.branch = argument or "visual"
        if self.branch not in BRANCHES:
            raise SystemExit(f"unknown branch {self.branch!r}; available: {', '.join(BRANCHES)}")

    def route(self, subdata):
        return self.branch, {"reason": "fixed"}

    def describe(self):
        return f"fixed:{self.branch}"


@register("router", "rule")
class RuleRouter(Router):
    name = "rule"

    def __init__(self, argument, **_):
        self.argument = argument

    def score(self, subdata):
        raise NotImplementedError("score function not decided")

    def cost(self, branch, subdata):
        raise NotImplementedError("cost function not decided")

    def route(self, subdata):
        raise NotImplementedError(
            "the score/cost rule is deliberately unimplemented. Pick the branch with --router fixed:<branch> "
            "until the routing rule is agreed.")
