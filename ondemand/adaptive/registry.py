REGISTRY = {}


def register(kind, name):
    def wrap(target):
        REGISTRY.setdefault(kind, {})[name] = target
        return target
    return wrap


def names(kind):
    return sorted(REGISTRY.get(kind, {}))


def build(kind, spec, **context):
    name, _, argument = str(spec).partition(":")
    options = REGISTRY.get(kind, {})
    if name not in options:
        raise SystemExit(f"unknown {kind} {name!r}; available: {', '.join(sorted(options)) or 'none'}")
    return options[name](argument, **context)
