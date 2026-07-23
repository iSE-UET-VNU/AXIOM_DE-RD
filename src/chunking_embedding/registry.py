"""Name registries for chunkers and embedders.

Register a component with one decorator; select it with one config key. Plugin
modules are auto-imported by their package, so adding a chunker or embedder is a
single new file. A chunker may declare ``needs`` (a tuple like
``("embedder",)`` / ``("llm",)``); the field router injects a matching
``Resources`` handle when it runs.

Chunkers can also define aliases so experiment configs stay readable while old
names continue to resolve.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any, Callable

_CHUNKERS: dict[str, Callable[..., Any]] = {}
_CHUNKER_NEEDS: dict[str, tuple[str, ...]] = {}
_CHUNKER_ALIASES: dict[str, str] = {}
_EMBEDDERS: dict[str, type] = {}


def chunker(
    name: str,
    *,
    needs: tuple[str, ...] = (),
    aliases: tuple[str, ...] = (),
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def register(fn: Callable[..., Any]) -> Callable[..., Any]:
        if name in _CHUNKERS or name in _CHUNKER_ALIASES:
            raise ValueError(f"Duplicate chunker registration: {name!r}")
        _CHUNKERS[name] = fn
        _CHUNKER_NEEDS[name] = tuple(needs)
        for alias in aliases:
            if alias in _CHUNKERS or alias in _CHUNKER_ALIASES:
                raise ValueError(f"Duplicate chunker alias registration: {alias!r}")
            _CHUNKER_ALIASES[alias] = name
        return fn

    return register


def embedder(name: str) -> Callable[[type], type]:
    def register(cls: type) -> type:
        if name in _EMBEDDERS:
            raise ValueError(f"Duplicate embedder registration: {name!r}")
        _EMBEDDERS[name] = cls
        return cls

    return register


def get_chunker(name: str) -> Callable[..., Any]:
    return _lookup(_CHUNKERS, "chunker", _resolve_chunker_name(name))


def get_embedder(name: str) -> type:
    return _lookup(_EMBEDDERS, "embedder", name)


def chunker_needs(name: str) -> tuple[str, ...]:
    return _CHUNKER_NEEDS.get(_resolve_chunker_name(name), ())


def chunker_names() -> list[str]:
    return sorted(_CHUNKERS)


def chunker_aliases(name: str) -> tuple[str, ...]:
    canonical = _resolve_chunker_name(name)
    return tuple(
        sorted(alias for alias, target in _CHUNKER_ALIASES.items() if target == canonical)
    )


def chunker_info(name: str) -> dict[str, Any]:
    canonical = _resolve_chunker_name(name)
    return {
        "name": canonical,
        "aliases": chunker_aliases(canonical),
        "needs": _CHUNKER_NEEDS.get(canonical, ()),
    }


def embedder_names() -> list[str]:
    return sorted(_EMBEDDERS)


def load_plugins(package: str, path: list[str]) -> None:
    for info in pkgutil.iter_modules(path):
        if not info.name.startswith("_"):
            importlib.import_module(f"{package}.{info.name}")


def _lookup(target: dict[str, Any], kind: str, name: str) -> Any:
    try:
        return target[name]
    except KeyError:
        raise KeyError(f"Unknown {kind} {name!r}; registered: {sorted(target)}") from None


def _resolve_chunker_name(name: str) -> str:
    return _CHUNKER_ALIASES.get(name, name)
