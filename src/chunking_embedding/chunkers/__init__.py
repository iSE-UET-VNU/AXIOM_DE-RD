"""Built-in and plugin text chunkers, auto-registered via ``@chunker``."""

from ..registry import load_plugins

load_plugins(__name__, __path__)
