"""Text chunkers — one module per strategy, auto-registered via @chunker."""

from ..registry import load_plugins

load_plugins(__name__, __path__)
