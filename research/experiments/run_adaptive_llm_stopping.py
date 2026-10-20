"""Compatibility entrypoint for the organized Adaptive LLM experiment."""

from __future__ import annotations

from pathlib import Path
import runpy


TARGET = (
    Path(__file__).resolve().parents[1]
    / "adaptive/llm/scripts/run_adaptive_llm_stopping.py"
)


if __name__ == "__main__":
    runpy.run_path(str(TARGET), run_name="__main__")
