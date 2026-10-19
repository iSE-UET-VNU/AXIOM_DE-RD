"""CLI wrapper for importing offline Adaptive-K predictions into a gate."""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ondemand.adaptive.topk_policy import main  # noqa: E402


if __name__ == "__main__":
    main()
