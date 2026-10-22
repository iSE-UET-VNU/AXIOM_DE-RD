"""Independent adaptive page-budget prediction module.

This package deliberately does not import or modify the adaptive processor
router. It predicts how many pages from a frozen light-retrieval ranking to
pass downstream.
"""

__all__ = ["features", "dataset", "model", "inference"]
