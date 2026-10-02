#!/usr/bin/env python3
"""Apply / revert the T3 seeded defect.

    t3_seed.py apply    # inject the bug, after stashing a pristine copy
    t3_seed.py revert   # restore the pristine copy
    t3_seed.py status

The defect: BM25Index.search normalizes its query terms to NFD while the index
was built from NFC text, so every Vietnamese term misses the postings map and
scores nothing. ASCII and CJK are byte-identical under NFD, so English and
Chinese retrieval are unaffected and all 85 existing tests still pass. The
symptom the agent is given is only "hit@k is zero for Vietnamese queries".

A literal string swap rather than a unified diff, because src/retrieval/sparse.py
is currently untracked - `git checkout --` would not bring it back.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import sys

SEEDS = Path(__file__).resolve().parent
REPO = SEEDS.parents[2]
TARGET = REPO / "src" / "retrieval" / "sparse.py"
PRISTINE = SEEDS / "sparse.py.pristine"

CLEAN = "        for term in self.analyzer(query):"
BUGGY = '        for term in (unicodedata.normalize("NFD", t) for t in self.analyzer(query)):'


def apply() -> int:
    source = TARGET.read_text(encoding="utf-8")
    if BUGGY in source:
        print("already seeded")
        return 0
    if CLEAN not in source:
        print(f"ERROR: anchor line not found in {TARGET}; the seed is stale", file=sys.stderr)
        return 1
    if not PRISTINE.exists():
        shutil.copy2(TARGET, PRISTINE)
    TARGET.write_text(source.replace(CLEAN, BUGGY, 1), encoding="utf-8")
    print(f"seeded {TARGET.relative_to(REPO)} (pristine copy at {PRISTINE.relative_to(REPO)})")
    return 0


def revert() -> int:
    if not PRISTINE.exists():
        print("ERROR: no pristine copy to restore from", file=sys.stderr)
        return 1
    shutil.copy2(PRISTINE, TARGET)
    print(f"reverted {TARGET.relative_to(REPO)}")
    return 0


def status() -> int:
    source = TARGET.read_text(encoding="utf-8")
    print("seeded" if BUGGY in source else "clean" if CLEAN in source else "unknown")
    return 0


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "status"
    raise SystemExit({"apply": apply, "revert": revert, "status": status}[action]())
