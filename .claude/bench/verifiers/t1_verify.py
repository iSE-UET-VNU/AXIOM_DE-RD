#!/usr/bin/env python3
"""T1 verifier: did the trace name every stage handoff, with a correct file:line?

Usage:  t1_verify.py <answer-file>

Ground truth lives in tasks/t1_anchors.json as (file, symbol) pairs. Line numbers
are resolved *here, at verify time*, by locating the symbol in the current source
via ast. That way an ordinary refactor moves the expected line with the code
instead of silently failing every future run.

Exit 0 = pass (every anchor cited and correctly located). Exit 1 = fail.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import ast
import json
import re
import sys

BENCH = Path(__file__).resolve().parents[1]
REPO = BENCH.parents[1]

# "src/foo/bar.py:123", "`src/foo/bar.py:123`", "src/foo/bar.py#L123", "(src/foo/bar.py:123)"
CITATION = re.compile(r"([\w./\\-]+\.py)\s*(?::|#L)\s*(\d+)")


def resolve_symbol_line(path: Path, symbol: str) -> int | None:
    """Line of a module-level def/class named `symbol`, else None."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == symbol:
                # Point at `def`, not at the first decorator.
                return node.lineno
    return None


def resolve_literal_line(path: Path, needle: str) -> int | None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for number, line in enumerate(lines, start=1):
        if needle in line:
            return number
    return None


def parse_citations(text: str) -> list[tuple[str, int]]:
    out = []
    for raw_path, raw_line in CITATION.findall(text):
        out.append((raw_path.replace("\\", "/").lstrip("./"), int(raw_line)))
    return out


def cited(citations: list[tuple[str, int]], want_file: str, want_line: int, tol: int) -> int | None:
    """Return the matching cited line, or None. Path matches on suffix, so both
    `src/pipeline.py:38` and an absolute path count."""
    for path, line in citations:
        if path.endswith(want_file) and abs(line - want_line) <= tol:
            return line
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("answer", type=Path, help="File containing the agent's answer")
    ap.add_argument("--anchors", type=Path, default=BENCH / "tasks" / "t1_anchors.json")
    ap.add_argument("--json", action="store_true", help="Machine-readable result")
    args = ap.parse_args()

    if not args.answer.exists():
        print(f"FAIL: answer file not found: {args.answer}", file=sys.stderr)
        return 1

    spec = json.loads(args.anchors.read_text(encoding="utf-8"))
    tol = int(spec.get("tolerance_lines", 10))
    citations = parse_citations(args.answer.read_text(encoding="utf-8", errors="replace"))

    results = []
    for anchor in spec["anchors"]:
        source = REPO / anchor["file"]
        if anchor["kind"] == "symbol":
            truth = resolve_symbol_line(source, anchor["symbol"])
            label = f"{anchor['file']}::{anchor['symbol']}"
        else:
            truth = resolve_literal_line(source, anchor["match"])
            label = f"{anchor['file']} :: {anchor['match'][:40]}"

        if truth is None:
            # The answer key itself no longer resolves - the harness is stale,
            # which is a harness failure, not an agent failure. Say so loudly.
            results.append((anchor["id"], label, None, None, "ANCHOR-STALE"))
            continue

        hit = cited(citations, anchor["file"], truth, tol)
        results.append((anchor["id"], label, truth, hit, "ok" if hit is not None else "MISSING"))

    stale = [r for r in results if r[4] == "ANCHOR-STALE"]
    missing = [r for r in results if r[4] == "MISSING"]
    passed = not stale and not missing

    if args.json:
        print(json.dumps({
            "task": "T1",
            "pass": passed,
            "anchors_total": len(results),
            "anchors_found": sum(1 for r in results if r[4] == "ok"),
            "missing": [r[0] for r in missing],
            "stale": [r[0] for r in stale],
            "citations_in_answer": len(citations),
        }))
    else:
        for anchor_id, label, truth, hit, status in results:
            mark = "  ok " if status == "ok" else "FAIL "
            expected = f"expected line {truth}" if truth else "anchor did not resolve"
            got = f"cited {hit}" if hit else "not cited"
            print(f"{mark} {anchor_id:<18} {label:<62} {expected:<22} {got}")
        found = sum(1 for r in results if r[4] == "ok")
        print(f"\n{found}/{len(results)} anchors located (tolerance +/-{tol} lines)")
        if stale:
            print("HARNESS STALE: anchors no longer resolve in source - update t1_anchors.json")
        print("PASS" if passed else "FAIL")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
