#!/usr/bin/env python3
"""T2 verifier: is the new chunker registered, contract-correct, and did the suite stay green?

Usage:  t2_verify.py [--skip-suite]

Checks the behavioural contract only - it never reads the implementation, so any
correct implementation passes and no implementation can be written "to the test"
by matching its source.

Exit 0 = pass. Exit 1 = fail.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import subprocess
import sys
import unicodedata

BENCH = Path(__file__).resolve().parents[1]
REPO = BENCH.parents[1]
sys.path.insert(0, str(REPO))

NAME = "token_window"
ALIAS = "tok_win"

SAMPLE = (
    "Ingestion parses a PDF into a ParsedData record. The cleaning stage passes it "
    "through unchanged. Enrichment does the same. The chunking stage then routes each "
    "field to a chunker, which returns character spans over the source text.\n\n"
    "Spans are converted into chunks, embedded, and written to the embedded artifact "
    "directory. Every stage handoff is a plain function call, which is what makes the "
    "trace checkable at all. Tieng Viet cung duoc xu ly o day.\n\n"
    "A token-budgeted window must respect its budget, overlap its neighbours when asked "
    "to, and return the same answer twice in a row."
)


class Failure(Exception):
    pass


def check(condition: bool, message: str) -> None:
    if not condition:
        raise Failure(message)


def token_len(text: str) -> int | None:
    """Budget is measured with tiktoken cl100k_base, as the task states."""
    try:
        import tiktoken
    except ImportError:
        return None
    return len(tiktoken.get_encoding("cl100k_base").encode(text))


def run_checks() -> list[str]:
    from src.chunking_embedding import chunker_info, get_chunker  # noqa: PLC0415
    from src.chunking_embedding.registry import chunker_names  # noqa: PLC0415

    notes: list[str] = []

    # 1. Registration, by canonical name and by alias.
    check(NAME in chunker_names(), f"{NAME!r} is not registered; registered: {chunker_names()}")
    fn = get_chunker(NAME)
    alias_fn = get_chunker(ALIAS)
    check(fn is alias_fn, f"alias {ALIAS!r} does not resolve to the same callable as {NAME!r}")

    info = chunker_info(ALIAS)
    check(info["name"] == NAME, f"chunker_info({ALIAS!r}) reports name {info['name']!r}")
    check(ALIAS in info["aliases"], f"chunker_info does not list alias {ALIAS!r}: {info['aliases']}")
    check(tuple(info["needs"]) == (), f"{NAME!r} should declare no resource needs, got {info['needs']}")

    # 2. Span contract, matching the one the existing chunker tests enforce.
    spans = fn(SAMPLE)
    check(isinstance(spans, list) and spans, "chunker returned no spans")
    check(all(isinstance(s, tuple) and len(s) == 2 for s in spans), "spans must be (start, end) tuples")
    check(spans == sorted(spans), "spans must be sorted")
    for start, end in spans:
        check(0 <= start < end <= len(SAMPLE), f"span ({start}, {end}) out of bounds for len {len(SAMPLE)}")
        check(SAMPLE[start:end].strip() != "", f"span ({start}, {end}) is whitespace-only")

    # 3. Determinism.
    check(fn(SAMPLE) == spans, "chunker is not deterministic across two identical calls")

    # 4. Coverage: no non-whitespace character may be dropped.
    covered = bytearray(len(SAMPLE))
    for start, end in spans:
        for i in range(start, end):
            covered[i] = 1
    dropped = [i for i, ch in enumerate(SAMPLE) if not ch.isspace() and not covered[i]]
    check(not dropped, f"{len(dropped)} non-whitespace characters are covered by no span (first at {dropped[:1]})")

    # 5. The budget is actually a budget.
    tight = fn(SAMPLE, max_tokens=16, overlap_tokens=0)
    check(len(tight) > len(spans), "a 16-token budget should produce more spans than the default")
    lengths = [token_len(SAMPLE[s:e]) for s, e in tight]
    if any(length is None for length in lengths):
        notes.append("tiktoken unavailable - budget size not enforced this run")
    else:
        over = [(s, e, n) for (s, e), n in zip(tight, lengths) if n > 16]
        check(not over, f"{len(over)} span(s) exceed max_tokens=16, largest {max(n for *_, n in over) if over else 0}")

    # 6. Overlap behaves as named, in both directions.
    check(
        all(nxt[0] >= cur[1] for cur, nxt in zip(tight, tight[1:])),
        "with overlap_tokens=0 consecutive spans must not overlap",
    )
    lapped = fn(SAMPLE, max_tokens=16, overlap_tokens=8)
    check(len(lapped) > 1, "expected multiple spans at max_tokens=16")
    check(
        any(nxt[0] < cur[1] for cur, nxt in zip(lapped, lapped[1:])),
        "with overlap_tokens=8 consecutive spans must overlap",
    )

    # 7. Degenerate inputs must not raise.
    for edge in ("", "   ", "one", unicodedata.normalize("NFC", "Tieng Viet")):
        try:
            fn(edge)
        except Exception as exc:  # noqa: BLE001
            raise Failure(f"chunker raised on input {edge!r}: {exc!r}") from exc

    return notes


def run_suite() -> tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "-p", "no:cacheprovider"],
        cwd=REPO, capture_output=True, text=True,
    )
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else proc.stderr[-200:]
    return proc.returncode == 0, tail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-suite", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    failures: list[str] = []
    notes: list[str] = []
    try:
        notes = run_checks()
    except Failure as exc:
        failures.append(str(exc))
    except Exception as exc:  # import errors, bad signature, ...
        failures.append(f"{type(exc).__name__}: {exc}")

    suite_ok, suite_tail = (True, "skipped") if args.skip_suite else run_suite()
    if not suite_ok:
        failures.append(f"existing test suite is not green: {suite_tail}")

    passed = not failures
    if args.json:
        print(json.dumps({"task": "T2", "pass": passed, "failures": failures,
                          "notes": notes, "suite": suite_tail}))
    else:
        for note in notes:
            print(f"note: {note}")
        for failure in failures:
            print(f"FAIL: {failure}")
        print(f"suite: {suite_tail}")
        print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
