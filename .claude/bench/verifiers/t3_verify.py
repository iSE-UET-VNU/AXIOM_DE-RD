#!/usr/bin/env python3
"""T3 verifier: does Vietnamese retrieval work again, without breaking anything else?

Usage:  t3_verify.py [--skip-suite]

Purely behavioural: it builds a small mixed-language BM25 index through the real
public API and scores it with the repo's own hit_at_k. It never inspects the
source, so it accepts any correct fix and rejects a fix that only special-cases
this fixture.

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

# Written NFC. A correct implementation must also match the NFD spelling of the
# same query, which is what a mobile keyboard or a macOS filename actually emits.
CHUNKS = [
    {"chunk_id": "c1", "doc_id": "vi_fpt",   "text": "Báo cáo thường niên của tập đoàn FPT năm 2024 trình bày kết quả kinh doanh hợp nhất."},
    {"chunk_id": "c2", "doc_id": "vi_hanoi", "text": "Trụ sở chính đặt tại Hà Nội, với các chi nhánh ở Đà Nẵng và Thành phố Hồ Chí Minh."},
    {"chunk_id": "c3", "doc_id": "vi_chunk",  "text": "Quy trình chia nhỏ văn bản thành các đoạn ngữ nghĩa trước khi nhúng vector."},
    {"chunk_id": "c4", "doc_id": "en_report", "text": "The annual report presents consolidated revenue and operating profit for the group."},
    {"chunk_id": "c5", "doc_id": "zh_report", "text": "年度报告列出了集团的合并收入和营业利润。"},
]

# The last field is the analyzers this query is expected to work under. The CJK
# query is "auto" only, on purpose: sparse.py documents that the plain analyzer
# collapses a Chinese phrase into a single token and cannot match it. Asserting
# otherwise would be testing a behaviour the module deliberately does not have.
QUERIES = [
    ("q1", "báo cáo thường niên FPT", ["vi_fpt"], ("auto", "plain")),
    ("q2", "chi nhánh Đà Nẵng", ["vi_hanoi"], ("auto", "plain")),
    ("q3", "chia nhỏ văn bản ngữ nghĩa", ["vi_chunk"], ("auto", "plain")),
    ("q4", "consolidated revenue", ["en_report"], ("auto", "plain")),
    ("q5", "合并收入", ["zh_report"], ("auto",)),
]

K = 3


class Failure(Exception):
    pass


def build_index(analyzer_name: str):
    from src.retrieval.sparse import BM25Index  # noqa: PLC0415

    return BM25Index(analyzer_name=analyzer_name).build(CHUNKS)


def ranked_docs(index, query: str) -> list[str]:
    return [index.doc_ids[position] for position, _score in index.search(query, K)]


def run_checks() -> list[str]:
    from benchmark.metrics import QueryResult, hit_at_k  # noqa: PLC0415
    from src.retrieval.sparse import analyze  # noqa: PLC0415

    notes: list[str] = []

    # 1. The analyzer must be normalization-stable: the same word spelled NFC and
    #    NFD has to produce the same tokens, or index and query can never agree.
    word = "thường niên"
    if analyze(unicodedata.normalize("NFD", word)) != analyze(unicodedata.normalize("NFC", word)):
        raise Failure("analyze() is not normalization-stable: NFC and NFD spellings tokenize differently")

    # 2. End-to-end hit@3, under every analyzer the service can be configured with.
    for analyzer_name in ("auto", "plain"):
        index = build_index(analyzer_name)
        for qid, query, gold, analyzers in QUERIES:
            if analyzer_name not in analyzers:
                continue
            for form in ("NFC", "NFD"):
                spelled = unicodedata.normalize(form, query)
                result = QueryResult(qid=qid, ranked_docs=ranked_docs(index, spelled), gold_docs=gold)
                score = hit_at_k(result, K)
                if score != 1.0:
                    raise Failure(
                        f"hit@{K} = {score} for {qid} ({form}, analyzer={analyzer_name!r}): "
                        f"query {spelled!r} ranked {result.ranked_docs}, expected one of {gold}"
                    )

    # 3. A round-trip through save/load must not change any of that.
    import tempfile  # noqa: PLC0415

    index = build_index("auto")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bm25.json"
        index.save(path)
        from src.retrieval.sparse import BM25Index  # noqa: PLC0415

        reloaded = BM25Index.load(path)
        for qid, query, gold, _analyzers in QUERIES:
            result = QueryResult(qid=qid, ranked_docs=ranked_docs(reloaded, query), gold_docs=gold)
            if hit_at_k(result, K) != 1.0:
                raise Failure(f"hit@{K} dropped to 0 for {qid} after save/load round-trip")

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
    except Exception as exc:  # noqa: BLE001
        failures.append(f"{type(exc).__name__}: {exc}")

    suite_ok, suite_tail = (True, "skipped") if args.skip_suite else run_suite()
    if not suite_ok:
        failures.append(f"existing test suite is not green: {suite_tail}")

    passed = not failures
    if args.json:
        print(json.dumps({"task": "T3", "pass": passed, "failures": failures,
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
