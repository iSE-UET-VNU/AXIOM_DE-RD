"""Every arm we intend to compare must land on a distinct cache key."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from src.evaluation.run_retrieval import index_identity
from src.retrieval import runs

QIDS = ["q1", "q2", "q3"]


def arm(**overrides) -> argparse.Namespace:
    base = dict(
        benchmark="ise", level="page", text_source="vlm_text",
        chunker="", embedder="", prefix=False,
        rerank="", rerank_model="llm-rerank", rerank_depth=20,
        chunk_param=[], embedder_param=[], depth=100,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


class Corpus:
    def __init__(self, path: Path) -> None:
        self.corpus_path = path


def corpus(tmp_path: Path, name: str, text: str) -> Corpus:
    path = tmp_path / name
    path.write_text(json.dumps({"doc_id": "d0", "blocks": [{"text": text}]}), encoding="utf-8")
    return Corpus(path)


def key(args: argparse.Namespace, bench: Corpus, retriever_id: str, params: dict) -> str:
    identity = index_identity(args, bench)
    return str(runs.cache_path(Path("/runs"), identity, retriever_id,
                               {**params, "depth": args.depth}, QIDS))


@pytest.fixture
def ladder(tmp_path: Path):
    """The eight arms, exactly as they will be run."""
    extract = corpus(tmp_path, "corpus.extract.jsonl", "parsed by extract.py")
    chandra = corpus(tmp_path, "corpus.chandra.jsonl", "parsed by chandra2")
    full = dict(chunker="fixed_overlap", embedder="openrouter_te3s", prefix=True,
                rerank="llm", rerank_model="llm-rerank")
    return {
        "1  baseline":        (arm(), extract, "bm25", {"k1": 1.2}),
        "2  dense":           (arm(embedder="openrouter_te3s"), extract, "dense", {}),
        "3  full":            (arm(**full), chandra, "alpha0.7", {"alpha": 0.7}),
        "3a no rerank":       (arm(**{**full, "rerank": ""}), chandra, "alpha0.7", {"alpha": 0.7}),
        "3b no prefix":       (arm(**{**full, "prefix": False}), chandra, "alpha0.7", {"alpha": 0.7}),
        "3c dense not alpha": (arm(**full), chandra, "dense", {}),
        "3d no chunking":     (arm(**{**full, "chunker": ""}), chandra, "alpha0.7", {"alpha": 0.7}),
        "3e extract corpus":  (arm(**full), extract, "alpha0.7", {"alpha": 0.7}),
    }


def test_eight_arms_get_eight_distinct_cache_keys(ladder):
    keys = {name: key(*spec) for name, spec in ladder.items()}
    collisions = {}
    for name, value in keys.items():
        collisions.setdefault(value, []).append(name)
    duplicated = {v: names for v, names in collisions.items() if len(names) > 1}
    assert not duplicated, f"arms sharing a cache key: {list(duplicated.values())}"
    assert len(set(keys.values())) == 8


def test_the_parser_comparison_is_the_pair_that_was_broken(ladder):
    """3 vs 3e differ in nothing but the corpus. This is bug #7 in one assertion."""
    assert key(*ladder["3  full"]) != key(*ladder["3e extract corpus"])


# -- the knobs the audit found missing after the corpus one ---------------------


def test_chunker_parameters_are_in_the_identity(tmp_path: Path):
    """Two fixed_overlap runs at 512 and 1024 shared an identity."""
    bench = corpus(tmp_path, "c.jsonl", "x")
    small = arm(chunker="fixed_overlap", chunk_param=["size=512"])
    large = arm(chunker="fixed_overlap", chunk_param=["size=1024"])
    assert index_identity(small, bench) != index_identity(large, bench)


def test_embedder_parameters_are_in_the_identity(tmp_path: Path):
    """``model=`` selects a different upstream model under one embedder name."""
    bench = corpus(tmp_path, "c.jsonl", "x")
    a = arm(embedder="axiom_gateway", embedder_param=["model=openrouter-embedding"])
    b = arm(embedder="axiom_gateway", embedder_param=["model=embedding-default"])
    assert index_identity(a, bench) != index_identity(b, bench)


def test_rerank_model_is_in_the_identity(tmp_path: Path):
    """``rr-llm`` said reranking happened, not what did it."""
    bench = corpus(tmp_path, "c.jsonl", "x")
    weak = arm(rerank="llm", rerank_model="llm-rerank")
    strong = arm(rerank="llm", rerank_model="llm-rerank-strong")
    assert index_identity(weak, bench) != index_identity(strong, bench)


def test_rerank_depth_is_in_the_identity(tmp_path: Path):
    bench = corpus(tmp_path, "c.jsonl", "x")
    shallow = arm(rerank="llm", rerank_depth=20)
    deep = arm(rerank="llm", rerank_depth=50)
    assert index_identity(shallow, bench) != index_identity(deep, bench)


def test_depth_is_in_the_run_key(tmp_path: Path):
    """``depth`` changes what each record contains, so it changes the run."""
    bench = corpus(tmp_path, "c.jsonl", "x")
    shallow = key(arm(depth=20), bench, "bm25", {})
    deep = key(arm(depth=100), bench, "bm25", {})
    assert shallow != deep


def test_query_set_is_in_the_run_key():
    """A --limit run must not be served as the full run."""
    a = runs.cache_path(Path("/r"), "idx", "bm25", {}, ["q1", "q2", "q3"])
    b = runs.cache_path(Path("/r"), "idx", "bm25", {}, ["q1", "q2"])
    assert a != b


def test_query_order_does_not_change_the_key():
    """Otherwise every reordering is a cache miss and a re-spend."""
    a = runs.cache_path(Path("/r"), "idx", "bm25", {}, ["q1", "q2"])
    b = runs.cache_path(Path("/r"), "idx", "bm25", {}, ["q2", "q1"])
    assert a == b


def test_language_and_subset_separate_vidore_arms(tmp_path: Path):
    """ViDoRe has no corpus file; the same hole would open on its own axes."""
    from src.evaluation.benchmarks.vidore_v3 import ViDoreV3

    class Fake(ViDoreV3):
        def __init__(self, subset: str, language: str) -> None:  # noqa: D107
            self.subset, self.language = subset, language

    args = arm(benchmark="vidore_v3")
    assert index_identity(args, Fake("hr", "english")) != index_identity(args, Fake("hr", "french"))
    assert index_identity(args, Fake("hr", "english")) != index_identity(args, Fake("physics", "english"))
