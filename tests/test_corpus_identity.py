"""The seam between which corpus an arm ran over and the id its runs cache under."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.evaluation.run_retrieval import corpus_token


def write_corpus(path: Path, texts: list[str]) -> Path:
    path.write_text(
        "\n".join(
            json.dumps({"doc_id": f"d{i}", "title": "", "modality": "text",
                        "blocks": [{"text": t}]})
            for i, t in enumerate(texts)
        ),
        encoding="utf-8",
    )
    return path


class Adapter:
    """Minimal stand-in for the ise adapter: it carries a corpus path."""

    def __init__(self, corpus_path: Path) -> None:
        self.corpus_path = corpus_path


def test_two_corpora_get_different_tokens(tmp_path: Path):
    """extract.py vs chandra2 over the same lake -- the ladder's arm 3e vs arm 3."""
    extract = write_corpus(tmp_path / "corpus.extract.jsonl", ["page one", "page two"])
    chandra = write_corpus(tmp_path / "corpus.chandra.jsonl", ["page one parsed better", "page two"])

    assert corpus_token(Adapter(extract)) != corpus_token(Adapter(chandra))


def test_same_content_under_a_different_name_is_still_a_different_arm(tmp_path: Path):
    """The stem is part of the token, so an arm is named by what it is called."""
    a = write_corpus(tmp_path / "corpus.a.jsonl", ["same"])
    b = write_corpus(tmp_path / "corpus.b.jsonl", ["same"])
    assert corpus_token(Adapter(a)) != corpus_token(Adapter(b))
    assert corpus_token(Adapter(a)).startswith("corpus.a-")


def test_rebuilding_in_place_changes_the_token(tmp_path: Path):
    """Content-hashed, not path-named."""
    path = write_corpus(tmp_path / "corpus.jsonl", ["before"])
    before = corpus_token(Adapter(path))
    write_corpus(path, ["after"])
    assert corpus_token(Adapter(path)) != before


def test_identical_bytes_at_one_path_are_stable(tmp_path: Path):
    """Otherwise every re-run misses its own cache and pays for it again."""
    path = write_corpus(tmp_path / "corpus.jsonl", ["stable"])
    assert corpus_token(Adapter(path)) == corpus_token(Adapter(path))


def test_adapter_without_a_corpus_file_describes_itself(tmp_path: Path):
    """ViDoRe has no single corpus file; subset and language are its identity."""
    from src.evaluation.benchmarks.vidore_v3 import ViDoreV3

    class Fake(ViDoreV3):
        def __init__(self, subset: str, language: str) -> None:  # noqa: D107
            self.subset, self.language = subset, language

    assert corpus_token(Fake("physics", "english")) == "physics-english"
    assert corpus_token(Fake("physics", "french")) != corpus_token(Fake("physics", "english"))
    assert corpus_token(Fake("hr", "english")) != corpus_token(Fake("physics", "english"))


def test_missing_corpus_is_named_not_silently_equal(tmp_path: Path):
    """Two absent corpora must not collapse to one token and share a cache."""
    a = corpus_token(Adapter(tmp_path / "corpus.gone.jsonl"))
    b = corpus_token(Adapter(tmp_path / "corpus.other.jsonl"))
    assert a != b
    assert a.endswith("-missing")


# -- build_corpus refuses to clobber a baseline --------------------------------


def test_claim_refuses_an_existing_corpus(tmp_path: Path):
    from src.evaluation.build_corpus import claim

    path = write_corpus(tmp_path / "corpus.extract.jsonl", ["baseline"])
    with pytest.raises(SystemExit, match="already exists"):
        claim(path, overwrite=False)
    assert claim(path, overwrite=True) == path
    assert path.read_text(encoding="utf-8")  # not truncated by the check itself


def test_report_travels_with_its_corpus(tmp_path: Path):
    from src.evaluation.build_corpus import report_path_for

    assert report_path_for(tmp_path / "corpus.chandra.jsonl").name == "corpus.chandra_report.json"
    assert report_path_for(tmp_path / "corpus.extract.jsonl").name == "corpus.extract_report.json"
