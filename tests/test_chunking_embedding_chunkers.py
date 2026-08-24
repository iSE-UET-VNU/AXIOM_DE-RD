from __future__ import annotations

import pytest

from src.chunking_embedding import get_chunker

recursive = get_chunker("recursive")
paragraph = get_chunker("paragraph")
fixed_overlap = get_chunker("fixed_overlap")
sentence_group = get_chunker("sentence_group")

SAMPLE = (
    "First sentence here. Second sentence follows. Third one too.\n\n"
    "A new paragraph starts. It has more sentences. And another one. "
    "Yet another sentence. The last of this group.\n\n"
    "Final short paragraph."
)


@pytest.mark.parametrize("name", ["recursive", "paragraph", "fixed_overlap", "sentence_group"])
def test_structure_chunkers_are_registered_and_deterministic(name: str) -> None:
    chunker = get_chunker(name)
    spans_a = chunker(SAMPLE)
    spans_b = chunker(SAMPLE)
    assert spans_a == spans_b
    assert spans_a == sorted(spans_a)
    for start, end in spans_a:
        assert 0 <= start < end <= len(SAMPLE)
        assert SAMPLE[start:end].strip()


def test_structure_aliases_resolve_to_structure_implementations() -> None:
    assert get_chunker("recursive_400") is get_chunker("recursive")
    assert get_chunker("fixed_512_ol") is get_chunker("fixed_overlap")
    assert get_chunker("sentence_5") is get_chunker("sentence_group")


def test_non_paper_llm_approximations_are_not_registered() -> None:
    for name in ("raptor_lite", "raptor_tree", "proposition_llm", "lumber", "pic", "semantic_window"):
        with pytest.raises(KeyError):
            get_chunker(name)


def test_paragraph_splits_on_blank_lines() -> None:
    spans = paragraph(SAMPLE)
    assert len(spans) == 3
    assert SAMPLE[spans[2][0]:spans[2][1]] == "Final short paragraph."


def test_sentence_group_groups() -> None:
    spans = sentence_group(SAMPLE, n=2)
    texts = [SAMPLE[a:b] for a, b in spans]
    assert texts[0].startswith("First sentence here.")
    assert "Second sentence follows." in texts[0]
    assert "Third one too." not in texts[0]


def test_fixed_overlap_windows_overlap() -> None:
    text = " ".join(f"w{i}" for i in range(1000))
    spans = fixed_overlap(text, n_words=100, overlap=20)
    assert len(spans) > 1
    first_words = text[spans[0][0]:spans[0][1]].split()
    second_words = text[spans[1][0]:spans[1][1]].split()
    assert len(first_words) == 100
    assert first_words[80:] == second_words[:20]


def test_recursive_respects_limit() -> None:
    text = "\n\n".join("Sentence %d is here. It keeps going for a while now." % i for i in range(200))
    spans = recursive(text, target=400)
    assert all(end - start <= 800 for start, end in spans)
    covered = sum(end - start for start, end in spans)
    assert covered >= 0.9 * len(text)


def test_structure_chunkers_return_no_spans_for_empty_text() -> None:
    for name in ("recursive", "paragraph", "fixed_overlap", "sentence_group"):
        assert get_chunker(name)("") == []
        assert get_chunker(name)("   \n\n  ") == []
