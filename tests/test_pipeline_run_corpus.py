"""A parsed run must be indexable, and two parsers must never share an identity.

Corpus identity keyed on the parser *name* would give chandra2 and kdl the same
token whenever their configs differ, so a config change would report as "the
parsers are identical" -- bug #7 on the parser axis.
"""

from __future__ import annotations

import json

import pytest

from src.evaluation.corpus_source import PipelineRunCorpus, parser_identity

CHANDRA = {"parser": "chandra2", "model_name": "chandra", "backend": "vllm",
           "max_workers": 48, "render_processes": 32, "request_batch_size": 1,
           "table_refinement": {"enabled": False, "prompt_sha256": "d4cc6a21"},
           "page_count": 17, "token_count": 9001, "latency_seconds": 3.5}
KDL = {"parser": "kdl", "model_name": "KDL-Frontier-Parser-nano", "backend": "vllm",
       "max_workers": 48, "render_processes": 32, "bbox_max_workers": 32,
       "max_output_tokens": 4096, "request_timeout_seconds": 3600,
       "page_count": 17, "token_count": 12000, "latency_seconds": 4.1}


def write_run(tmp_path, name, meta, docs):
    run = tmp_path / name / "run1"
    (run / "documents").mkdir(parents=True)
    (run / "metadata.json").write_text(json.dumps({"run_id": "run1", "document_count": len(docs)}))
    for i, (file_name, blocks) in enumerate(docs):
        (run / "documents" / f"d{i}.json").write_text(json.dumps({
            "document": {"file_name": file_name},
            "content": {"blocks": blocks,
                        "reading_order": [b["component_id"] for b in blocks]},
            "parsed": {"metadata": {**meta, "token_count": meta["token_count"] + i}},
        }))
    return run


def block(cid, page, index, type_, text):
    return {"component_id": cid, "page": page, "block_index": index, "type": type_, "text": text}


DOC = ("pdfs/Deck_One.pdf", [
    block("/page/0/SectionHeader/0", 0, 0, "SectionHeader", "Chapitre 1"),
    block("/page/0/Text/1", 0, 1, "Text", "corps de la page zero"),
    block("/page/1/Text/0", 1, 0, "Text", "corps de la page un"),
])


@pytest.fixture
def chandra_run(tmp_path):
    return write_run(tmp_path, "chandra2", CHANDRA, [DOC])


def test_units_are_pages_not_blocks(chandra_run):
    units = list(PipelineRunCorpus(chandra_run, subset="physics").units())
    assert len(units) == 2


def test_unit_ids_match_the_benchmark_convention(chandra_run):
    ids = {u.doc_id for u in PipelineRunCorpus(chandra_run, subset="physics").units()}
    assert ids == {"physics::Deck_One#page=0", "physics::Deck_One#page=1"}


def test_the_pdf_suffix_and_directory_prefix_are_stripped(chandra_run):
    """Real runs emit ``pdfs/<name>.pdf``; stripping only .pdf matched 0/42."""
    ids = {u.doc_id for u in PipelineRunCorpus(chandra_run, subset="physics").units()}
    assert all("pdfs/" not in i and ".pdf" not in i for i in ids)


def test_blocks_are_joined_in_reading_order(chandra_run):
    units = {u.doc_id: u.text for u in PipelineRunCorpus(chandra_run, subset="physics").units()}
    assert units["physics::Deck_One#page=0"].startswith("Chapitre 1")


def test_two_parsers_do_not_share_a_corpus_identity(tmp_path):
    a = PipelineRunCorpus(write_run(tmp_path, "c", CHANDRA, [DOC]), subset="physics")
    b = PipelineRunCorpus(write_run(tmp_path, "k", KDL, [DOC]), subset="physics")
    assert a.corpus_identity() != b.corpus_identity()


def test_the_same_parser_with_a_different_config_does_not_share_an_identity(tmp_path):
    """The failure this test exists for: identity on the parser name alone."""
    tweaked = {**CHANDRA, "max_workers": 256}
    a = PipelineRunCorpus(write_run(tmp_path, "a", CHANDRA, [DOC]), subset="physics")
    b = PipelineRunCorpus(write_run(tmp_path, "b", tweaked, [DOC]), subset="physics")
    assert a.corpus_identity() != b.corpus_identity()


def test_per_document_counters_do_not_change_the_identity(tmp_path):
    """token_count varies per document; folding it in would make identity unstable."""
    a = PipelineRunCorpus(write_run(tmp_path, "a", CHANDRA, [DOC]), subset="physics")
    b = PipelineRunCorpus(write_run(tmp_path, "b", CHANDRA, [DOC, DOC]), subset="physics")
    assert parser_identity(a.run_dir) == parser_identity(b.run_dir)


def test_the_identity_names_the_subset_and_parser_readably(chandra_run):
    assert PipelineRunCorpus(chandra_run, subset="physics").corpus_identity().startswith("physics-chandra2-")


def test_the_identity_is_stable_across_reads(chandra_run):
    source = PipelineRunCorpus(chandra_run, subset="physics")
    assert source.corpus_identity() == PipelineRunCorpus(chandra_run, subset="physics").corpus_identity()


def test_chunk_granularity_refuses_a_chunker(chandra_run):
    """Chunks are fixed at parse time; silently ignoring --chunker reports a lie."""
    with pytest.raises(ValueError, match="chunker"):
        PipelineRunCorpus(chandra_run, subset="physics", granularity="chunk",
                          chunker="fixed_overlap")


def test_page_granularity_still_allows_a_chunker(chandra_run):
    source = PipelineRunCorpus(chandra_run, subset="physics", granularity="page",
                               chunker="fixed_overlap", params={"n_words": 2, "overlap": 0})
    assert len(list(source.units())) > 2


def test_an_unknown_granularity_is_refused(chandra_run):
    with pytest.raises(ValueError, match="granularity"):
        PipelineRunCorpus(chandra_run, subset="physics", granularity="paragraph")


def test_a_missing_run_directory_is_refused(tmp_path):
    with pytest.raises((FileNotFoundError, ValueError)):
        list(PipelineRunCorpus(tmp_path / "nope", subset="physics").units())


def test_units_carry_the_page_number(chandra_run):
    pages = {u.page for u in PipelineRunCorpus(chandra_run, subset="physics").units()}
    assert pages == {"0", "1"}


def test_empty_pages_are_dropped(tmp_path):
    doc = ("pdfs/D.pdf", [block("/page/0/Text/0", 0, 0, "Text", "   "),
                          block("/page/1/Text/0", 1, 0, "Text", "réel")])
    units = list(PipelineRunCorpus(write_run(tmp_path, "c", CHANDRA, [doc]), subset="physics").units())
    assert [u.doc_id for u in units] == ["physics::D#page=1"]


def test_the_same_parser_over_two_subsets_is_two_corpora(tmp_path):
    run = write_run(tmp_path, "c", CHANDRA, [DOC])
    a = PipelineRunCorpus(run, subset="physics").corpus_identity()
    b = PipelineRunCorpus(run, subset="pharmaceuticals").corpus_identity()
    assert a != b


def test_output_stage_documents_without_a_parser_block_are_refused(tmp_path):
    """The real shape: settings live in ingested/, output/ drops them."""
    run = tmp_path / "out" / "run1"
    (run / "documents").mkdir(parents=True)
    (run / "documents" / "d0.json").write_text(json.dumps(
        {"document": {"file_name": "pdfs/D.pdf"}, "content": {"blocks": [], "reading_order": []}}))
    with pytest.raises(ValueError, match="parser metadata"):
        PipelineRunCorpus(run, subset="physics").corpus_identity()


# -- content granularity: keep the structure blocks[].text flattens ------------

TABLE_HTML = ('<table border="1"><thead><tr><th>Milieu</th><th>Vitesse</th></tr></thead>'
              '<tbody><tr><td>Air</td><td>330</td></tr>'
              '<tr><td>Eau douce</td><td>1460</td></tr></tbody></table>')
TABLE_FLAT = "Milieu Vitesse Air 330 Eau douce 1460"

RICH = ("pdfs/Rich.pdf", [
    {**block("/page/0/Text/0", 0, 0, "Text", "corps du texte"), "html": "<p>corps du texte</p>"},
    {**block("/page/0/Table/1", 0, 1, "Table", TABLE_FLAT), "html": TABLE_HTML},
    {**block("/page/0/ListGroup/2", 0, 2, "ListGroup", "un deux"), "html": "<ul><li>un</li><li>deux</li></ul>"},
])


@pytest.fixture
def rich_run(tmp_path):
    return write_run(tmp_path, "rich", CHANDRA, [RICH])


def text_of(run, granularity):
    units = list(PipelineRunCorpus(run, subset="physics", granularity=granularity).units())
    return "\n".join(u.text for u in units)


def test_page_granularity_still_flattens_the_table(rich_run):
    """The frozen chandra_page baseline depends on this staying unchanged."""
    assert TABLE_FLAT in text_of(rich_run, "page")
    assert "<table" not in text_of(rich_run, "page")


def test_content_granularity_keeps_the_table_structure(rich_run):
    body = text_of(rich_run, "content")
    assert "<table" in body and "<td>Eau douce</td>" in body


def test_content_granularity_keeps_list_structure(rich_run):
    assert "<li>un</li>" in text_of(rich_run, "content")


def test_content_granularity_leaves_prose_as_plain_text(rich_run):
    """A <p> wrapper adds no structure and only pollutes the term statistics."""
    body = text_of(rich_run, "content")
    assert "corps du texte" in body and "<p>" not in body


def test_content_granularity_keeps_the_same_units(rich_run):
    page = {u.doc_id for u in PipelineRunCorpus(rich_run, subset="physics").units()}
    content = {u.doc_id for u in PipelineRunCorpus(rich_run, subset="physics",
                                                   granularity="content").units()}
    assert page == content


def test_content_granularity_falls_back_when_html_is_absent(chandra_run):
    body = text_of(chandra_run, "content")
    assert "Chapitre 1" in body


def test_content_granularity_is_a_different_corpus(rich_run):
    """Same parser, different projection: it must not reuse the page cache."""
    a = PipelineRunCorpus(rich_run, subset="physics", granularity="page").corpus_identity()
    b = PipelineRunCorpus(rich_run, subset="physics", granularity="content").corpus_identity()
    assert a != b


def test_content_granularity_allows_a_chunker(rich_run):
    source = PipelineRunCorpus(rich_run, subset="physics", granularity="content",
                               chunker="fixed_overlap", params={"n_words": 5, "overlap": 0})
    assert len(list(source.units())) > 1


def test_chunk_granularity_is_refused_not_silently_empty(chandra_run):
    """Real retrieval.items are main_text spans with no page; 0 units read as a valid run."""
    with pytest.raises(NotImplementedError, match="no page"):
        list(PipelineRunCorpus(chandra_run, subset="physics", granularity="chunk").units())
