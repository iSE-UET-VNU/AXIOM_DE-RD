"""Oracle must be an ordinary run, or the ceiling is scored by different code.

Oracle context was built inside a one-off script while retrieved context came
from the harness. Two paths to the same measurement drift, and the Oracle-minus-
retrieved delta is the comparable part of every end-to-end table we report.
"""

from __future__ import annotations

from src.evaluation.run_retrieval import oracle_records
from src.retrieval.protocol import ChunkRecord


class Index:
    index_id = "ix"

    def __init__(self, units):
        self.records = [ChunkRecord(chunk_id=u, doc_id=u, text=t, page="0", meta={})
                        for u, t in units.items()]


class Bench:
    def __init__(self, gold):
        self._gold = gold

    def qrels(self):
        return self._gold


class Q:
    def __init__(self, qid, query="q"):
        self.qid, self.query = qid, query


UNITS = {"physics::A#page=0": "page zero", "physics::A#page=1": "page un",
         "physics::B#page=0": "autre page"}


def test_gold_units_become_the_ranking():
    records = oracle_records(Bench({"q1": {"physics::A#page=0": 1}}), [Q("q1")], Index(UNITS), 10)
    assert [c["chunk_id"] for c in records[0].chunks] == ["physics::A#page=0"]


def test_the_text_comes_from_the_index_not_an_empty_string():
    """Empty context would score as a generator failure, not a corpus gap."""
    records = oracle_records(Bench({"q1": {"physics::A#page=1": 1}}), [Q("q1")], Index(UNITS), 10)
    assert records[0].chunks[0]["text"] == "page un"


def test_higher_graded_gold_ranks_first():
    gold = {"q1": {"physics::A#page=0": 1, "physics::A#page=1": 2}}
    records = oracle_records(Bench(gold), [Q("q1")], Index(UNITS), 10)
    assert [c["chunk_id"] for c in records[0].chunks][0] == "physics::A#page=1"


def test_a_gold_unit_missing_from_the_index_is_dropped_not_faked():
    gold = {"q1": {"physics::A#page=0": 1, "physics::MISSING#page=9": 1}}
    records = oracle_records(Bench(gold), [Q("q1")], Index(UNITS), 10)
    assert [c["chunk_id"] for c in records[0].chunks] == ["physics::A#page=0"]


def test_depth_bounds_the_ranking():
    gold = {"q1": {u: 1 for u in UNITS}}
    records = oracle_records(Bench(gold), [Q("q1")], Index(UNITS), 2)
    assert len(records[0].chunks) == 2


def test_a_question_without_gold_still_yields_a_record():
    """A dropped row would silently shrink the denominator against other arms."""
    records = oracle_records(Bench({}), [Q("q1")], Index(UNITS), 10)
    assert len(records) == 1 and records[0].chunks == []


def test_records_are_tagged_as_the_oracle_arm():
    records = oracle_records(Bench({"q1": {"physics::A#page=0": 1}}), [Q("q1")], Index(UNITS), 10)
    assert records[0].retriever_id == "oracle"


def test_ranks_are_one_based_and_contiguous():
    gold = {"q1": {u: 1 for u in UNITS}}
    records = oracle_records(Bench(gold), [Q("q1")], Index(UNITS), 10)
    assert [c["rank"] for c in records[0].chunks] == [1, 2, 3]
