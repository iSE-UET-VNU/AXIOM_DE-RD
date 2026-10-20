from __future__ import annotations

import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from run_adaptive_llm_stopping import (  # noqa: E402
    accepted_stop,
    next_state,
    normalize_result,
    parse_json_object,
    stable_hash,
)


def test_parse_fenced_json() -> None:
    parsed = parse_json_object('```json\n{"decision":"CONTINUE"}\n```')
    assert parsed["decision"] == "CONTINUE"


def test_stop_requires_seen_citation_and_empty_missing() -> None:
    result = normalize_result({
        "decision": "STOP",
        "confidence": 0.9,
        "provisional_answer": "Grounded answer",
        "cited_page_ids": ["p1"],
        "resolved_claims": [{"claim": "x", "page_ids": ["p1"]}],
        "missing_information": [],
        "safe_to_use_current_k": True,
    }, "hard-only")
    assert accepted_stop(result, {"p1"}, "hard-only") == (True, [])
    allowed, reasons = accepted_stop(result, {"p2"}, "hard-only")
    assert not allowed and "invalid_citation" in reasons


def test_low_confidence_stop_is_rejected() -> None:
    result = normalize_result({
        "decision": "STOP", "confidence": 0.84,
        "provisional_answer": "x", "cited_page_ids": ["p1"],
        "missing_information": [],
    }, "general")
    allowed, reasons = accepted_stop(result, {"p1"}, "general")
    assert not allowed and "confidence_below_0.85" in reasons


def test_state_does_not_contain_raw_pages() -> None:
    result = normalize_result({
        "decision": "CONTINUE", "confidence": 0.5,
        "provisional_answer": "draft", "cited_page_ids": ["p1"],
        "resolved_claims": [], "missing_information": ["value"],
        "state_summary": "short",
    }, "general")
    state = next_state(result, "general")
    assert "pages" not in json.dumps(state)


def test_hash_is_order_stable() -> None:
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})
