from __future__ import annotations

from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from run_general_pagewise_ohr import parse_decision, request_hash  # noqa: E402


def test_parse_stop() -> None:
    assert parse_decision('{"decision":"STOP","reason":"enough"}') == ("STOP", "enough", "")


def test_parse_failure_continues() -> None:
    decision, _, error = parse_decision("not json")
    assert decision == "CONTINUE"
    assert error


def test_request_hash_stable() -> None:
    messages = [{"role": "user", "content": "x"}]
    assert request_hash("m", messages) == request_hash("m", messages)
