import json

from ..bench import BENCH, work
from ..openrouter import decide
from .qa import cache_path, cached_or_refuse

JEV = "typesafe/jev-1.13-20260917"
JEV_QUESTIONS = {
    "all_covered": {"type": "noul", "instructions": "Probability that the excerpts completely and directly answer every part of the question, with no required value missing, cut off, only referenced elsewhere, or hidden in a figure, chart, table or formula that the text does not reproduce."},
    "route_action": {"type": "choice", "instructions": "Given only these excerpts, which next step does answering the question need?",
                     "criteria": {"complete": "Nothing more is needed: the excerpts contain everything required to answer the question fully.",
                                  "bridge": "Some required information is referenced or continued elsewhere in the same documents (another table, section or page, or the rest of a list), and a targeted text search could find it.",
                                  "escalate": "Some required information is missing from the text: it sits in a figure, chart, table or formula that the text does not reproduce, or nothing in the excerpts points to it, so the page images must be inspected."}},
    "gap_severity": {"type": "score", "instructions": "How much of the evidence needed to answer the question is missing from the excerpts?",
                     "criteria": ["Complete: the text contains all evidence needed.",
                                  "Minor: a small detail is missing but the core answer is supported.",
                                  "Major: an important part of the answer is missing or only referenced.",
                                  "Critical: the core evidence is missing, or sits in a visual element or page the text does not reproduce."]}}


def excerpts(chunks):
    return "\n\n".join(f"[{i + 1}] {c.text}" for i, c in enumerate(chunks))


def jev_check(query, chunks, allow_api, bench=BENCH):
    state = {"question": query, "excerpts": excerpts(chunks)}
    key = f"{JEV}|{json.dumps(JEV_QUESTIONS, sort_keys=True)}\n{json.dumps(state, ensure_ascii=False, sort_keys=True)}"
    folder = work("qa_cache", "jev", bench=bench)
    return cached_or_refuse(folder, key, lambda: decide(JEV, state, JEV_QUESTIONS), allow_api, "the Jev escalation check")


def route_action(verdict):
    return (verdict.get("answers") or {}).get("route_action", {}).get("choice")
