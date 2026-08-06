"""Build the evaluation set from the question sheet.

    python -m research.harness.build_evalset [--questions CSV --lake DIR --out DIR]

Writes ``questions.jsonl`` (every question, with resolution state) plus a
coverage report. Selection of scored subsets happens at run time from the
``resolvable`` / ``is_text_only`` flags, so nothing is lost here.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import argparse
import json

from .evalset import load_questions, write_jsonl
from .lake import Lake

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LAKE = PROJECT_ROOT / "[iSE Summer Challenge 2026] Data Lake"
DEFAULT_QUESTIONS = PROJECT_ROOT / "[iSE Summer Challenge 2026] Questions - Q&A(1).csv"
DEFAULT_OUT = PROJECT_ROOT / "data" / "benchmark"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS))
    parser.add_argument("--lake", default=str(DEFAULT_LAKE))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    lake = Lake.index(args.lake)
    questions = load_questions(args.questions, lake)
    out_dir = Path(args.out)
    write_jsonl(out_dir / "questions.jsonl", questions)

    resolvable = [q for q in questions if q.resolvable]
    text_only = [q for q in questions if q.is_text_only]
    refs = sum(len(q.evidence_refs) for q in questions)
    unresolved = sum(len(q.unresolved_refs) for q in questions)

    report = {
        "lake_files": len(lake.by_rel),
        "questions": len(questions),
        "evidence_refs": refs,
        "evidence_resolved": refs - unresolved,
        "questions_resolvable": len(resolvable),
        "questions_text_only": len(text_only),
        "modality_mix": dict(Counter(m for q in resolvable for m in q.modalities).most_common()),
        "text_only_by_level": dict(Counter(q.level for q in text_only)),
        "text_only_by_answer_type": dict(Counter(q.answer_type for q in text_only)),
        "gold_docs_text_only": len({d for q in text_only for d in q.gold_doc_ids}),
        # How each reference was matched. Anything other than exact/case was
        # inferred from a malformed reference, so a reader can audit the labels
        # rather than trust them.
        "resolution_strategies": dict(
            Counter(s for q in questions for s in q.resolution.values()).most_common()
        ),
        "questions_with_any_of": sum(1 for q in resolvable if q.gold_any_of),
        "questions_multi_gold": sum(
            1 for q in resolvable if len(q.gold_doc_ids) + len(q.gold_any_of) > 1
        ),
        "unresolved_examples": [
            {"qid": q.qid, "refs": q.unresolved_refs} for q in questions if q.unresolved_refs
        ][:15],
    }
    (out_dir / "evalset_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"lake files            : {report['lake_files']}")
    print(f"questions             : {report['questions']}")
    print(f"evidence resolved     : {report['evidence_resolved']}/{report['evidence_refs']}")
    print(f"questions resolvable  : {report['questions_resolvable']}")
    print(f"questions TEXT-ONLY   : {report['questions_text_only']}  <- benchmark subset")
    print(f"gold docs (text-only) : {report['gold_docs_text_only']}")
    print(f"modality mix          : {report['modality_mix']}")
    print(f"text-only by level    : {report['text_only_by_level']}")
    print(f"artifacts             : {out_dir}")


if __name__ == "__main__":
    main()
