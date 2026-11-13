import argparse
import json
from collections import Counter
from pathlib import Path

from ..bench import FLAWED_QIDS, load_jsonl, work
from .llm import EndpointChat, OpenRouterChat
from .qa_reader import answer_prompt


def prompts(tag, n):
    records = [r for r in load_jsonl(work("results", tag) / "per_query.jsonl") if r["query_id"] not in FLAWED_QIDS]
    records.sort(key=lambda r: r["query_id"])
    return [(r["query_id"], answer_prompt(r["query"], [c["text"] for c in r["chunks"]])) for r in records[:n]]


def main():
    ap = argparse.ArgumentParser(prog="python -m ondemand.adaptive.determinism_check")
    ap.add_argument("--model", required=True)
    ap.add_argument("--provider", help="OpenRouter provider slug to pin, e.g. novita")
    ap.add_argument("--quantization", help="OpenRouter quantization filter, e.g. bf16")
    ap.add_argument("--endpoint", help="OpenAI-compatible /v1 URL instead of OpenRouter (vLLM, DashScope)")
    ap.add_argument("--tag", default="xk_evie_v2_stop10_214", help="run whose top-10 chunks make the prompts")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--out", default="determinism_check.json")
    args = ap.parse_args()

    if args.endpoint:
        client = EndpointChat(args.endpoint, args.model, args.max_tokens)
    else:
        provider = {"order": [args.provider], "allow_fallbacks": False, "require_parameters": True}
        if args.quantization:
            provider["quantizations"] = [args.quantization]
        client = OpenRouterChat(args.model, provider if args.provider else None, args.max_tokens)

    rows, served, tokens = [], Counter(), Counter()
    for qid, prompt in prompts(args.tag, args.n):
        answers = []
        for _ in range(args.repeats):
            reply = client(prompt)
            answers.append(reply["text"])
            served[str(reply["served_provider"])] += 1
            for key, value in (reply.get("usage") or {}).items():
                if isinstance(value, (int, float)):
                    tokens[key] += value
        identical = len(set(answers)) == 1
        rows.append({"qid": qid, "identical": identical, "distinct_answers": len(set(answers)), "answers": answers})
        print(f"{qid}: {'identical' if identical else f'{len(set(answers))} different answers'}", flush=True)

    report = {"model": args.model, "settings": client.settings(), "prompts": len(rows), "repeats": args.repeats,
              "identical_prompts": sum(r["identical"] for r in rows), "served_by": dict(served),
              "usage_totals": dict(tokens), "rows": rows}
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("model", "prompts", "repeats", "identical_prompts", "served_by",
                                             "usage_totals")}, indent=1))
    print(f"full answers in {args.out}")


if __name__ == "__main__":
    main()
