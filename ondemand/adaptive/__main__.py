import argparse
import json

from . import pipeline


def main():
    ap = argparse.ArgumentParser(prog="python -m ondemand.adaptive")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="gate -> adaptive k -> router -> branch -> chunk&rank -> QA")
    p.add_argument("--tag", required=True)
    p.add_argument("--gate", default="gate_locked_all.json")
    p.add_argument("--pages", default="pages_ocr_ppocr.jsonl")
    p.add_argument("--k", default="fixed:20")
    p.add_argument("--router", default="fixed:visual")
    p.add_argument("--refine-k", default="fixed:10")
    p.add_argument("--branch-arg", action="append", default=[], metavar="BRANCH=SPEC")
    p.add_argument("--chunk", default="fixed:512:128")
    p.add_argument("--ranker", default="hybrid")
    p.add_argument("--top-chunks", type=int, default=10)
    p.add_argument("--cache", default="te3s_ppocr")
    p.add_argument("--store", help="precomputed chunk vectors under data/work/<fp>/embeddings/<name>, e.g. kdl")
    p.add_argument("--limit", type=int)
    p.add_argument("--qa", action="store_true")
    p.add_argument("--allow-api", action="store_true")

    sub.add_parser("list", help="show the registered components")
    p = sub.add_parser("artifacts", help="show which cached artifacts this bundle has")
    p.add_argument("--pages", default="pages_ocr_ppocr.jsonl")

    args = ap.parse_args()
    if args.cmd == "run":
        branch_args = dict(item.split("=", 1) for item in args.branch_arg)
        pipeline.run(args.tag, args.gate, k=args.k, router=args.router, branch_args=branch_args,
                     refine_k=args.refine_k, chunker=args.chunk, ranker=args.ranker,
                     top_chunks=args.top_chunks, pages_name=args.pages, cache_name=args.cache, store=args.store,
                     allow_api=args.allow_api, limit=args.limit, with_qa=args.qa)
    elif args.cmd == "list":
        print(json.dumps(pipeline.available(), indent=1))
    elif args.cmd == "artifacts":
        print(json.dumps(pipeline.artifacts(pages_name=args.pages), indent=1))


if __name__ == "__main__":
    main()
