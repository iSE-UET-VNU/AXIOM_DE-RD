import argparse
import json

from . import pipeline
from .config import doctor


def main():
    ap = argparse.ArgumentParser(prog="python -m ondemand.adaptive")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # 1. Full pipeline benchmark run
    p = sub.add_parser("run", help="gate -> adaptive k -> router -> branch -> chunk&rank -> QA")
    p.add_argument("--tag", required=True)
    p.add_argument("--gate", default="gate_ppocrv5_all.json")
    p.add_argument("--pages", default="pages_ppocrv5.jsonl")
    p.add_argument("--k", default="fixed:20")
    p.add_argument("--router", default="rule")
    p.add_argument("--refine-k", default="fixed:10")
    p.add_argument("--branch-arg", action="append", default=[], metavar="BRANCH=SPEC")
    p.add_argument("--chunk", default="fixed:512:128")
    p.add_argument("--ranker", default="hybrid")
    p.add_argument("--top-chunks", type=int, default=10)
    p.add_argument("--cache", default="te3s_ppocr")
    p.add_argument("--store", default="kdl", help="precomputed chunk vectors under data/work/<fp>/embeddings/<name>, e.g. kdl")
    p.add_argument("--limit", type=int)
    p.add_argument("--qa", action="store_true")
    p.add_argument("--allow-api", action="store_true")
    p.add_argument("--exclude-flawed", action="store_true", help="exclude the 2 confirmed-flawed benchmark reference questions")
    p.add_argument("--chandra-endpoint", help="override Chandra OCR endpoint, e.g. http://localhost:8000/v1")
    p.add_argument("--colpali-endpoint", help="live ColPali endpoint; only used if no ColVec cache is found (default scorer)")

    # 2. Interactive / Single Query Tool Calling
    p = sub.add_parser("query", help="run adaptive retrieval on a single query or benchmark query ID")
    p.add_argument("query_text", help="query string or query ID (e.g. vidore_physics::326)")
    p.add_argument("--gate", default="gate_ppocrv5_all.json")
    p.add_argument("--pages", default="pages_ppocrv5.jsonl")
    p.add_argument("--router", default="rule")
    p.add_argument("--k", default="fixed:20")
    p.add_argument("--chandra-endpoint", help="override Chandra endpoint")
    p.add_argument("--colpali-endpoint", help="live ColPali endpoint; only used if no ColVec cache is found (default scorer)")
    p.add_argument("--ranker", default="hybrid", help="chunk ranker: hybrid, page_order, lexical")
    p.add_argument("--store", default="kdl", help="precomputed chunk vectors, e.g. kdl")
    p.add_argument("--allow-api", action="store_true", help="permit embedding calls for uncached texts")

    # 3. Diagnostics and introspection
    sub.add_parser("status", help="check endpoints, API keys, and local cache status")
    sub.add_parser("doctor", help="diagnose environment, endpoints, and tool health")
    sub.add_parser("list", help="show the registered components")
    p = sub.add_parser("artifacts", help="show which cached artifacts this bundle has")
    p.add_argument("--pages", default="pages_ppocrv5.jsonl")

    args = ap.parse_args()
    if args.cmd == "run":
        branch_args = dict(item.split("=", 1) for item in args.branch_arg)
        pipeline.run(args.tag, args.gate, k=args.k, router=args.router, branch_args=branch_args,
                     refine_k=args.refine_k, chunker=args.chunk, ranker=args.ranker,
                     top_chunks=args.top_chunks, pages_name=args.pages, cache_name=args.cache, store=args.store,
                     allow_api=args.allow_api, limit=args.limit, with_qa=args.qa,
                     exclude_flawed=args.exclude_flawed, chandra_endpoint=args.chandra_endpoint,
                     colpali_endpoint=args.colpali_endpoint)
    elif args.cmd == "query":
        pipe = pipeline.AdaptivePipeline(gate_name=args.gate, k=args.k, router=args.router,
                                        pages_name=args.pages, ranker=args.ranker, store=args.store,
                                        chandra_endpoint=args.chandra_endpoint,
                                        colpali_endpoint=args.colpali_endpoint, allow_api=args.allow_api)
        res = pipe.query(args.query_text)
        print("=" * 60)
        print(f"Query:  {res['query']}")
        print(f"Branch: {res['branch']} (k={res['k']})")
        print(f"Route:  {res['route_notes']}")
        print(f"Ranked Pages ({len(res['passages'])}):")
        for p in res['passages'][:5]:
            print(f"  #{p.rank}: {p.page_id}")
        print(f"\nTop Chunks ({len(res['chunks'])}):")
        for i, c in enumerate(res['chunks'][:3], 1):
            sample = c['text'].replace('\n', ' ')[:100]
            print(f"  [{i}] ({c['page_id']}) {sample}...")
        print("=" * 60)
    elif args.cmd in ("status", "doctor"):
        doctor()
    elif args.cmd == "list":
        print(json.dumps(pipeline.available(), indent=1))
    elif args.cmd == "artifacts":
        print(json.dumps(pipeline.artifacts(pages_name=args.pages), indent=1))


if __name__ == "__main__":
    main()
