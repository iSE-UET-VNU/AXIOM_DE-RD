import argparse
import hashlib
import json
import sys
from pathlib import Path

from ..bench import queries as load_queries
from . import pipeline, prep, services
from .config import MODES, doctor, get_config
from .services.base import console


def add_service_args(p):
    g = p.add_argument_group("model services")
    for name, modes in MODES.items():
        g.add_argument(f"--{name}", dest=f"{name}_mode", choices=modes, help=f"{name} mode (default: from .env, else cached)")
    g.add_argument("--colvec-model", help="ColVec checkpoint (default webAI-Official/webAI-ColVec1.1-8b)")
    g.add_argument("--colvec-8bit", action="store_true", help="load ColVec in 8-bit (fits smaller GPUs; different scores)")
    g.add_argument("--colvec-endpoint", help="ColVec /rerank server; implies --colvec endpoint")
    g.add_argument("--kdl-endpoint", help="running KDL vLLM server, e.g. http://host:8000/v1; implies --kdl vllm")
    g.add_argument("--chandra-endpoint", help="running Chandra vLLM server; implies --chandra vllm")
    g.add_argument("--ppocr-endpoint", help="PP-OCRv5 /ocr server; implies --ppocr endpoint")
    g.add_argument("--ppocr-python", help="python interpreter with paddleocr, if it cannot share this env with torch")
    g.add_argument("--all-local-gpu", action="store_true", help="colvec local, kdl vllm, chandra vllm, ppocr local")
    g.add_argument("--colvec-local", action="store_true", help=argparse.SUPPRESS)
    g.add_argument("--chandra-local", action="store_true", help=argparse.SUPPRESS)
    g.add_argument("--ppocr-local", action="store_true", help=argparse.SUPPRESS)


def overrides(args):
    keys = ["colvec_mode", "kdl_mode", "chandra_mode", "ppocr_mode", "colvec_model", "colvec_8bit", "colvec_endpoint",
            "kdl_endpoint", "chandra_endpoint", "ppocr_endpoint", "ppocr_python", "all_local_gpu", "colvec_local",
            "chandra_local", "ppocr_local"]
    return {k: getattr(args, k) for k in keys if hasattr(args, k)}


def read_pages(args):
    pages = list(args.pages or [])
    if args.pages_file:
        pages += [line.strip() for line in Path(args.pages_file).read_text().splitlines() if line.strip()]
    if not pages:
        raise SystemExit("give --pages or --pages-file")
    return pages


def call(args):
    cfg = get_config().apply(**overrides(args))
    service = services.create(args.service, cfg)
    with service:
        if args.service == "colvec":
            if args.verify:
                gate = pipeline.load_gate(args.gate)
                queries = load_queries(service.bench)
                qids = sorted(gate["gate"])[:args.verify]
                return service.verify([(q, queries[q]["query"], gate["gate"][q]) for q in qids])
            query = args.query or load_queries(service.bench)[args.qid]["query"]
            qid = args.qid or "custom::" + hashlib.sha1(query.encode()).hexdigest()[:10]
            scores, missing = service.score(qid, query, read_pages(args))
            return {"service": service.describe(), "qid": qid, "scores": scores, "missing": missing}
        pages = read_pages(args)
        text, missing = (service.ocr if args.service == "ppocr" else service.parse)(pages)
        return {"service": service.describe(), "text": text, "missing": missing}


def main():
    ap = argparse.ArgumentParser(prog="python -m ondemand.adaptive")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="gate -> k -> router -> per-model stages -> chunk&rank -> QA, resumable")
    p.add_argument("--tag", required=True)
    p.add_argument("--gate", default="gate_ppocrv5_ungated.json")
    p.add_argument("--pages", default="pages_ppocrv5.jsonl")
    p.add_argument("--k", default="fixed:20")
    p.add_argument("--router", default="rule")
    p.add_argument("--escalate", choices=("none", "jev"), default="none",
                   help="after light's chunks are ranked, ask Jev whether to re-route that query to visual")
    p.add_argument("--refine-k", default="fixed:10")
    p.add_argument("--branch-arg", action="append", default=[], metavar="BRANCH=SPEC")
    p.add_argument("--chunk", default="fixed:512:128")
    p.add_argument("--ranker", default="hybrid")
    p.add_argument("--top-chunks", type=int, default=10)
    p.add_argument("--cache", default="te3s_ppocr")
    p.add_argument("--store", default="kdl")
    p.add_argument("--limit", type=int)
    p.add_argument("--qa", action="store_true")
    p.add_argument("--allow-api", action="store_true")
    p.add_argument("--exclude-flawed", action="store_true")
    p.add_argument("--resume", action="store_true", help="continue results/<tag>, skipping finished queries")
    p.add_argument("--keep-loaded", action="store_true", help="keep every model resident (only on large GPUs)")
    add_service_args(p)

    p = sub.add_parser("query", help="one benchmark query id or free-text query")
    p.add_argument("query_text")
    p.add_argument("--gate", default="gate_ppocrv5_ungated.json")
    p.add_argument("--pages", default="pages_ppocrv5.jsonl")
    p.add_argument("--router", default="rule")
    p.add_argument("--branch", choices=("light", "enrich", "visual"), help="skip the router")
    p.add_argument("--k", default="fixed:20")
    p.add_argument("--ranker", default="hybrid")
    p.add_argument("--store", default="kdl")
    p.add_argument("--allow-api", action="store_true")
    p.add_argument("--keep-loaded", action="store_true")
    p.add_argument("--json", action="store_true", help="print the full result as JSON")
    add_service_args(p)

    p = sub.add_parser("call", help="call one model service directly and print JSON")
    p.add_argument("service", choices=tuple(services.SERVICES))
    p.add_argument("--pages", nargs="*", help="page ids, e.g. 'vidore_physics::doc#page=3'")
    p.add_argument("--pages-file", help="file with one page id per line")
    p.add_argument("--qid", help="colvec: benchmark query id")
    p.add_argument("--query", help="colvec: free-text query")
    p.add_argument("--verify", type=int, metavar="N", help="colvec: rescore N gate queries, compare with the corpus matrix")
    p.add_argument("--gate", default="gate_ppocrv5_ungated.json")
    p.add_argument("--out", help="write the JSON here instead of stdout")
    add_service_args(p)

    p = sub.add_parser("prep", help="one-time light prep: pdf-inspector -> PP-OCRv5 -> pages file -> gate")
    p.add_argument("--pages-name", default="pages_ppocrv5_live.jsonl")
    p.add_argument("--gate-name", default="gate_ppocrv5_live.json")
    p.add_argument("--cache", default="te3s_ppocr")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--allow-api", action="store_true", help="embed uncached page chunks for the gate (te3-small)")
    p.add_argument("--overwrite", action="store_true")
    add_service_args(p)

    sub.add_parser("doctor", help="hardware, packages, service modes, endpoints and artifacts")
    sub.add_parser("status", help="same as doctor")
    sub.add_parser("list", help="registered components")
    p = sub.add_parser("artifacts", help="cached artifacts for this bundle")
    p.add_argument("--pages", default="pages_ppocrv5.jsonl")

    args = ap.parse_args()
    if args.cmd == "run":
        pipeline.run(args.tag, args.gate, k=args.k, router=args.router, escalate=args.escalate,
                     branch_args=dict(item.split("=", 1) for item in args.branch_arg), refine_k=args.refine_k,
                     chunker=args.chunk, ranker=args.ranker, top_chunks=args.top_chunks, pages_name=args.pages,
                     cache_name=args.cache, store=args.store, allow_api=args.allow_api, limit=args.limit,
                     with_qa=args.qa, exclude_flawed=args.exclude_flawed, resume=args.resume,
                     keep_loaded=args.keep_loaded, **overrides(args))
    elif args.cmd == "query":
        pipe = pipeline.AdaptivePipeline(gate_name=args.gate, k=args.k, router=args.router, pages_name=args.pages,
                                         ranker=args.ranker, store=args.store, allow_api=args.allow_api,
                                         keep_loaded=args.keep_loaded, **overrides(args))
        try:
            res = pipe.query(args.query_text, branch=args.branch)
        finally:
            pipe.close()
        if args.json:
            res["passages"] = [{"page_id": p.page_id, "rank": p.rank, "text": p.text} for p in res["passages"]]
            print(json.dumps(res, ensure_ascii=False, indent=1))
            return
        print("=" * 60)
        print(f"Query:  {res['query']}")
        print(f"Branch: {res['branch']} (k={res['k']})")
        print(f"Route:  {res['route_notes']}")
        print(f"Cost:   {res['cost']}")
        print(f"Ranked pages ({len(res['passages'])}):")
        for p in res["passages"][:5]:
            print(f"  #{p.rank}: {p.page_id}")
        print(f"\nTop chunks ({len(res['chunks'])}):")
        for i, c in enumerate(res["chunks"][:3], 1):
            print(f"  [{i}] ({c['page_id']}) {c['text'].replace(chr(10), ' ')[:100]}...")
        if res["fallbacks"]:
            print(f"\nFALLBACKS USED: {res['fallbacks']}")
        print("=" * 60)
    elif args.cmd == "call":
        console()
        try:
            result = call(args)
        finally:
            services.close_all()
        text = json.dumps(result, ensure_ascii=False, indent=1)
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
            print(args.out)
        else:
            sys.stdout.write(text + "\n")
    elif args.cmd == "prep":
        print(json.dumps(prep.run(pages_name=args.pages_name, gate_name=args.gate_name, cache_name=args.cache,
                                  allow_api=args.allow_api, workers=args.workers, overwrite=args.overwrite,
                                  **overrides(args)), indent=1))
    elif args.cmd in ("status", "doctor"):
        doctor()
    elif args.cmd == "list":
        print(json.dumps(pipeline.available(), indent=1))
    elif args.cmd == "artifacts":
        print(json.dumps(pipeline.artifacts(pages_name=args.pages), indent=1))


if __name__ == "__main__":
    main()
