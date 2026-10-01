import argparse
import json

from . import arms, bundle, chunks, gate, light_ocr, light_prep, light_retrieval, ocr_bundle, qa, report


def main():
    ap = argparse.ArgumentParser(prog="python -m ondemand")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("light-prep")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--ocr", choices=("tesseract", "none"), default="tesseract")
    p.add_argument("--name")
    p = sub.add_parser("ocr-bundle")
    p.add_argument("--select", choices=tuple(ocr_bundle.SELECTORS), default="weak")
    p.add_argument("--name")
    p = sub.add_parser("merge-ocr")
    p.add_argument("--results", required=True)
    p.add_argument("--base", choices=("sparse", "inspector"), default="inspector")
    p.add_argument("--pages", default="pages_inspector.jsonl")
    p.add_argument("--name")
    p = sub.add_parser("light-retrieval")
    p.add_argument("--pages", default="pages_ocr_ppocr.jsonl")
    p.add_argument("--name")
    p.add_argument("--signals", choices=tuple(light_retrieval.POLICIES), default="all")
    p.add_argument("--analyzer", choices=tuple(light_retrieval.ANALYZERS), default="enfr")
    p.add_argument("--file-mode", choices=("dense_pool", "bm25_blend"), default="dense_pool")
    p.add_argument("--file-k", type=int, default=light_retrieval.FILE_K)
    p.add_argument("--k", type=int, default=light_retrieval.PAGE_K)
    p.add_argument("--w-dense", type=float, default=light_retrieval.W_DENSE)
    p.add_argument("--parent", type=float, default=light_retrieval.PARENT)
    p.add_argument("--file-direct", type=float, default=light_retrieval.FILE_DIRECT)
    p.add_argument("--cache", default="te3s_ppocr")
    p.add_argument("--allow-api", action="store_true")
    sub.add_parser("gate").add_argument("--name", default="gate_k20_best.json")
    sub.add_parser("chunks")
    p = sub.add_parser("arms")
    p.add_argument("--tag", required=True)
    p.add_argument("--gate", default="gate_k20_best.json")
    p = sub.add_parser("qa")
    p.add_argument("--tag", required=True)
    p.add_argument("--arm", choices=("A", "B"), required=True)
    p = sub.add_parser("report")
    p.add_argument("--tag", required=True)
    p.add_argument("--gate", default="gate_k20_best.json")
    sub.add_parser("bundle").add_argument("--target", choices=("colvec", "kdl"), required=True)
    args = ap.parse_args()

    if args.cmd == "light-prep":
        print(light_prep.run(workers=args.workers, ocr=args.ocr, name=args.name))
    elif args.cmd == "ocr-bundle":
        print(ocr_bundle.build(select=args.select, name=args.name))
    elif args.cmd == "merge-ocr":
        if args.base == "inspector":
            print(light_ocr.merge_pure(results=args.results, pages_name=args.pages, name=args.name))
        else:
            print(light_ocr.merge(results=args.results))
    elif args.cmd == "light-retrieval":
        print(light_retrieval.build(pages_name=args.pages, name=args.name, signals=args.signals,
                                    analyzer=args.analyzer, file_mode=args.file_mode, file_k=args.file_k,
                                    page_k=args.k, w_dense=args.w_dense, parent=args.parent,
                                    file_direct=args.file_direct, cache_name=args.cache, allow_api=args.allow_api))
    elif args.cmd == "gate":
        print(gate.build(name=args.name))
    elif args.cmd == "chunks":
        print(chunks.run())
    elif args.cmd == "arms":
        print(json.dumps({k: v["all"] for k, v in arms.run(args.tag, args.gate)["arms"].items()}, indent=1))
    elif args.cmd == "qa":
        print(json.dumps(qa.run(args.tag, args.arm), indent=1))
    elif args.cmd == "report":
        print(report.sheet(args.tag, args.gate))
    elif args.cmd == "bundle":
        print(bundle.build(args.target))


if __name__ == "__main__":
    main()
