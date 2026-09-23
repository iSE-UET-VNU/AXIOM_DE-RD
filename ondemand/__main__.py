import argparse
import json

from . import arms, bundle, chunks, gate, light_prep, qa, report


def main():
    ap = argparse.ArgumentParser(prog="python -m ondemand")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("light-prep").add_argument("--workers", type=int, default=8)
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
        print(light_prep.run(workers=args.workers))
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
