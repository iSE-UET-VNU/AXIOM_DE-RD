import argparse

from . import bundle, light_prep


def main():
    ap = argparse.ArgumentParser(prog="python -m ondemand")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("light-prep", help="pdf-inspector text layer over the corpus (PP-OCRv5 runs in adaptive prep)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--ocr", choices=("tesseract", "none"), default="none")
    p.add_argument("--name")
    sub.add_parser("bundle", help="zip code + benchmark + cached phase artifacts for a GPU machine") \
        .add_argument("--target", choices=("adaptive", "handoff"), default="adaptive")
    args = ap.parse_args()
    if args.cmd == "light-prep":
        print(light_prep.run(workers=args.workers, ocr=args.ocr, name=args.name))
    elif args.cmd == "bundle":
        print(bundle.build(args.target))


if __name__ == "__main__":
    main()
