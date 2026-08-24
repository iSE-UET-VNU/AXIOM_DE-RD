"""Browse the actual rows behind a ViDoRe number.

    python inspect.py retrieval computer_science bm25 --n 5 --only miss
    python inspect.py generation computer_science --n 5 --only "Partially Correct"
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "data/benchmark/runs"
GEN = Path(__file__).resolve().parents[2] / "data/benchmark/vidore_v3/results" / "english_oracle_generation"


def run_dir(subset):
    hits = sorted(RUNS.glob(f"vidore_v3.*.{subset}-english.*"))
    dirs = [p for p in hits if p.is_dir()]
    if not dirs:
        raise SystemExit(f"no run dir for {subset}; have: {[p.name for p in RUNS.iterdir() if p.is_dir()]}")
    return dirs[-1]


def retrieval(args):
    from src.evaluation.benchmarks import load

    bench = load("vidore_v3", subset=args.subset, language="english")
    qrels = bench.qrels()
    directory = run_dir(args.subset)
    files = sorted(directory.glob(f"{args.arm}__*.jsonl"))
    if not files:
        raise SystemExit(f"no {args.arm} arm in {directory.name}; have {[f.name.split('__')[0] for f in directory.iterdir()]}")
    print(f"# {directory.name} / {files[-1].name}\n")

    shown = 0
    with files[-1].open(encoding="utf-8") as handle:
        for line in handle:
            if shown >= args.n:
                break
            record = json.loads(line)
            gold = qrels.get(record["qid"]) or {}
            top = record["chunks"][: args.k]
            found = [c for c in top if c["doc_id"] in gold]
            if args.only == "miss" and found:
                continue
            if args.only == "hit" and not found:
                continue
            shown += 1
            print(f"=== {record['qid']}  hit@{args.k}: {len(found)}/{len(gold)}")
            print(f"    Q: {record['query']}")
            print(f"    gold ({len(gold)}):")
            for unit, gain in gold.items():
                rank = next((c["rank"] for c in record["chunks"] if c["doc_id"] == unit), None)
                where = f"rank {rank}" if rank is not None else "NOT RETRIEVED"
                print(f"      gain={gain}  {unit}   -> {where}")
            print(f"    top {args.k} retrieved:")
            for chunk in top:
                mark = "*" if chunk["doc_id"] in gold else " "
                print(f"     {mark}{chunk['rank']:>3}  {chunk['score']:>9.3f}  {chunk['doc_id']}")
                if args.text:
                    print(f"          {chunk['text'][:args.text].replace(chr(10), ' ')}")
            print()


def generation(args):
    path = GEN / f"{args.subset}.json"
    if not path.exists():
        raise SystemExit(f"{path} not written yet; have {[p.name for p in GEN.iterdir()]}")
    rows = json.loads(path.read_text())
    labels = {}
    for row in rows:
        labels[row["judgment"]] = labels.get(row["judgment"], 0) + 1
    print(f"# {path}  n={len(rows)}  {labels}\n")
    shown = 0
    for row in rows:
        if shown >= args.n:
            break
        if args.only and row["judgment"] != args.only:
            continue
        shown += 1
        print(f"=== {row['qid']}  [{row['judgment']}]  gold_pages={row['gold_pages']}  ctx={row['context_chars']} chars")
        print(f"    Q       : {row['query']}")
        print(f"    gold    : {row['gold_answer']}")
        print(f"    model   : {row['answer']}")
        if row.get("error"):
            print(f"    ERROR   : {row['error']}")
        print()


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
sub = parser.add_subparsers(dest="mode", required=True)

r = sub.add_parser("retrieval")
r.add_argument("subset")
r.add_argument("arm", choices=["bm25", "dense", "rrf"])
r.add_argument("--n", type=int, default=5)
r.add_argument("--k", type=int, default=10)
r.add_argument("--only", choices=["hit", "miss"])
r.add_argument("--text", type=int, default=0, help="chars of retrieved page text to print")
r.set_defaults(func=retrieval)

g = sub.add_parser("generation")
g.add_argument("subset")
g.add_argument("--n", type=int, default=5)
g.add_argument("--only", choices=["Correct", "Partially Correct", "Incorrect"])
g.set_defaults(func=generation)

args = parser.parse_args()
args.func(args)
