import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .bench import BENCH, load_jsonl, work, write_jsonl
from .openrouter import _post
from .routing_score import at_rate

REPOS = {
    "retrieval-router": ("https://github.com/emrekuruu/retrieval-router", "6fca6dd498161d0a49820b1b0b45060b1e976306"),
    "UniversalRAG": ("https://github.com/wgcyeo/UniversalRAG", "d29ab3acaebfbaebdc7f095457076585bec88065"),
    "automix": ("https://github.com/automix-llm/automix", "531af3ee3c4e4f4a53da1ffbc91dec4d6bb73781"),
    "RankGPT": ("https://github.com/sunnweiwei/RankGPT", "0d62bc3855c7c118048a7c47c18e719b938e291a"),
}
VISUAL_CATEGORIES = {"table", "image"}
WORKERS = 8
SLM, LLM = "llama13b_f1", "llama70b_f1"


def repo_dir(name):
    return work("routing", "repos", bench=BENCH) / name


def slug(model):
    return re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")


def load_defs(sources, names):
    namespace, found = {}, set()
    for source in sources:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.Import) and all(a.name in sys.stdlib_module_names for a in node.names):
                exec(compile(ast.Module([node], []), "<repo>", "exec"), namespace)
                continue
            if isinstance(node, ast.FunctionDef) and node.name in names:
                found.add(node.name)
            elif isinstance(node, ast.Assign) and any(getattr(t, "id", None) in names for t in node.targets):
                found.update(t.id for t in node.targets if getattr(t, "id", None) in names)
            else:
                continue
            exec(compile(ast.Module([node], []), "<repo>", "exec"), namespace)
    missing = set(names) - found
    if missing:
        raise SystemExit(f"not found in repo source: {sorted(missing)}")
    return namespace


def notebook_cells(path):
    return ["".join(c["source"]) for c in json.loads(Path(path).read_text())["cells"] if c["cell_type"] == "code"]


def call(folder, body, allow_api):
    path = folder / (hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() + ".json")
    if path.exists():
        return json.loads(path.read_text())["text"]
    if not allow_api:
        raise SystemExit("this call is not cached (would call a live API); rerun with --allow-api to permit it")
    for _ in range(3):
        payload = _post("chat/completions", body)
        text = ((payload.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        if text.strip():
            path.write_text(json.dumps({"body": body, "text": text}, ensure_ascii=False))
            return text
    return ""


def parallel(fn, items):
    with ThreadPoolExecutor(WORKERS) as pool:
        return list(pool.map(fn, items))


def clone(args):
    for name, (url, commit) in REPOS.items():
        target = repo_dir(name)
        if not (target / ".git").exists():
            subprocess.run(["git", "clone", "-q", url, str(target)], check=True)
        subprocess.run(["git", "-C", str(target), "checkout", "-q", commit], check=True)
        print(name, commit, target)


def decision_dir(labels):
    return work("routing", "decisions", Path(labels).stem.removeprefix("labels_"), bench=BENCH)


def universalrag(args):
    repo = repo_dir("UniversalRAG")
    sys.path.insert(0, str(repo))
    from route.prompt import ROUTER_PROMPT
    defs = load_defs([(repo / "route/gpt/route_gpt.py").read_text()],
                     {"ALLOWED_CATEGORIES", "_clean_category_token", "normalize_retrieval"})
    labels = load_jsonl(work("routing", bench=BENCH) / args.labels)
    folder = work("routing", "cache", "universalrag", bench=BENCH)

    def one(row):
        body = {"model": args.model, "messages": [{"role": "user", "content": ROUTER_PROMPT.format(query=row["query"])}]}
        if args.seed is not None:
            body["seed"] = args.seed
        if args.temperature is not None:
            body["temperature"] = args.temperature
        raw = call(folder, body, args.allow_api)
        category = defs["normalize_retrieval"](raw)
        if category == "error":
            branch = args.on_error
        else:
            branch = "visual" if set(category.split("+")) & VISUAL_CATEGORIES else "light"
        return {"query_id": row["query_id"], "branch": branch, "category": category, "raw": raw}

    rows = parallel(one, labels)
    out = decision_dir(args.labels) / f"universalrag-gpt_{slug(args.model)}.jsonl"
    write_jsonl(out, rows)
    counts = pd.Series([r["category"] for r in rows]).value_counts().to_dict()
    counts["empty_reply"] = sum(not r["raw"].strip() for r in rows)
    print(json.dumps({"out": str(out), "categories": counts,
                      "visual": sum(r["branch"] == "visual" for r in rows)}))


def verify(args):
    cells = notebook_cells(repo_dir("automix") / "colabs/Step2_SelfVerify.ipynb")
    defs = load_defs(cells, {"verifier_prompt", "compute_fraction_correct"})
    records = {r["query_id"]: r for r in load_jsonl(work("results", args.light, bench=BENCH) / "per_query.jsonl")}
    qa_path = work("results", args.light, bench=BENCH) / "qa.jsonl"
    if not qa_path.exists():
        raise SystemExit(f"{qa_path} is missing; generate the light answers first (Run 2)")
    answers = {r["query_id"]: r["generated_answer"] for r in load_jsonl(qa_path)}
    labels = load_jsonl(work("routing", bench=BENCH) / args.labels)
    folder = work("routing", "cache", "automix_verify", bench=BENCH)

    def one(row):
        record = records[row["query_id"]]
        context = "\n\n".join(f"[{i + 1}] {c['text']}" for i, c in enumerate(record["chunks"]))
        prompt = defs["verifier_prompt"].format(context=context, question=record["query"],
                                                generated_answer=answers[row["query_id"]].strip())
        samples = [call(folder, {"model": args.model, "messages": [{"role": "user", "content": prompt}],
                                 "temperature": args.temperature, "max_tokens": args.max_tokens, "stop": ["---"],
                                 "seed": s}, args.allow_api) for s in range(args.samples)]
        return {"query_id": row["query_id"], "p_ver": defs["compute_fraction_correct"](samples), "samples": samples}

    rows = parallel(one, labels)
    out = work("routing", bench=BENCH) / f"automix_verify_{slug(args.model)}.jsonl"
    write_jsonl(out, rows)
    samples = [s for r in rows for s in r["samples"]]
    print(json.dumps({"out": str(out), "mean_p_ver": round(float(np.mean([r["p_ver"] for r in rows])), 4),
                      "samples": len(samples), "empty": sum(not s.strip() for s in samples),
                      "no_decision": sum("the ai generated answer is" not in s.lower() for s in samples),
                      "queries_without_any_decision": sum(
                          all("the ai generated answer is" not in s.lower() for s in r["samples"]) for r in rows)}))


def metric_columns(labels, metric, light_tag, visual_tag):
    if metric != "qa":
        return ({r["query_id"]: r[f"light_{metric}@10"] / 100 for r in labels},
                {r["query_id"]: r[f"visual_{metric}@10"] / 100 for r in labels})
    out = []
    for tag in (light_tag, visual_tag):
        path = work("results", tag, bench=BENCH) / "qa.jsonl"
        if not path.exists():
            raise SystemExit(f"{path} is missing; run QA for {tag} first")
        out.append({r["query_id"]: float(bool(r["credited"])) for r in load_jsonl(path)})
    return tuple(out)


def automix(args):
    sys.path.insert(0, str(repo_dir("automix")))
    from automix import POMDP, Automix, Threshold
    methods = {"threshold": Threshold, "pomdp": POMDP}
    labels = load_jsonl(work("routing", bench=BENCH) / args.labels)
    verified = {r["query_id"]: r["p_ver"] for r in load_jsonl(work("routing", bench=BENCH) / args.verify)}
    light, visual = metric_columns(labels, args.metric, args.light, args.visual)
    df = pd.DataFrame([{"query_id": r["query_id"], "fold": r["fold"], SLM: light[r["query_id"]],
                        LLM: visual[r["query_id"]], "p_ver": verified[r["query_id"]]} for r in labels])
    stem = Path(args.verify).stem.replace("automix_verify_", "")

    rows = []
    for fold in sorted(df["fold"].unique()):
        mixer = Automix(methods[args.method](num_bins=args.bins), slm_column=SLM, llm_column=LLM,
                        verifier_column="p_ver", costs=args.costs, verifier_cost=args.verifier_cost)
        mixer.train(df[df["fold"] != fold])
        test = df[df["fold"] == fold]
        retry = np.asarray(mixer.method.run(test, mixer.best_param, verifier_column="p_ver"), dtype=bool)
        rows += [{"query_id": q, "branch": "visual" if r else "light", "fold": int(fold), "p_ver": p,
                  "best_param": str(mixer.best_param)} for q, r, p in zip(test["query_id"], retry, test["p_ver"])]
    out = decision_dir(args.labels) / f"automix-{args.method}-{args.metric}_{stem}.jsonl"
    write_jsonl(out, rows)
    print(json.dumps({"out": str(out), "visual": sum(r["branch"] == "visual" for r in rows)}))

    curve = []
    for t in sorted(set(df["p_ver"])):
        pick = (df["p_ver"] <= t).to_numpy()
        curve.append({"threshold": float(t), "rate": round(float(pick.mean()), 4),
                      args.metric: at_rate(100 * df[SLM].to_numpy(), 100 * df[LLM].to_numpy(), pick)})
    curve_path = work("routing", bench=BENCH) / f"automix_curve-{args.metric}_{stem}.jsonl"
    write_jsonl(curve_path, curve)
    for point in curve:
        print(json.dumps(point))


def main():
    ap = argparse.ArgumentParser(prog="python -m ondemand.routing_api")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("clone", help="clone the router repos at pinned commits into routing/repos/")

    p = sub.add_parser("universalrag", help="UniversalRAG training-free LLM router, their prompt and parser")
    p.add_argument("--model", required=True)
    p.add_argument("--temperature", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--on-error", choices=("light", "visual"), default="light")
    p.add_argument("--labels", default="labels_r10_k100.jsonl")
    p.add_argument("--allow-api", action="store_true")

    p = sub.add_parser("verify", help="AutoMix few-shot self-verification of the light answers")
    p.add_argument("--model", required=True)
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=250)
    p.add_argument("--light", default="light_r20")
    p.add_argument("--labels", default="labels_r10_k100.jsonl")
    p.add_argument("--allow-api", action="store_true")

    p = sub.add_parser("automix", help="AutoMix meta-verifier, 5-fold CV decisions plus a threshold curve (no API)")
    p.add_argument("--verify", required=True, help="automix_verify_<model>.jsonl in routing/")
    p.add_argument("--method", choices=("threshold", "pomdp"), default="pomdp")
    p.add_argument("--metric", choices=("qa", "ndcg", "recall"), default="recall")
    p.add_argument("--bins", type=int, default=8)
    p.add_argument("--costs", type=float, nargs=2, default=[1, 50])
    p.add_argument("--verifier-cost", type=float, default=1)
    p.add_argument("--light", default="light_r20")
    p.add_argument("--visual", default="ppocrv5_locked")
    p.add_argument("--labels", default="labels_r10_k100.jsonl")

    args = ap.parse_args()
    {"clone": clone, "universalrag": universalrag, "verify": verify, "automix": automix}[args.cmd](args)


if __name__ == "__main__":
    main()
