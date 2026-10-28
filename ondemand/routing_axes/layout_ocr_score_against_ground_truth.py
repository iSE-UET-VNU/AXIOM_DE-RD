import argparse
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import lxml.html
from apted import APTED, Config
from rapidfuzz.distance import Levenshtein

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from ondemand.bench import load_jsonl, work

W = work()
OUT = W / "routing_axes/k14"
FLAT = "ppocrv5_flat"
L4_RATIO = 1.86
LOOP_N, LOOP_RUNS = 8, 4
ORDER_MIN_JACCARD = 0.3
WORD = re.compile(r"\w+")
MD_ROW = re.compile(r"^\s*\|.*\|\s*$")
MD_RULE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
TABULAR = re.compile(r"\\begin\{tabular\}(?:\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})?(.*?)\\end\{tabular\}", re.S)
MULTI = re.compile(r"\\multi(column|row)\{(\d+)\}\{[^{}]*\}\{(.*)\}\s*$", re.S)


def norm(text):
    t = unicodedata.normalize("NFKC", text or "")
    t = "\n".join(" " if MD_RULE.match(line) else line for line in t.split("\n"))
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"\\begin\{\w+\*?\}(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})?|\\end\{\w+\*?\}", " ", t)
    t = re.sub(r"\\multi(?:column|row)\{[^{}]*\}\{[^{}]*\}", " ", t)
    t = re.sub(r"\\\\|\\(?:hline|toprule|midrule|bottomrule)|\\cline\{[^{}]*\}", " ", t)
    t = re.sub(r"\\([$%&#_{}])", r"\1", t)
    t = re.sub(r"\\[a-zA-Z]+\*?", " ", t)
    t = re.sub(r"[|${}&]", " ", t)
    return " ".join(t.split()).lower()


def words(text):
    return WORD.findall(norm(text))


def word_pr(pred, ref):
    p, r = Counter(words(pred)), Counter(words(ref))
    hit = sum((p & r).values())
    return hit / max(1, sum(p.values())), hit / max(1, sum(r.values()))


def looped(text):
    w = words(text)
    grams = Counter(tuple(w[i:i + LOOP_N]) for i in range(len(w) - LOOP_N + 1))
    return bool(grams) and max(grams.values()) >= LOOP_RUNS


def cell(text, colspan=1, rowspan=1):
    return {"text": norm(text), "colspan": int(colspan), "rowspan": int(rowspan)}


def md_tables(text):
    tables, rows = [], []
    for line in (text or "").split("\n") + [""]:
        if MD_ROW.match(line):
            if not MD_RULE.match(line):
                rows.append([cell(c) for c in line.strip().strip("|").split("|")])
        elif rows:
            tables.append(rows)
            rows = []
    return tables


def latex_tables(text):
    tables = []
    for body in TABULAR.findall(text or ""):
        body = re.sub(r"\\(?:hline|toprule|midrule|bottomrule)|\\cline\{[^{}]*\}", " ", body)
        rows = []
        for line in re.split(r"\\\\", body):
            if not line.strip():
                continue
            row = []
            for part in re.split(r"(?<!\\)&", line):
                m = MULTI.match(part.strip())
                if m:
                    kind, n, inner = m.groups()
                    row.append(cell(inner, n if kind == "column" else 1, n if kind == "row" else 1))
                else:
                    row.append(cell(part))
            rows.append(row)
        if rows:
            tables.append(rows)
    return tables


def html_tables(text):
    if "<table" not in (text or "").lower():
        return []
    tables = []
    for table in lxml.html.fromstring(f"<div>{text}</div>").iter("table"):
        rows = [[cell(c.text_content(), c.get("colspan") or 1, c.get("rowspan") or 1) for c in tr if c.tag in ("td", "th")]
                for tr in table.iter("tr")]
        if rows:
            tables.append(rows)
    return tables


OTSL = re.compile(r"<(fcel|ecel|lcel|ucel|xcel|nl|ched|rhed|srow)>")


def otsl_to_html(text):
    if "<table" in (text or "").lower() or not OTSL.search(text or ""):
        return text
    parts = OTSL.split(text)
    rows, row = [], []
    for tag, body in zip(parts[1::2], parts[2::2]):
        if tag == "nl":
            rows.append(row)
            row = []
        else:
            row.append((tag, body.strip()))
    if row:
        rows.append(row)
    out = []
    for i, r in enumerate(rows):
        cells = []
        for j, (tag, body) in enumerate(r):
            if tag in ("lcel", "ucel", "xcel"):
                continue
            cs = 1
            while j + cs < len(r) and r[j + cs][0] in ("lcel", "xcel"):
                cs += 1
            rs = 1
            while i + rs < len(rows) and j < len(rows[i + rs]) and rows[i + rs][j][0] in ("ucel", "xcel"):
                rs += 1
            span = (f' colspan="{cs}"' if cs > 1 else "") + (f' rowspan="{rs}"' if rs > 1 else "")
            body = body.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            cells.append(f"<td{span}>{body}</td>")
        out.append("<tr>" + "".join(cells) + "</tr>")
    return "<table>" + "".join(out) + "</table>"


def tables_of(text):
    return html_tables(text) or latex_tables(text) or md_tables(text)


class Node:
    def __init__(self, tag, children=(), colspan=1, rowspan=1, text=""):
        self.tag, self.children, self.colspan, self.rowspan, self.text = tag, list(children), colspan, rowspan, text


def tree(rows):
    return Node("table", [Node("tr", [Node("td", (), c["colspan"], c["rowspan"], c["text"]) for c in r]) for r in rows])


def size(node):
    return 1 + sum(size(c) for c in node.children)


class TedsConfig(Config):
    def rename(self, a, b):
        if (a.tag, a.colspan, a.rowspan) != (b.tag, b.colspan, b.rowspan):
            return 1.0
        return Levenshtein.normalized_distance(a.text, b.text) if a.tag == "td" else 0.0

    def children(self, node):
        return node.children


def teds(a, b):
    ta, tb = tree(a), tree(b)
    return 1.0 - APTED(ta, tb, TedsConfig()).compute_edit_distance() / max(size(ta), size(tb))


def page_teds(ref_text, pred_tables):
    ref = tables_of(ref_text)
    if not ref:
        return None
    return sum(max((teds(r, p) for p in pred_tables), default=0.0) for r in ref) / len(ref)


def order_distance(ref_text, regions):
    paras = [set(words(p)) for p in re.split(r"\n\s*\n", ref_text or "")]
    paras = [p for p in paras if p]
    if len(paras) < 2:
        return None
    seq = []
    for r in regions:
        w = set(words(r["text"]))
        if not w or "table" in r["type"]:
            continue
        score, best = max((len(w & p) / len(w | p), i) for i, p in enumerate(paras))
        if score >= ORDER_MIN_JACCARD and (not seq or seq[-1] != best):
            seq.append(best)
    return Levenshtein.normalized_distance(seq, sorted(seq)) if len(seq) >= 2 else None


def assemble(regions):
    regions = sorted(regions, key=lambda r: (r["order"], r["region_idx"]))
    return regions, "\n\n".join(r["text"] for r in regions if r["text"])


def score_page(ref, regions, encoder=None):
    regions, text = assemble(regions)
    p, r = word_pr(text, ref["text"])
    pred_tables = [t for x in regions if "table" in x["type"] for t in tables_of(x["text"])]
    return {"page_id": ref["page_id"], "group": ref["group"],
            "ned": Levenshtein.normalized_distance(norm(text), norm(ref["text"])), "precision": p, "recall": r,
            "teds": page_teds(ref["text"], pred_tables), "order": order_distance(ref["text"], regions),
            "loop": looped(text), "seconds": sum(x.get("seconds") or 0.0 for x in regions),
            "tokens": len(encoder.encode(text, disallowed_special=())) if encoder else None,
            "empty": not text.strip()}


def readers_in(folder):
    return sorted(p.stem.removeprefix("regions_") for p in folder.glob("regions_*.jsonl"))


def load_regions(folder, reader):
    by_page = defaultdict(list)
    for r in load_jsonl(folder / f"regions_{reader}.jsonl"):
        by_page[r["page_id"]].append(r)
    return by_page


def flat_regions(pages):
    text = {r["page_id"]: r.get("text") or "" for r in load_jsonl(work("light_prep") / "pages_ppocrv5.jsonl")}
    return {p: [{"region_idx": 0, "order": 0, "type": "page", "text": text.get(p, ""), "seconds": None}] for p in pages}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs), len(xs)) if xs else (None, 0)


def summarize(rows, layout_seconds):
    out = {}
    groups = defaultdict(list)
    for r in rows:
        groups["all"].append(r)
        groups[r["group"]].append(r)
    for g, rs in sorted(groups.items()):
        out[g] = {"n": len(rs), "empty": sum(r["empty"] for r in rs), "loop_pct": 100 * sum(r["loop"] for r in rs) / len(rs)}
        for key in ("ned", "precision", "recall", "teds", "order", "seconds", "tokens"):
            m, n = mean(r[key] for r in rs)
            out[g][key], out[g][f"n_{key}"] = m, n
        m, _ = mean(layout_seconds.get(r["page_id"]) for r in rs)
        out[g]["layout_seconds"] = m
        for key in ("seconds", "layout_seconds"):
            out[g][f"{key}_l4_est"] = None if out[g][key] is None else out[g][key] / L4_RATIO
    return out


def score(folder, refs, readers=None, flat=None, encoder=None):
    layout = {r["page_id"]: r for r in load_jsonl(folder / "layout.jsonl")}
    pages = sorted(layout)
    unknown = [p for p in pages if p not in refs]
    if unknown:
        raise SystemExit(f"{len(unknown)} layout pages have no reference, e.g. {unknown[:3]}")
    layout_s = {p: (r.get("seconds_orient") or 0.0) + (r.get("seconds_layout") or 0.0) for p, r in layout.items()}
    sources = {reader: load_regions(folder, reader) for reader in (readers or readers_in(folder))}
    excluded = {reader: [p for p in pages if len(by_page.get(p, [])) != layout[p]["n_kept"]]
                for reader, by_page in sources.items()}
    common = [p for p in pages if not any(p in e for e in excluded.values())]
    sources[FLAT] = flat if flat is not None else flat_regions(pages)
    excluded[FLAT] = []
    results = {}
    for reader, by_page in sources.items():
        rows = [score_page(refs[p], by_page.get(p, []), encoder) for p in common]
        results[reader] = summarize(rows, {} if reader == FLAT else layout_s)
    return results, excluded


def table(results):
    head = f"{'reader':16} {'group':22} {'n':>4} {'NED':>6} {'P':>6} {'R':>6} {'TEDS':>6} {'nT':>4} {'order':>6} " \
           f"{'nO':>4} {'loop%':>6} {'s/pg':>6} {'L4':>6} {'lay s':>6} {'tok':>6} {'empty':>5}"
    lines = [head]
    f = lambda v, d=3: "-" if v is None else f"{v:.{d}f}"
    for reader, groups in results.items():
        for g, s in groups.items():
            lines.append(f"{reader:16} {g:22} {s['n']:>4} {f(s['ned']):>6} {f(s['precision']):>6} {f(s['recall']):>6} "
                         f"{f(s['teds']):>6} {s['n_teds']:>4} {f(s['order']):>6} {s['n_order']:>4} "
                         f"{s['loop_pct']:>6.1f} {f(s['seconds'], 2):>6} {f(s['seconds_l4_est'], 2):>6} "
                         f"{f(s['layout_seconds'], 2):>6} {f(s['tokens'], 0):>6} {s['empty']:>5}")
    return "\n".join(lines)


if __name__ == "__main__":
    import tiktoken

    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--readers", help="comma list; default every regions_*.jsonl in the folder")
    ap.add_argument("--tag", help="output name; default smoke or full from meta.json")
    args = ap.parse_args()
    refs = {r["page_id"]: r for r in load_jsonl(OUT / "reference.jsonl")}
    results, excluded = score(args.folder, refs, args.readers.split(",") if args.readers else None,
                             encoder=tiktoken.get_encoding("o200k_base"))
    print(table(results))
    for reader, pages in excluded.items():
        if pages:
            print(f"{reader}: {len(pages)} pages not finished, left out for every reader, e.g. {pages[:2]}")
    meta = args.folder / "meta.json"
    tag = args.tag or ("smoke" if not meta.exists() or json.loads(meta.read_text())["smoke"] else "full")
    name = OUT / f"score_{tag}.json"
    name.write_text(json.dumps({"results": results, "excluded": excluded}, indent=1))
    print(f"-> {name}")
