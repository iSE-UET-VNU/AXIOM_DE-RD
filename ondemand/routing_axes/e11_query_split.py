import argparse
import json
import re
from pathlib import Path

from ondemand.openrouter import chat
from ondemand.routing_axes.tight_coupling_common import GROUPS, queries

MODEL, SEED, MAX_TOKENS = "deepseek/deepseek-v4-flash", 0, 512
OUT = Path("outputs/e11_splits_v2.jsonl")
FINAL = Path("outputs/e11_splits_final.jsonl")
RETRY_TOKENS = 2048
WORD = re.compile(r"[^\W_]+(?:'[^\W_]+)*")
PROMPT = """You split a document-search query into the separate things it asks about.

Query: <QUERY>

If the query asks about two or more separate topics or items that could be found on different pages, return each one as a part.
If it asks about one thing, return one part containing the whole query.
Each part must make sense as a search by itself. Words that only modify a part, such as a source, format or follow-up about it, stay in that part.
When unsure, return one part.
Put words that apply to every part, such as a leading phrase, in "shared". Leave "shared" empty if there are none.
Copy every span word for word from the query. Spans must not overlap.
Reply with JSON only: {"shared": "", "parts": ["..."]}"""


def place(span, query, used):
    at = query.find(span)
    while at >= 0:
        end = at + len(span)
        if all(end <= a or b <= at for a, b in used):
            return at, end
        at = query.find(span, at + 1)
    return None


def parse(raw, query):
    found = re.search(r"\{.*\}", raw, re.S)
    if not found:
        return None
    try:
        data = json.loads(found.group())
    except json.JSONDecodeError:
        return None
    shared, parts = data.get("shared", ""), data.get("parts")
    if not isinstance(shared, str) or not isinstance(parts, list) or not parts or not all(isinstance(p, str) and p.strip() for p in parts):
        return None
    used, shared_span = [], None
    if shared.strip():
        shared_span = place(shared.strip(), query, used)
        if shared_span is None:
            return None
        used.append(shared_span)
    spans = []
    for part in parts:
        span = place(part.strip(), query, used)
        if span is None:
            return None
        used.append(span)
        spans.append(span)
    return shared_span, sorted(spans)


def load_reply(raw):
    found = re.search(r"\{.*\}", raw, re.S)
    if not found:
        return None
    try:
        data = json.loads(found.group())
    except json.JSONDecodeError:
        return None
    shared, parts = data.get("shared", ""), data.get("parts")
    if not isinstance(shared, str) or not isinstance(parts, list) or not parts or not all(isinstance(p, str) and p.strip() for p in parts):
        return None
    return shared, parts


def word_spans(text, query):
    wanted = {m.group().casefold() for m in WORD.finditer(text)}
    return [list(m.span()) for m in WORD.finditer(query) if m.group().casefold() in wanted]


def finalize(raw, query):
    reply = load_reply(raw)
    if reply is None:
        return "invalid", []
    placed = parse(raw, query)
    if placed is not None:
        shared_span, spans = placed
        if len(spans) < 2:
            return "single", []
        parts = []
        for span in spans:
            pieces = sorted([list(span)] + ([list(shared_span)] if shared_span else []))
            parts.append({"text": " ".join(query[a:b] for a, b in pieces), "spans": pieces, "mode": "verbatim"})
        return "split", parts
    shared, texts = reply
    seen, parts = set(), []
    for text in texts:
        pieces = word_spans(f"{shared} {text}", query)
        key = tuple(map(tuple, pieces))
        if pieces and key not in seen:
            seen.add(key)
            parts.append({"text": f"{shared} {text}".strip(), "spans": pieces, "mode": "words"})
    if len(parts) < 2:
        return "single", []
    return "split", sorted(parts, key=lambda part: part["spans"][0])


def freeze():
    if FINAL.exists():
        raise SystemExit(f"{FINAL} already exists; frozen splits are not rewritten")
    rows = [json.loads(line) for line in OUT.read_text(encoding="utf-8").splitlines()]
    final = []
    for row in rows:
        query, raw, retried = row["query"], row["raw"], False
        if row["status"] == "failed" or (row["status"] == "invalid" and load_reply(raw) is None):
            retried = True
            try:
                raw = chat(MODEL, PROMPT.replace("<QUERY>", query), max_tokens=RETRY_TOKENS, temperature=0.0, seed=SEED)
            except RuntimeError as error:
                raw = str(error)
        status, parts = finalize(raw, query)
        final.append({"query_id": row["query_id"], "query": query, "raw": raw, "retried": retried, "status": status, "parts": parts})
    with FINAL.open("x", encoding="utf-8") as handle:
        for row in final:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print({s: sum(r["status"] == s for r in final) for s in ("split", "single", "invalid")}, "retried", sum(r["retried"] for r in final), "total", len(final))


def toy_check():
    q = "energy of waves and energy of particles"
    raw = '```json\n{"shared": "", "parts": ["energy of waves", "energy of particles"]}\n```'
    if parse(raw, q) != (None, [(0, 15), (20, 39)]):
        raise SystemExit("toy check failed: verbatim parts")
    if parse('{"shared": "", "parts": ["wave energy"]}', q) is not None:
        raise SystemExit("toy check failed: paraphrase accepted")
    if parse('{"shared": "energy", "parts": ["energy", "energy", "energy"]}', q) is not None:
        raise SystemExit("toy check failed: overlap accepted")
    q = "Compare energy transfer by conduction and convection?"
    status, parts = finalize('{"shared": "", "parts": ["energy transfer by conduction", "energy transfer by convection"]}', q)
    if status != "split" or [q[a:b] for a, b in parts[0]["spans"]] != ["energy", "transfer", "by", "conduction"]:
        raise SystemExit(f"toy check failed: word mapping {parts}")
    if finalize('{"shared": "", "parts": ["conduction", "Conduction"]}', q)[0] != "single":
        raise SystemExit("toy check failed: same tokens not merged")
    if finalize('{"shared": "", "parts": ["trunc', q)[0] != "invalid":
        raise SystemExit("toy check failed: cut-off reply accepted")
    print("toy check: verbatim, paraphrase, overlap, word mapping and cut-off cases behave")


def main():
    toy_check()
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true")
    if ap.parse_args().freeze:
        freeze()
        return
    done = {json.loads(line)["query_id"] for line in OUT.read_text(encoding="utf-8").splitlines()} if OUT.exists() else set()
    members = sorted(q for q in GROUPS["vidore"] if q not in done)
    print(f"to call {len(members)}, already saved {len(done)}")
    with OUT.open("a", encoding="utf-8") as handle:
        for i, q in enumerate(members, 1):
            query = queries[q]["query"]
            row = {"query_id": q, "query": query, "model": MODEL, "seed": SEED}
            try:
                row["raw"] = chat(MODEL, PROMPT.replace("<QUERY>", query), max_tokens=MAX_TOKENS, temperature=0.0, seed=SEED)
            except RuntimeError as error:
                row.update(raw=str(error), status="failed", shared=None, parts=[])
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                continue
            placed = parse(row["raw"], query)
            if placed is None:
                row.update(status="invalid", shared=None, parts=[])
            else:
                row.update(status="split" if len(placed[1]) > 1 else "single", shared=placed[0], parts=placed[1])
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            if i % 25 == 0:
                print(f"{i}/{len(members)}", flush=True)
    rows = [json.loads(line) for line in OUT.read_text(encoding="utf-8").splitlines()]
    print({s: sum(r["status"] == s for r in rows) for s in ("split", "single", "invalid", "failed")}, "total", len(rows))


if __name__ == "__main__":
    main()
