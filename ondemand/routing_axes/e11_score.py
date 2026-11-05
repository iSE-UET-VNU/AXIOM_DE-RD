import io
import json
import zipfile
from pathlib import Path

import numpy as np

from ondemand.bench import BENCH, load_jsonl, work
from ondemand.light_retrieval import DENSE_MODEL, Hierarchy, cache_misses, chunk_vectors, page_scores
from ondemand.openrouter import embed
from ondemand.routing_axes.e1_patch_text import query_word_of_tokens, token_bytes
from ondemand.routing_axes.tight_coupling_common import (ENC, GROUPS, POOL, V2, W, boot, col, fmt, gate, mean_r, metrics, queries, row, stop_rule,
                                                         stored)

SPLITS = Path("outputs/e11_splits_final.jsonl")
PART_CACHE = Path("outputs/e11_te3s_cache")
K18 = "outputs/k18_out_v2.zip"
SLOTS = 10


def take_turns(rankings, slots=SLOTS):
    chosen, cursor = [], [0] * len(rankings)
    while len(chosen) < slots:
        moved = False
        for i, ranking in enumerate(rankings):
            while cursor[i] < len(ranking) and ranking[cursor[i]] in chosen:
                cursor[i] += 1
            if cursor[i] < len(ranking) and len(chosen) < slots:
                chosen.append(ranking[cursor[i]])
                cursor[i] += 1
                moved = True
        if not moved:
            break
    return chosen


def part_tokens(tokens, input_ids, query, spans, special):
    pieces = [token_bytes(t) for t in tokens]
    text = b"".join(pieces)
    target = query.encode()
    at = text.find(target)
    if at < 0:
        at = text.lower().find(target.lower())
    if at < 0:
        raise SystemExit(f"query not found in its tokens: {query}")
    words = query_word_of_tokens(tokens, query)
    bytes_of = [(at + len(query[:a].encode()), at + len(query[:b].encode())) for a, b in spans]
    out, pos = [], 0
    for i, piece in enumerate(pieces):
        a, b = pos, pos + len(piece)
        pos = b
        if words[i] is not None and input_ids[i] not in special and any(a < e and s < b for s, e in bytes_of):
            out.append(i)
    return out


def toy_check():
    if take_turns([["a", "b", "c"], ["a", "d"]], 4) != ["a", "d", "b", "c"]:
        raise SystemExit("toy check failed: turns skip taken pages")
    if take_turns([["a"], ["b"]], 10) != ["a", "b"]:
        raise SystemExit("toy check failed: short rankings stop")
    print("toy check: turn-taking behaves")


def main():
    toy_check()
    members = GROUPS["vidore"]
    splits = {r["query_id"]: r for r in load_jsonl(SPLITS)}
    if sorted(splits) != sorted(members):
        raise SystemExit("splits do not cover the 154 queries")
    split_ids = [q for q in members if splits[q]["status"] == "split"]

    rows = load_jsonl(W / "light_prep/pages_ppocrv5.jsonl")
    hierarchy = Hierarchy(rows, file_k=gate["file_k"], page_k=gate["gate_k"], w_dense=gate["w_dense"], parent=gate["parent_weight"],
                          signals=gate["signals"], analyzer=gate["analyzer"], file_mode=gate["file_mode"], file_direct=gate["file_direct"])
    main_cache = work("embedding_cache", "te3s_ppocr", bench=BENCH)
    whole = [queries[q]["query"] for q in members]
    if cache_misses(whole, main_cache):
        raise SystemExit("whole-query embeddings are not all cached")
    owners, vectors = chunk_vectors(rows, main_cache, False)
    n_pages = len(rows)

    def light(text, vector):
        return hierarchy.rank(text, page_scores(owners, vectors, vector / (np.linalg.norm(vector) + 1e-9), n_pages))[0]

    whole_vectors = embed(whole, DENSE_MODEL, main_cache)
    wrong = [q for q, t, v in zip(members, whole, whole_vectors) if light(t, v) != gate["light_run_top100"][q]]
    print(f"Light check: whole query differs from light_run_top100 on {len(wrong)} queries")
    if wrong:
        raise SystemExit(f"Light does not reproduce the gate: {wrong[:5]}")

    archive = zipfile.ZipFile(K18)
    index = json.loads(archive.read("k18_out/q_index.json"))
    win_sim = np.load(io.BytesIO(archive.read("k18_out/win_sim.npy")))
    start = np.concatenate([[0], np.cumsum(index["lengths"])])
    position = {q: i for i, q in enumerate(index["query_ids"])}
    e2 = W / "routing_axes/e2/e2_out"
    token_index = json.loads((e2 / "q_128_index.json").read_text())
    token_position = {q: i for i, q in enumerate(token_index["query_ids"])}
    special = set(json.loads((e2 / "meta.json").read_text())["special_ids"])

    texts = sorted({p["text"] for q in split_ids for p in splits[q]["parts"]})
    PART_CACHE.mkdir(parents=True, exist_ok=True)
    part_vectors = dict(zip(texts, embed(texts, DENSE_MODEL, PART_CACHE)))
    print(f"split queries {len(split_ids)}, distinct part texts {len(texts)}")

    lists, encoded = {}, {}
    largest = 0.0
    for q in members:
        if q not in split_ids:
            lists[q] = sorted(ENC[q], key=lambda p: (-ENC[q][p], p))[:SLOTS]
            encoded[q] = len(ENC[q])
            continue
        block = win_sim[:, start[position[q]]:start[position[q] + 1]].astype(np.float64)
        largest = max(largest, float(np.abs(block.sum(1) - V2[row[q]]).max()))
        i = token_position[q]
        seen, part_scores = set(ENC[q]), []
        for part in splits[q]["parts"]:
            chosen = part_tokens(token_index["tokens"][i], token_index["input_ids"][i], queries[q]["query"], part["spans"], special)
            if not chosen:
                raise SystemExit(f"part with no tokens: {q} {part['text']}")
            scores = block[:, chosen].sum(1)
            part_scores.append(scores)
            pool = light(part["text"], part_vectors[part["text"]])[:len(POOL[q])]
            seen |= set(stop_rule(lambda p, s=scores: float(s[col[p]]), pool))
        rankings = [sorted(seen, key=lambda p, s=s: (-float(s[col[p]]), p)) for s in part_scores]
        lists[q] = take_turns(rankings)
        encoded[q] = len(seen)
    print(f"token score check: max |sum win_sim - V2| on split queries {largest}")
    if largest > 1e-3:
        raise SystemExit("per-token scores do not add up to V2")
    changed = [q for q in members if q not in split_ids and lists[q] != stored[q]]
    print(f"unsplit check: differs from stored baseline on {len(changed)} queries")
    if changed:
        raise SystemExit(f"unsplit queries changed: {changed[:5]}")

    base, got = metrics({q: stored[q] for q in members}), metrics(lists)
    ci = boot([got[q][0] - base[q][0] for q in members])
    only = boot([got[q][0] - base[q][0] for q in split_ids])
    print(f"RESULT R@10 {mean_r(got, members):.2f} vs baseline {mean_r(base, members):.2f}: {fmt(ci)} {'PASS' if ci[1] > 0 else 'FAIL'}")
    print(f"encoded pages/q {np.mean([encoded[q] for q in members]):.2f} vs baseline {np.mean([len(ENC[q]) for q in members]):.2f}")
    print(f"split queries only, n {len(split_ids)}: R@10 {mean_r(got, split_ids):.2f} vs {mean_r(base, split_ids):.2f}: {fmt(only)}")


if __name__ == "__main__":
    main()
