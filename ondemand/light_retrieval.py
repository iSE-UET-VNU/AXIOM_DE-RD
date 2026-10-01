import hashlib
import json
from collections import Counter, defaultdict
from time import perf_counter

import numpy as np

from .bench import BENCH, doc_of, fingerprint, gold as load_gold, load_jsonl, queries as load_queries, work
from .chunks import page_chunks
from .evaluate import as_run, evaluate, file_hit, groups, page_recall
from .openrouter import embed
from .text import EN_STOP, FR_STOP, PageBM25, enfr, plain, real_text

DENSE_MODEL = "openai/text-embedding-3-small"
DENSE_DIM, DENSE_SLICE = 1536, 256
FILE_K, PAGE_K, POOL = 3, 20, 100
W_DENSE, PARENT, FILE_DIRECT = 0.70, 0.15, 0.50
STOPWORD_MIN = 3

ANALYZERS = {"enfr": enfr, "plain": plain}
POLICIES = {}


def policy(name):
    def wrap(fn):
        POLICIES[name] = fn
        return fn
    return wrap


def language(text):
    tokens = plain(text)
    if not tokens:
        return None
    english = sum(t in EN_STOP for t in tokens)
    french = sum(t in FR_STOP for t in tokens)
    if max(english, french) < STOPWORD_MIN or english == french:
        return None
    return "en" if english > french else "fr"


@policy("all")
def all_signals(query_language, page_languages):
    return np.ones(len(page_languages), dtype=bool)


@policy("lang")
def language_match(query_language, page_languages):
    if query_language is None:
        return np.ones(len(page_languages), dtype=bool)
    return np.array([p is None or p == query_language for p in page_languages], dtype=bool)


def normalise(values):
    values = np.asarray(values, dtype=np.float32)
    positive = values[values > 0]
    top = positive.max() if positive.size else 0.0
    if top <= 0:
        return np.zeros_like(values)
    return np.clip(values, 0, None) / top


def dense_signal(rows, qids, queries, bench, cache_name, allow_api):
    cache = work("embedding_cache", cache_name, bench=bench)
    owners, chunks = [], []
    for i, row in enumerate(rows):
        for chunk in page_chunks(real_text(row["text"])):
            owners.append(i)
            chunks.append(chunk)
    owners = np.array(owners, dtype=int)
    wanted = chunks + [queries[q]["query"] for q in qids]
    if not allow_api:
        missing = sum(not (cache / (hashlib.sha256((DENSE_MODEL + "\n" + t).encode()).hexdigest() + ".json")).exists()
                      for t in wanted)
        if missing:
            raise SystemExit(f"{missing} of {len(wanted)} texts are not in {cache}; rerun with --allow-api to embed them")
    vectors = np.zeros((len(chunks), DENSE_DIM), dtype=np.float32)
    for start in range(0, len(chunks), DENSE_SLICE):
        vectors[start:start + DENSE_SLICE] = embed(chunks[start:start + DENSE_SLICE], DENSE_MODEL, cache)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
    qvecs = embed([queries[q]["query"] for q in qids], DENSE_MODEL, cache)
    qvecs /= np.linalg.norm(qvecs, axis=1, keepdims=True) + 1e-9
    out = {}
    for i, q in enumerate(qids):
        page = np.full(len(rows), -1e9, dtype=np.float32)
        np.maximum.at(page, owners, vectors @ qvecs[i])
        out[q] = page
    return out


class Hierarchy:
    def __init__(self, rows, file_k=FILE_K, page_k=PAGE_K, w_dense=W_DENSE, parent=PARENT, signals="all",
                 analyzer="enfr", file_mode="dense_pool", file_direct=FILE_DIRECT):
        self.units = [r["page_id"] for r in rows]
        self.file_k, self.page_k = file_k, page_k
        self.w_dense, self.w_bm25 = w_dense, 1.0 - w_dense
        self.parent, self.file_mode, self.file_direct = parent, file_mode, file_direct
        self.signals, self.analyzer_name = signals, analyzer
        self.policy, self.analyze = POLICIES[signals], ANALYZERS[analyzer]
        grouped = defaultdict(list)
        for r in rows:
            grouped[doc_of(r["page_id"])].append(real_text(r["text"]))
        self.file_ids = sorted(grouped)
        index = {f: i for i, f in enumerate(self.file_ids)}
        self.owner = np.array([index[doc_of(u)] for u in self.units])
        self.page_bm25 = PageBM25([self.analyze(real_text(r["text"])) for r in rows])
        self.file_bm25 = (PageBM25([self.analyze("\n".join(grouped[f]).strip()) for f in self.file_ids])
                          if file_mode == "bm25_blend" else None)
        self.page_language = [language(real_text(r["text"])) for r in rows]
        self.file_language = [self._vote(i) for i in range(len(self.file_ids))]

    def _vote(self, position):
        votes = Counter(l for l, o in zip(self.page_language, self.owner) if o == position and l)
        return votes.most_common(1)[0][0] if votes else None

    def file_scores(self, tokens, base, dense, query_language):
        pooled = np.zeros(len(self.file_ids), dtype=np.float32)
        np.maximum.at(pooled, self.owner, dense if self.file_mode == "dense_pool" else base)
        pooled = normalise(pooled)
        if self.file_mode == "dense_pool":
            return pooled
        mask = self.policy(query_language, self.file_language)
        direct = normalise(self.file_bm25.scores(tokens))
        blended = self.file_direct * direct + (1 - self.file_direct) * pooled
        return normalise(np.where(mask, blended, pooled))

    def rank(self, query, dense):
        tokens = self.analyze(query)
        query_language = language(query)
        mask = self.policy(query_language, self.page_language)
        bm25 = normalise(self.page_bm25.scores(tokens))
        dense = normalise(dense)
        base = np.where(mask, self.w_bm25 * bm25 + self.w_dense * dense, dense)
        files = self.file_scores(tokens, base, dense, query_language)
        selected = set(sorted(range(len(self.file_ids)), key=lambda i: (-files[i], self.file_ids[i]))[:self.file_k])
        scores = (1 - self.parent) * base + self.parent * files[self.owner]
        keep = [i for i, o in enumerate(self.owner) if o in selected]
        order = sorted(keep, key=lambda i: (-scores[i], self.units[i]))[:POOL]
        notes = {"query_language": query_language, "bm25_pages": int(mask.sum()),
                 "files": sorted(self.file_ids[i] for i in selected)}
        return [self.units[i] for i in order], {self.units[i]: float(scores[i]) for i in order}, notes


def build(bench=BENCH, pages_name="pages_ocr_ppocr.jsonl", name=None, signals="all", file_k=FILE_K, page_k=PAGE_K,
          w_dense=W_DENSE, parent=PARENT, analyzer="enfr", file_mode="dense_pool", file_direct=FILE_DIRECT,
          cache_name="te3s_ppocr", allow_api=False):
    out = work("light_prep", bench=bench)
    rows = load_jsonl(out / pages_name)
    gold, queries = load_gold(bench), load_queries(bench)
    qids = sorted(gold)
    hierarchy = Hierarchy(rows, file_k, page_k, w_dense, parent, signals, analyzer, file_mode, file_direct)
    dense = dense_signal(rows, qids, queries, bench, cache_name, allow_api)
    ranked, scores, notes, seconds = {}, {}, {}, []
    for q in qids:
        started = perf_counter()
        ranked[q], scores[q], notes[q] = hierarchy.rank(queries[q]["query"], dense[q])
        seconds.append(perf_counter() - started)

    gate = {q: ranked[q][:page_k] for q in qids}
    metrics = {}
    for group, members in groups(qids, queries).items():
        sub = {q: as_run(ranked[q]) for q in members}
        n10, r10, _ = evaluate(sub, gold, 10)
        n20, r20, _ = evaluate(sub, gold, 20)
        metrics[group] = {"n": len(members), "light_ndcg@10": round(n10, 2), "light_recall@10": round(r10, 2),
                          "light_ndcg@20": round(n20, 2), "light_recall@20": round(r20, 2),
                          "gate_recall@20": round(100 * float(np.mean([page_recall(gate[q], gold[q]) for q in members])), 2),
                          "file_recall@20": round(100 * float(np.mean([file_hit(gate[q], gold[q]) for q in members])), 2)}

    variant = f"hier_{file_mode}_k{file_k}_{analyzer}bm25{1 - w_dense:.2f}_te3s{w_dense:.2f}_{signals}"
    payload = {"variant": variant, "signals": signals, "analyzer": analyzer, "file_mode": file_mode,
               "gate_k": page_k, "file_k": file_k, "w_dense": w_dense, "parent_weight": parent,
               "file_direct": file_direct, "pages_source": pages_name, "dense_model": DENSE_MODEL,
               "bundle_fingerprint": fingerprint(bench), "n_queries": len(qids), "n_corpus_pages": len(rows),
               "n_files": len(hierarchy.file_ids), "n_union_pages": len(set().union(*gate.values())),
               "metrics": metrics, "page_languages": dict(Counter(hierarchy.page_language).most_common()),
               "query_languages": dict(Counter(n["query_language"] for n in notes.values()).most_common()),
               "query_seconds_mean": round(float(np.mean(seconds)), 4),
               "gate": gate, "gate_scores": {q: [round(scores[q][u], 6) for u in gate[q]] for q in qids},
               "union": sorted(set().union(*gate.values())), "signal_notes": notes,
               "gold_rank": {q: {u: (ranked[q].index(u) + 1 if u in ranked[q] else None) for u in gold[q]} for q in qids},
               "light_run_top100": {q: ranked[q][:POOL] for q in qids}}
    path = out / (name or f"gate_{variant}.json")
    path.write_text(json.dumps(payload), encoding="utf-8")
    for group, m in metrics.items():
        print(f"{group:22s} n={m['n']:3d} gate_r@20 {m['gate_recall@20']:6.2f} file@20 {m['file_recall@20']:6.2f} "
              f"ndcg@10 {m['light_ndcg@10']:6.2f} r@10 {m['light_recall@10']:6.2f} r@20 {m['light_recall@20']:6.2f}")
    print(path)
    return path


if __name__ == "__main__":
    build()
