import math
import re
import unicodedata
from collections import Counter, defaultdict

import numpy as np
import scipy.sparse as sp
import Stemmer

PLACEHOLDER = re.compile(r"\[Image:[^\]]*\]")
IMAGE_ONLY = re.compile(r"^(\[Image:[^\]]*\]\s*)+$")
TOKEN = re.compile(r"[^\W_]+(?:'[^\W_]+)*", flags=re.UNICODE)
WORD = re.compile(r"\S+")
SPARSE_CHARS = 200
K1, B = 1.2, 0.75

EN_STOP = frozenset("""a about above after again against all am an and any are as at be because been before being below
between both but by can could did do does doing down during each few for from further had has have having he her here
hers herself him himself his how i if in into is it its itself just me more most my myself no nor not now of off on once
only or other our ours ourselves out over own same she should so some such than that the their theirs them themselves then
there these they this those through to too under until up very was we were what when where which while who whom why will
with you your yours yourself yourselves s t don also may might must shall would describe explain according give provide
list show shows shown discuss""".split())

FR_STOP = frozenset((
    "ai", "aie", "aient", "aies", "ait", "as", "au", "aura", "aurai", "auraient", "aurais",
    "aurait", "auras", "aurez", "auriez", "aurions", "aurons", "auront", "aux", "avaient",
    "avais", "avait", "avec", "avez", "aviez", "avions", "avons", "ayant", "ayante", "ayantes",
    "ayants", "ayez", "ayons", "c", "ce", "ces", "d", "dans", "de", "des", "du", "elle", "en",
    "es", "est", "et", "eu", "eue", "eues", "eurent", "eus", "eusse", "eussent", "eusses",
    "eussiez", "eussions", "eut", "eux", "eûmes", "eût", "eûtes", "furent", "fus", "fusse",
    "fussent", "fusses", "fussiez", "fussions", "fut", "fûmes", "fût", "fûtes", "il", "ils",
    "j", "je", "l", "la", "le", "les", "leur", "lui", "m", "ma", "mais", "me", "mes", "moi",
    "mon", "même", "n", "ne", "nos", "notre", "nous", "on", "ont", "ou", "par", "pas", "pour",
    "qu", "que", "qui", "s", "sa", "se", "sera", "serai", "seraient", "serais", "serait",
    "seras", "serez", "seriez", "serions", "serons", "seront", "ses", "soient", "sois", "soit",
    "sommes", "son", "sont", "soyez", "soyons", "suis", "sur", "t", "ta", "te", "tes", "toi",
    "ton", "tu", "un", "une", "vos", "votre", "vous", "y", "à", "étaient", "étais", "était",
    "étant", "étante", "étantes", "étants", "étiez", "étions", "été", "étée", "étées", "étés",
    "êtes",
))

_EN, _FR = Stemmer.Stemmer("english"), Stemmer.Stemmer("french")


def real_text(text):
    return PLACEHOLDER.sub("", text).strip()


def is_uninformative(text):
    stripped = text.strip()
    return not stripped or bool(IMAGE_ONLY.fullmatch(stripped))


def mostly_image(text):
    return bool(PLACEHOLDER.search(text)) and len(real_text(text)) < SPARSE_CHARS


def plain(text):
    normalized = unicodedata.normalize("NFC", str(text)).casefold().replace("’", "'")
    return TOKEN.findall(normalized)


def enfr(text):
    parts = [p for token in plain(text) for p in token.split("'") if p]
    return _FR.stemWords(_EN.stemWords([p for p in parts if p not in EN_STOP and p not in FR_STOP]))


def windows(text, n_words=512, overlap=128):
    words = list(WORD.finditer(text))
    spans, step, index = [], max(1, n_words - overlap), 0
    while index < len(words):
        end = min(index + n_words, len(words))
        spans.append((words[index].start(), words[end - 1].end()))
        if end == len(words):
            break
        index += step
    return spans


class PageBM25:
    def __init__(self, docs):
        vocab, r, c, v = {}, [], [], []
        for i, d in enumerate(docs):
            for term, tf in Counter(d).items():
                r.append(i); c.append(vocab.setdefault(term, len(vocab))); v.append(tf)
        tf = sp.csr_matrix((v, (r, c)), shape=(len(docs), len(vocab)), dtype=np.float32)
        dl = np.array([len(d) for d in docs], dtype=np.float32)
        avg = dl.mean() if dl.mean() > 0 else 1
        df = np.bincount(tf.indices, minlength=len(vocab))
        self.idf = np.log(1 + (len(docs) - df + 0.5) / (df + 0.5)).astype(np.float32)
        tf = tf.tocoo()
        w = tf.data * (K1 + 1) / (tf.data + K1 * (1 - B + B * dl[tf.row] / avg))
        self.W = sp.csc_matrix((w, (tf.row, tf.col)), shape=tf.shape)
        self.vocab, self.n = vocab, len(docs)

    def scores(self, q):
        s = np.zeros(self.n, dtype=np.float32)
        for j, _ in Counter(self.vocab[t] for t in q if t in self.vocab).items():
            col = self.W[:, j]
            s[col.indices] += self.idf[j] * col.data
        return s


class ChunkBM25:
    def __init__(self, texts):
        self.postings, self.lengths = defaultdict(list), []
        for position, text in enumerate(texts):
            tokens = plain(text)
            for term, count in Counter(tokens).items():
                self.postings[term].append((position, count))
            self.lengths.append(len(tokens))
        self.avgdl = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0

    def search(self, query, top_k, allowed=None):
        total = len(self.lengths)
        if not total:
            return []
        scores = defaultdict(float)
        for term in plain(query):
            posting = self.postings.get(term)
            if not posting:
                continue
            idf = math.log(1.0 + (total - len(posting) + 0.5) / (len(posting) + 0.5))
            for position, freq in posting:
                if allowed is not None and position not in allowed:
                    continue
                length = self.lengths[position] or 1
                scores[position] += idf * (freq * (K1 + 1)) / (freq + K1 * (1 - B + B * length / (self.avgdl or 1.0)))
        ranked = sorted(scores.items(), key=lambda item: -item[1])
        return [(p, s) for p, s in ranked[:top_k] if s > 0]
