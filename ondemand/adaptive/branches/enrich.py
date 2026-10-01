import base64
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv
import fitz
import requests

from ...bench import BENCH, ROOT, documents, load_jsonl, work
from .. import fallback
from ..config import get_config
from ..registry import build, register
from ..subdata import BranchOutput, Passage

load_dotenv(ROOT / ".env")

PROMPT = "OCR this document page. Convert everything to clean Markdown with full tables, reading order, and formulas."


class Enricher:
    name = "base"

    def parse(self, page_ids):
        raise NotImplementedError

    def describe(self):
        return self.name


@register("enricher", "cached")
class CachedEnricher(Enricher):
    name = "cached"

    def __init__(self, argument, bench=BENCH, **_):
        source = work("enrich", bench=bench) / (argument or "enriched_pages.jsonl")
        if not source.exists():
            raise SystemExit(f"{source} is missing; no enriched parse is cached for this bundle")
        self.source = source
        self.text = {r["page_id"]: r["text"] for r in load_jsonl(source)}

    def parse(self, page_ids):
        return {p: self.text.get(p, "") for p in page_ids}

    def describe(self):
        return f"cached:{self.source.name}"


@register("enricher", "chandra")
class ChandraEnricher(Enricher):
    name = "chandra"

    def __init__(self, argument, bench=BENCH, pages_name="pages_ppocrv5.jsonl", **context):
        cfg = get_config()
        self.endpoint = (argument if argument and (argument.startswith("http://") or argument.startswith("https://"))
                         else context.get("chandra_endpoint") or cfg.chandra_endpoint)
        self.api_key = context.get("chandra_api_key") or cfg.chandra_api_key
        self.model = context.get("chandra_model") or cfg.chandra_model
        self.bench = bench
        self.cache_dir = work("enrich", "cache", bench=bench)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.docs = {d["doc_id"]: bench / d["path"] for d in documents(bench)}

        # Optional bulk pre-cached JSONL
        source = work("enrich", bench=bench) / "enriched_pages.jsonl"
        self.memory_cache = {r["page_id"]: r["text"] for r in load_jsonl(source)} if source.exists() else {}

        # Fallback text source in case endpoint is not provided or fails
        lp_source = work("light_prep", bench=bench) / pages_name
        self.fallback = {r["page_id"]: r["text"] for r in load_jsonl(lp_source)} if lp_source.exists() else {}

    def _render_page_b64(self, page_id):
        if "#page=" not in page_id:
            return None
        doc_id, page_str = page_id.split("#page=")
        path = self.docs.get(doc_id)
        if not path or not path.exists():
            return None
        try:
            with fitz.open(path) as doc:
                page = doc.load_page(int(page_str))
                pix = page.get_pixmap(dpi=144)
                return base64.b64encode(pix.tobytes("png")).decode("utf-8")
        except Exception:
            return None

    def _call_endpoint(self, page_id, b64_img):
        url = self.endpoint.rstrip("/")
        if not url.endswith("/chat/completions"):
            url = f"{url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}}
                    ]
                }
            ],
            "max_tokens": 4096,
            "temperature": 0.0
        }
        res = requests.post(url, json=payload, headers=headers, timeout=120)
        res.raise_for_status()
        data = res.json()
        return data["choices"][0]["message"]["content"]

    def parse(self, page_ids):
        out = {}
        for p in page_ids:
            if p in self.memory_cache:
                out[p] = self.memory_cache[p]
                continue

            cache_file = self.cache_dir / (hashlib.sha256(f"{self.model}\n{PROMPT}\n{p}".encode("utf-8")).hexdigest() + ".json")
            if cache_file.exists():
                text = json.loads(cache_file.read_text(encoding="utf-8")).get("text", "")
                self.memory_cache[p] = text
                out[p] = text
                continue

            if self.endpoint:
                b64 = self._render_page_b64(p)
                if not b64:
                    fallback.warn("chandra_render", f"could not render {p} to an image, so Chandra never saw it")
                if b64:
                    try:
                        text = self._call_endpoint(p, b64)
                        cache_file.write_text(json.dumps({"page_id": p, "text": text}, ensure_ascii=False), encoding="utf-8")
                        self.memory_cache[p] = text
                        out[p] = text
                        continue
                    except Exception as err:
                        fallback.warn("chandra_error", f"the call to {self.endpoint} failed ({err}); using the light OCR "
                                                       f"text for this page instead of Chandra")

            if not self.endpoint:
                fallback.warn("chandra_endpoint", "CHANDRA_ENDPOINT is not set (.env or --chandra-endpoint); using the "
                                                  "light OCR text instead of Chandra")

            out[p] = self.fallback.get(p, "")
        return out

    def describe(self):
        ep = self.endpoint or "fallback_local"
        return f"chandra:{ep}"


@register("branch", "enrich")
class EnrichBranch:
    name = "enrich"

    def __init__(self, argument, enricher="chandra", **context):
        self.enricher = build("enricher", argument or enricher, **context)

    def run(self, subdata):
        started = perf_counter()
        pages = subdata.top()
        text = self.enricher.parse(pages)
        passages = [Passage(page, text.get(page, ""), rank) for rank, page in enumerate(pages, 1)]
        return BranchOutput(self.name, passages,
                            cost={"seconds": perf_counter() - started, "pages": len(passages),
                                  "parsed_pages": len(passages)},
                            notes={"enricher": self.enricher.describe()})

    def describe(self):
        return f"enrich:{self.enricher.describe()}"
