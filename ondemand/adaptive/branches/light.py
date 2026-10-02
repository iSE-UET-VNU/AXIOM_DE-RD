import base64
import os
import tempfile
from time import perf_counter

import fitz
import requests

from ...bench import BENCH, documents, load_jsonl, work
from ..config import get_config
from ..registry import register
from ..subdata import BranchOutput, Passage


@register("branch", "light")
class LightBranch:
    name = "light"

    def __init__(self, argument=None, bench=BENCH, pages_name="pages_ppocrv5.jsonl", **context):
        cfg = get_config()
        source = work("light_prep", bench=bench) / (argument or pages_name)
        if not source.exists():
            # Try secondary fallback
            fallback_source = work("light_prep", bench=bench) / "pages_ocr_ppocr.jsonl"
            if fallback_source.exists():
                source = fallback_source
            else:
                raise SystemExit(f"{source} is missing; run light-prep first")
        self.source = source
        self.text = {r["page_id"]: r["text"] for r in load_jsonl(source)}
        self.bench = bench
        self.ppocr_endpoint = context.get("ppocr_endpoint") or cfg.ppocr_endpoint
        self.ppocr_local = context.get("ppocr_local") or cfg.ppocr_local
        self.ocr_engine = None
        self.docs = {d["doc_id"]: bench / d["path"] for d in documents(bench)}

    def _get_ocr_engine(self):
        if self.ocr_engine is not None:
            return self.ocr_engine
        try:
            from paddleocr import PaddleOCR
            import torch
            device = "gpu:0" if torch.cuda.is_available() else "cpu"
            self.ocr_engine = PaddleOCR(
                text_detection_model_name="PP-OCRv5_server_det",
                text_recognition_model_name="PP-OCRv5_server_rec",
                use_doc_orientation_classify=True, use_doc_unwarping=False,
                use_textline_orientation=False, device=device
            )
            return self.ocr_engine
        except Exception as err:
            raise RuntimeError(f"Local PP-OCRv5 initialization failed: {err}")

    def _render_page_png(self, page_id, temp_path):
        if "#page=" not in page_id:
            return False
        doc_id, page_str = page_id.split("#page=")
        path = self.docs.get(doc_id)
        if not path or not path.exists():
            return False
        try:
            with fitz.open(path) as doc:
                page = doc.load_page(int(page_str))
                dpi = 200
                side = max(page.rect.width, page.rect.height) * dpi / 72
                if side > 4000:
                    dpi = int(dpi * 4000 / side)
                page.get_pixmap(dpi=dpi).save(temp_path)
            return True
        except Exception:
            return False

    def _ocr_page(self, page_id):
        # 1. Remote microservice endpoint if configured
        if self.ppocr_endpoint:
            url = self.ppocr_endpoint.rstrip("/")
            if not url.endswith("/ocr"):
                url = f"{url}/ocr"
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                tmp_png = tf.name
            try:
                if self._render_page_png(page_id, tmp_png):
                    with open(tmp_png, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode("utf-8")
                    res = requests.post(url, json={"image": b64, "page_id": page_id}, timeout=30)
                    if res.status_code == 200:
                        text = res.json().get("text", "")
                        if text:
                            return text
            except Exception:
                pass
            finally:
                if os.path.exists(tmp_png):
                    os.unlink(tmp_png)

        # 2. Local in-process PaddleOCR if requested
        if self.ppocr_local:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                tmp_png = tf.name
            try:
                if self._render_page_png(page_id, tmp_png):
                    ocr = self._get_ocr_engine()
                    texts = [t for p in ocr.predict(tmp_png) for t in p.get("rec_texts", [])]
                    return "\n".join(texts)
            except Exception:
                pass
            finally:
                if os.path.exists(tmp_png):
                    os.unlink(tmp_png)

        return self.text.get(page_id, "")

    def run(self, subdata):
        started = perf_counter()
        passages = []
        for rank, page in enumerate(subdata.top(), 1):
            if page in self.text and not self.ppocr_local and not self.ppocr_endpoint:
                text = self.text[page]
            elif self.ppocr_local or self.ppocr_endpoint:
                text = self._ocr_page(page)
            else:
                text = self.text.get(page, "")
            passages.append(Passage(page, text, rank))

        empty = sum(not p.text.strip() for p in passages)
        return BranchOutput(self.name, passages,
                            cost={"seconds": perf_counter() - started, "pages": len(passages), "parsed_pages": 0},
                            notes={"source": self.source.name, "empty_pages": empty})

    def describe(self):
        mode = "local_gpu" if self.ppocr_local else ("endpoint" if self.ppocr_endpoint else "cache")
        return f"light:{self.source.name}:{mode}"
