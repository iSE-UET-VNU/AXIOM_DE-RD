import logging

from ..bench import BENCH, load_jsonl, work, write_jsonl
from .. import light_prep, light_retrieval
from ..light_ocr import merge_row
from . import services
from .config import get_config
from .services.base import attach, console, detach

ENGINE = "ppocrv5_doc"
log = logging.getLogger("ondemand.prep")


def run(pages_name="pages_ppocrv5_live.jsonl", gate_name="gate_ppocrv5_live.json", cache_name="te3s_ppocr",
        allow_api=False, workers=4, overwrite=False, bench=BENCH, **service_overrides):
    out = work("light_prep", bench=bench)
    pages_path, gate_path = out / pages_name, out / gate_name
    for path in (pages_path, gate_path):
        if path.exists() and not overwrite:
            raise SystemExit(f"{path} exists; pass --overwrite or pick another name")
    console()
    handler = attach(work("logs", bench=bench) / "prep.log")
    try:
        if not (service_overrides.get("ppocr_mode") or service_overrides.get("ppocr_endpoint")):
            service_overrides["ppocr_mode"] = "local"
        cfg = get_config().apply(**service_overrides)
        inspector = out / "pages_inspector.jsonl"
        if inspector.exists():
            log.info("reusing %s", inspector)
        else:
            log.info("pdf-inspector over the corpus -> %s", inspector)
            light_prep.run(bench=bench, workers=workers, ocr="none")
        rows = load_jsonl(inspector)
        targets = [r["page_id"] for r in rows if light_prep.wants_ocr(r)]
        log.info("%d of %d pages need OCR; PP-OCRv5 mode %s", len(targets), len(rows), cfg.ppocr_mode)
        service = services.create("ppocr", cfg, bench)
        with service:
            service.prepare(targets)
        found = service.rows(targets)
        missing = [p for p in targets if p not in found or found[p].get("error")]
        if missing:
            raise SystemExit(f"{len(missing)} pages have no PP-OCRv5 text yet, e.g. {missing[:3]}; see "
                             f"logs/ppocr.log and rerun (finished pages are cached)")
        merged = [merge_row(r, found[r["page_id"]], found[r["page_id"]].get("lines", []), ENGINE)
                  if r["page_id"] in found else r for r in rows]
        write_jsonl(pages_path, merged)
        log.info("wrote %s (%d pages carry PP-OCRv5 text)", pages_path, len(found))
        path = light_retrieval.build(bench=bench, pages_name=pages_name, name=gate_name, cache_name=cache_name,
                                     allow_api=allow_api)
        log.info("wrote %s", path)
        return {"pages": str(pages_path), "gate": str(path), "ocr_pages": len(found)}
    finally:
        services.close_all()
        detach(handler)
