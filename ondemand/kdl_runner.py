import argparse
import json
import os
import sys
import threading
from pathlib import Path
from time import perf_counter

from .bench import ROOT, fingerprint, load_jsonl, source_of, unit_id
from .timing import Timings

META_KEYS = ("page_count", "latency_seconds", "label_counts", "kdl_usage", "pdf_type", "native_text_regions",
             "kdl_text_fallback_regions", "text_routing_fallback_reasons", "inference_mode")


def make_provider(name, config):
    from .kdl.kdl import KDLConfig, KDLProvider

    kdl_config = KDLConfig.from_mapping(config)
    if name == "kdl":
        return KDLProvider(kdl_config)
    if name == "kdl_pdf_inspector":
        from .kdl.kdl_pdf_inspector import KdlPdfInspectorProvider

        return KdlPdfInspectorProvider(kdl_config)
    raise SystemExit(f"unknown provider {name!r}")


def safe_id(doc_id):
    return doc_id.replace("::", "__").replace("/", "_")


def load_checkpoint(out):
    done_path = out / "kdl_done.json"
    done = set(json.loads(done_path.read_text())["documents"]) if done_path.exists() else set()
    for name in ("kdl_pages.jsonl", "kdl_doc_meta.jsonl"):
        path = out / name
        if path.exists():
            kept = [r for r in load_jsonl(path) if r["doc_id"] in done]
            path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), encoding="utf-8")
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="0.1. BENCHMARK")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--provider", choices=("kdl_pdf_inspector", "kdl"), default="kdl_pdf_inspector")
    ap.add_argument("--limit-docs", type=int, default=0)
    ap.add_argument("--endpoint", default=os.environ.get("VLLM_API_BASE", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--model", default=os.environ.get("VLLM_MODEL_NAME", "kdl-frontier-parser-nano"))
    ap.add_argument("--scheduler", choices=("parsebench_document", "global_two_phase"), default="parsebench_document")
    ap.add_argument("--max-workers", type=int, default=16)
    ap.add_argument("--render-processes", type=int, default=8)
    ap.add_argument("--bbox-max-workers", type=int, default=32)
    ap.add_argument("--request-workers", type=int, default=8)
    ap.add_argument("--request-batch-size", type=int, default=1)
    ap.add_argument("--max-model-sequences", type=int, default=32)
    ap.add_argument("--retries", type=int, default=1)
    args = ap.parse_args()

    from .kdl.models import DataObject

    bench = Path(args.bench) if Path(args.bench).is_absolute() else ROOT / args.bench
    documents = load_jsonl(bench / "documents.jsonl")
    if args.limit_docs:
        documents = documents[:args.limit_docs]
    by_id = {d["doc_id"]: d for d in documents}
    args.out.mkdir(parents=True, exist_ok=True)
    raw_dir = args.out / "raw"
    bundle = fingerprint(bench)
    timings = Timings(args.out / "timings.jsonl", stage_group="kdl", bundle=bundle, provider=args.provider,
                      gpu=os.environ.get("KDL_GPU_NAME", "unknown"))
    done = load_checkpoint(args.out)
    todo = [d for d in documents if d["doc_id"] not in done]
    print(f"{len(documents)} documents, {len(done & set(by_id))} already done, {len(todo)} to parse", flush=True)
    if not todo:
        return

    max_pages = max(1024, max(d["metadata"]["page_count"] for d in documents))
    provider = make_provider(args.provider, {
        "endpoint_url": args.endpoint, "model": args.model, "max_pages": max_pages, "dpi": 144,
        "scheduler": args.scheduler, "continuous_page_queue": True, "max_workers": args.max_workers,
        "render_processes": args.render_processes, "bbox_max_workers": args.bbox_max_workers,
        "request_workers": args.request_workers, "request_batch_size": args.request_batch_size,
        "max_model_sequences": args.max_model_sequences, "request_timeout_seconds": 3600, "max_retries": 2,
        "layout_max_output_tokens": 6000, "text_max_output_tokens": 2048, "table_max_output_tokens": 5500,
        "picture_max_output_tokens": 4096, "formula_max_output_tokens": 128,
        "save_raw_outputs": True, "output_dir": str(raw_dir)})

    lock = threading.Lock()
    state = {"finished": len(done & set(by_id)), "pages": 0}

    def record_success(doc, parsed):
        expected = doc["metadata"]["page_count"]
        result = json.loads((raw_dir / safe_id(doc["doc_id"]) / "result.json").read_text(encoding="utf-8"))
        text_of = {int(p["page_number"]) - 1: p.get("content") or "" for p in result.get("markdown_pages") or []}
        rows = [{"page_id": unit_id(doc["doc_id"], p), "doc_id": doc["doc_id"], "source": doc["source"],
                 "page_index": p, "page_number": p + 1, "text": text_of.get(p, ""),
                 "kdl_status": "ok" if text_of.get(p, "").strip() else "empty", "provider": args.provider}
                for p in range(expected)]
        meta = {k: parsed.metadata.get(k) for k in META_KEYS}
        meta.update(doc_id=doc["doc_id"], source=doc["source"], expected_pages=expected,
                    pages_with_text=sum(r["kdl_status"] == "ok" for r in rows),
                    pages_beyond_manifest=sorted(p for p in text_of if p >= expected), provider=args.provider)
        with lock:
            with (args.out / "kdl_pages.jsonl").open("a", encoding="utf-8") as f:
                f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
            with (args.out / "kdl_doc_meta.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(meta, ensure_ascii=False) + "\n")
            done.add(doc["doc_id"])
            tmp = args.out / "kdl_done.json.tmp"
            tmp.write_text(json.dumps({"documents": sorted(done), "bundle": bundle}), encoding="utf-8")
            tmp.replace(args.out / "kdl_done.json")
            state["finished"] += 1
            state["pages"] += expected
            timings.record("kdl_doc", "page", expected, float(meta["latency_seconds"] or 0.0),
                           doc_id=doc["doc_id"], pages_with_text=meta["pages_with_text"])
            print(f"DOC {state['finished']}/{len(documents)} ok {doc['doc_id']} pages={expected} "
                  f"with_text={meta['pages_with_text']} latency={meta['latency_seconds']}s", flush=True)

    def run(batch):
        failures = []
        pairs = [(bench / d["path"], DataObject(object_id=safe_id(d["doc_id"]), uri=str(bench / d["path"]),
                                                metadata={"format": "pdf"})) for d in batch]

        def on_complete(index, outcome):
            doc = batch[index]
            if isinstance(outcome, Exception):
                failures.append((doc, outcome))
                print(f"DOC fail {doc['doc_id']}: {type(outcome).__name__}: {str(outcome)[:200]}", flush=True)
            else:
                record_success(doc, outcome)

        provider.parse_files_with_errors(pairs, on_document_complete=on_complete)
        return failures

    started = perf_counter()
    failures = run(todo)
    for attempt in range(args.retries):
        if not failures:
            break
        print(f"retrying {len(failures)} failed documents ({attempt + 1}/{args.retries})", flush=True)
        failures = run([d for d, _ in failures])
    timings.record("kdl_corpus_wall", "corpus", state["pages"], perf_counter() - started, docs=len(todo))
    (args.out / "kdl_failed.jsonl").write_text("".join(
        json.dumps({"doc_id": d["doc_id"], "source": source_of(d["doc_id"]), "error": f"{type(e).__name__}: {e}"},
                   ensure_ascii=False) + "\n" for d, e in failures), encoding="utf-8")
    print(f"finished {state['finished']}/{len(documents)} documents in {perf_counter() - started:.0f}s; "
          f"{len(failures)} failed", flush=True)
    if failures:
        sys.exit(2)


if __name__ == "__main__":
    main()
