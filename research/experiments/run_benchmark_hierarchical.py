"""Run generic raw-input BM25 + V-SPLADE hierarchical retrieval.

The command is intentionally stageable so a Colab runtime can resume after an
interruption.  The default cascade is the fixed Physics arm:

    page = .70 BM25 + .30 V-SPLADE
    file = .50 direct-file-BM25 + .50 max(page)
    page_final = .85 page + .15 parent-file

The runner stops after the light stage and writes a slim top-20 run.  It does
not call KDL, a reranker, ColVec, an LLM, or a QA service.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Any, Iterable, Mapping, Sequence
import urllib.parse
import urllib.request
import zipfile

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.data_discovery.pipeline import PdfInspectorPageParser  # noqa: E402
from research.data_discovery.raw_hierarchical import (  # noqa: E402
    VECTOR_DIM,
    RawDocument,
    RawPage,
    aggregate_file_scores,
    atomic_json_dump,
    atomic_write_text,
    build_file_bm25,
    build_page_bm25,
    canonical_page_id,
    load_documents,
    load_queries,
    normalise_scores,
    read_jsonl,
    safe_name,
    sha256_file,
    sort_scores,
    write_jsonl,
)
from src.retrieval.sparse import BM25Index  # noqa: E402


DEFAULT_DATASET_ROOT = ROOT / "data/raw/BENCHMARK"
DEFAULT_OUTPUT_ROOT = ROOT / "data/work/benchmark_hierarchical"
DEFAULT_MODEL = "naver/v-splade-efficient"
STAGES = (
    "validate",
    "normalize-qrels",
    "parse",
    "encode-vsplade",
    "build-bm25",
    "retrieve",
    "evaluate",
    "package",
)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


class EventLog:
    def __init__(self, root: Path) -> None:
        self.path = root / "logs/events.jsonl"
        self.error_path = root / "logs/errors.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, status: str, **payload: Any) -> None:
        row = {"timestamp": _now(), "event": event, "status": status, **payload}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    def error(self, event: str, error: BaseException, **payload: Any) -> None:
        row = {
            "timestamp": _now(),
            "event": event,
            "status": "error",
            "error_type": type(error).__name__,
            "error": str(error),
            **payload,
        }
        with self.error_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _save_csr(path: Path, matrix: sparse.csr_matrix) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    np.savez_compressed(
        temporary,
        indices=matrix.indices.astype(np.int32, copy=False),
        indptr=matrix.indptr.astype(np.int32, copy=False),
        format=np.asarray("csr").astype("S3"),
        shape=np.asarray(matrix.shape, dtype=np.int64),
        data=matrix.data.astype(np.float32, copy=False),
    )
    # numpy appends .npz to a path without that suffix.
    generated = temporary if temporary.suffix == ".npz" else temporary.with_suffix(temporary.suffix + ".npz")
    generated.replace(path)


def _load_csr(path: Path) -> sparse.csr_matrix:
    payload = np.load(path, allow_pickle=True)
    shape = tuple(int(value) for value in payload["shape"])
    return sparse.csr_matrix(
        (
            payload["data"].astype(np.float32, copy=False),
            payload["indices"].astype(np.int32, copy=False),
            payload["indptr"].astype(np.int32, copy=False),
        ),
        shape=shape,
    )


def _package_versions() -> dict[str, str]:
    names = (
        "numpy",
        "scipy",
        "PyMuPDF",
        "pdf-inspector",
        "sentence-transformers",
        "transformers",
        "torch",
    )
    output: dict[str, str] = {}
    for name in names:
        try:
            output[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            output[name] = "not-installed"
    try:
        import torch

        output["torch.cuda"] = str(bool(torch.cuda.is_available()))
        if torch.cuda.is_available():
            output["torch.gpu"] = str(torch.cuda.get_device_name(0))
    except Exception as exc:  # pragma: no cover - diagnostics only
        output["torch.error"] = str(exc)
    return output


def _config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "contract_version": "benchmark-hierarchical-v1",
        "dataset_root": str(args.dataset_root.resolve()),
        "documents": str(args.documents.resolve()),
        "queries": str(args.queries.resolve()),
        "qrels": str(args.qrels.resolve()) if args.qrels else None,
        "model": args.model,
        "device": args.device,
        "batch_size": args.batch_size,
        "query_batch_size": args.query_batch_size,
        "render_dpi": args.render_dpi,
        "file_k": args.file_k,
        "metric_page_k": args.metric_page_k,
        "saved_page_k": args.saved_page_k,
        "limit_documents": args.limit_documents,
        "limit_queries": args.limit_queries,
        "no_qrels": args.no_qrels,
        "page_bm25_weight": 0.70,
        "file_direct_weight": 0.50,
        "parent_weight": 0.15,
        "input_image_policy": "visual-only",
        "qrel_policy": "source-corpus-map-or-mpdoc-page",
    }


def _config_hash(config: Mapping[str, Any]) -> str:
    raw = json.dumps(dict(config), sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


class Runner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.root = args.output_dir.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.events = EventLog(self.root)
        self.documents = load_documents(self.args.dataset_root, self.args.documents)
        self.queries = load_queries(self.args.queries)
        if self.args.limit_documents is not None:
            self.documents = self.documents[: self.args.limit_documents]
        if self.args.limit_queries is not None:
            self.queries = self.queries[: self.args.limit_queries]
        self.input_signature = self._input_signature()
        self.config = _config(args)
        self.config["input_signature"] = self.input_signature
        self.config_hash = _config_hash(self.config)
        self.config_path = self.root / "config.json"
        atomic_json_dump(self.config_path, {**self.config, "config_hash": self.config_hash})
        self.pages: list[RawPage] | None = None

    def _input_signature(self) -> str:
        payload = {
            "documents": [
                {
                    "doc_id": document.doc_id,
                    "path": str(document.path),
                    "sha256": sha256_file(document.path),
                }
                for document in self.documents
            ],
            "queries_sha256": sha256_file(self.args.queries),
            "qrels_sha256": (
                sha256_file(self.args.qrels)
                if self.args.qrels is not None and self.args.qrels.is_file()
                else None
            ),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()

    def load_inputs(self) -> tuple[list[RawDocument], list[dict[str, Any]]]:
        return self.documents, self.queries

    def marker(self, stage: str) -> Path:
        return self.root / f".stage_{safe_name(stage)}.json"

    def complete(self, stage: str) -> bool:
        if stage in self.args.force_stage:
            return False
        marker = self.marker(stage)
        if not marker.is_file():
            return False
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
            if payload.get("config_hash") != self.config_hash:
                return False
            if payload.get("input_signature") != self.input_signature:
                return False
            return all((self.root / relative).is_file() for relative in self._stage_artifacts(stage))
        except Exception:
            return False

    def _stage_artifacts(self, stage: str) -> list[str]:
        run_name = f"hierarchical_kf{self.args.file_k}_top{self.args.saved_page_k}.jsonl"
        return {
            "validate": ["manifest.json", "input_inventory.jsonl"],
            "normalize-qrels": ["qrels_page_level.jsonl", "qrels_mapping.json"],
            "parse": ["parsing/pages.jsonl", "parsing/summary.json"],
            "encode-vsplade": [
                "vsplade/page_vectors.npz",
                "vsplade/page_metadata.json",
                "vsplade/query_vectors.npz",
                "vsplade/query_metadata.json",
                "vsplade/timing.json",
            ],
            "build-bm25": [
                "indexes/page_bm25.json",
                "indexes/file_bm25.json",
                "indexes/summary.json",
            ],
            "retrieve": [f"runs/{run_name}", "runs/timing.json"],
            "evaluate": ["reports/report.json", "reports/report.md", "reports/per_query.jsonl"],
            "package": [f"bundle/{self.root.name}.zip"],
        }[stage]

    def mark_complete(self, stage: str, started: float, **payload: Any) -> None:
        elapsed = time.perf_counter() - started
        result_payload = dict(payload)
        result_seconds = result_payload.pop("seconds", None)
        if result_seconds is not None:
            result_payload["result_seconds"] = result_seconds
        atomic_json_dump(
            self.marker(stage),
            {
                "stage": stage,
                "config_hash": self.config_hash,
                "input_signature": self.input_signature,
                "seconds": elapsed,
                **result_payload,
            },
        )
        timing_path = self.root / "timing.json"
        timing = (
            json.loads(timing_path.read_text(encoding="utf-8"))
            if timing_path.is_file()
            else {"stages": {}}
        )
        timing["stages"][stage] = {
            "wall_seconds": elapsed,
            "completed_at": _now(),
            **result_payload,
        }
        timing["total_completed_stage_seconds"] = sum(
            float(item.get("wall_seconds", 0.0)) for item in timing["stages"].values()
        )
        atomic_json_dump(timing_path, timing)
        self.events.write(stage, "complete", seconds=elapsed, **result_payload)

    def run_stage(self, stage: str, callback: Any) -> None:
        if self.complete(stage):
            self.events.write(stage, "skipped", reason="matching-complete-marker")
            print(f"[{stage}] skipped: matching artifact marker", flush=True)
            return
        started = time.perf_counter()
        self.events.write(stage, "started", config_hash=self.config_hash)
        print(f"[{stage}] started", flush=True)
        try:
            payload = callback() or {}
            self.mark_complete(stage, started, **payload)
            print(f"[{stage}] complete in {time.perf_counter() - started:.2f}s", flush=True)
        except Exception as exc:
            self.events.error(stage, exc)
            raise

    def validate(self) -> dict[str, Any]:
        documents, queries = self.load_inputs()
        inventory: list[dict[str, Any]] = []
        page_total = 0
        for document in documents:
            if not document.is_pdf and not document.is_image:
                raise ValueError(f"Unsupported raw input: {document.path}")
            if document.is_pdf:
                import fitz

                with fitz.open(str(document.path)) as pdf:
                    page_count = len(pdf)
            else:
                from PIL import Image

                with Image.open(document.path) as image:
                    image.verify()
                page_count = 1
            if document.page_count_hint is not None and page_count != document.page_count_hint:
                raise ValueError(
                    f"Page count mismatch for {document.doc_id}: actual={page_count}, "
                    f"manifest={document.page_count_hint}"
                )
            page_total += page_count
            inventory.append({**document.manifest_record(self.args.dataset_root), "page_count": page_count})
        write_jsonl(self.root / "input_inventory.jsonl", inventory)
        atomic_json_dump(
            self.root / "manifest.json",
            {
                "contract_version": "benchmark-hierarchical-v1",
                "config_hash": self.config_hash,
                "dataset_root": str(self.args.dataset_root.resolve()),
                "documents": len(documents),
                "queries": len(queries),
                "pages": page_total,
                "sources": dict(_counts_by(documents, lambda item: item.source)),
                "package_versions": _package_versions(),
                "git_sha": _git_sha(),
                "input_inventory": str(self.root / "input_inventory.jsonl"),
            },
        )
        return {"documents": len(documents), "queries": len(queries), "pages": page_total}

    def normalize_qrels(self) -> dict[str, Any]:
        if self.args.no_qrels or self.args.qrels is None or not self.args.qrels.is_file():
            atomic_json_dump(self.root / "qrels_mapping.json", {"available": False, "reason": "no qrels"})
            write_jsonl(self.root / "qrels_page_level.jsonl", [])
            return {"available": False}
        documents, queries = self.load_inputs()
        doc_by_id = {doc.doc_id: doc for doc in documents}
        inventory_path = self.root / "input_inventory.jsonl"
        inventory_counts = {
            str(row["doc_id"]): int(row["page_count"])
            for row in read_jsonl(inventory_path)
        } if inventory_path.is_file() else {}
        rows = read_jsonl(self.args.qrels)
        source_maps: dict[str, dict[str, dict[str, Any]]] = {}
        output: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        query_ids = {str(row["query_id"]) for row in queries}
        for subset in sorted({document.source.removeprefix("vidore_") for document in documents if document.source.startswith("vidore_")}):
            source_maps[subset] = _download_source_page_map(subset, self.root / "qrels_source_maps")
        for row in rows:
            qid = str(row.get("query_id") or row.get("qid") or "")
            doc_id = str(row.get("doc_id") or "")
            document = doc_by_id.get(doc_id)
            if not qid or qid not in query_ids or document is None:
                if self.args.limit_documents is None and self.args.limit_queries is None:
                    failures.append({"row": row, "reason": "unknown query or document"})
                continue
            evidence = row.get("evidence") or []
            for item in evidence:
                try:
                    if document.source == "mpdocvqa":
                        page_index = int(item["pdf_page_number"]) - 1
                        source_page = str(item.get("source_page_path") or "")
                    elif "source_corpus_id" in item:
                        source_id = str(item["source_corpus_id"])
                        subset = document.source.removeprefix("vidore_")
                        source = source_maps[subset].get(source_id)
                        if source is None:
                            raise ValueError(f"source_corpus_id {source_id} not found in {subset}")
                        expected_source_doc = document.source_doc_id or ""
                        if str(source["doc_id"]) != expected_source_doc:
                            raise ValueError(
                                f"source document mismatch: qrel={source['doc_id']!r}, "
                                f"manifest={expected_source_doc!r}"
                            )
                        page_index = int(source["page_number_in_doc"])
                        source_page = source_id
                    else:
                        raise ValueError("evidence has neither pdf_page_number nor source_corpus_id")
                    page_count = int(inventory_counts.get(document.doc_id, document.page_count_hint or 0))
                    if page_index < 0 or (page_count and page_index >= page_count):
                        raise ValueError(f"page {page_index} outside {document.doc_id} ({page_count})")
                    relevance = int(item.get("score", row.get("relevance", 1)))
                    output.append(
                        {
                            "query_id": qid,
                            "page_id": canonical_page_id(doc_id, page_index),
                            "doc_id": doc_id,
                            "page_index": page_index,
                            "relevance": relevance,
                            "source_evidence_id": source_page,
                        }
                    )
                except Exception as exc:
                    failures.append({"query_id": qid, "doc_id": doc_id, "evidence": item, "reason": str(exc)})
        if failures:
            atomic_json_dump(self.root / "qrels_mapping_failures.json", {"failures": failures})
            raise RuntimeError(f"Could not normalize {len(failures)} qrel evidence records; see qrels_mapping_failures.json")
        dedup: dict[tuple[str, str], dict[str, Any]] = {}
        for row in output:
            key = (row["query_id"], row["page_id"])
            if key not in dedup or row["relevance"] > dedup[key]["relevance"]:
                dedup[key] = row
        output = sorted(dedup.values(), key=lambda row: (row["query_id"], row["page_id"]))
        write_jsonl(self.root / "qrels_page_level.jsonl", output)
        atomic_json_dump(
            self.root / "qrels_mapping.json",
            {
                "available": True,
                "input_rows": len(rows),
                "output_page_qrels": len(output),
                "queries_with_qrels": len({row["query_id"] for row in output}),
                "source_maps": sorted(source_maps),
                "failures": 0,
            },
        )
        manifest_path = self.root / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["qrels_mapping_sha256"] = sha256_file(self.root / "qrels_mapping.json")
            atomic_json_dump(manifest_path, manifest)
        return {"available": True, "page_qrels": len(output)}

    def parse(self) -> dict[str, Any]:
        documents, _ = self.load_inputs()
        parser = PdfInspectorPageParser()
        checkpoint_dir = self.root / "parsing/files"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        all_pages: list[RawPage] = []
        file_stats: list[dict[str, Any]] = []
        errors = 0
        for document in documents:
            target = checkpoint_dir / f"{safe_name(document.doc_id)}.jsonl"
            stat_path = checkpoint_dir / f"{safe_name(document.doc_id)}.json"
            if target.is_file() and stat_path.is_file() and document.path.stat().st_size > 0:
                stat = json.loads(stat_path.read_text(encoding="utf-8"))
                if stat.get("input_sha256") == sha256_file(document.path):
                    cached = [RawPage(**row) for row in read_jsonl(target)]
                    all_pages.extend(cached)
                    file_stats.append({**stat, "resumed": True})
                    self.events.write("parse_document", "skipped", doc_id=document.doc_id, pages=len(cached))
                    continue
            started = time.perf_counter()
            status = "ok"
            error_text: str | None = None
            if document.is_image:
                pages = [
                    RawPage(
                        page_id=canonical_page_id(document.doc_id, 0),
                        doc_id=document.doc_id,
                        source=document.source,
                        source_path=str(document.path),
                        relative_path=document.relative_path,
                        page_index=0,
                        page_number=1,
                        text="",
                        visual_only=True,
                    )
                ]
            else:
                try:
                    extracted = parser.parse(document.path, source_uri=document.doc_id)
                    by_index = {int(item.page_index): item for item in extracted}
                    import fitz

                    with fitz.open(str(document.path)) as pdf:
                        actual_count = len(pdf)
                    pages = []
                    for page_index in range(actual_count):
                        item = by_index.get(page_index)
                        pages.append(
                            RawPage(
                                page_id=canonical_page_id(document.doc_id, page_index),
                                doc_id=document.doc_id,
                                source=document.source,
                                source_path=str(document.path),
                                relative_path=document.relative_path,
                                page_index=page_index,
                                page_number=page_index + 1,
                                text=item.text if item else "",
                                visual_only=False,
                                needs_ocr=bool(item.needs_ocr) if item else False,
                                parse_status="ok" if item else "missing-page-record",
                                parse_error=None if item else "pdf-inspector returned no record",
                            )
                        )
                except Exception as exc:
                    errors += 1
                    status = "error"
                    error_text = str(exc)
                    import fitz

                    with fitz.open(str(document.path)) as pdf:
                        actual_count = len(pdf)
                    pages = [
                        RawPage(
                            page_id=canonical_page_id(document.doc_id, page_index),
                            doc_id=document.doc_id,
                            source=document.source,
                            source_path=str(document.path),
                            relative_path=document.relative_path,
                            page_index=page_index,
                            page_number=page_index + 1,
                            text="",
                            visual_only=False,
                            parse_status="error",
                            parse_error=error_text,
                        )
                        for page_index in range(actual_count)
                    ]
                    self.events.error("parse_document", exc, doc_id=document.doc_id)
            write_jsonl(target, (page.as_dict() for page in pages))
            stat = {
                "doc_id": document.doc_id,
                "input_sha256": sha256_file(document.path),
                "status": status,
                "error": error_text,
                "pages": len(pages),
                "text_pages": sum(bool(page.text.strip()) for page in pages),
                "empty_pages": sum(not page.text.strip() for page in pages),
                "seconds": time.perf_counter() - started,
            }
            atomic_json_dump(stat_path, stat)
            self.events.write("parse_document", status, **stat)
            all_pages.extend(pages)
            file_stats.append(stat)
        document_order = {document.doc_id: index for index, document in enumerate(documents)}
        all_pages.sort(key=lambda page: (document_order[page.doc_id], page.page_index))
        write_jsonl(self.root / "parsing/pages.jsonl", (page.as_dict() for page in all_pages))
        atomic_json_dump(
            self.root / "parsing/summary.json",
            {
                "documents": len(documents),
                "pages": len(all_pages),
                "text_pages": sum(bool(page.text.strip()) for page in all_pages),
                "visual_only_pages": sum(page.visual_only for page in all_pages),
                "parse_errors": errors,
                "files": file_stats,
            },
        )
        return {"documents": len(documents), "pages": len(all_pages), "parse_errors": errors}

    def encode_vsplade(self) -> dict[str, Any]:
        documents, queries = self.load_inputs()
        pages = [RawPage(**row) for row in read_jsonl(self.root / "parsing/pages.jsonl")]
        self.pages = pages
        visual_root = self.root / "vsplade"
        checkpoint_root = visual_root / "page_checkpoints"
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        rendering_root = self.root / "rendering"
        rendering_root.mkdir(parents=True, exist_ok=True)

        from PIL import Image
        import fitz
        from sentence_transformers import SparseEncoder

        model_started = time.perf_counter()
        model = SparseEncoder(self.args.model, trust_remote_code=True, device=self.args.device)
        model_seconds = time.perf_counter() - model_started

        matrices: list[sparse.csr_matrix] = []
        metadata: list[dict[str, Any]] = []
        doc_stats: list[dict[str, Any]] = []
        page_by_doc: dict[str, list[RawPage]] = defaultdict(list)
        for page in pages:
            page_by_doc[page.doc_id].append(page)
        document_by_id = {document.doc_id: document for document in documents}
        vector_offset = 0
        for document in documents:
            checkpoint = checkpoint_root / f"{safe_name(document.doc_id)}.npz"
            checkpoint_meta = checkpoint_root / f"{safe_name(document.doc_id)}.metadata.json"
            doc_pages = page_by_doc[document.doc_id]
            if checkpoint.is_file() and checkpoint_meta.is_file():
                matrix = _load_csr(checkpoint)
                doc_meta = json.loads(checkpoint_meta.read_text(encoding="utf-8"))
                if (
                    matrix.shape[0] != len(doc_pages)
                    or len(doc_meta) != len(doc_pages)
                    or any(item.get("input_sha256") != sha256_file(document.path) for item in doc_meta)
                ):
                    raise RuntimeError(f"V-SPLADE checkpoint shape mismatch for {document.doc_id}")
                matrices.append(matrix)
                metadata.extend(
                    [{**item, "vector_row": vector_offset + index} for index, item in enumerate(doc_meta)]
                )
                vector_offset += matrix.shape[0]
                doc_stats.append({"doc_id": document.doc_id, "pages": len(doc_pages), "seconds": 0.0, "resumed": True})
                self.events.write("vsplade_document", "skipped", doc_id=document.doc_id, pages=len(doc_pages))
                continue
            started = time.perf_counter()
            images: list[Image.Image] = []
            if document.is_pdf:
                with fitz.open(str(document.path)) as pdf:
                    for page in doc_pages:
                        pix = pdf[page.page_index].get_pixmap(dpi=self.args.render_dpi, alpha=False)
                        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        images.append(image)
            else:
                with Image.open(document.path) as source_image:
                    images.append(source_image.convert("RGB"))
            encoded_batches: list[sparse.csr_matrix] = []
            try:
                for begin in range(0, len(images), self.args.batch_size):
                    batch = images[begin : begin + self.args.batch_size]
                    encoded = model.encode_document(
                        batch,
                        batch_size=len(batch),
                        show_progress_bar=False,
                        convert_to_tensor=True,
                    )
                    dense = encoded.to_dense().float().cpu().numpy()
                    encoded_batches.append(sparse.csr_matrix(dense, dtype=np.float32))
            finally:
                for image in images:
                    image.close()
            matrix = sparse.vstack(encoded_batches, format="csr") if encoded_batches else sparse.csr_matrix((0, VECTOR_DIM), dtype=np.float32)
            if matrix.shape[1] != VECTOR_DIM:
                raise RuntimeError(f"Unexpected V-SPLADE dimension for {document.doc_id}: {matrix.shape}")
            doc_meta = [
                {
                    "vector_row": index,
                    "page_id": page.page_id,
                    "doc_id": page.doc_id,
                    "input_sha256": sha256_file(document.path),
                    "source": page.source,
                    "relative_path": page.relative_path,
                    "page_index": page.page_index,
                    "page_number": page.page_number,
                    "visual_only": page.visual_only,
                }
                for index, page in enumerate(doc_pages)
            ]
            _save_csr(checkpoint, matrix)
            atomic_json_dump(checkpoint_meta, doc_meta)
            seconds = time.perf_counter() - started
            stat = {"doc_id": document.doc_id, "pages": len(doc_pages), "seconds": seconds, "nnz": int(matrix.nnz), "resumed": False}
            atomic_json_dump(checkpoint_root / f"{safe_name(document.doc_id)}.timing.json", stat)
            self.events.write("vsplade_document", "complete", **stat)
            matrices.append(matrix)
            metadata.extend(
                [{**item, "vector_row": vector_offset + index} for index, item in enumerate(doc_meta)]
            )
            vector_offset += matrix.shape[0]
            doc_stats.append(stat)
            print(f"[vsplade] {document.doc_id}: {len(doc_pages)} pages in {seconds:.2f}s", flush=True)
        page_matrix = sparse.vstack(matrices, format="csr") if matrices else sparse.csr_matrix((0, VECTOR_DIM), dtype=np.float32)
        if len(metadata) != len(pages):
            raise RuntimeError("V-SPLADE metadata/page count mismatch")
        page_position = {page["page_id"]: index for index, page in enumerate(metadata)}
        expected = [page.page_id for page in pages]
        if list(page_position) != expected:
            raise RuntimeError("V-SPLADE page ordering does not match parsing order")
        _save_csr(visual_root / "page_vectors.npz", page_matrix)
        atomic_json_dump(visual_root / "page_metadata.json", metadata)
        query_started = time.perf_counter()
        query_encoded = model.encode_query(
            [row["query"] for row in queries],
            batch_size=self.args.query_batch_size,
            show_progress_bar=False,
            convert_to_tensor=True,
        )
        query_matrix = sparse.csr_matrix(query_encoded.to_dense().float().cpu().numpy(), dtype=np.float32)
        if query_matrix.shape != (len(queries), VECTOR_DIM):
            raise RuntimeError(f"Unexpected query vector shape: {query_matrix.shape}")
        _save_csr(visual_root / "query_vectors.npz", query_matrix)
        atomic_json_dump(
            visual_root / "query_metadata.json",
            [{"row": index, "query_id": row["query_id"], "query": row["query"]} for index, row in enumerate(queries)],
        )
        query_seconds = time.perf_counter() - query_started
        atomic_json_dump(
            rendering_root / "page_metadata.json",
            {
                "dpi": self.args.render_dpi,
                "pages": len(pages),
                "documents": len(documents),
                "rendered_in_memory": True,
                "note": "Rendered images are not persisted; page vector checkpoints are persisted.",
            },
        )
        atomic_json_dump(
            visual_root / "timing.json",
            {
                "model_load_seconds": model_seconds,
                "page_documents": doc_stats,
                "query_encoding_seconds": query_seconds,
                "page_count": len(pages),
                "query_count": len(queries),
                "page_nnz": int(page_matrix.nnz),
                "query_nnz": int(query_matrix.nnz),
                "device": self.args.device,
                "model": self.args.model,
            },
        )
        return {"pages": len(pages), "queries": len(queries), "model_load_seconds": model_seconds, "query_encoding_seconds": query_seconds}

    def build_indexes(self) -> dict[str, Any]:
        pages = [RawPage(**row) for row in read_jsonl(self.root / "parsing/pages.jsonl")]
        started = time.perf_counter()
        page_index = build_page_bm25(pages)
        file_index = build_file_bm25(pages)
        page_index.save(self.root / "indexes/page_bm25.json")
        file_index.save(self.root / "indexes/file_bm25.json")
        stats = {
            "page_count": len(pages),
            "file_count": len({page.doc_id for page in pages}),
            "page_index_bytes": (self.root / "indexes/page_bm25.json").stat().st_size,
            "file_index_bytes": (self.root / "indexes/file_bm25.json").stat().st_size,
            "seconds": time.perf_counter() - started,
        }
        atomic_json_dump(self.root / "indexes/summary.json", stats)
        return stats

    def retrieve(self) -> dict[str, Any]:
        pages = [RawPage(**row) for row in read_jsonl(self.root / "parsing/pages.jsonl")]
        documents, queries = self.load_inputs()
        page_index = BM25Index.load(self.root / "indexes/page_bm25.json")
        file_index = BM25Index.load(self.root / "indexes/file_bm25.json")
        page_vectors = _load_csr(self.root / "vsplade/page_vectors.npz")
        query_vectors = _load_csr(self.root / "vsplade/query_vectors.npz")
        if page_vectors.shape[0] != len(pages) or query_vectors.shape[0] != len(queries):
            raise RuntimeError("Vector/index/query count mismatch before retrieval")
        pages_by_id = {page.page_id: page for page in pages}
        page_to_file = {page.page_id: page.doc_id for page in pages}
        file_ids = [document.doc_id for document in documents]
        page_positions = {page.page_id: index for index, page in enumerate(pages)}
        score_matrix = (query_vectors @ page_vectors.T).toarray()
        output_rows: list[dict[str, Any]] = []
        timings: list[dict[str, Any]] = []
        started_all = time.perf_counter()
        for query_row, query in enumerate(queries):
            started = time.perf_counter()
            bm25_hits = page_index.search(query["query"], len(pages))
            page_bm25 = {page_index.chunk_ids[position]: float(score) for position, score in bm25_hits}
            bm25_norm = normalise_scores(page_bm25)
            visual = {page.page_id: float(score_matrix[query_row, position]) for position, page in enumerate(pages)}
            visual_norm = normalise_scores(visual)
            page_base = {
                page.page_id: 0.70 * bm25_norm.get(page.page_id, 0.0) + 0.30 * visual_norm.get(page.page_id, 0.0)
                for page in pages
            }
            direct_hits = file_index.search(query["query"], len(file_ids))
            file_direct = {file_index.chunk_ids[position]: float(score) for position, score in direct_hits}
            direct_norm = normalise_scores(file_direct)
            file_pool = aggregate_file_scores(page_base, page_to_file, file_ids)
            pool_norm = normalise_scores(file_pool)
            file_scores = {
                file_id: 0.50 * direct_norm.get(file_id, 0.0) + 0.50 * pool_norm.get(file_id, 0.0)
                for file_id in file_ids
            }
            file_ranked = sort_scores(file_scores)[: min(self.args.file_k, len(file_ids))]
            selected_files = {file_id for file_id, _ in file_ranked}
            parent_norm = normalise_scores(file_scores)
            page_scores = {
                page.page_id: 0.85 * page_base[page.page_id] + 0.15 * parent_norm.get(page.doc_id, 0.0)
                for page in pages
                if page.doc_id in selected_files
            }
            page_ranked = sort_scores(page_scores)[: self.args.saved_page_k]
            chunks = []
            for rank, (page_id, score) in enumerate(page_ranked, start=1):
                page = pages_by_id[page_id]
                chunks.append(
                    {
                        "page_id": page_id,
                        "doc_id": page.doc_id,
                        "source": page.source,
                        "relative_path": page.relative_path,
                        "page_index": page.page_index,
                        "page_number": page.page_number,
                        "rank": rank,
                        "score": round(float(score), 8),
                        "components": {
                            "page_bm25": round(float(bm25_norm.get(page_id, 0.0)), 8),
                            "vsplade": round(float(visual_norm.get(page_id, 0.0)), 8),
                            "page_base": round(float(page_base[page_id]), 8),
                            "parent_file": round(float(parent_norm.get(page.doc_id, 0.0)), 8),
                        },
                    }
                )
            output_rows.append(
                {
                    "query_id": query["query_id"],
                    "query": query["query"],
                    "file_k": self.args.file_k,
                    "metric_page_k": self.args.metric_page_k,
                    "saved_page_k": self.args.saved_page_k,
                    "selected_files": [
                        {"file_id": file_id, "score": round(float(score), 8)}
                        for file_id, score in file_ranked
                    ],
                    "candidate_page_count": sum(page.doc_id in selected_files for page in pages),
                    "chunks": chunks,
                }
            )
            timings.append(
                {
                    "query_id": query["query_id"],
                    "seconds": time.perf_counter() - started,
                    "candidate_files": len(selected_files),
                    "candidate_pages": sum(page.doc_id in selected_files for page in pages),
                }
            )
        run_path = self.root / f"runs/hierarchical_kf{self.args.file_k}_top{self.args.saved_page_k}.jsonl"
        write_jsonl(run_path, output_rows)
        atomic_json_dump(
            self.root / "runs/timing.json",
            {
                "queries": timings,
                "retrieval_seconds": time.perf_counter() - started_all,
                "mean_query_seconds": sum(row["seconds"] for row in timings) / len(timings),
                "mean_candidate_pages": sum(row["candidate_pages"] for row in timings) / len(timings),
            },
        )
        return {"queries": len(output_rows), "saved_pages": self.args.saved_page_k, "seconds": time.perf_counter() - started_all}

    def evaluate(self) -> dict[str, Any]:
        run_name = f"hierarchical_kf{self.args.file_k}_top{self.args.saved_page_k}.jsonl"
        run_rows = read_jsonl(self.root / "runs" / run_name)
        qrels_path = self.root / "qrels_page_level.jsonl"
        if not qrels_path.is_file() or not read_jsonl(qrels_path):
            report = {"available": False, "reason": "no normalized qrels", "queries": len(run_rows)}
            atomic_json_dump(self.root / "reports/report.json", report)
            atomic_write_text(self.root / "reports/report.md", "# Benchmark hierarchical retrieval\n\nEvaluation unavailable: no normalized qrels.\n")
            return report
        qrels: dict[str, dict[str, int]] = defaultdict(dict)
        for row in read_jsonl(qrels_path):
            qrels[str(row["query_id"])][str(row["page_id"])] = int(row["relevance"])
        page_recall10: list[float] = []
        page_recall20: list[float] = []
        page_hit10: list[float] = []
        ndcg10: list[float] = []
        candidate_file_recall: list[float] = []
        derived_file_recall: list[float] = []
        per_query: list[dict[str, Any]] = []
        for row in run_rows:
            qid = str(row["query_id"])
            gold = qrels.get(qid, {})
            gold_pages = set(gold)
            chunks = list(row["chunks"])
            top10 = [str(item["page_id"]) for item in chunks[: self.args.metric_page_k]]
            top20 = [str(item["page_id"]) for item in chunks[: self.args.saved_page_k]]
            found10 = set(top10) & gold_pages
            found20 = set(top20) & gold_pages
            gold_files = {page_id.split("#page=", 1)[0] for page_id in gold_pages}
            selected = [str(item["file_id"]) for item in row["selected_files"]]
            derived: list[str] = []
            for page_id in top20:
                file_id = page_id.split("#page=", 1)[0]
                if file_id not in derived:
                    derived.append(file_id)
            candidate_file_recall.append(len(set(selected) & gold_files) / len(gold_files) if gold_files else 0.0)
            derived_file_recall.append(len(set(derived[:3]) & gold_files) / len(gold_files) if gold_files else 0.0)
            page_recall10.append(len(found10) / len(gold_pages) if gold_pages else 0.0)
            page_recall20.append(len(found20) / len(gold_pages) if gold_pages else 0.0)
            page_hit10.append(float(bool(found10)))
            ndcg10.append(_ndcg(top10, gold, self.args.metric_page_k))
            per_query.append(
                {
                    "query_id": qid,
                    "gold_pages": sorted(gold_pages),
                    "gold_files": sorted(gold_files),
                    "top10_pages": top10,
                    "top20_pages": top20,
                    "selected_files": selected,
                    "page_recall@10": page_recall10[-1],
                    "page_recall@20": page_recall20[-1],
                    "page_hit@10": bool(found10),
                    "candidate_file_recall@3": candidate_file_recall[-1],
                    "derived_file_recall@3": derived_file_recall[-1],
                }
            )
        mean = lambda values: sum(values) / len(values) if values else 0.0
        parsing = json.loads((self.root / "parsing/summary.json").read_text(encoding="utf-8"))
        run_timing = json.loads((self.root / "runs/timing.json").read_text(encoding="utf-8"))
        report = {
            "available": True,
            "dataset": str(self.args.dataset_root.resolve()),
            "queries": len(run_rows),
            "pages": len(read_jsonl(self.root / "parsing/pages.jsonl")),
            "files": len(self.load_inputs()[0]),
            "config": self.config,
            "metrics": {
                "page_recall@10": mean(page_recall10),
                "page_recall@20": mean(page_recall20),
                "page_hit@10": mean(page_hit10),
                "ndcg@10": 100.0 * mean(ndcg10),
                "file_candidate_recall@3": mean(candidate_file_recall),
                "derived_file_recall@3": mean(derived_file_recall),
            },
            "cost": {
                "mean_selected_files": sum(len(row["selected_files"]) for row in run_rows) / len(run_rows),
                "mean_candidate_pages": run_timing["mean_candidate_pages"],
                "mean_saved_pages": sum(len(row["chunks"]) for row in run_rows) / len(run_rows),
            },
            "timing": run_timing,
            "diagnostics": {
                "parser_empty_pages": int(parsing.get("empty_pages", 0)),
                "visual_only_pages": int(parsing.get("visual_only_pages", 0)),
                "parse_errors": int(parsing.get("parse_errors", 0)),
                "mapping_failures": int(
                    json.loads((self.root / "qrels_mapping.json").read_text(encoding="utf-8")).get("failures", 0)
                    if (self.root / "qrels_mapping.json").is_file()
                    else 0
                ),
                "resumed_parse_files": sum(bool(item.get("resumed")) for item in parsing.get("files", [])),
                "retried_parse_files": sum(
                    item.get("status") == "error" for item in parsing.get("files", [])
                ),
            },
            "parsing": parsing,
            "per_query": per_query,
        }
        atomic_json_dump(self.root / "reports/report.json", report)
        report_md = [
            "# Generic benchmark hierarchical retrieval",
            "",
            f"- Files: **{report['files']}**; pages: **{report['pages']}**; queries: **{report['queries']}**.",
            f"- Config: Kf={self.args.file_k}, evaluate top-{self.args.metric_page_k}, save top-{self.args.saved_page_k}.",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Page recall@10 | {report['metrics']['page_recall@10']:.2%} |",
            f"| Page recall@20 | {report['metrics']['page_recall@20']:.2%} |",
            f"| Page hit@10 | {report['metrics']['page_hit@10']:.2%} |",
            f"| nDCG@10 | {report['metrics']['ndcg@10']:.2f} |",
            f"| File candidate recall@3 | {report['metrics']['file_candidate_recall@3']:.2%} |",
            f"| Derived file recall@3 | {report['metrics']['derived_file_recall@3']:.2%} |",
            f"| Mean candidate pages | {report['cost']['mean_candidate_pages']:.2f} |",
            f"| Mean retrieval seconds/query | {report['timing']['mean_query_seconds']:.6f} |",
            "",
            f"Full per-query diagnostics are in `report.json`; saved run rows are in `runs/{run_name}`.",
        ]
        atomic_write_text(self.root / "reports/report.md", "\n".join(report_md) + "\n")
        write_jsonl(self.root / "reports/per_query.jsonl", per_query)
        return report["metrics"]

    def package(self) -> dict[str, Any]:
        package_root = self.root / "bundle"
        package_root.mkdir(parents=True, exist_ok=True)
        zip_path = package_root / f"{self.root.name}.zip"
        include = [
            "manifest.json",
            "config.json",
            "timing.json",
            "input_inventory.jsonl",
            "qrels_page_level.jsonl",
            "qrels_mapping.json",
            "parsing/pages.jsonl",
            "parsing/summary.json",
            "rendering/page_metadata.json",
            "vsplade/page_vectors.npz",
            "vsplade/page_metadata.json",
            "vsplade/query_vectors.npz",
            "vsplade/query_metadata.json",
            "vsplade/timing.json",
            "indexes/page_bm25.json",
            "indexes/file_bm25.json",
            "indexes/summary.json",
            f"runs/hierarchical_kf{self.args.file_k}_top{self.args.saved_page_k}.jsonl",
            "runs/timing.json",
            "reports/report.json",
            "reports/report.md",
            "reports/per_query.jsonl",
            "logs/events.jsonl",
            "logs/errors.jsonl",
        ]
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for relative in include:
                source = self.root / relative
                if source.is_file():
                    archive.write(source, relative)
        return {"zip": str(zip_path), "bytes": zip_path.stat().st_size}


def _counts_by(items: Sequence[Any], key: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        value = str(key(item))
        counts[value] = counts.get(value, 0) + 1
    return counts


def _git_sha() -> str:
    try:
        import subprocess

        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _download_source_page_map(subset: str, cache_root: Path) -> dict[str, dict[str, Any]]:
    target = cache_root / f"{safe_name(subset)}.json"
    if target.is_file():
        return json.loads(target.read_text(encoding="utf-8"))
    local_parquet = ROOT / "data/benchmark/vidore_v3" / subset / "corpus.parquet"
    if local_parquet.is_file():
        try:
            import pandas as pd

            frame = pd.read_parquet(
                local_parquet,
                engine="fastparquet",
                columns=["corpus_id", "doc_id", "page_number_in_doc"],
            )
            output = {
                str(row.corpus_id): {
                    "doc_id": str(row.doc_id),
                    "page_number_in_doc": int(row.page_number_in_doc),
                }
                for row in frame.itertuples(index=False)
            }
            if output:
                atomic_json_dump(target, output)
                return output
        except Exception:
            # The public datasets-server route below is the portable fallback.
            pass
    dataset = f"vidore/vidore_v3_{subset}"
    output: dict[str, dict[str, Any]] = {}
    partial_root = cache_root / f"{safe_name(subset)}_batches"

    def request_payload(offset: int) -> dict[str, Any]:
        query = urllib.parse.urlencode(
            {
                "dataset": dataset,
                "config": "corpus",
                "split": "test",
                "offset": offset,
                "length": 100,
                "columns": "corpus_id,doc_id,page_number_in_doc",
            }
        )
        url = f"https://datasets-server.huggingface.co/rows?{query}"
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                request = urllib.request.Request(
                    url,
                    headers={"User-Agent": "AXIOM-DE-RD/benchmark-hierarchical"},
                )
                with urllib.request.urlopen(request, timeout=90) as response:
                    return json.loads(response.read().decode("utf-8"))
            except Exception as exc:  # pragma: no cover - network retry path
                last_error = exc
                retry_after = None
                if hasattr(exc, "headers"):
                    try:
                        retry_after = float(exc.headers.get("Retry-After", "0"))
                    except (TypeError, ValueError):
                        retry_after = None
                time.sleep(max(retry_after or 0.0, min(60.0, 3.0 * (attempt + 1))))
        raise RuntimeError(f"Could not fetch {url}: {last_error}")

    def slim_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
        rows = []
        for item in payload.get("rows") or []:
            row = item["row"]
            rows.append(
                {
                    "row": {
                        "corpus_id": row["corpus_id"],
                        "doc_id": row["doc_id"],
                        "page_number_in_doc": row["page_number_in_doc"],
                    }
                }
            )
        return {"num_rows_total": payload.get("num_rows_total"), "rows": rows}

    def fetch_cached(offset: int) -> dict[str, Any]:
        partial = partial_root / f"offset_{offset}.json"
        if partial.is_file():
            return json.loads(partial.read_text(encoding="utf-8"))
        payload = slim_payload(request_payload(offset))
        atomic_json_dump(partial, payload)
        return payload

    first_payload = fetch_cached(0)
    first_rows = first_payload.get("rows") or []
    total = int(first_payload.get("num_rows_total") or len(first_rows))
    offsets = list(range(100, total, 100))
    batches: dict[int, list[dict[str, Any]]] = {0: first_rows}
    for offset in offsets:
        batches[offset] = fetch_cached(offset).get("rows") or []
        # The endpoint rate-limits bursts. Serial requests are deliberate: a
        # small pause is faster than retrying a fan-out after HTTP 429.
        time.sleep(0.75)
    for offset in sorted(batches):
        for item in batches[offset]:
            row = item["row"]
            output[str(row["corpus_id"])] = {
                "doc_id": str(row["doc_id"]),
                "page_number_in_doc": int(row["page_number_in_doc"]),
            }
    if not output:
        raise RuntimeError(f"Hugging Face source page map is empty for {dataset}")
    atomic_json_dump(target, output)
    return output


def _ndcg(ranked: Sequence[str], qrels: Mapping[str, int], k: int) -> float:
    gains = {str(page): int(score) for page, score in qrels.items()}
    actual = sum(
        (2.0 ** gains[page] - 1.0) / np.log2(rank + 1)
        for rank, page in enumerate(ranked[:k], 1)
        if page in gains
    )
    ideal_scores = sorted(gains.values(), reverse=True)[:k]
    ideal = sum(
        (2.0 ** score - 1.0) / np.log2(rank + 1)
        for rank, score in enumerate(ideal_scores, 1)
    )
    return actual / ideal if ideal else 0.0


def _arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--documents", type=Path)
    parser.add_argument("--queries", type=Path)
    parser.add_argument("--qrels", type=Path)
    parser.add_argument("--no-qrels", action="store_true", help="Ignore qrels even when dataset root contains qrels.jsonl")
    parser.add_argument("--limit-documents", type=int, help="Optional deterministic prefix for smoke tests")
    parser.add_argument("--limit-queries", type=int, help="Optional deterministic prefix for smoke tests")
    parser.add_argument("--stage", choices=["all", *STAGES], default="all")
    parser.add_argument("--force-stage", action="append", default=[])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--query-batch-size", type=int, default=64)
    parser.add_argument("--render-dpi", type=int, default=144)
    parser.add_argument("--file-k", type=int, default=3)
    parser.add_argument("--metric-page-k", type=int, default=10)
    parser.add_argument("--saved-page-k", type=int, default=20)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments().parse_args(argv)
    args.dataset_root = args.dataset_root.resolve()
    args.documents = (args.documents or args.dataset_root / "documents.jsonl").resolve()
    args.queries = (args.queries or args.dataset_root / "queries.jsonl").resolve()
    default_qrels = args.dataset_root / "qrels.jsonl"
    if args.no_qrels:
        args.qrels = None
    elif args.qrels is not None:
        args.qrels = args.qrels.resolve()
    elif default_qrels.is_file():
        args.qrels = default_qrels.resolve()
    if args.file_k <= 0 or args.metric_page_k <= 0 or args.saved_page_k < args.metric_page_k:
        raise SystemExit("Require file-k > 0 and saved-page-k >= metric-page-k > 0")
    if args.batch_size <= 0 or args.query_batch_size <= 0 or args.render_dpi <= 0:
        raise SystemExit("Batch sizes and render DPI must be positive")
    if args.limit_documents is not None and args.limit_documents <= 0:
        raise SystemExit("limit-documents must be positive")
    if args.limit_queries is not None and args.limit_queries <= 0:
        raise SystemExit("limit-queries must be positive")
    runner = Runner(args)
    callbacks = {
        "validate": runner.validate,
        "normalize-qrels": runner.normalize_qrels,
        "parse": runner.parse,
        "encode-vsplade": runner.encode_vsplade,
        "build-bm25": runner.build_indexes,
        "retrieve": runner.retrieve,
        "evaluate": runner.evaluate,
        "package": runner.package,
    }
    stages = STAGES if args.stage == "all" else (args.stage,)
    for stage in stages:
        runner.run_stage(stage, callbacks[stage])
    print(json.dumps({"output_dir": str(runner.root), "config_hash": runner.config_hash, "stages": list(stages)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
