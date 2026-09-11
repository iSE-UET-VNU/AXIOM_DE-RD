"""Decode cached V-SPLADE page vectors for Physics retrieval diagnostics.

This is intentionally an offline diagnostic. It does not run model inference:
the page vectors were already produced for all 1,674 Physics pages and the
English query vectors for all 302 queries are already cached. The script
decodes vector coordinates through the local tokenizer vocabulary and compares
them with the PDF-inspector text attached to the paired retrieval run.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAGE_VECTOR_DIR = ROOT / "data/output/vsplade/vidore_v3_physics_48q"
DEFAULT_QUERY_VECTOR_DIR = ROOT / "data/output/vsplade/vidore_v3_physics_english_302q"
DEFAULT_PAIR_DIR = ROOT / "data/benchmark/vidore_v3/results/physics_vsplade_bm25_fusion"
DEFAULT_ANALYSIS = DEFAULT_PAIR_DIR / "per_query_file_recall_at3/report.json"
DEFAULT_OUTPUT_DIR = DEFAULT_PAIR_DIR / "sparse_token_diagnostics"

REPRESENTATIVE_QIDS = (
    "physics::107",
    "physics::160",
    "physics::61",
    "physics::213",
    "physics::182",
    "physics::239",
    "physics::10",
    "physics::192",
)

FRENCH_STOPWORDS = set(
    "a au aux avec ce ceci cela ces cette comme dans de des du elle en et est eu eux "
    "il ils je la le les leur leurs lui ma mais me mes moi mon ne nos notre nous on ou "
    "par pas pour que quel quelle quelles quels qui sa se sera son sont sur ta te tes toi "
    "ton tu un une vos votre vous y d l".split()
)
ENGLISH_STOPWORDS = set(
    "a an and are as at be been being but by can could did do does for from had has have "
    "how if in is it its may might of on or should that the their them then there these "
    "this those to was were what when where which who why will with would according "
    "according described mainly mentioned".split()
)


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


def _load_id_to_token(model_dir: Path) -> dict[int, str]:
    payload = json.loads((model_dir / "tokenizer.json").read_text(encoding="utf-8"))
    id_to_token = {int(value): str(token) for token, value in payload["model"]["vocab"].items()}
    id_to_token.update(
        {int(item["id"]): str(item["content"]) for item in payload.get("added_tokens", [])}
    )
    return id_to_token


def _pretty_token(token: str) -> str:
    return token.replace("Ġ", " ").replace("Ċ", "\\n").strip()


def _top_tokens(
    row: sparse.csr_matrix,
    id_to_token: dict[int, str],
    limit: int = 20,
) -> list[dict[str, Any]]:
    if row.nnz == 0:
        return []
    order = np.argsort(-row.data, kind="stable")[:limit]
    return [
        {
            "id": int(row.indices[position]),
            "token": _pretty_token(id_to_token.get(int(row.indices[position]), "<unknown>")),
            "weight": round(float(row.data[position]), 4),
        }
        for position in order
    ]


def _vector_values(row: sparse.csr_matrix) -> dict[int, float]:
    return {int(index): float(value) for index, value in zip(row.indices, row.data)}


def _shared_tokens(
    query_row: sparse.csr_matrix,
    page_row: sparse.csr_matrix,
    id_to_token: dict[int, str],
    limit: int = 20,
) -> list[dict[str, Any]]:
    query_values = _vector_values(query_row)
    page_values = _vector_values(page_row)
    shared = set(query_values) & set(page_values)
    ordered = sorted(
        shared,
        key=lambda index: (
            -(query_values[index] * page_values[index]),
            -page_values[index],
            index,
        ),
    )[:limit]
    return [
        {
            "id": index,
            "token": _pretty_token(id_to_token.get(index, "<unknown>")),
            "query_weight": round(query_values[index], 4),
            "page_weight": round(page_values[index], 4),
            "score_contribution": round(query_values[index] * page_values[index], 6),
        }
        for index in ordered
    ]


def _normalised_terms(text: str, stopwords: set[str]) -> set[str]:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(character for character in text if not unicodedata.combining(character))
    return {
        token
        for token in re.findall(r"[a-z0-9]{3,}", text)
        if token not in stopwords
    }


def _sparse_query_matches(
    query: str,
    page_row: sparse.csr_matrix,
    id_to_token: dict[int, str],
) -> list[dict[str, Any]]:
    """Find readable page-vector tokens that overlap English query terms.

    V-SPLADE uses subword tokens, so this is a diagnostic substring match,
    not a replacement for the exact sparse dot product. Exact contribution is
    reported separately by ``_shared_tokens``.
    """
    query_terms = _normalised_terms(query, ENGLISH_STOPWORDS)
    matches: list[dict[str, Any]] = []
    for index, weight in zip(page_row.indices, page_row.data):
        token = _pretty_token(id_to_token.get(int(index), "<unknown>"))
        normalised = _normalised_terms(token, set())
        if not normalised or any(
            piece == term or (len(piece) >= 3 and piece in term) or (len(term) >= 4 and term in piece)
            for piece in normalised
            for term in query_terms
        ):
            matched_terms = sorted(
                term
                for term in query_terms
                if any(
                    piece == term or (len(piece) >= 3 and piece in term) or (len(term) >= 4 and term in piece)
                    for piece in normalised
                )
            )
            if matched_terms:
                matches.append(
                    {
                        "id": int(index),
                        "token": token,
                        "weight": round(float(weight), 4),
                        "query_terms": matched_terms,
                    }
                )
    return sorted(matches, key=lambda item: (-item["weight"], item["id"]))[:20]


def _read_text_by_page(path: Path) -> dict[str, str]:
    text_by_page: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        for chunk in record.get("chunks", []):
            page_id = str(chunk.get("chunk_id", ""))
            text = str(chunk.get("text", ""))
            if page_id and text.strip() and page_id not in text_by_page:
                text_by_page[page_id] = text
    return text_by_page


def _read_score_by_page(path: Path) -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = defaultdict(dict)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        for chunk in record.get("chunks", []):
            page_id = str(chunk.get("chunk_id", ""))
            scores[str(record["qid"])][page_id] = {
                "rank": float(chunk.get("rank", 0)),
                "score": float(chunk.get("score", 0.0)),
            }
    return scores


def _page_choice(row: dict[str, Any]) -> tuple[str, str]:
    if row["page_class"] == "vsplade_only" and row["vsplade"]["gold_pages_found"]:
        return row["vsplade"]["gold_pages_found"][0], "vsplade"
    if row["page_class"] == "bm25_only" and row["bm25"]["gold_pages_found"]:
        return row["bm25"]["gold_pages_found"][0], "bm25"
    if row["vsplade"]["gold_pages_found"]:
        return row["vsplade"]["gold_pages_found"][0], "vsplade"
    if row["bm25"]["gold_pages_found"]:
        return row["bm25"]["gold_pages_found"][0], "bm25"
    return row["gold_pages"][0], "gold"


def _inspect_row(
    row: dict[str, Any],
    page_index_by_id: dict[str, int],
    page_vectors: sparse.csr_matrix,
    query_vectors: sparse.csr_matrix,
    id_to_token: dict[int, str],
    text_by_page: dict[str, str],
    bm25_scores: dict[str, dict[str, float]],
    vsplade_scores: dict[str, dict[str, float]],
) -> dict[str, Any]:
    page_id, selected_by = _page_choice(row)
    if page_id not in page_index_by_id:
        raise RuntimeError(f"Page vector not found for {page_id}")
    query_index = int(row["qid"].rsplit("::", 1)[1])
    page_index = page_index_by_id[page_id]
    query_row = query_vectors.getrow(query_index)
    page_row = page_vectors.getrow(page_index)
    text = text_by_page.get(page_id, "")
    query_text_terms = _normalised_terms(row["query_french"], FRENCH_STOPWORDS)
    text_terms = _normalised_terms(text, FRENCH_STOPWORDS)
    shared = _shared_tokens(query_row, page_row, id_to_token)
    return {
        "qid": row["qid"],
        "query_french": row["query_french"],
        "query_english": row["query_english"],
        "metadata": row["metadata"],
        "page_class": row["page_class"],
        "file_class": row["file_class"],
        "winner": row["winner"],
        "gold_pages": row["gold_pages"],
        "selected_page": page_id,
        "selected_page_by": selected_by,
        "pdf_inspector_text_available": bool(text.strip()),
        "pdf_inspector_query_term_overlap": sorted(query_text_terms & text_terms),
        "pdf_inspector_query_term_overlap_count": len(query_text_terms & text_terms),
        "pdf_inspector_query_term_count": len(query_text_terms),
        "pdf_inspector_text_excerpt": text[:1200],
        "bm25_page_rank": bm25_scores.get(row["qid"], {}).get(page_id, {}).get("rank"),
        "bm25_page_score": bm25_scores.get(row["qid"], {}).get(page_id, {}).get("score"),
        "vsplade_page_rank": vsplade_scores.get(row["qid"], {}).get(page_id, {}).get("rank"),
        "vsplade_page_score": vsplade_scores.get(row["qid"], {}).get(page_id, {}).get("score"),
        "query_sparse_nnz": int(query_row.nnz),
        "page_sparse_nnz": int(page_row.nnz),
        "shared_sparse_dimensions": len(set(query_row.indices) & set(page_row.indices)),
        "sparse_dot_product": round(float((query_row @ page_row.T).toarray()[0, 0]), 6),
        "query_top_tokens": _top_tokens(query_row, id_to_token),
        "page_top_tokens": _top_tokens(page_row, id_to_token),
        "shared_top_tokens": shared,
        "english_query_terms_found_in_page_sparse_vector": _sparse_query_matches(
            row["query_english"], page_row, id_to_token
        ),
    }


def _group_diagnostics(
    rows: list[dict[str, Any]],
    page_index_by_id: dict[str, int],
    page_vectors: sparse.csr_matrix,
    query_vectors: sparse.csr_matrix,
    id_to_token: dict[int, str],
    text_by_page: dict[str, str],
    bm25_scores: dict[str, dict[str, float]],
    vsplade_scores: dict[str, dict[str, float]],
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group = row["page_class"]
        try:
            groups[group].append(
                _inspect_row(
                    row, page_index_by_id, page_vectors, query_vectors, id_to_token,
                    text_by_page, bm25_scores, vsplade_scores,
                )
            )
        except (IndexError, RuntimeError):
            continue
    result: dict[str, dict[str, Any]] = {}
    for group, inspected in sorted(groups.items()):
        result[group] = {
            "queries": len(inspected),
            "mean_query_sparse_nnz": sum(x["query_sparse_nnz"] for x in inspected) / len(inspected),
            "mean_page_sparse_nnz": sum(x["page_sparse_nnz"] for x in inspected) / len(inspected),
            "mean_shared_sparse_dimensions": sum(x["shared_sparse_dimensions"] for x in inspected) / len(inspected),
            "mean_sparse_dot_product": sum(x["sparse_dot_product"] for x in inspected) / len(inspected),
            "pdf_inspector_text_available": sum(x["pdf_inspector_text_available"] for x in inspected),
            "zero_pdf_inspector_query_term_overlap": sum(
                x["pdf_inspector_text_available"] and not x["pdf_inspector_query_term_overlap"]
                for x in inspected
            ),
            "mean_english_query_terms_found_in_sparse_vector": sum(
                len(x["english_query_terms_found_in_page_sparse_vector"])
                for x in inspected
            ) / len(inspected),
        }
    return result


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    def pct(value: float) -> str:
        return f"{value:.1%}"

    def token_list(items: list[dict[str, Any]], field: str = "token") -> str:
        values = []
        for item in items:
            value = str(item.get(field, "")).replace("|", "\\|").replace("\n", " ")
            weight = item.get("weight", item.get("score_contribution", ""))
            values.append(f"`{value}` ({weight})")
        return ", ".join(values) if values else "(none)"

    lines = [
        "# Physics V-SPLADE sparse-token diagnostics",
        "",
        "This report decodes cached V-SPLADE page vectors. It does not rerun inference. "
        "The page vectors cover all 1,674 Physics pages, while the query vectors are the "
        "English-query vectors used by the paired V-SPLADE ranking.",
        "",
        "## Aggregate diagnostics by page class",
        "",
        "`pdf_inspector_query_term_overlap` is a simple French lexical check. "
        "`english_query_terms_found_in_sparse_vector` is a readable substring diagnostic "
        "over the fixed sparse vocabulary; the exact evidence for ranking is the sparse dot product.",
        "",
        "| Page class | Queries | Mean shared sparse dims | Mean dot product | Text available | Zero French-term overlap | Mean English terms represented |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group, stats in report["group_diagnostics"].items():
        lines.append(
            f"| {group} | {stats['queries']} | {stats['mean_shared_sparse_dimensions']:.1f} | "
            f"{stats['mean_sparse_dot_product']:.3f} | {stats['pdf_inspector_text_available']} | "
            f"{stats['zero_pdf_inspector_query_term_overlap']} | "
            f"{stats['mean_english_query_terms_found_in_sparse_vector']:.2f} |"
        )

    lines += ["", "## Representative pages", ""]
    for item in report["representative_pages"]:
        lines += [
            f"### {item['qid']} - {item['selected_page']}",
            "",
            f"- French query: {item['query_french']}",
            f"- English query used by V-SPLADE: {item['query_english']}",
            f"- Page class: `{item['page_class']}`; file class: `{item['file_class']}`; winner at file recall@3: `{item['winner']}`.",
            f"- Selected gold page was reached by: `{item['selected_page_by']}`; BM25 rank: `{item['bm25_page_rank']}`; V-SPLADE rank: `{item['vsplade_page_rank']}`.",
            f"- PDF-inspector text terms: `{', '.join(item['pdf_inspector_query_term_overlap']) or '(none)'}`; "
            f"text available: `{item['pdf_inspector_text_available']}`.",
            f"- Sparse vector: query nnz `{item['query_sparse_nnz']}`, page nnz `{item['page_sparse_nnz']}`, "
            f"shared dimensions `{item['shared_sparse_dimensions']}`, dot product `{item['sparse_dot_product']}`.",
            f"- English query terms represented in page vector: {token_list(item['english_query_terms_found_in_page_sparse_vector'])}",
            f"- Top page activations: {token_list(item['page_top_tokens'])}",
            f"- Top query-page shared activations: {token_list(item['shared_top_tokens'], 'token')}",
            "",
            "PDF-inspector excerpt:",
            "",
            "```text",
            item["pdf_inspector_text_excerpt"].replace("```", "'''") or "(no PDF-inspector text attached)",
            "```",
            "",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--page-vector-dir", type=Path, default=DEFAULT_PAGE_VECTOR_DIR)
    parser.add_argument("--query-vector-dir", type=Path, default=DEFAULT_QUERY_VECTOR_DIR)
    parser.add_argument("--pair-dir", type=Path, default=DEFAULT_PAIR_DIR)
    parser.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--representative-qids", nargs="*", default=list(REPRESENTATIVE_QIDS))
    args = parser.parse_args()

    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    rows = analysis["per_query"]
    page_vectors = _load_csr(args.page_vector_dir / "page_vectors.npz")
    query_vectors = _load_csr(args.query_vector_dir / "query_vectors.npz")
    id_to_token = _load_id_to_token(args.page_vector_dir.parent / "efficient")
    page_metadata = json.loads((args.page_vector_dir / "page_metadata.json").read_text(encoding="utf-8"))
    page_index_by_id = {str(item["unit_id"]): index for index, item in enumerate(page_metadata)}
    if page_vectors.shape[0] != len(page_metadata):
        raise RuntimeError(f"Page vector/metadata mismatch: {page_vectors.shape} vs {len(page_metadata)}")
    if query_vectors.shape[0] != len(rows):
        raise RuntimeError(f"Query vector/report mismatch: {query_vectors.shape} vs {len(rows)}")

    text_path = args.pair_dir / "vsplade_french_bm25-french_vs-english.jsonl"
    bm25_path = args.pair_dir / "bm25_french_bm25-french_vs-english.jsonl"
    text_by_page = _read_text_by_page(text_path)
    bm25_scores = _read_score_by_page(bm25_path)
    vsplade_scores = _read_score_by_page(text_path)
    row_by_qid = {row["qid"]: row for row in rows}
    representative_pages = []
    for qid in args.representative_qids:
        if qid not in row_by_qid:
            raise RuntimeError(f"Representative QID not found: {qid}")
        representative_pages.append(
            _inspect_row(
                row_by_qid[qid], page_index_by_id, page_vectors, query_vectors,
                id_to_token, text_by_page, bm25_scores, vsplade_scores,
            )
        )

    report = {
        "setup": {
            "benchmark": "vidore_v3/physics",
            "page_vector_dir": str(args.page_vector_dir),
            "query_vector_dir": str(args.query_vector_dir),
            "analysis_report": str(args.analysis),
            "page_vector_shape": list(page_vectors.shape),
            "query_vector_shape": list(query_vectors.shape),
            "vocabulary_size": page_vectors.shape[1],
            "text_source": str(text_path),
            "query_vectors_are": "English query vectors paired by French query ordinal",
        },
        "group_diagnostics": _group_diagnostics(
            rows, page_index_by_id, page_vectors, query_vectors, id_to_token,
            text_by_page, bm25_scores, vsplade_scores,
        ),
        "representative_pages": representative_pages,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_markdown(args.output_dir / "report.md", report)
    print(json.dumps(report["group_diagnostics"], ensure_ascii=True, indent=2))
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
