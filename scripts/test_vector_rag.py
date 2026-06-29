"""Run a small retrieval smoke test against vector_records.json."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import json
import math
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.indexing_cataloging.embeddings import EmbeddingConfig, OpenRouterEmbeddingProvider  # noqa: E402
from src.utils.env import load_dotenv_file  # noqa: E402
from src.utils.paths import portable_path  # noqa: E402


RAG_SMOKE_QUERIES = [
    {
        "question": "Canada's goods trade surplus with the United States increased to what amount in November, and what was it in October?",
        "expected_terms": ["8.2-billion", "6.6-billion", "November", "October"],
    },
    {
        "question": "What tariff percentage did Donald Trump promise on Canadian imports, and what concern did he say Ottawa must address?",
        "expected_terms": ["25-per-cent", "border security", "tariffs"],
    },
    {
        "question": "What share of Canadian exports goes to the U.S., and what were the 2023 oil and gas and vehicle export amounts mentioned?",
        "expected_terms": ["75 per cent", "166-billion", "85-billion"],
    },
    {
        "question": "During John Hunkin's CIBC career, what U.S. expansion work and technology-company financing role did the article describe?",
        "expected_terms": ["CIBC", "U.S. market", "technology companies", "1999 to 2005"],
    },
    {
        "question": "What did the Canadian Medical Association recommend about membership fees or user fees for publicly insured primary-care services?",
        "expected_terms": ["Canadian Medical Association", "ban membership fees", "primary-care services"],
    },
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke-test vector_records retrieval.")
    parser.add_argument(
        "--vector-records",
        default="data/output/final_test/data/vector_records.json",
        help="Path to vector_records.json.",
    )
    parser.add_argument(
        "--output",
        default="data/output/final_test/reports/rag_smoke_report.json",
        help="Path to write the retrieval report.",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--model", default="openai/text-embedding-3-small")
    parser.add_argument("--dimension", type=int, default=1536)
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    args = parser.parse_args(argv)

    load_dotenv_file(PROJECT_ROOT)
    vector_path = _resolve_project_path(args.vector_records)
    output_path = _resolve_project_path(args.output)

    vector_records = json.loads(vector_path.read_text(encoding="utf-8"))
    provider = OpenRouterEmbeddingProvider(
        model=args.model,
        dimension=args.dimension,
        api_key_env=args.api_key_env,
        app_title="AXIOM_DE-RD RAG smoke",
    )
    questions = [item["question"] for item in RAG_SMOKE_QUERIES]
    question_embeddings = provider.embed_batch(questions)

    results = []
    for query, embedding in zip(RAG_SMOKE_QUERIES, question_embeddings):
        hits = _search(vector_records, embedding, args.top_k)
        results.append(
            {
                "question": query["question"],
                "expected_terms": query["expected_terms"],
                "hits": hits,
                "top_hit_contains_expected_term": _contains_any_expected_term(
                    hits[0]["text"] if hits else "",
                    query["expected_terms"],
                ),
            }
        )

    report = {
        "contract_version": "rag-smoke-report-v1",
        "vector_records_path": portable_path(vector_path, PROJECT_ROOT),
        "embedding_model": args.model,
        "embedding_dimension": args.dimension,
        "top_k": args.top_k,
        "query_count": len(results),
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {portable_path(output_path, PROJECT_ROOT)}")
    for index, result in enumerate(results, start=1):
        top = result["hits"][0] if result["hits"] else {}
        print(f"{index}. {result['question']}")
        print(
            "   top_chunk="
            f"{top.get('chunk_index')} score={top.get('score')} "
            f"contains_expected={result['top_hit_contains_expected_term']}"
        )
        print(f"   snippet={top.get('snippet', '')}")
    return 0


def _search(vector_records: list[dict[str, Any]], query_embedding: list[float], top_k: int) -> list[dict[str, Any]]:
    scored = []
    for record in vector_records:
        embedding = record.get("embedding")
        if not isinstance(embedding, list):
            continue
        score = _cosine_similarity(query_embedding, embedding)
        text = str(record.get("text") or "")
        scored.append(
            {
                "score": round(score, 6),
                "record_id": record.get("record_id"),
                "chunk_id": record.get("chunk_id"),
                "chunk_index": record.get("chunk_index"),
                "source_uri": record.get("source_uri"),
                "start_char": record.get("start_char"),
                "end_char": record.get("end_char"),
                "snippet": _snippet(text),
                "text": text,
            }
        )
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _snippet(text: str, max_length: int = 320) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_length:
        return compact
    return compact[: max_length - 3] + "..."


def _contains_any_expected_term(text: str, terms: list[str]) -> bool:
    lower = text.lower()
    return any(term.lower() in lower for term in terms)


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
