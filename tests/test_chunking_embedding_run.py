from __future__ import annotations

import json
import shutil
from pathlib import Path

from src.chunking_embedding import run_cli

FIXTURE = Path(__file__).parent / "fixtures" / "indexing_enriched_data.json"


def _config() -> dict:
    return {
        "chunking_embedding": {
            "chunker": "paragraph",
            "chunker_params": {},
            "max_rows_per_chunk": 20,
            "embedder": "local_hash",
            "embedder_params": {"dimension": 16},
            "retrieval_profile": "hybrid_default",
            "embeddings": {"enabled": True, "provider": "openrouter"},  # unknown key: ignored
        }
    }


def _run(tmp_path: Path, config: dict | None = None) -> dict:
    input_dir = tmp_path / "processed"
    input_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURE, input_dir / "enriched_data.json")
    return run_cli(config or _config(), input_dir, tmp_path / "output")


def test_end_to_end_outputs_and_report(tmp_path: Path) -> None:
    report = _run(tmp_path)
    out = tmp_path / "output"
    chunk_records = json.loads((out / "chunk_records.json").read_text())
    vector_records = json.loads((out / "vector_records.json").read_text())

    assert report["docs"] == {"total": 3, "indexed": 2, "skipped_existing": 0, "skipped_failed": 1}
    assert [s["doc_id"] for s in report["skipped_docs"]] == ["bbbb333344445555"]
    assert "no rows" in report["skipped_docs"][0]["reason"]

    assert len(chunk_records) == len(vector_records) == 7
    ids = {r["chunk_id"] for r in chunk_records}
    assert ids == {r["id"] for r in vector_records}
    for record in chunk_records:
        assert record["chunk_type"] in ("text_chunk", "table_chunk", "figure_chunk")
        assert record["metadata"]["config_hash"] == report["config_hash"]
    for vector in vector_records:
        assert vector["dim"] == 16
        assert vector["model"] == "local-hash-embedding-v1"

    assert report["chunk_counts"] == {"text_chunk": 5, "table_chunk": 1, "figure_chunk": 1}
    assert report["chars_mean"] > 0
    assert report["chars_p90"] >= report["chars_mean"]
    assert set(report["wall_time_seconds"]) == {"load", "chunk", "embed", "write"}
    profile = report["retrieval_profile"]
    assert profile["name"] == "hybrid_default"
    assert profile["embedder"] == "local_hash"
    assert profile["dim"] == 16
    assert "hybrid_weights" in profile and "boost_fields" in profile


def test_rerun_is_idempotent(tmp_path: Path) -> None:
    first = _run(tmp_path)
    first_records = json.loads((tmp_path / "output" / "chunk_records.json").read_text())

    second = _run(tmp_path)
    second_records = json.loads((tmp_path / "output" / "chunk_records.json").read_text())

    assert second["docs"]["indexed"] == 0
    assert second["docs"]["skipped_existing"] == 2
    assert [r["chunk_id"] for r in first_records] == [r["chunk_id"] for r in second_records]
    assert first["config_hash"] == second["config_hash"]


def test_config_change_triggers_reindex_with_new_ids_kept_separate(tmp_path: Path) -> None:
    _run(tmp_path)
    changed = _config()
    changed["chunking_embedding"]["chunker"] = "sentence_group"
    report = _run(tmp_path, changed)
    assert report["docs"]["indexed"] == 2

    records = json.loads((tmp_path / "output" / "chunk_records.json").read_text())
    hashes = {r["metadata"]["config_hash"] for r in records}
    assert len(hashes) == 2  # both configs' records coexist, stamped distinctly


def test_ids_stable_across_runs_in_fresh_dirs(tmp_path: Path) -> None:
    report_a = _run(tmp_path / "a")
    report_b = _run(tmp_path / "b")
    ids_a = [r["chunk_id"] for r in json.loads((tmp_path / "a" / "output" / "chunk_records.json").read_text())]
    ids_b = [r["chunk_id"] for r in json.loads((tmp_path / "b" / "output" / "chunk_records.json").read_text())]
    assert ids_a == ids_b
    assert report_a["config_hash"] == report_b["config_hash"]
