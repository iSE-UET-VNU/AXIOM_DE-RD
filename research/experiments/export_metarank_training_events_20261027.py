"""Export the selected feature matrices to Metarank ranking events/configs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/work/metarank_feature_ablation_20261027"
GATE = ROOT / "data/work/adaptive_routers_handoff/adaptive_routers_handoff/data/work/ondemand_v2/ef7f11811803/light_prep/gate_ppocrv5_ungated.json"
OUT = DATA / "metarank_native"


def write_config(path: Path, features: list[str]) -> None:
    lines = [
        "state:", "  type: memory", "models:", "  ltr:", "    type: lambdamart", "    backend:",
        "      type: xgboost", "      iterations: 180", "      learningRate: 0.05",
        "      maxDepth: 3", "      sampling: 0.85", "      seed: 42", "      ndcgCutoff: 20",
        "    weights:", "      rel0: 0.0", "      rel1: 1.0", "    features:",
    ]
    lines.extend(f"      - {name}" for name in features)
    lines += ["    split: random=80%", "    eval: [ndcg@20]", "features:"]
    for name in features:
        lines += [f"  - name: {name}", "    type: number", "    scope: item", f"    field: ranking.{name}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    gate = json.loads(GATE.read_text(encoding="utf-8"))
    meta = json.loads((DATA / "feature_matrix_metadata.json").read_text(encoding="utf-8"))
    qids = meta["qids"]
    candidates = meta["candidate_page_ids_by_qid"]
    if len(qids) != 220:
        raise ValueError("Expected 220 qids")
    definitions: dict[str, tuple[str, list[str]]] = {
        "discovery_plus_cohere_page_shape": (
            "discovery_plus_cohere_page_shape_feature_matrix.npy",
            ["discovery_score_minmax", "discovery_score_z", "discovery_rank_pct",
             "cohere4pro_score_minmax", "cohere4pro_score_z", "cohere4pro_rank_pct",
             "page_log_token_count", "page_unique_token_ratio", "page_numeric_density",
             "page_table_marker_count", "page_ocr_marker_count"],
        ),
        "score_rank_plus_page_shape": (
            "score_rank_plus_page_shape_feature_matrix.npy",
            [f"{mod}_{suffix}" for mod in ("discovery", "cohere4pro", "qwen3_8b", "bge_m3_local")
             for suffix in ("score_minmax", "score_z", "rank_pct")]
            + ["page_log_token_count", "page_unique_token_ratio", "page_numeric_density",
               "page_table_marker_count", "page_ocr_marker_count"],
        ),
        "all_engineered": ("all_engineered_feature_matrix.npy", meta["feature_names"]),
    }
    for name, (matrix_file, feature_names) in definitions.items():
        matrix = np.load(DATA / matrix_file, mmap_mode="r")
        if matrix.shape != (220 * 100, len(feature_names)):
            raise ValueError(f"{name} matrix shape mismatch: {matrix.shape}")
        folder = OUT / name
        folder.mkdir(parents=True, exist_ok=True)
        event_file = folder / "events.jsonl"
        with event_file.open("w", encoding="utf-8") as stream:
            for qi, qid in enumerate(qids):
                items = []
                for pi, page in enumerate(candidates[qid]):
                    row = matrix[qi * 100 + pi]
                    items.append({
                        "id": page,
                        "label": int(page in set(gate["gold_rank"][qid])),
                        "fields": [{"name": feature_names[j], "value": float(row[j])}
                                   for j in range(len(feature_names))],
                    })
                event = {"event": "ranking", "id": f"{name}-{qi+1:04d}",
                         "timestamp": str(1_798_100_000_000 + qi), "items": items}
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        write_config(folder / "config.yml", feature_names)
        (folder / "README.md").write_text(
            f"# {name}\n\n220 ranking events, one query per event, with 100 frozen candidates each. "
            "Each item has an explicit binary relevance label from the benchmark qrels and numeric feature fields. "
            "Labels are for offline training/evaluation only. The generated `random=80%` Metarank config is a runnable "
            "smoke configuration, not the primary five-fold OOF result; use the adjacent report for that result.\n",
            encoding="utf-8",
        )
    print(f"Exported native event/config pairs to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
