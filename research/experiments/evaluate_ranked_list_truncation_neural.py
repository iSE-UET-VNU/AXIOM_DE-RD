"""Evaluate BiCut, Choppy and AttnCut on the canonical 220-query ranking.

The public Ranked-List-Truncation repository contains model definitions but no
checkpoint that can be transferred to this benchmark.  This runner therefore
re-trains the three paper-style models from scratch with five query-level OOF
folds.  Qrels from a test fold are used only for evaluation; training-fold
qrels create the cutoff rewards/labels.

The input ranking is the cached PP-OCRv5 + Qwen3-Embedding-8B hybrid ranking
used by the earlier adaptive-k experiments.  The default quick screen maps the
top-200 candidates into 50 four-page cutoff bins; ``--model-depth 200`` keeps
one model position per candidate page.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "data/work/ranked_list_truncation_neural_20261005"
ADAPTIVE_RUNNER = ROOT / "experiments/adaptive-k-retrieval/run_canonical_adaptive_methods.py"
TRUNCATION_REPO = ROOT / "experiments/ranked-list-truncation"


def load_canonical_rows(candidate_depth: int) -> list[dict[str, Any]]:
    """Reuse the already checked-in loader and its cached ranking artifacts."""
    module = runpy.run_path(str(ADAPTIVE_RUNNER))
    rows, _query_vectors = module["load_rows"]("hybrid", candidate_depth)
    return rows


def build_features(
    rows: Sequence[Mapping[str, Any]], model_depth: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return paper-style inputs and binary relevance reward curves.

    Choppy uses the score alone.  BiCut and AttnCut receive score, adjacent
    score gap, and normalized rank; these are cheap signals available at
    truncation time and substitute for unavailable Robust04 document stats.
    """
    all_features: list[np.ndarray] = []
    all_rewards: list[np.ndarray] = []
    cutoff_map: np.ndarray | None = None
    for row in rows:
        scores = np.asarray(row["scores"], dtype=np.float32)
        length = len(scores)
        if length == 0:
            raise ValueError("An input row has an empty ranked list")
        if model_depth > length:
            raise ValueError(f"model_depth={model_depth} exceeds ranked-list length {length}")
        # A shorter model sequence makes the CPU screen practical.  Each model
        # position represents a contiguous ranked-list bin and maps to the bin
        # end, so a predicted cutoff still returns real page counts.
        cutoff_map = np.ceil(
            np.arange(1, model_depth + 1, dtype=np.float32) * length / model_depth
        ).astype(int)
        starts = np.floor(
            np.arange(model_depth, dtype=np.float32) * length / model_depth
        ).astype(int)
        score_min = float(np.min(scores))
        score_max = float(np.max(scores))
        score = (scores - score_min) / max(score_max - score_min, 1e-6)
        sampled_score = score[starts]
        gaps = np.zeros(length, dtype=np.float32)
        if length > 1:
            gaps[1:] = np.maximum(scores[:-1] - scores[1:], 0.0)
        gap_max = float(np.max(gaps))
        sampled_gap = gaps[starts] / max(gap_max, 1e-6)
        rank = cutoff_map.astype(np.float32) / max(length - 1, 1)
        # Keep the first score as the bin's representative.  The local gap and
        # bin-end rank are available without qrels at inference time.
        all_features.append(
            np.column_stack((sampled_score, sampled_gap, rank)).astype(np.float32)
        )

        labels = np.asarray(
            [1.0 if page_id in set(row["gold"]) else 0.0 for page_id in row["ranked"]],
            dtype=np.float32,
        )
        cumulative_hits = np.cumsum(labels)[cutoff_map - 1]
        cutoff = cutoff_map.astype(np.float32)
        gold_count = float(max(len(row["gold"]), 1))
        # F1@k = 2*TP/(k + number_of_gold_pages).  This is the reward used by
        # the public Choppy/AttnCut loss implementations.
        reward = 2.0 * cumulative_hits / (cutoff + gold_count)
        all_rewards.append(reward.astype(np.float32))
    assert cutoff_map is not None
    return np.stack(all_features), np.stack(all_rewards), cutoff_map


def make_models(seq_len: int, device: str, model_scale: str) -> dict[str, Any]:
    """Build corrected equivalents of the three public model definitions."""
    import torch
    from torch import nn

    if model_scale == "paper":
        choppy_d_model, choppy_heads, choppy_layers, choppy_dropout = 128, 8, 3, 0.2
        attn_hidden, attn_d_model, attn_heads, attn_layers, attn_dropout = 128, 256, 4, 1, 0.4
    else:
        # The paper settings are expensive on CPU at sequence length 200.  The
        # fast setting preserves the attention/LSTM and decision structure,
        # while making a short exploratory run practical on this workstation.
        choppy_d_model, choppy_heads, choppy_layers, choppy_dropout = 64, 4, 1, 0.1
        attn_hidden, attn_d_model, attn_heads, attn_layers, attn_dropout = 32, 4 * 16, 4, 1, 0.1

    class ChoppyPaper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            d_model = choppy_d_model
            self.position_encoding = nn.Parameter(
                torch.randn(seq_len, d_model - 1) * 0.02
            )
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=choppy_heads,
                dropout=choppy_dropout,
                batch_first=True,
                norm_first=False,
            )
            self.attention_layer = nn.TransformerEncoder(layer, num_layers=choppy_layers)
            self.decision_layer = nn.Linear(d_model, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            pe = self.position_encoding.unsqueeze(0).expand(x.shape[0], -1, -1)
            encoded = self.attention_layer(torch.cat((x, pe), dim=2))
            return torch.softmax(self.decision_layer(encoded).squeeze(-1), dim=1)

    class AttnCutPaper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoding_layer = nn.LSTM(
                input_size=3,
                hidden_size=attn_hidden,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
            )
            layer = nn.TransformerEncoderLayer(
                d_model=attn_d_model,
                nhead=attn_heads,
                dropout=attn_dropout,
                batch_first=True,
                norm_first=False,
            )
            self.attention_layer = nn.TransformerEncoder(layer, num_layers=attn_layers)
            self.decision_layer = nn.Linear(attn_d_model, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            encoded = self.encoding_layer(x)[0]
            encoded = self.attention_layer(encoded)
            return torch.softmax(self.decision_layer(encoded).squeeze(-1), dim=1)

    # Import the repository's BiCut class itself.  It has no Transformer shape
    # issue and its documented output order is [EOL, Continue].
    if str(TRUNCATION_REPO) not in sys.path:
        sys.path.insert(0, str(TRUNCATION_REPO))
    from models.Bicut import BiCut  # type: ignore

    bicut = BiCut(input_size=3, dropout=0.2).to(device)
    return {
        "choppy": ChoppyPaper().to(device),
        "attncut": AttnCutPaper().to(device),
        "bicut": bicut,
    }


def set_seed(seed: int) -> None:
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one(
    method: str,
    train_x: np.ndarray,
    train_rewards: np.ndarray,
    train_epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: str,
    model_scale: str,
) -> Any:
    """Train one fold using the public paper losses/adapted cutoff labels."""
    import torch
    import torch.nn.functional as F

    set_seed(seed)
    model = make_models(train_x.shape[1], device, model_scale)[method]
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    x_tensor = torch.as_tensor(train_x, dtype=torch.float32, device=device)
    reward_tensor = torch.as_tensor(train_rewards, dtype=torch.float32, device=device)
    n_items = x_tensor.shape[0]
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    model.train()
    for _epoch in range(train_epochs):
        order = torch.randperm(n_items, generator=generator, device=device)
        for start in range(0, n_items, batch_size):
            indices = order[start : start + batch_size]
            xb = x_tensor[indices]
            rb = reward_tensor[indices]
            optimizer.zero_grad(set_to_none=True)
            if method == "choppy":
                # ChoopyLoss: maximize expected F1 of the predicted cutoff.
                probabilities = model(xb[:, :, :1])
                loss = -(probabilities * rb).sum(dim=1).mean()
            elif method == "attncut":
                # AttnCutLoss: RAML/reward distribution with tau=0.95.
                probabilities = model(xb)
                target = torch.softmax(rb / 0.95, dim=1)
                loss = -(target * torch.log(probabilities.clamp_min(1e-8))).sum(dim=1).mean()
            else:
                # BiCut's sequential decision: learn Continue before the
                # fold-train oracle F1 cutoff and EOL at that position.
                probabilities = model(xb)
                losses = []
                for row_index in range(len(indices)):
                    best_k = int(torch.argmax(rb[row_index]).item()) + 1
                    labels = torch.ones(best_k, dtype=torch.long, device=device)
                    labels[-1] = 0  # class 0 = EOL, class 1 = Continue
                    log_probs = torch.log(probabilities[row_index, :best_k].clamp_min(1e-8))
                    losses.append(F.nll_loss(log_probs, labels))
                loss = torch.stack(losses).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite {method} loss: {loss.item()}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model


def predict_choices(
    method: str,
    model: Any,
    test_x: np.ndarray,
    cutoff_map: np.ndarray,
    device: str,
) -> np.ndarray:
    import torch

    model.eval()
    with torch.no_grad():
        x_tensor = torch.as_tensor(test_x, dtype=torch.float32, device=device)
        if method == "choppy":
            probabilities = model(x_tensor[:, :, :1])
            positions = torch.argmax(probabilities, dim=1).cpu().numpy()
            return cutoff_map[positions].astype(int)
        if method == "attncut":
            probabilities = model(x_tensor)
            positions = torch.argmax(probabilities, dim=1).cpu().numpy()
            return cutoff_map[positions].astype(int)
        probabilities = model(x_tensor)
        # First position whose most likely class is EOL; if none, consume all.
        predicted_eol = torch.argmax(probabilities, dim=2).cpu().numpy() == 0
        choices = []
        for row in predicted_eol:
            positions = np.flatnonzero(row)
            choices.append(int(cutoff_map[positions[0]]) if len(positions) else int(cutoff_map[-1]))
        return np.asarray(choices, dtype=int)


def recall(row: Mapping[str, Any], k: int) -> float:
    gold = set(row["gold"])
    return len(set(row["ranked"][:k]) & gold) / max(len(gold), 1)


def evaluate(rows: Sequence[Mapping[str, Any]], choices: Sequence[int], name: str) -> dict[str, Any]:
    choices = [int(np.clip(k, 1, len(row["ranked"]))) for row, k in zip(rows, choices)]
    adaptive = [recall(row, k) for row, k in zip(rows, choices)]
    mean_k = float(np.mean(choices))
    matched_k = int(np.clip(round(mean_k), 1, len(rows[0]["ranked"])))
    fixed = [recall(row, matched_k) for row in rows]
    return {
        "name": name,
        "queries": len(rows),
        "recall_at_adaptive_k": float(np.mean(adaptive)),
        "matched_fixed_k": matched_k,
        "matched_fixed_recall": float(np.mean(fixed)),
        "delta_vs_matched_fixed": float(np.mean(adaptive) - np.mean(fixed)),
        "recall@10": float(np.mean([recall(row, 10) for row in rows])),
        "recall@20": float(np.mean([recall(row, 20) for row in rows])),
        "candidate_recall@200": float(np.mean([recall(row, len(row["ranked"])) for row in rows])),
        "mean_k": mean_k,
        "median_k": float(np.median(choices)),
        "p90_k": float(np.percentile(choices, 90)),
        "min_k": int(min(choices)),
        "max_k": int(max(choices)),
        "choice_histogram": {str(k): int(choices.count(k)) for k in sorted(set(choices))},
    }


def by_source(rows: Sequence[Mapping[str, Any]], choices: Sequence[int], name: str) -> dict[str, Any]:
    result = {}
    for source in sorted({str(row["source"]) for row in rows}):
        indices = [i for i, row in enumerate(rows) if str(row["source"]) == source]
        result[source] = evaluate(
            [rows[i] for i in indices], [choices[i] for i in indices], name
        )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--candidate-depth", type=int, default=200)
    parser.add_argument(
        "--model-depth",
        type=int,
        default=50,
        help="Number of model positions; each position maps to a ranked-list bin.",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--model-scale", choices=("fast", "paper"), default="fast")
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("bicut", "choppy", "attncut"),
        default=["bicut", "choppy", "attncut"],
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    import torch
    from sklearn.model_selection import KFold

    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    rows = load_canonical_rows(args.candidate_depth)
    if len(rows) != 220:
        raise ValueError(f"Expected 220 queries, found {len(rows)}")
    features, rewards, cutoff_map = build_features(rows, args.model_depth)
    if features.shape[1] != args.model_depth:
        raise ValueError(f"Expected model depth {args.model_depth}, got {features.shape[1]}")

    splitter = KFold(n_splits=args.folds, shuffle=True, random_state=2026)
    choices_by_method = {method: np.zeros(len(rows), dtype=int) for method in args.models}
    fold_records: list[dict[str, Any]] = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(features), start=1):
        fold_record: dict[str, Any] = {
            "fold": fold,
            "train_queries": int(len(train_idx)),
            "test_queries": int(len(test_idx)),
            "methods": {},
        }
        for method in args.models:
            model = train_one(
                method,
                features[train_idx],
                rewards[train_idx],
                args.epochs,
                args.batch_size,
                args.learning_rate,
                args.seed + fold * 100 + len(method),
                args.device,
                args.model_scale,
            )
            choices = predict_choices(
                method, model, features[test_idx], cutoff_map, args.device
            )
            choices_by_method[method][test_idx] = choices
            fold_record["methods"][method] = {
                "mean_k": float(np.mean(choices)),
                "median_k": float(np.median(choices)),
                "min_k": int(np.min(choices)),
                "max_k": int(np.max(choices)),
            }
        fold_records.append(fold_record)
        print(f"completed fold {fold}/{args.folds}")

    metrics: list[dict[str, Any]] = []
    source_metrics: dict[str, Any] = {}
    policies: dict[str, list[int]] = {}
    # Fixed references use the same original ranking and are included so the
    # report can be copied directly into the adaptive-k comparison table.
    for fixed_k in (10, 20, 50, 100, 150):
        name = f"fixed_k{fixed_k}"
        choices = [fixed_k] * len(rows)
        policies[name] = choices
        metrics.append(evaluate(rows, choices, name))
        source_metrics[name] = by_source(rows, choices, name)
    for method, choices_array in choices_by_method.items():
        name = method
        choices = choices_array.astype(int).tolist()
        policies[name] = choices
        metrics.append(evaluate(rows, choices, name))
        source_metrics[name] = by_source(rows, choices, name)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol": {
            "benchmark": "Physics 80 + Industrial 75 + OHR-Bench 65",
            "queries": len(rows),
            "pages": 5091,
            "ranking": "PP-OCRv5 + Qwen3-Embedding-8B; 0.2 normalized BM25 + 0.8 normalized dense",
            "candidate_depth": args.candidate_depth,
            "model_depth": args.model_depth,
            "cutoff_map": cutoff_map.tolist(),
            "folds": args.folds,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "model_scale": args.model_scale,
            "qrels_used_for": "training-fold cutoff rewards/labels and final evaluation only",
            "qrels_not_used_for": "test-fold model training, feature construction, ranking, or OOF fold assignment",
            "transfer_caveat": "Models are re-trained from scratch on this benchmark; no Robust04/MQ2007 checkpoint or original document-statistics fields are transferred.",
        },
        "models": {
            "bicut": {
                "paper": "An Assumption-Free Approach to the Dynamic Truncation of Ranked Lists (ICTIR 2019)",
                "input": "score + adjacent score gap + normalized rank",
                "loss": "prefix cross-entropy to Continue before train-fold best-F1 cutoff and EOL at cutoff",
                "inference": "first predicted EOL, otherwise top-200",
            },
            "choppy": {
                "paper": "Choppy: Cut Transformers For Ranked List Truncation (SIGIR 2020)",
                "input": "normalized score only",
                "loss": "negative expected F1 under predicted cutoff distribution",
                "inference": "argmax cutoff distribution",
                "implementation_note": "batch_first=True equivalent; fast scale uses one 64-dimensional encoder layer, paper scale is available via --model-scale paper",
            },
            "attncut": {
                "paper": "Learning to Truncate Ranked Lists for Information Retrieval (AAAI 2021)",
                "input": "score + adjacent score gap + normalized rank",
                "loss": "RAML/reward distribution cross-entropy with tau=0.95",
                "inference": "argmax cutoff distribution",
                "implementation_note": "batch_first=True equivalent; fast scale uses a 64-dimensional encoder with a 32-unit BiLSTM, paper scale is available via --model-scale paper",
            },
        },
        "metrics": metrics,
        "by_source": source_metrics,
        "folds_detail": fold_records,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "per_query.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for method, choices in policies.items():
            for row, choice in zip(rows, choices):
                handle.write(
                    json.dumps(
                        {
                            "method": method,
                            "qid": row["qid"],
                            "source": row["source"],
                            "adaptive_k": int(choice),
                            "recall_at_k": recall(row, int(choice)),
                            "gold_pages": len(row["gold"]),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    lines = [
        "# Neural ranked-list truncation screen",
        "",
        "All neural rows are five-fold query-level OOF fits on the canonical 220-query benchmark.",
        "",
        "| Method | Recall at selected K | Matched fixed K | Fixed recall | Delta | Mean K | Median | P90 | Recall@10 | Recall@20 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for metric in metrics:
        lines.append(
            f"| {metric['name']} | {metric['recall_at_adaptive_k']:.2%} | {metric['matched_fixed_k']} | "
            f"{metric['matched_fixed_recall']:.2%} | {metric['delta_vs_matched_fixed']:+.2%} | "
            f"{metric['mean_k']:.2f} | {metric['median_k']:.0f} | {metric['p90_k']:.0f} | "
            f"{metric['recall@10']:.2%} | {metric['recall@20']:.2%} |"
        )
    lines += [
        "",
        "## By source",
        "",
        "| Method | Source | Queries | Recall at selected K | Matched K | Fixed recall | Delta | Mean K |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in policies:
        for source, metric in source_metrics[method].items():
            lines.append(
                f"| {method} | {source} | {metric['queries']} | {metric['recall_at_adaptive_k']:.2%} | "
                f"{metric['matched_fixed_k']} | {metric['matched_fixed_recall']:.2%} | "
                f"{metric['delta_vs_matched_fixed']:+.2%} | {metric['mean_k']:.2f} |"
            )
    lines += [
        "",
        "## Interpretation limits",
        "",
        "- This is a from-scratch transfer experiment, not a zero-shot reproduction of an author checkpoint.",
        "- The original papers use Robust04/MQ2007-style training data; this run substitutes the available score/gap/rank features and current binary page qrels.",
        "- OOF prevents a query's own qrels from selecting its model, but 220 queries remain a small training/evaluation sample.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(args.output_dir / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
